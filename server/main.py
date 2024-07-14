import os
from typing import Optional
import uvicorn
import spacy
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from loguru import logger

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
)
from datastore.factory import get_datastore
from services.file import get_document_from_file

from models.models import DocumentMetadata, Source
import numpy as np
from scipy.spatial.distance import cosine
from langchain_openai import OpenAIEmbeddings
from services.summarization import SummarizeOutput, SummarizeInput, TranscriptElement
from services.summarization import create_sentences, create_chunks, summarize_stage_1, get_topics, summarize_stage_2

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

# Use spaCy to segment the full text into sentences
nlp = spacy.load('en_core_web_sm')

app = FastAPI(dependencies=[Depends(validate_token)])
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Retrieval Plugin API for ChatVector",
    description="A retrieval API for querying and filtering documents based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://chatvector.fly.dev"}],
    dependencies=[Depends(validate_token)],
)
app.mount("/sub", sub_app)


@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        metadata_obj = (
            DocumentMetadata.parse_raw(metadata)
            if metadata
            else DocumentMetadata(source=Source.file)
        )
    except:
        metadata_obj = DocumentMetadata(source=Source.file)

    document = await get_document_from_file(file, metadata_obj)

    try:
        ids = await datastore.upsert([document])
        return UpsertResponse(ids=ids)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"str({e})")


@app.post(
    "/upsert",
    response_model=UpsertResponse,
)
async def upsert(
    request: UpsertRequest = Body(...),
):
    try:
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_main(
    request: QueryRequest = Body(...),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@sub_app.post(
    "/query",
    response_model=QueryResponse,
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="Accepts search query objects array each with query and optional filter. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
)
async def query(
    request: QueryRequest = Body(...),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@app.post("/summarize/", response_model=SummarizeOutput)
async def summarize(input_data: SummarizeInput):
    
    full_text = ' '.join(segment.text for segment in input_data.transcript)

    doc = nlp(full_text)
    inputsentences = [sent.text for sent in doc.sents]

    segments = [sentence.split(',') for sentence in inputsentences]
    segments = [item.strip() for sublist in segments for item in sublist]

    sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
    chunks = create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
    chunks_text = [chunk['text'] for chunk in chunks]

    print('================sentences===========================')
    print(sentences)
    print('==================chunks=========================')
    print(chunks)
    # Run Stage 1 Summarizing
    stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']
    # Split the titles and summaries
    stage_1_summaries = [e['summary'] for e in stage_1_outputs]
    stage_1_titles = [e['title'] for e in stage_1_outputs]

    print('============stage_1_summaries and stage_1_titles==============')
    print(stage_1_summaries)
    print(stage_1_titles)
    num_1_chunks = len(stage_1_summaries)

    # Use OpenAI to embed the summaries and titles. Size of _embeds: (num_chunks x 1536)
    openai_embed = OpenAIEmbeddings()

    summary_embeds = np.array(openai_embed.embed_documents(stage_1_summaries))
    title_embeds = np.array(openai_embed.embed_documents(stage_1_titles))

    # Get similarity matrix between the embeddings of the chunk summaries
    summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
    summary_similarity_matrix[:] = np.nan
    for row in range(num_1_chunks):
        for col in range(row, num_1_chunks):
            # Calculate cosine similarity between the two vectors
            similarity = 1- cosine(summary_embeds[row], summary_embeds[col])
            summary_similarity_matrix[row, col] = similarity
            summary_similarity_matrix[col, row] = similarity

    # Set num_topics to be 1/4 of the number of chunks, or 8, which ever is smaller
    num_topics = min(int(num_1_chunks / 4), 8)
    topics_out = get_topics(summary_similarity_matrix, num_topics = num_topics, bonus_constant = 0.2)

    chunk_topics = topics_out['chunk_topics']
    topics = topics_out['topics']

    out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250)
    stage_2_outputs = out['stage_2_outputs']
    # stage_2_titles = [e['title'] for e in stage_2_outputs]
    # stage_2_summaries = [e['summary'] for e in stage_2_outputs]
    final_summary = out['final_summary']
    return SummarizeOutput(stage_2_outputs=stage_2_outputs, final_summary=final_summary)

@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)

def load_and_process_data():
  testdata = """[
    {
      text: 'uh hi guys so my name is nick herring',
      start: 0.56,
      duration: 3.28
    },
    { text: "i'm the technical director of", start: 2.72, duration: 2.8 },
    {
      text: 'infrastructure for eve online at ccp',
      start: 3.84,
      duration: 3.76
    },
    {
      text: 'games which is a very long title for',
      start: 5.52,
      duration: 4.72
    },
    {
      text: "cosmic plumber um we're going to be",
      start: 7.6,
      duration: 5.6
    },
    {
      text: 'going over kind of eve online',
      start: 10.24,
      duration: 4.399
    },
    {
      text: 'the last 20 years of development what',
      start: 13.2,
      duration: 3.6
    },
    {
      text: 'that looks like what we had to work on',
      start: 14.639,
      duration: 5.041
    },
    {
      text: 'to kind of modernize the eve online',
      start: 16.8,
      duration: 4.319
    },
    {
      text: 'tech stack at least from the perspective',
      start: 19.68,
      duration: 3.519
    },
    {
      text: 'of the server side and the network',
      start: 21.119,
      duration: 3.681
    },
    {
      text: 'portions of the client side',
      start: 23.199,
      duration: 3.361
    },
    { text: 'um', start: 24.8, duration: 4.08 },
    {
      text: "and we'll go over more of what the",
      start: 26.56,
      duration: 4.4
    },
    {
      text: 'topology is in the evolution of that',
      start: 28.88,
      duration: 4.24
    },
    {
      text: "topology kind of how you've originally",
      start: 30.96,
      duration: 4.16
    },
    {
      text: "started and how it's gone from",
      start: 33.12,
      duration: 3.04
    },
    { text: 'uh', start: 35.12, duration: 3.279 },
    {
      text: 'from 2003 to what we have now',
      start: 36.16,
      duration: 3.12
    },
    { text: 'and', start: 38.399, duration: 2.48 },
    {
      text: "we're going to talk about how we tried",
      start: 39.28,
      duration: 3.84
    },
    {
      text: 'to fundamentally or are fundamentally',
      start: 40.879,
      duration: 4
    },
    {
      text: 'changing how we actually work on eve',
      start: 43.12,
      duration: 3.279
    },
    { text: 'online', start: 44.879, duration: 2.881 },
    {
      text: "there's multiple pieces to that but the",
      start: 46.399,
      duration: 2.64
    },
    {
      text: 'two biggest ones being a technical',
      start: 47.76,
      duration: 3.68
    },
    {
      text: 'aspect and a cultural aspect',
      start: 49.039,
      duration: 4.16
    },
    {
      text: 'and the cultural aspect is a pretty big',
      start: 51.44,
      duration: 3.599
    },
    { text: 'part of it', start: 53.199, duration: 3.441 },
    { text: 'and hopefully', start: 55.039, duration: 3.121 },
    {
      text: "we don't have to go too fast here",
      start: 56.64,
      duration: 2.8
    },
    {
      text: 'because right after this we have a round',
      start: 58.16,
      duration: 3.76
    },
    {
      text: 'table um but the round table is more for',
      start: 59.44,
      duration: 4.24
    },
    {
      text: 'anything else so any kind of quasar',
      start: 61.92,
      duration: 3.279
    },
    {
      text: 'specific stuff we can talk about here',
      start: 63.68,
      duration: 3.119
    },
    {
      text: "hopefully at the end of this if there's",
      start: 65.199,
      duration: 4.321
    },
    {
      text: 'time for questions uh and then',
      start: 66.799,
      duration: 3.921
    },
    {
      text: 'afterwards in the round type we can talk',
      start: 69.52,
      duration: 3.52
    },
    {
      text: 'more about other things like quasar and',
      start: 70.72,
      duration: 3.759
    },
    {
      text: 'easy and how they interact and what',
      start: 73.04,
      duration: 3.52
    },
    {
      text: 'makes sense there and the any kind of',
      start: 74.479,
      duration: 3.68
    },
    {
      text: "other technology that we're using on the",
      start: 76.56,
      duration: 3.28
    },
    { text: 'server side', start: 78.159, duration: 3.121 },
    { text: 'so we can start with', start: 79.84, duration: 3.76 },
    {
      text: '20 years of eve development uh it was',
      start: 81.28,
      duration: 6.56
    },
    {
      text: 'released in 2003. you guys all know this',
      start: 83.6,
      duration: 6
    },
    {
      text: 'right now there are roughly over 2',
      start: 87.84,
      duration: 4.319
    },
    {
      text: 'million changeless in perforce',
      start: 89.6,
      duration: 4.8
    },
    {
      text: 'that number is probably growing faster',
      start: 92.159,
      duration: 3.521
    },
    {
      text: 'and faster as we add more and more',
      start: 94.4,
      duration: 3.28
    },
    {
      text: "automation into the ecosystem so there's",
      start: 95.68,
      duration: 3.439
    },
    {
      text: 'less and less humans actually making',
      start: 97.68,
      duration: 3.84
    },
    { text: 'changes to the code base', start: 99.119, duration: 3.36 },
    { text: 'and', start: 101.52, duration: 3.12 },
    {
      text: "we've kind of added a little reference",
      start: 102.479,
      duration: 4.64
    },
    {
      text: "of how much code there is and i've added",
      start: 104.64,
      duration: 5.2
    },
    {
      text: 'this silly reference of the ue4s so if',
      start: 107.119,
      duration: 4.401
    },
    {
      text: 'you take the code base of unreal engine',
      start: 109.84,
      duration: 2.559
    },
    { text: '4', start: 111.52, duration: 3.279 },
    {
      text: 'you can kind of get an idea of how much',
      start: 112.399,
      duration: 4.561
    },
    {
      text: 'of that code is is being used there it',
      start: 114.799,
      duration: 3.441
    },
    {
      text: "means absolutely nothing it's just fun",
      start: 116.96,
      duration: 2.24
    },
    { text: 'to think about', start: 118.24, duration: 2.32 },
    { text: 'um', start: 119.2, duration: 2.72 },
    {
      text: 'and so if we think about like the cec',
      start: 120.56,
      duration: 3.599
    },
    {
      text: "plus plus that's where a lot of the",
      start: 121.92,
      duration: 3.6
    },
    {
      text: "rendering code is that's where the",
      start: 124.159,
      duration: 3.041
    },
    { text: 'simulation code is', start: 125.52, duration: 3.599 },
    {
      text: "and that's a lot of where the the glue",
      start: 127.2,
      duration: 3.52
    },
    { text: 'is from uh', start: 129.119, duration: 3.441 },
    {
      text: 'c to python marshalling and back and',
      start: 130.72,
      duration: 3.28
    },
    { text: 'forth', start: 132.56, duration: 4.16 },
    {
      text: "so that's roughly 1.7 million lines of",
      start: 134,
      duration: 4.08
    },
    { text: 'code', start: 136.72, duration: 4.4 },
    {
      text: 'then the next up would be sql so a lot',
      start: 138.08,
      duration: 5.92
    },
    {
      text: 'of eve is run by basically sql procs a',
      start: 141.12,
      duration: 4.479
    },
    { text: 'lot of the logic is', start: 144, duration: 4.56 },
    { text: 'unfortunately um', start: 145.599, duration: 4.481 },
    { text: 'and so we can see that', start: 148.56, duration: 4.16 },
    {
      text: 'just our sql code alone is the size of',
      start: 150.08,
      duration: 5.28
    },
    { text: 'ue4', start: 152.72, duration: 2.64 },
    {
      text: 'and then if we continue on',
      start: 155.519,
      duration: 5.281
    },
    {
      text: 'where a bulk of the logic was written uh',
      start: 157.68,
      duration: 5.36
    },
    {
      text: 'in python in stackless python',
      start: 160.8,
      duration: 4.48
    },
    {
      text: "um we can see that there's 3.4 million",
      start: 163.04,
      duration: 4.16
    },
    {
      text: 'lines of code a little bit more than a',
      start: 165.28,
      duration: 3.28
    },
    { text: 'ue4', start: 167.2, duration: 2.72 },
    {
      text: "and then if you weren't worried enough",
      start: 168.56,
      duration: 2.88
    },
    { text: 'yet', start: 169.92, duration: 3.2 },
    { text: 'we have roughly', start: 171.44, duration: 4.4 },
    { text: '53 million lines of yaml', start: 173.12, duration: 4.08 },
    { text: 'um', start: 175.84, duration: 4.56 },
    {
      text: 'this is 24 unreal engines uh worth of',
      start: 177.2,
      duration: 8.16
    },
    {
      text: "code code's a strong word but um",
      start: 180.4,
      duration: 7.04
    },
    {
      text: 'this this is what holds everything to do',
      start: 185.36,
      duration: 4.64
    },
    {
      text: 'with uh how the universe is authored any',
      start: 187.44,
      duration: 4.4
    },
    {
      text: 'anywhere from how the spaceships are',
      start: 190,
      duration: 3.84
    },
    {
      text: 'made uh and authored as far the',
      start: 191.84,
      duration: 3.759
    },
    {
      text: "attributes are concerned and there's a",
      start: 193.84,
      duration: 3.679
    },
    {
      text: 'lot of work done there and and this this',
      start: 195.599,
      duration: 4.241
    }
  ]"""

  # Using eval to convert the data into Python objects
  # Notice the replacement of unquoted keys with quoted ones to make it a valid Python dict
  safe_data = testdata.replace('text:', '"text":').replace('start:', '"start":').replace('duration:', '"duration":')
  parsed_data = eval(safe_data)
  return [TranscriptElement(**item) for item in parsed_data]