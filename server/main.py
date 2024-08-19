import sys
import os
import re
from typing import Optional
import uvicorn
import spacy
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from loguru import logger

from rpunct import RestorePuncts

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


def remove_bracketed_text(input_text):
    # Use regular expression to find and remove text within square brackets
    return re.sub(r'\[.*?\]', '', input_text)

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

def generate_chunks(texts, chunk_size=20, overlap=0):
    cleaned_texts = [remove_bracketed_text(text) for text in texts]
    for i in range(0, len(cleaned_texts), chunk_size - overlap):
        yield ' '.join(cleaned_texts[i:i + chunk_size])

# Use spaCy to segment the full text into sentences
nlp = spacy.load('en_core_web_trf')
rpunct = RestorePuncts()

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
    
    #full_text = ' '.join(segment.text for segment in input_data.transcript)
    #doc = nlp(full_text)
    #inputsentences = [sent.text for sent in doc.sents]
    #transcript_chunks = list(generate_chunks([segment.text for segment in input_data.transcript]))
    
    # Step 1: Generate the transcript chunks
    transcript_chunks = list(generate_chunks([segment.text for segment in input_data.transcript]))

    print('============transcript_chunks==============')
    for chunk in transcript_chunks:
        print("-----------")
        print(chunk)

    # Step 2: Punctuate each chunk
    punctuated_chunks = [rpunct.punctuate(chunk) for chunk in transcript_chunks]

    #docs = list(nlp.pipe(transcript_chunks))
    docs = list(nlp.pipe(punctuated_chunks))
    full_doc = spacy.tokens.Doc.from_docs(docs)

    segments = []
    for sentence in full_doc.sents:  # `sents` provides sentences from the `Doc`
        segments.extend([item.strip() for item in sentence.text.split(',')])


    print('============segments==============')
    for line in segments:
        print("-----------")
        print(line)

    sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
    #sentences = create_sentences(segments, MIN_WORDS=10, MAX_WORDS=50)
    chunks = create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
    chunks_text = [chunk['text'] for chunk in chunks]

    # Run Stage 1 Summarizing
    stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']
    # Split the titles and summaries
    stage_1_summaries = [e['summary'] for e in stage_1_outputs]
    stage_1_titles = [e['title'] for e in stage_1_outputs]
    num_1_chunks = len(stage_1_summaries)

    #print('============stage_1_summaries and stage_1_titles==============')
    #print(stage_1_summaries)
    #print(stage_1_titles)

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
            similarity = 1 - cosine(summary_embeds[row], summary_embeds[col])
            summary_similarity_matrix[row, col] = similarity
            summary_similarity_matrix[col, row] = similarity

    # Set num_topics to be 1/4 of the number of chunks, or 8, which ever is smaller
    num_topics = min(int(num_1_chunks / 4), 8)
    topics_out = get_topics(summary_similarity_matrix, num_topics = num_topics, bonus_constant = 0.2)
    chunk_topics = topics_out['chunk_topics']
    topics = topics_out['topics']

    out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250)
    stage_2_outputs = out['stage_2_outputs']
    stage_2_titles = [e['title'] for e in stage_2_outputs]
    stage_2_summaries = [e['summary'] for e in stage_2_outputs]
    final_summary = out['final_summary']
    return SummarizeOutput(stage_2_outputs=stage_2_outputs, final_summary=final_summary)

@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)