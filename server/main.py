import sys
import os
import re
from typing import Optional
import uvicorn
import spacy
import numpy as np
from loguru import logger
from rpunct import RestorePuncts
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from scipy.spatial.distance import cosine, pdist, squareform
from langchain_openai import OpenAIEmbeddings

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
from services.summarization import SummarizeOutput, SummarizeInput, TranscriptElement
from services.summarization import create_sentences, create_chunks, summarize_stage_1, get_topics, summarize_stage_2

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None

def compute_similarity_matrix2(embeddings):
    """Compute the cosine similarity matrix for a list of embeddings."""
    num_chunks = embeddings.shape[0]
    similarity_matrix = np.zeros((num_chunks, num_chunks))
    for row in range(num_chunks):
        for col in range(row, num_chunks):
            similarity = 1 - cosine(embeddings[row], embeddings[col])
            similarity_matrix[row, col] = similarity
            similarity_matrix[col, row] = similarity
    return similarity_matrix

def compute_similarity_matrix(embeddings):
    """Compute the cosine similarity matrix for a list of embeddings."""
    # Compute the pairwise cosine distances and convert them to similarity
    similarity_matrix = 1 - squareform(pdist(embeddings, metric='cosine'))
    return similarity_matrix

def remove_bracketed_text(input_text):
    # Use regular expression to find and remove text within square brackets
    return re.sub(r'\[.*?\]', '', input_text)

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

def process_transcript(input_data):
    # Step 1: Clean the full transcript
    full_text = ' '.join(segment.text for segment in input_data.transcript)
    cleaned_text = remove_bracketed_text(full_text)
    
    # Step 2: Punctuate the cleaned full text
    punctuated_text = rpunct.punctuate(cleaned_text)
    
    return punctuated_text

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
    punctuated_chunks = process_transcript(input_data)
    doc = nlp(punctuated_chunks)

    #docs = list(nlp.pipe(punctuated_chunks))
    #full_doc = spacy.tokens.Doc.from_docs(docs)
    #segments = []
    #for sentence in full_doc.sents:  # `sents` provides sentences from the `Doc`
    #    segments.extend([item.strip() for item in sentence.text.split(',')])
    #sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)

    sentences = []
    for sentence_num, sentence in enumerate(doc.sents):
        sentence_text = sentence.text.strip()
        sentence_length = len(sentence_text.split())
        sentences.append({
            'sentence_num': sentence_num,
            'text': sentence_text,
            'sentence_length': sentence_length
        })
    chunks = create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
    chunks_text = [chunk['text'] for chunk in chunks]

    # Run Stage 1 Summarizing
    stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']
    # Split the titles and summaries
    stage_1_summaries = [e['Rewrite'] for e in stage_1_outputs]
    stage_1_titles = [e['Title'] for e in stage_1_outputs]
    num_1_chunks = len(stage_1_summaries)

    # Use OpenAI to embed the summaries and titles. Size of _embeds: (num_chunks x 1536)
    openai_embed = OpenAIEmbeddings()

    summary_embeds = np.array(openai_embed.embed_documents(stage_1_summaries))
    title_embeds = np.array(openai_embed.embed_documents(stage_1_titles))

    summary_similarity_matrix = compute_similarity_matrix(summary_embeds)
    title_similarity_matrix = compute_similarity_matrix(title_embeds)

    # Set num_topics
    num_topics = min(int(num_1_chunks / 4), 20)
    topics_out = get_topics(summary_similarity_matrix, num_topics = num_topics, bonus_constant = 0.2)
    chunk_topics = topics_out['chunk_topics']
    topics = topics_out['topics']

    out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250)
    topic_summary_items = out['topic_summary_items']
    stage_2_titles = [e['Title'] for e in topic_summary_items]
    stage_2_summaries = [e['Rewrite'] for e in topic_summary_items]
    video_summary = out['video_summary']
    return SummarizeOutput(topic_summary_items=topic_summary_items, video_summary=video_summary)

@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)