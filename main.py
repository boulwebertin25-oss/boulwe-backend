"""
Point d'entrée de l'API boulwe-backend.

Lancer en local avec :
    uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from ingestion import ingest_document
from rag import answer_question
from models import IngestResponse, AskRequest, AskResponse


load_dotenv()  # charge les clés API depuis .env

app = FastAPI(
    title="Boulwe Backend",
    description="IA basée sur RAG : sources internes",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"status": "ok", "service": "boulwe-backend"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    collection: str = Form(default="general"),
):
    """Indexe un document (PDF ou texte)"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    file_bytes = await file.read()
    chunks_count = ingest_document(file_bytes, file.filename, collection)

    if chunks_count == 0:
        raise HTTPException(status_code=400, detail="Aucun contenu indexé")

    return IngestResponse(filename=file.filename, chunks_indexed=chunks_count)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Pose une question à l'IA, qui répond à partir des documents indexés"""
    result = answer_question(request.question)
    return AskResponse(**result)
