""" Point d'entrée de l'API boulwe-backend. Lancer en local avec : uvicorn main:app --reload """
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from app.ingestion import ingest_document
from app.rag import answer_question
from app.models import IngestResponse, AskRequest, AskResponse
load_dotenv()  # charge les clés API depuis le fichier .env
app = FastAPI(
    title="Boulwe Backend",
    description="IA basée sur RAG : sources scientifiques fiables + projets humanitaires",
    version="0.1.0",
)
@app.get("/")
def root():
    return {"status": "ok", "service": "boulwe-backend"}
@app.post("/ingest", response_model=IngestResponse)
async def ingest( file: UploadFile = File(...), collection: str = Form(default="general"), ):
    """Indexe un document (PDF ou texte) dans une collection Qdrant."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")
file_bytes = await file.read()
    chunks_count = ingest_document(file_bytes, file.filename, collection=collection)
if chunks_count == 0:
        raise HTTPException(status_code=422, detail="Aucun texte exploitable trouvé dans le document.")
return IngestResponse(filename=file.filename, chunks_indexed=chunks_count, collection=collection)
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Pose une question à l'IA, qui répond en s'appuyant sur les documents indexés."""
    result = answer_question(request.question, collection=request.collection, top_k=request.top_k)
    return AskResponse(**result) 
