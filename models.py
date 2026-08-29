"""
Schémas Pydantic utilisés par l'API (validation des entrées/sorties).
"""

from pydantic import BaseModel, Field
from typing import Optional


class IngestResponse(BaseModel):
    """Réponse renvoyée après l'indexation d'un document."""
    filename: str
    chunks_indexed: int
    collection: str


class AskRequest(BaseModel):
    """Requête pour poser une question à l'IA."""
    question: str = Field(..., min_length=3, description="La question posée par l'utilisateur")
    collection: Optional[str] = Field(
        default="general",
        description="Domaine de recherche : 'general', 'scientifique', 'humanitaire', etc."
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Nombre de passages sources à récupérer")


class SourceChunk(BaseModel):
    """Un passage source retrouvé et utilisé pour répondre."""
    text: str
    source_document: str
    score: float


class AskResponse(BaseModel):
    """Réponse finale envoyée à l'utilisateur."""
    answer: str
    sources: list[SourceChunk]
    collection: str
