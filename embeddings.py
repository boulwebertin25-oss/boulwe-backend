"""
Génération d'embeddings via Gemini (utilisé pour la recherche RAG).
"""

import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

genai.configure(api_key=GEMINI_API_KEY)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Génère les embeddings pour une liste de textes."""
    result = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
    return result["embedding"]


def embed_query(text: str) -> list[float]:
    """Génère l'embedding d'une question."""
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return result["embedding"]
