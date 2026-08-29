"""
Génération d'embeddings via Voyage AI (utilisés pour l'indexation et la recherche).
"""

import os
import voyageai

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
EMBEDDING_MODEL = "voyage-3"

vo = voyageai.Client(api_key=VOYAGE_API_KEY)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Génère les embeddings pour une liste de chunks à indexer."""
    result = vo.embed(texts, model=EMBEDDING_MODEL, input_type="document")
    return result.embeddings


def embed_query(text: str) -> list[float]:
    """Génère l'embedding d'une question posée par l'utilisateur."""
    result = vo.embed([text], model=EMBEDDING_MODEL, input_type="query")
    return result.embeddings[0]
