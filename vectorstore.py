"""
Connexion à Qdrant : création de collection, insertion et recherche de vecteurs.
"""

import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Taille des embeddings Voyage AI (voyage-3 -> 1024 dimensions)
EMBEDDING_DIM = 1024

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # None en local, requis si Qdrant Cloud

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(collection_name: str) -> None:
    """Crée la collection Qdrant si elle n'existe pas déjà."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def upsert_chunks(collection_name: str, chunks: list[str], embeddings: list[list[float]], source_document: str) -> int:
    """Insère une liste de chunks (texte + vecteur) dans une collection Qdrant."""
    ensure_collection(collection_name)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, "source_document": source_document},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search_chunks(collection_name: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Recherche les chunks les plus proches d'une question dans une collection."""
    ensure_collection(collection_name)

    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "source_document": hit.payload.get("source_document", "inconnu"),
            "score": hit.score,
        }
        for hit in results
    ]
