"""
Logique RAG : recherche des passages pertinents puis génération de la réponse
avec Gemini, en forçant l'ancrage sur les sources fournies.
"""

import os
import google.generativeai as genai

from app.embeddings import embed_query
from app.vectorstore import search_chunks
from app.models import SourceChunk

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """Tu es un assistant qui répond UNIQUEMENT à partir des sources fournies ci-dessous.

Règles strictes :
- Si la réponse ne se trouve pas dans les sources, dis clairement que tu ne disposes pas de cette information dans les documents indexés. N'invente jamais de réponse.
- Cite la source (nom du document) pour chaque affirmation importante.
- Sois précis, concis et rigoureux, comme pour une recherche scientifique ou un rapport humanitaire.
"""

model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    system_instruction=SYSTEM_PROMPT,
)


def build_context(chunks: list[dict]) -> str:
    """Assemble les passages retrouvés en un contexte lisible pour le prompt."""
    if not chunks:
        return "Aucune source pertinente trouvée."

    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Source {i} - {c['source_document']}]\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(question: str, collection: str = "general", top_k: int = 5) -> dict:
    """
    Pipeline complet : embedding de la question -> recherche Qdrant ->
    génération de la réponse par Gemini ancrée sur les sources.
    """
    query_embedding = embed_query(question)
    chunks = search_chunks(collection, query_embedding, top_k=top_k)
    context = build_context(chunks)

    user_message = f"Sources disponibles :\n\n{context}\n\nQuestion : {question}"

    response = model.generate_content(user_message)

    answer_text = response.text

    sources = [
        SourceChunk(text=c["text"], source_document=c["source_document"], score=c["score"])
        for c in chunks
    ]

    return {"answer": answer_text, "sources": sources, "collection": collection}
