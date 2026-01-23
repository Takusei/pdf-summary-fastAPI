from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.rag.config import COLLECTION, DB_DIR, OPENAI_EMBEDDINGS_MODEL


def initialize_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL)


def get_vector_store() -> Chroma:
    embeddings = initialize_embeddings()
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
