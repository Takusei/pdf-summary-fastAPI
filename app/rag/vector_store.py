from __future__ import annotations

from pathlib import Path
from typing import Union

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.rag.config import COLLECTION, DB_DIR, OPENAI_EMBEDDINGS_MODEL


def initialize_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL)


def get_vector_store(folder: Union[str, Path, None] = None) -> Chroma:
    embeddings = initialize_embeddings()
    if folder is None:
        persist_directory = DB_DIR
    else:
        persist_directory = str(Path(folder) / DB_DIR)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
