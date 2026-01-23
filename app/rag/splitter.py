from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
