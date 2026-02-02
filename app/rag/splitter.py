from __future__ import annotations

import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import log_event
from app.rag.config import CHUNK_OVERLAP, CHUNK_SIZE


class TimedRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def split_documents(self, documents):
        start = time.perf_counter()
        chunks = super().split_documents(documents)
        log_event(
            "chunking",
            duration_s=time.perf_counter() - start,
            doc_count=len(documents),
            chunk_count=len(chunks),
        )
        return chunks


def get_splitter() -> RecursiveCharacterTextSplitter:
    return TimedRecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
