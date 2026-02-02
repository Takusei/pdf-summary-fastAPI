import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import log_event

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


def split_docs(docs: list[Document]) -> list[Document]:
    start = time.perf_counter()
    chunks = _splitter.split_documents(docs)
    log_event(
        "chunking",
        duration_s=time.perf_counter() - start,
        doc_count=len(docs),
        chunk_count=len(chunks),
    )
    return chunks
