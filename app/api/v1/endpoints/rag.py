from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.rag.agent import answer_question
from app.rag.indexer import index_folder
from app.schemas.rag import (
    IndexFolderRequest,
    IndexFolderResponse,
    RagQueryRequest,
    RagQueryResponse,
)

router = APIRouter()


@router.post("/index", response_model=IndexFolderResponse)
async def index_folder_endpoint(request: IndexFolderRequest):
    """
    Index all supported files under a folder and store embeddings in Chroma.
    """
    folder_path = Path(request.folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(
            status_code=400, detail="folder_path must be an existing directory"
        )

    start = time.perf_counter()
    result = index_folder(folder_path, regenerate=request.regenerate)
    duration = time.perf_counter() - start

    return {
        "folder_path": str(folder_path),
        "added": result["added"],
        "updated": result["updated"],
        "skipped": result["skipped"],
        "duration": duration,
    }


@router.post("/query", response_model=RagQueryResponse)
async def rag_query_endpoint(request: RagQueryRequest):
    """
    Ask a question and retrieve relevant chunks using a RAG agent.
    """
    folder_path = Path(request.folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(
            status_code=400, detail="folder_path must be an existing directory"
        )

    start = time.perf_counter()
    answer, sources = answer_question(
        request.question, folder=str(folder_path), k=request.top_k
    )
    duration = time.perf_counter() - start

    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in sources
        ],
        "duration": duration,
    }
