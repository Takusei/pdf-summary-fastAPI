from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.rag.indexer import index_folder
from app.schemas.rag import IndexFolderRequest, IndexFolderResponse

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
    result = index_folder(folder_path)
    duration = time.perf_counter() - start

    return {
        "folder_path": str(folder_path),
        "added": result["added"],
        "updated": result["updated"],
        "skipped": result["skipped"],
        "duration": duration,
    }
