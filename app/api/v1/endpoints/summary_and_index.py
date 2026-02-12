from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.logging import log_base_dir
from app.llm.models import initialize_model
from app.schemas.summary_and_index import (
    SummaryAndIndexRequest,
    SummaryAndIndexResponse,
)
from app.services.summary_and_index import summarize_and_index_folder

router = APIRouter()

llm_model = initialize_model()


@router.post("/folder", response_model=SummaryAndIndexResponse)
async def summarize_and_index_endpoint(request: SummaryAndIndexRequest):
    """
    Summarize and index all supported files under a folder in one pass.
    """
    folder_path = Path(request.folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(
            status_code=400, detail="folder_path must be an existing directory"
        )

    with log_base_dir(folder_path):
        return summarize_and_index_folder(
            folder_path=str(folder_path),
            regenerate=request.regenerate,
            sync=request.sync,
            llm=llm_model,
        )
