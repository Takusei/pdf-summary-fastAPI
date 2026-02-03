from pathlib import Path

from fastapi import APIRouter

from app.core.logging import log_base_dir
from app.llm.models import initialize_model
from app.schemas.summarize import (
    FilePathRequest,
    FolderPathRequest,
    MultipleSummariesResponse,
    SingleSummaryResponse,
)
from app.services.summarizer.file import summarize_single_file
from app.services.summarizer.folder import summarize_folder

router = APIRouter()

llm_model = initialize_model()


@router.post("/file", response_model=SingleSummaryResponse)
async def summarize_file_endpoint(request: FilePathRequest):
    """
    Summarizes a single file from its path.
    """
    file_path = Path(request.file_path)
    with log_base_dir(file_path.parent):
        summary, duration = summarize_single_file(
            str(file_path),
            llm=llm_model,
            method="stuff",
            base_dir=str(file_path.parent),
        )
    stat = file_path.stat()
    file_type = "directory" if file_path.is_dir() else file_path.suffix
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_size": stat.st_size,
        "last_modified_time": stat.st_mtime,
        "file_type": file_type,
        "summary": summary,
        "duration": duration,
    }


@router.post(
    "/folder",
    response_model=MultipleSummariesResponse,
)
async def summarize_folder_endpoint(request: FolderPathRequest):
    """
    Summarizes all files in a given folder path recursively and in parallel using asyncio.
    """
    with log_base_dir(request.folder_path):
        return await summarize_folder(
            folder_path=request.folder_path,
            regenerate=request.regenerate,
            sync=request.sync,
            llm=llm_model,
            base_dir=request.folder_path,
        )
