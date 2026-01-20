import asyncio
import os
import time
from pathlib import Path

from fastapi import APIRouter

from app.llm.models import initialize_model
from app.schemas.summarize import (
    FilePathRequest,
    FolderPathRequest,
    MultipleSummariesResponse,
    SingleSummaryResponse,
)
from app.services.summarizer import summarize_single_file, summarize_single_file_async

router = APIRouter()

llm_model = initialize_model()


@router.post("/file", response_model=SingleSummaryResponse)
async def summarize_file_endpoint(request: FilePathRequest):
    """
    Summarizes a single file from its path.
    """
    file_path = Path(request.file_path)
    summary, duration = summarize_single_file(
        str(file_path), llm=llm_model, method="stuff"
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
    start_time = time.time()
    all_files_meta = []
    for root, _, files in os.walk(request.folder_path):
        for file in files:
            file_path = Path(root) / file
            try:
                stat = file_path.stat()
                file_type = "directory" if file_path.is_dir() else file_path.suffix
                all_files_meta.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_size": stat.st_size,
                        "last_modified_time": stat.st_mtime,
                        "file_type": file_type,
                    }
                )
            except FileNotFoundError:
                # Handle cases where file might be deleted during the walk
                continue

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(10)

    # Create a list of coroutines for summarizing each file
    tasks = [
        summarize_single_file_async(
            file_meta["file_path"], semaphore, llm_model, method="stuff"
        )
        for file_meta in all_files_meta
    ]

    # Run all summarization tasks concurrently
    results = await asyncio.gather(*tasks)

    summaries = [
        {
            **all_files_meta[i],
            "summary": results[i][0],  # Unpack summary
            "duration": results[i][1],  # Unpack duration
        }
        for i in range(len(all_files_meta))
    ]
    total_duration = time.time() - start_time
    return {"summaries": summaries, "duration": total_duration}
