import asyncio
import os
import time

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
    file_path = request.file_path
    summary, duration = summarize_single_file(file_path, llm=llm_model, method="stuff")
    return {"file_path": file_path, "summary": summary, "duration": duration}


@router.post(
    "/folder",
    response_model=MultipleSummariesResponse,
)
async def summarize_folder_endpoint(request: FolderPathRequest):
    """
    Summarizes all files in a given folder path recursively and in parallel using asyncio.
    """
    start_time = time.time()
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(request.folder_path)
        for file in files
    ]
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(10)

    # Create a list of coroutines for summarizing each file
    tasks = [
        summarize_single_file_async(file_path, semaphore, llm_model, method="stuff")
        for file_path in all_files
    ]

    # Run all summarization tasks concurrently
    results = await asyncio.gather(*tasks)

    summaries = [
        {
            "file_path": all_files[i],
            "summary": results[i][0],  # Unpack summary
            "duration": results[i][1],  # Unpack duration
        }
        for i in range(len(all_files))
    ]
    total_duration = time.time() - start_time
    return {"summaries": summaries, "duration": total_duration}
