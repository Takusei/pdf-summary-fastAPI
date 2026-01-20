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
from app.services.summarizer import summarize_single_pdf, summarize_single_pdf_async

router = APIRouter()

llm_model = initialize_model()


@router.post("/api/files/summary", response_model=SingleSummaryResponse)
async def summarize_file_endpoint(request: FilePathRequest):
    """
    Summarizes a single file from its path.
    """
    file_path = request.file_path
    summary, duration = summarize_single_pdf(file_path, llm=llm_model)
    return {"file_path": file_path, "summary": summary, "duration": duration}


@router.post(
    "/api/folders/summary",
    response_model=MultipleSummariesResponse,
)
async def summarize_folder_endpoint(request: FolderPathRequest):
    """
    Summarizes all files in a given folder path recursively and in parallel using asyncio.
    """
    start_time = time.time()
    pdf_files = []
    for root, _, files in os.walk(request.folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(10)

    # Create a list of coroutines for summarizing each file
    tasks = [
        summarize_single_pdf_async(file_path, semaphore, llm_model)
        for file_path in pdf_files
    ]

    # Run all summarization tasks concurrently
    results = await asyncio.gather(*tasks)

    summaries = [
        {
            "file_path": pdf_files[i],
            "summary": results[i][0],  # Unpack summary
            "duration": results[i][1],  # Unpack duration
        }
        for i in range(len(pdf_files))
    ]
    total_duration = time.time() - start_time
    return {"summaries": summaries, "duration": total_duration}
