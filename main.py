import asyncio
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from libs.agent import initialize_model
from libs.summary import summarize_single_pdf, summarize_single_pdf_async

app = FastAPI()
llm_model = initialize_model()


class SummaryResponse(BaseModel):
    file_path: str
    summary: str
    duration: float


class FilePathRequest(BaseModel):
    file_path: str


class FolderPathRequest(BaseModel):
    folder_path: str


@app.post("/summarize-file/", response_model=SummaryResponse)
async def summarize_file_endpoint(request: FilePathRequest):
    """
    Summarizes a single PDF file from its path.
    """
    file_path = request.file_path
    summary, duration = summarize_single_pdf(file_path, llm=llm_model)
    return {"file_path": file_path, "summary": summary, "duration": duration}


@app.post("/summarize-folder/")
async def summarize_folder_endpoint(request: FolderPathRequest):
    """
    Summarizes all PDF files in a given folder path recursively and in parallel using asyncio.
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
    return {"summaries": summaries, "total_duration": total_duration}
