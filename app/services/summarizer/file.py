import asyncio
import time

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.core.logging import log_event
from app.services.file_loader import load_file
from app.services.summarizer.utils import (
    choose_method,
    summarize_with_map_reduce,
    summarize_with_stuff,
)


def summarize_single_file(
    file_path: str,
    llm: ChatOpenAI | AzureChatOpenAI,
    method: str = "auto",
) -> tuple[str, float]:
    start = time.perf_counter()

    try:
        docs = load_file(file_path)

        use_method = choose_method(docs, method)
        log_event("summary_method", file_path=file_path, method=use_method)

        if use_method == "map-reduce":
            summary = summarize_with_map_reduce(docs, llm)
        else:
            summary = summarize_with_stuff(docs, llm)
    except Exception as e:
        summary = f"Error during summarization: {str(e)}"
        log_event("summary_error", file_path=file_path, error=str(e))

    duration = time.perf_counter() - start
    log_event("summary_file", duration_s=duration, file_path=file_path)
    return summary, duration


async def summarize_single_file_async(
    file_path: str,
    semaphore: asyncio.Semaphore,
    llm: ChatOpenAI | AzureChatOpenAI,
    method: str = "auto",
):
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, summarize_single_file, file_path, llm, method
        )
