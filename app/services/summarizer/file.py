import asyncio
import time

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.core.logging import log_base_dir, log_event
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
    base_dir: str | None = None,
) -> tuple[str, float]:
    start = time.perf_counter()
    if base_dir:
        with log_base_dir(base_dir):
            return _summarize_single_file(file_path, llm, method, start)
    return _summarize_single_file(file_path, llm, method, start)


def _summarize_single_file(
    file_path: str,
    llm: ChatOpenAI | AzureChatOpenAI,
    method: str,
    start: float,
) -> tuple[str, float]:
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
    base_dir: str | None = None,
):
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, summarize_single_file, file_path, llm, method, base_dir
        )
