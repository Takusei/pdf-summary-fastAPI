import asyncio
import time

from langchain_openai import AzureChatOpenAI, ChatOpenAI

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
    start = time.time()

    try:
        docs = load_file(file_path)

        use_method = choose_method(docs, method)
        print(f"Using summarization method: {use_method}")

        if use_method == "map-reduce":
            summary = summarize_with_map_reduce(docs, llm)
        else:
            summary = summarize_with_stuff(docs, llm)
    except Exception as e:
        summary = f"Error during summarization: {str(e)}"

    return summary, time.time() - start


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
