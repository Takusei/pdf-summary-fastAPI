import asyncio
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from app.llm.chains import (
    build_map_chain,
    build_reduce_chain,
    build_stuff_chain,
)
from app.services.chunking import split_docs
from app.services.file_loader import load_file


def get_file_name_from_docs(docs: list) -> str:
    if not docs:
        return "unknown"

    file_path = docs[0].metadata.get("file_path") or docs[0].metadata.get("source")

    if not file_path:
        return "unknown"

    return Path(file_path).name


def summarize_with_map_reduce(docs, llm: ChatOpenAI) -> str:
    chunks = split_docs(docs)

    # Filter out empty chunks and check if there's any content left
    non_empty_chunks = [d for d in chunks if d.page_content.strip()]
    if not non_empty_chunks:
        raise ValueError("No text to summarize after chunking")

    map_chain = build_map_chain(llm)
    reduce_chain = build_reduce_chain(llm)

    map_inputs = [{"text": d.page_content} for d in non_empty_chunks]
    map_summaries = map_chain.batch(map_inputs)

    combined = "\n".join(map_summaries)
    file_name = get_file_name_from_docs(docs)

    return reduce_chain.invoke({"text": combined, "file_name": file_name})


def summarize_with_stuff(docs, llm: ChatOpenAI) -> str:
    chain = build_stuff_chain(llm)
    file_name = get_file_name_from_docs(docs)

    text = "\n\n".join(d.page_content for d in docs)

    # In case of empty documents or scanned PDFs
    if not text.strip():
        raise ValueError("No text to summarize")
    return chain.invoke({"text": text, "file_name": file_name})


def choose_method(docs, method: str) -> str:
    if method == "map-reduce":
        return "map-reduce"
    if method == "stuff":
        return "stuff"
    # auto
    return "map-reduce" if len(docs) > 20 else "stuff"


def summarize_single_file(
    file_path: str,
    llm: ChatOpenAI,
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
    llm: ChatOpenAI,
    method: str = "auto",
):
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, summarize_single_file, file_path, llm, method
        )
