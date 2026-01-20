import asyncio
import time

from langchain_openai import ChatOpenAI

from app.llm.chains import (
    build_map_chain,
    build_reduce_chain,
    build_stuff_chain,
)
from app.services.chunking import split_docs
from app.services.file_loader import load_pdf


def summarize_with_map_reduce(docs, llm: ChatOpenAI) -> str:
    chunks = split_docs(docs)

    map_chain = build_map_chain(llm)
    reduce_chain = build_reduce_chain(llm)

    map_inputs = [{"text": d.page_content} for d in chunks]
    map_summaries = map_chain.batch(map_inputs)

    combined = "\n".join(map_summaries)

    return reduce_chain.invoke({"text": combined})


def summarize_with_stuff(docs, llm: ChatOpenAI) -> str:
    chain = build_stuff_chain(llm)

    text = "\n\n".join(d.page_content for d in docs)
    return chain.invoke({"text": text})


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

    docs = load_pdf(file_path)

    use_method = choose_method(docs, method)
    print(f"Using summarization method: {use_method}")

    if use_method == "map-reduce":
        summary = summarize_with_map_reduce(docs, llm)
    else:
        summary = summarize_with_stuff(docs, llm)

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
