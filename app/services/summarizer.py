import asyncio
import os
import time
from pathlib import Path

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.llm.chains import (
    build_map_chain,
    build_reduce_chain,
    build_stuff_chain,
)
from app.schemas.summarize import MultipleSummariesResponse
from app.services.cache_utils import get_json_from_cache, save_json_to_cache
from app.services.chunking import split_docs
from app.services.file_loader import load_file

DB_FILE_NAME = ".summarycache.db"


def get_file_name_from_docs(docs: list) -> str:
    if not docs:
        return "unknown"

    file_path = docs[0].metadata.get("file_path") or docs[0].metadata.get("source")

    if not file_path:
        return "unknown"

    return Path(file_path).name


def summarize_with_map_reduce(docs, llm: ChatOpenAI | AzureChatOpenAI) -> str:
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


def summarize_with_stuff(docs, llm: ChatOpenAI | AzureChatOpenAI) -> str:
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


async def summarize_folder(
    folder_path: str,
    regenerate: bool,
    llm: ChatOpenAI | AzureChatOpenAI,
) -> MultipleSummariesResponse:
    """
    Summarizes files in a folder with caching.
    - If regenerate=False: Returns cached data if it exists, otherwise generates all.
    - If regenerate=True: Intelligently updates the cache by summarizing only new or modified files.
    """
    path_obj = Path(folder_path)
    db_path = path_obj / DB_FILE_NAME
    start_time = time.time()

    # --- Regenerate=False: Fast Cache or Full Generation ---
    if not regenerate:
        cached_data = get_json_from_cache(db_path, "summaries")
        if cached_data:
            return MultipleSummariesResponse(**cached_data)
        # If no cache, fall through to the full generation logic below

    # --- Regenerate=True: Smart Update ---
    # Or initial generation if cache was empty
    cached_summaries_map = {}

    # We load cache if we are in smart update mode (regenerate=True) OR if we just missed a cache hit above (implicit in flow)
    # However, if regenerate=False and we are here, it means cache didn't exist, so map is empty.
    # If regenerate=True, we WANT to load the cache to diff against it.
    if regenerate:
        cached_data = get_json_from_cache(db_path, "summaries")
        if cached_data:
            cached_response = MultipleSummariesResponse(**cached_data)
            for summary_item in cached_response.summaries:
                cached_summaries_map[summary_item.file_path] = summary_item

    # 1. Get the current state of files on disk
    current_files_meta = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == DB_FILE_NAME:
                continue
            file_path = Path(root) / file
            try:
                stat = file_path.stat()
                file_type = "directory" if file_path.is_dir() else file_path.suffix
                current_files_meta[str(file_path)] = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": stat.st_size,
                    "last_modified_time": stat.st_mtime,
                    "file_type": file_type,
                }
            except FileNotFoundError:
                continue

    # 2. Decide which files to summarize
    files_to_summarize_meta = []
    final_summaries = []

    for path, meta in current_files_meta.items():
        cached_item = cached_summaries_map.get(path)
        # Summarize if:
        # - We are doing a full generation (cache was empty)
        # - We are in smart update mode AND the item is new or modified
        if not cached_item or (
            regenerate and cached_item.last_modified_time != meta["last_modified_time"]
        ):
            files_to_summarize_meta.append(meta)
        else:
            # This file is unchanged, reuse the cached summary
            final_summaries.append(cached_item.model_dump())

    # 3. Summarize only the necessary files
    if files_to_summarize_meta:
        semaphore = asyncio.Semaphore(10)
        tasks = [
            summarize_single_file_async(
                file_meta["file_path"], semaphore, llm, method="stuff"
            )
            for file_meta in files_to_summarize_meta
        ]
        new_results = await asyncio.gather(*tasks)

        for i, file_meta in enumerate(files_to_summarize_meta):
            summary, duration = new_results[i]
            summary_data = {
                **file_meta,
                "summary": summary,
                "duration": duration,
            }
            final_summaries.append(summary_data)

    # 4. Create final response and save to cache
    total_duration = time.time() - start_time
    response = MultipleSummariesResponse(
        summaries=final_summaries, duration=total_duration
    )

    save_json_to_cache(db_path, "summaries", response.model_dump())
    return response
