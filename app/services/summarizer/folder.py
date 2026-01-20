import asyncio
import os
import time
from pathlib import Path

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.cache.utils import (
    SAVED_SUMMARY_DB,
    get_json_from_cache,
    is_cache_file,
    save_json_to_cache,
)
from app.schemas.summarize import MultipleSummariesResponse
from app.services.summarizer.file import summarize_single_file_async


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
    db_path = path_obj / SAVED_SUMMARY_DB
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
            if is_cache_file(file):
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
        print("Summarizing files:", [f["file_path"] for f in files_to_summarize_meta])
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
    else:
        print("All summaries loaded from cache; no files needed summarization.")

    # 4. Create final response and save to cache
    total_duration = time.time() - start_time
    response = MultipleSummariesResponse(
        summaries=final_summaries, duration=total_duration
    )

    save_json_to_cache(db_path, "summaries", response.model_dump())
    return response
