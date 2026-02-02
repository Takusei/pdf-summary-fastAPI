import asyncio
import os
import time
from pathlib import Path

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.cache.utils import (
    SAVED_SUMMARY_DB,
    VDR_DB_DIR,
    get_json_from_cache,
    is_cache_file,
    save_json_to_cache,
)
from app.core.logging import log_event
from app.schemas.summarize import MultipleSummariesResponse
from app.services.summarizer.file import summarize_single_file_async


async def summarize_folder(
    folder_path: str,
    regenerate: bool,
    sync: bool,
    llm: ChatOpenAI | AzureChatOpenAI,
) -> MultipleSummariesResponse:
    """
    Summarizes files in a folder with caching.
    - If regenerate=True: Forces regeneration of all summaries, ignoring any cache.
    - If sync=True: Intelligently updates the cache by summarizing only new or modified files.
    - If regenerate=False and sync=False: Returns cached data if it exists, otherwise generates all.
    """
    path_obj = Path(folder_path)
    db_dir = path_obj / VDR_DB_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / SAVED_SUMMARY_DB
    start_time = time.perf_counter()

    # --- Force regenerate: ignore cache ---
    if regenerate:
        log_event("summary_regenerate", folder_path=folder_path)
        # The logic will proceed to summarize all files as if no cache exists.
        pass
    # --- Fast Cache check (if not regenerating or syncing) ---
    elif not sync:
        cached_data = get_json_from_cache(db_path, "summaries")
        if cached_data:
            log_event("summary_cache_hit", folder_path=folder_path)
            return MultipleSummariesResponse(**cached_data)
        # If no cache, fall through to the full generation logic below

    # --- Sync (Smart Update) or Initial Generation ---
    cached_summaries_map = {}
    if sync and not regenerate:
        cached_data = get_json_from_cache(db_path, "summaries")
        if cached_data:
            cached_response = MultipleSummariesResponse(**cached_data)
            for summary_item in cached_response.summaries:
                cached_summaries_map[summary_item.file_path] = summary_item

    # 1. Get the current state of files on disk
    current_files_meta = {}
    for root, dirs, files in os.walk(folder_path):
        if VDR_DB_DIR in dirs:
            dirs.remove(VDR_DB_DIR)
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

    if regenerate:
        # If regenerating, all current files must be summarized.
        files_to_summarize_meta = list(current_files_meta.values())
    else:
        for path, meta in current_files_meta.items():
            cached_item = cached_summaries_map.get(path)
            # Summarize if:
            # - No cached item exists.
            # - We are in sync mode and the file is modified.
            if not cached_item or (
                sync and cached_item.last_modified_time != meta["last_modified_time"]
            ):
                files_to_summarize_meta.append(meta)
            else:
                # This file is unchanged, reuse the cached summary
                final_summaries.append(cached_item.model_dump())

    # 3. Summarize only the necessary files
    if files_to_summarize_meta:
        log_event(
            "summary_batch_start",
            folder_path=folder_path,
            file_count=len(files_to_summarize_meta),
        )
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
        log_event("summary_noop", folder_path=folder_path)

    # 4. Create final response and save to cache
    total_duration = time.perf_counter() - start_time
    log_event(
        "summary_batch",
        duration_s=total_duration,
        folder_path=folder_path,
        file_count=len(final_summaries),
        regenerate=regenerate,
        sync=sync,
    )
    response = MultipleSummariesResponse(
        summaries=final_summaries, duration=total_duration
    )

    save_json_to_cache(db_path, "summaries", response.model_dump())
    return response
