from __future__ import annotations

import time
from pathlib import Path

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.cache.utils import (
    SAVED_SUMMARY_DB,
    VDR_DB_DIR,
    get_json_from_cache,
    save_json_to_cache,
)
from app.core.logging import log_event
from app.rag import indexer
from app.rag.loaders import load_file
from app.rag.splitter import get_splitter
from app.rag.vector_store import get_vector_store
from app.schemas.rag import IndexFolderResponse
from app.schemas.summarize import MultipleSummariesResponse, SingleSummaryResponse
from app.schemas.summary_and_index import SummaryAndIndexResponse
from app.services.summarizer.file import summarize_docs


def _build_file_meta(path: Path) -> dict:
    stat = path.stat()
    file_type = "directory" if path.is_dir() else path.suffix
    return {
        "file_path": str(path),
        "file_name": path.name,
        "file_size": stat.st_size,
        "last_modified_time": stat.st_mtime,
        "file_type": file_type,
    }


def summarize_and_index_folder(
    folder_path: str,
    regenerate: bool,
    sync: bool,
    llm: ChatOpenAI | AzureChatOpenAI,
) -> SummaryAndIndexResponse:
    folder = Path(folder_path)
    db_dir = folder / VDR_DB_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / SAVED_SUMMARY_DB

    start_total = time.perf_counter()
    summary_duration = 0.0

    cached_summaries_map: dict[str, SingleSummaryResponse] = {}
    cached_response = None
    if not regenerate:
        cached_data = get_json_from_cache(db_path, "summaries")
        if cached_data:
            cached_response = MultipleSummariesResponse(**cached_data)
            for summary_item in cached_response.summaries:
                cached_summaries_map[summary_item.file_path] = summary_item
            if not sync:
                summary_duration = cached_response.duration

    vector_store = get_vector_store(folder)
    splitter = get_splitter()

    added = 0
    updated = 0
    skipped = 0
    errors = 0

    final_summaries: list[dict] = []

    paths, current_sources = indexer.collect_paths(folder)

    for path in paths:
        source = str(path.resolve())
        try:
            meta = _build_file_meta(path)
        except FileNotFoundError:
            continue

        cached_item = cached_summaries_map.get(meta["file_path"])
        needs_summary = (
            regenerate
            or (
                sync
                and (
                    not cached_item
                    or cached_item.last_modified_time != meta["last_modified_time"]
                )
            )
            or (not sync and not regenerate and not cached_item)
        )

        log_event("index_start", source=source)
        existing = indexer.get_existing_for_source(vector_store, source)
        needs_index = indexer.should_update(
            existing, meta["last_modified_time"], regenerate
        )

        if not needs_summary and not needs_index:
            log_event("index_skip", source=source)
            skipped += 1
            if cached_item:
                final_summaries.append(cached_item.model_dump())
            continue

        try:
            docs = load_file(path)
        except Exception as exc:
            docs = []
            log_event("file_load_error", source=source, error=str(exc))

        if needs_summary:
            summary_start = time.perf_counter()
            try:
                summary = summarize_docs(docs, str(path), llm, method="stuff")
            except Exception as exc:
                summary = f"Error during summarization: {str(exc)}"
                log_event("summary_error", file_path=str(path), error=str(exc))
            summary_elapsed = time.perf_counter() - summary_start
            summary_duration += summary_elapsed
            log_event(
                "summary_file",
                duration_s=summary_elapsed,
                file_path=str(path),
            )

            summary_data = {
                **meta,
                "summary": summary,
                "duration": summary_elapsed,
            }
            final_summaries.append(summary_data)
        elif cached_item:
            final_summaries.append(cached_item.model_dump())

        if not needs_index:
            log_event("index_skip", source=source)
            skipped += 1
            continue

        log_event("index_update", source=source)
        index_start = time.perf_counter()
        try:
            indexed = indexer.upsert_docs(vector_store, splitter, path, docs, existing)
            index_elapsed = time.perf_counter() - index_start
            log_event(
                "index_file_duration",
                duration_s=index_elapsed,
                source=source,
                indexed=indexed,
            )
        except Exception as exc:
            index_elapsed = time.perf_counter() - index_start
            errors += 1
            log_event("index_error", source=source, error=str(exc))
            log_event(
                "index_file_duration",
                duration_s=index_elapsed,
                source=source,
                error=str(exc),
            )
            continue

        if not indexed:
            skipped += 1
            continue

        if existing["ids"]:
            updated += 1
        else:
            added += 1

    deleted_sources = indexer.delete_removed_sources(vector_store, current_sources)

    summary_response = MultipleSummariesResponse(
        summaries=final_summaries, duration=summary_duration
    )
    save_json_to_cache(db_path, "summaries", summary_response.model_dump())

    index_response = IndexFolderResponse(
        folder_path=str(folder),
        added=added,
        updated=updated,
        skipped=skipped,
        deleted=len(deleted_sources),
        duration=time.perf_counter() - start_total,
    )

    total_duration = time.perf_counter() - start_total
    log_event(
        "summary_and_index",
        duration_s=total_duration,
        folder_path=str(folder),
        added=added,
        updated=updated,
        deleted=len(deleted_sources),
        skipped=skipped,
        errors=errors,
        regenerate=regenerate,
        sync=sync,
    )

    return SummaryAndIndexResponse(
        folder_path=str(folder),
        summaries=summary_response.summaries,
        summary_duration=summary_duration,
        index=index_response,
        duration=total_duration,
    )
