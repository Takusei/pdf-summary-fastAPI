from __future__ import annotations

import time
from pathlib import Path

from langchain_community.vectorstores.utils import filter_complex_metadata

from app.core.logging import log_event
from app.rag.loaders import iter_supported_files, load_file
from app.rag.splitter import get_splitter
from app.rag.vector_store import get_vector_store


def _collect_paths(folder: Path) -> tuple[list[Path], set[str]]:
    # Cache file list once so we can both index and compare against existing DB entries.
    paths = list(iter_supported_files(folder))
    # Build a set of current file sources to detect deletions in the vector store.
    current_sources = {str(path.resolve()) for path in paths}
    return paths, current_sources


def collect_paths(folder: Path) -> tuple[list[Path], set[str]]:
    return _collect_paths(folder)


def _get_existing_for_source(vector_store, source: str):
    return vector_store.get(where={"source": source}, include=["metadatas"])


def get_existing_for_source(vector_store, source: str):
    return _get_existing_for_source(vector_store, source)


def _should_update(existing, mtime: float, regenerate: bool) -> bool:
    if regenerate or not existing["ids"]:
        return True
    old_mtime = existing["metadatas"][0].get("mtime")
    log_event("index_mtime_check", old_mtime=old_mtime, current_mtime=mtime)
    return old_mtime != mtime


def should_update(existing, mtime: float, regenerate: bool) -> bool:
    return _should_update(existing, mtime, regenerate)


def upsert_docs(vector_store, splitter, path: Path, docs, existing) -> bool:
    source = str(path.resolve())
    mtime = path.stat().st_mtime

    file_start = time.perf_counter()

    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])

    # Skip files that yield no content
    if not docs or all(not d.page_content.strip() for d in docs):
        return False

    split_start = time.perf_counter()
    splits = splitter.split_documents(docs)
    split_duration = time.perf_counter() - split_start
    for d in splits:
        d.metadata.update({"source": source, "mtime": mtime})

    filtered_splits = filter_complex_metadata(splits)
    embed_start = time.perf_counter()
    vector_store.add_documents(filtered_splits)
    embed_duration = time.perf_counter() - embed_start

    log_event(
        "index_file",
        duration_s=time.perf_counter() - file_start,
        source=source,
        chunk_count=len(filtered_splits),
        chunking_s=split_duration,
        embedding_s=embed_duration,
    )
    return True


def _upsert_file(vector_store, splitter, path: Path, existing) -> bool:
    docs = load_file(path)
    return upsert_docs(vector_store, splitter, path, docs, existing)


def _delete_removed_sources(vector_store, current_sources: set[str]) -> set[str]:
    # After indexing, remove any stored chunks whose source file no longer exists.
    existing_all = vector_store.get(include=["metadatas"])
    ids_to_delete: list[str] = []
    deleted_sources: set[str] = set()
    for doc_id, metadata in zip(
        existing_all.get("ids", []), existing_all.get("metadatas", [])
    ):
        if not metadata:
            continue
        source = str(metadata.get("source", ""))
        if source and source not in current_sources:
            ids_to_delete.append(doc_id)
            deleted_sources.add(source)

    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)

    return deleted_sources


def delete_removed_sources(vector_store, current_sources: set[str]) -> set[str]:
    return _delete_removed_sources(vector_store, current_sources)


def index_folder(folder: Path, regenerate: bool = False) -> dict[str, int]:
    """Index all supported files under a folder and store embeddings in Chroma.
    Args:
        folder (Path): The folder to index.
        regenerate (bool): If True, reprocess all files even if they haven't changed.
    Returns:
        dict[str, int]: A dictionary with counts of added, updated, skipped, and errors.
    """
    start_time = time.perf_counter()
    vector_store = get_vector_store(folder)
    splitter = get_splitter()

    added = 0
    updated = 0
    skipped = 0
    errors = 0

    paths, current_sources = _collect_paths(folder)

    for path in paths:
        source = str(path.resolve())
        mtime = path.stat().st_mtime
        log_event("index_start", source=source)
        existing = _get_existing_for_source(vector_store, source)
        if not _should_update(existing, mtime, regenerate):
            log_event("index_skip", source=source)
            skipped += 1
            continue
        log_event("index_update", source=source)
        try:
            indexed = _upsert_file(vector_store, splitter, path, existing)
        except Exception as exc:
            errors += 1
            log_event("index_error", source=source, error=str(exc))
            continue

        if not indexed:
            skipped += 1
            continue

        if existing["ids"]:
            updated += 1
        else:
            added += 1

    deleted_sources = _delete_removed_sources(vector_store, current_sources)

    log_event(
        "index_folder",
        duration_s=time.perf_counter() - start_time,
        added=added,
        updated=updated,
        deleted=len(deleted_sources),
        skipped=skipped,
        errors=errors,
    )

    # ToDO: When generate the DB, and remove the DB folder manually, the next time,
    # it will not create the persistent collection again.
    return {
        "added": added,
        "updated": updated,
        "deleted": len(deleted_sources),
        "skipped": skipped,
        "errors": errors,
    }
