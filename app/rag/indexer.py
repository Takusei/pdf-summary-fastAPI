from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores.utils import filter_complex_metadata

from app.rag.loaders import iter_supported_files, load_file
from app.rag.splitter import get_splitter
from app.rag.vector_store import get_vector_store


def _collect_paths(folder: Path) -> tuple[list[Path], set[str]]:
    # Cache file list once so we can both index and compare against existing DB entries.
    paths = list(iter_supported_files(folder))
    # Build a set of current file sources to detect deletions in the vector store.
    current_sources = {str(path.resolve()) for path in paths}
    return paths, current_sources


def _get_existing_for_source(vector_store, source: str):
    return vector_store.get(where={"source": source}, include=["metadatas"])


def _should_update(existing, mtime: float, regenerate: bool) -> bool:
    if regenerate or not existing["ids"]:
        return True
    old_mtime = existing["metadatas"][0].get("mtime")
    print(f"  Existing mtime: {old_mtime}, Current mtime: {mtime}")
    return old_mtime != mtime


def _upsert_file(vector_store, splitter, path: Path, existing) -> bool:
    source = str(path.resolve())
    mtime = path.stat().st_mtime

    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])

    docs = load_file(path)

    # Skip files that yield no content
    if not docs or all(not d.page_content.strip() for d in docs):
        return False

    splits = splitter.split_documents(docs)
    for d in splits:
        d.metadata.update({"source": source, "mtime": mtime})

    filtered_splits = filter_complex_metadata(splits)
    vector_store.add_documents(filtered_splits)
    return True


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


def index_folder(folder: Path, regenerate: bool = False) -> dict[str, int]:
    """Index all supported files under a folder and store embeddings in Chroma.
    Args:
        folder (Path): The folder to index.
        regenerate (bool): If True, reprocess all files even if they haven't changed.
    Returns:
        dict[str, int]: A dictionary with counts of added, updated, skipped, and errors.
    """
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
        print(f"Indexing: {source}")
        existing = _get_existing_for_source(vector_store, source)
        if not _should_update(existing, mtime, regenerate):
            print("  Skipping file (no changes)...", source)
            skipped += 1
            continue
        print("  Updating index for file...", source)
        try:
            indexed = _upsert_file(vector_store, splitter, path, existing)
        except Exception as exc:
            errors += 1
            print(f"  Error indexing file: {source} -> {exc}")
            continue

        if not indexed:
            skipped += 1
            continue

        if existing["ids"]:
            updated += 1
        else:
            added += 1

    deleted_sources = _delete_removed_sources(vector_store, current_sources)

    print("✅ Index finished")
    print(f"  Added:   {added}")
    print(f"  Updated: {updated}")
    print(f"  Deleted: {len(deleted_sources)}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")

    # ToDO: When generate the DB, and remove the DB folder manually, the next time,
    # it will not create the persistent collection again.
    return {
        "added": added,
        "updated": updated,
        "deleted": len(deleted_sources),
        "skipped": skipped,
        "errors": errors,
    }
