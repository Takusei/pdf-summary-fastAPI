from __future__ import annotations

from pathlib import Path

from app.rag.loaders import iter_supported_files, load_file
from app.rag.splitter import get_splitter
from app.rag.vector_store import get_vector_store


def index_folder(folder: Path) -> dict[str, int]:
    vector_store = get_vector_store(folder)
    splitter = get_splitter()

    added = 0
    updated = 0
    skipped = 0

    for path in iter_supported_files(folder):
        mtime = path.stat().st_mtime
        source = str(path.resolve())

        existing = vector_store.get(
            where={"source": source},
            include=["metadatas"],
        )

        needs_update = True

        if existing["ids"]:
            old_mtime = existing["metadatas"][0].get("mtime")
            if old_mtime == mtime:
                skipped += 1
                needs_update = False

        if not needs_update:
            continue

        if existing["ids"]:
            vector_store.delete(ids=existing["ids"])
            updated += 1
        else:
            added += 1

        docs = load_file(path)
        if not docs:
            continue

        splits = splitter.split_documents(docs)

        for d in splits:
            d.metadata.update(
                {
                    "source": source,
                    "mtime": mtime,
                }
            )

        vector_store.add_documents(splits)

    # vector_store.persist()

    print("✅ Index finished")
    print(f"  Added:   {added}")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")

    return {"added": added, "updated": updated, "skipped": skipped}
