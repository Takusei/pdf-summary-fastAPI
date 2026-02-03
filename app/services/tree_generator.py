import time
from pathlib import Path

from app.cache.utils import (
    SAVED_TREE_DB,
    VDR_DB_DIR,
    get_json_from_cache,
    is_cache_file,
    save_json_to_cache,
)
from app.core.logging import log_event


def get_tree(path: Path, regenerate: bool = False):
    """
    Get the directory tree structure.
    It first tries to get it from a local sqlite cache unless regeneration is requested.
    If not found or regeneration is forced, it generates the tree, saves it to the cache, and returns it.
    """
    db_dir = path / VDR_DB_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / SAVED_TREE_DB
    start = time.perf_counter()
    if not regenerate:
        cached_tree = get_json_from_cache(db_path, "tree")
        if cached_tree:
            log_event("tree_cache_hit", folder_path=str(path))
            return cached_tree

    tree = _generate_tree(path)
    save_json_to_cache(db_path, "tree", tree)
    log_event(
        "tree_generated",
        duration_s=time.perf_counter() - start,
        folder_path=str(path),
        regenerate=regenerate,
    )
    return tree


def _generate_tree(current_path: Path):
    """
    Recursively generate the directory tree structure.
    """
    tree = []
    try:
        for item in sorted(current_path.iterdir()):
            # Skip the cache file
            if is_cache_file(item.name) or item.name == VDR_DB_DIR:
                continue

            stat = item.stat()
            file_type = "directory" if item.is_dir() else item.suffix
            item_data = {
                "file_path": str(item),
                "file_name": item.name,
                "file_size": stat.st_size,
                "last_modified_time": stat.st_mtime,
                "file_type": file_type,
                "children": None,
            }

            if item.is_dir():
                item_data["children"] = _generate_tree(item)

            tree.append(item_data)
    except FileNotFoundError:
        return []
    return tree
