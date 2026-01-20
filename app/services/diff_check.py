from pathlib import Path

from app.cache.utils import (
    SAVED_TREE_DB,
    get_json_from_cache,
    is_cache_file,
)


def check_diff(folder_path: Path) -> bool:
    """
    Compares the live directory against the cached tree to check for changes.
    Returns True if there are any new, deleted, or modified files.
    """
    db_path = folder_path / SAVED_TREE_DB
    cached_tree = get_json_from_cache(db_path, "tree")

    # If there's no cache, there's no "difference" by default.
    if not cached_tree:
        return False

    # Flatten the cached tree into a dictionary for easy lookup.
    cached_files = {}

    def flatten_tree(nodes):
        for node in nodes:
            cached_files[node["file_path"]] = node["last_modified_time"]
            if node.get("children"):
                flatten_tree(node["children"])

    flatten_tree(cached_tree)
    print(f"Cached files: {len(cached_files)} items")
    # Get the current state of files on disk.
    current_files = {}
    try:
        for item in folder_path.rglob("*"):
            if is_cache_file(item.name):
                continue
            try:
                current_files[str(item)] = item.stat().st_mtime
            except FileNotFoundError:
                continue  # File might have been deleted during iteration.
    except FileNotFoundError:
        # If the whole folder is gone, that's a change.
        return True

    # Compare the two sets of files.
    if set(cached_files.keys()) != set(current_files.keys()):
        return True  # New or deleted files found.

    # Check for modifications in existing files.
    for path, mtime in current_files.items():
        if path in cached_files and cached_files[path] != mtime:
            return True  # A file was modified.

    return False
