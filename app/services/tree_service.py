from pathlib import Path

from app.services.cache_utils import get_json_from_cache, save_json_to_cache

SAVED_TREE_DB = ".treecache.db"


def get_tree(path: Path, regenerate: bool = False):
    """
    Get the directory tree structure.
    It first tries to get it from a local sqlite cache unless regeneration is requested.
    If not found or regeneration is forced, it generates the tree, saves it to the cache, and returns it.
    """
    db_path = path / SAVED_TREE_DB
    if not regenerate:
        cached_tree = get_json_from_cache(db_path, "tree")
        if cached_tree:
            return cached_tree

    tree = _generate_tree(path)
    save_json_to_cache(db_path, "tree", tree)
    return tree


def _generate_tree(current_path: Path):
    """
    Recursively generate the directory tree structure.
    """
    tree = []
    try:
        for item in sorted(current_path.iterdir()):
            # Skip the cache file
            if item.name == SAVED_TREE_DB:
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
