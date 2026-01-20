import json
import sqlite3
from pathlib import Path

SAVED_TREE_DB = ".treecache.db"


def get_tree(path: Path, regenerate: bool = False):
    """
    Get the directory tree structure.
    It first tries to get it from a local sqlite cache unless regeneration is requested.
    If not found or regeneration is forced, it generates the tree, saves it to the cache, and returns it.
    """
    db_path = path / SAVED_TREE_DB
    if not regenerate and db_path.exists():
        try:
            return _get_tree_from_db(db_path)
        except (sqlite3.Error, json.JSONDecodeError):
            # Handle potential DB corruption or invalid data
            pass

    tree = _generate_tree(path)
    _save_tree_to_db(db_path, tree)
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
            item_data = {
                "file_path": str(item),
                "file_name": item.name,
                "file_size": stat.st_size,
                "last_modified_time": stat.st_mtime,
                "children": None,
            }

            if item.is_dir():
                item_data["children"] = _generate_tree(item)

            tree.append(item_data)
    except FileNotFoundError:
        return []
    return tree


def _get_tree_from_db(db_path: Path):
    """
    Retrieve the tree structure from the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT value FROM cache WHERE key='tree'")
    result = c.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])
    return None


def _save_tree_to_db(db_path: Path, tree: dict):
    """
    Save the tree structure to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
    c.execute(
        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
        ("tree", json.dumps(tree)),
    )
    conn.commit()
    conn.close()
