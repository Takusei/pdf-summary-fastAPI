import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


def get_json_from_cache(db_path: Path, key: str) -> Optional[Any]:
    """
    Retrieve and deserialize a JSON value from the SQLite cache.
    Returns None if cache doesn't exist, key not found, or data is invalid.
    """
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM cache WHERE key=?", (key,))
        result = c.fetchone()
        conn.close()

        if result:
            return json.loads(result[0])
    except (sqlite3.Error, json.JSONDecodeError):
        # Fail gracefully on corruption
        return None

    return None


def save_json_to_cache(db_path: Path, key: str, data: Any):
    """
    Serialize and save a JSON value to the SQLite cache.
    Creates the table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

        # Serialize data if it's not already a string
        if not isinstance(data, str):
            json_data = json.dumps(data)
        else:
            json_data = data

        c.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, json_data),
        )
        conn.commit()
    finally:
        # Ensure connection is always closed
        if "conn" in locals():
            conn.close()
