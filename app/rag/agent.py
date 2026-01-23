from __future__ import annotations

from app.rag.config import DATA_DIR
from app.rag.indexer import index_folder

if __name__ == "__main__":
    index_folder(DATA_DIR)
