from __future__ import annotations

import os

DB_DIR = os.getenv("RAG_DB_DIR", "./chroma_db")
COLLECTION = os.getenv("RAG_COLLECTION", "my_rag_docs")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-nano")
