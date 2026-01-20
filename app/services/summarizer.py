import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.llm.chains import (
    build_map_chain,
    build_reduce_chain,
    build_stuff_chain,
)
from app.schemas.summarize import MultipleSummariesResponse
from app.services.chunking import split_docs
from app.services.file_loader import load_file

DB_FILE_NAME = ".summarycache.db"


def get_file_name_from_docs(docs: list) -> str:
    if not docs:
        return "unknown"

    file_path = docs[0].metadata.get("file_path") or docs[0].metadata.get("source")

    if not file_path:
        return "unknown"

    return Path(file_path).name


def summarize_with_map_reduce(docs, llm: ChatOpenAI | AzureChatOpenAI) -> str:
    chunks = split_docs(docs)

    # Filter out empty chunks and check if there's any content left
    non_empty_chunks = [d for d in chunks if d.page_content.strip()]
    if not non_empty_chunks:
        raise ValueError("No text to summarize after chunking")

    map_chain = build_map_chain(llm)
    reduce_chain = build_reduce_chain(llm)

    map_inputs = [{"text": d.page_content} for d in non_empty_chunks]
    map_summaries = map_chain.batch(map_inputs)

    combined = "\n".join(map_summaries)
    file_name = get_file_name_from_docs(docs)

    return reduce_chain.invoke({"text": combined, "file_name": file_name})


def summarize_with_stuff(docs, llm: ChatOpenAI | AzureChatOpenAI) -> str:
    chain = build_stuff_chain(llm)
    file_name = get_file_name_from_docs(docs)

    text = "\n\n".join(d.page_content for d in docs)

    # In case of empty documents or scanned PDFs
    if not text.strip():
        raise ValueError("No text to summarize")
    return chain.invoke({"text": text, "file_name": file_name})


def choose_method(docs, method: str) -> str:
    if method == "map-reduce":
        return "map-reduce"
    if method == "stuff":
        return "stuff"
    # auto
    return "map-reduce" if len(docs) > 20 else "stuff"


def summarize_single_file(
    file_path: str,
    llm: ChatOpenAI | AzureChatOpenAI,
    method: str = "auto",
) -> tuple[str, float]:
    start = time.time()

    try:
        docs = load_file(file_path)

        use_method = choose_method(docs, method)
        print(f"Using summarization method: {use_method}")

        if use_method == "map-reduce":
            summary = summarize_with_map_reduce(docs, llm)
        else:
            summary = summarize_with_stuff(docs, llm)
    except Exception as e:
        summary = f"Error during summarization: {str(e)}"

    return summary, time.time() - start


async def summarize_single_file_async(
    file_path: str,
    semaphore: asyncio.Semaphore,
    llm: ChatOpenAI | AzureChatOpenAI,
    method: str = "auto",
):
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, summarize_single_file, file_path, llm, method
        )


def _get_summaries_from_db(
    folder_path: Path,
) -> MultipleSummariesResponse | None:
    """
    Retrieve the summary structure from the SQLite database.
    """
    db_path = folder_path / DB_FILE_NAME
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("SELECT value FROM cache WHERE key='summaries'")
        result = c.fetchone()
        if not result:
            return None

        data = json.loads(result[0])
        return MultipleSummariesResponse(**data)
    except (sqlite3.Error, json.JSONDecodeError):
        # Handle potential DB corruption or invalid data
        return None
    finally:
        conn.close()


def _save_summaries_to_db(folder_path: Path, response: MultipleSummariesResponse):
    """
    Save the summary structure to the SQLite database.
    """
    db_path = folder_path / DB_FILE_NAME
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
    c.execute(
        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
        ("summaries", response.model_dump_json()),
    )
    conn.commit()
    conn.close()


async def summarize_folder(
    folder_path: str,
    regenerate: bool,
    llm: ChatOpenAI | AzureChatOpenAI,
) -> MultipleSummariesResponse:
    """
    Summarizes all files in a given folder path, using a cache if available.
    """
    path_obj = Path(folder_path)
    if not regenerate:
        cached_summaries = _get_summaries_from_db(path_obj)
        if cached_summaries:
            return cached_summaries

    start_time = time.time()
    all_files_meta = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            try:
                stat = file_path.stat()
                file_type = "directory" if file_path.is_dir() else file_path.suffix
                all_files_meta.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_size": stat.st_size,
                        "last_modified_time": stat.st_mtime,
                        "file_type": file_type,
                    }
                )
            except FileNotFoundError:
                continue

    semaphore = asyncio.Semaphore(10)
    tasks = [
        summarize_single_file_async(
            file_meta["file_path"], semaphore, llm, method="stuff"
        )
        for file_meta in all_files_meta
    ]

    results = await asyncio.gather(*tasks)

    summaries = [
        {
            **all_files_meta[i],
            "summary": results[i][0],
            "duration": results[i][1],
        }
        for i in range(len(all_files_meta))
    ]
    total_duration = time.time() - start_time
    response = MultipleSummariesResponse(summaries=summaries, duration=total_duration)

    _save_summaries_to_db(path_obj, response)
    return response
