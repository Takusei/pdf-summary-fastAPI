from pathlib import Path

from langchain_openai import AzureChatOpenAI, ChatOpenAI

from app.llm.chains import (
    build_map_chain,
    build_reduce_chain,
    build_stuff_chain,
)
from app.services.chunking import split_docs


def get_file_name_from_docs(docs: list) -> str:
    if not docs:
        return "unknown"

    file_path = docs[0].metadata.get("file_path") or docs[0].metadata.get("source")

    if not file_path:
        return "unknown"

    return Path(file_path).name


def choose_method(docs, method: str) -> str:
    if method == "map-reduce":
        return "map-reduce"
    if method == "stuff":
        return "stuff"
    # auto
    return "map-reduce" if len(docs) > 20 else "stuff"


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
