from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from app.services.file_loader import load_file as load_office_file


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_txt_file(path: Path) -> List[Document]:
    return TextLoader(str(path)).load()


def _load_pdf_file(path: Path) -> List[Document]:
    return PyMuPDFLoader(str(path)).load()


def load_file(path: Path) -> List[Document]:
    # suffix = path.suffix.lower()
    # if suffix == ".txt":
    #     return _load_txt_file(path)
    # if suffix == ".pdf":
    #     return _load_pdf_file(path)
    try:
        return load_office_file(str(path))
    except ValueError:
        return []


def iter_supported_files(folder: Path) -> Iterable[Path]:
    supported = {".pdf", ".pptx", ".docx", ".xls", ".xlsx"}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue
        yield path
