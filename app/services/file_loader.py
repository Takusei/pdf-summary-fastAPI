import os

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError("PDF is empty or unreadable")
    return docs


def load_pptx(file_path: str) -> list[Document]:
    loader = UnstructuredPowerPointLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError("PPTX is empty or unreadable")
    return docs


def load_docx(file_path: str) -> list[Document]:
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError("DOCX is empty or unreadable")
    return docs


def load_excel(file_path: str) -> list[Document]:
    loader = UnstructuredExcelLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError("Excel file is empty or unreadable")
    return docs


# Create a dispatch table for file loaders
FILE_LOADERS = {
    ".pdf": load_pdf,
    ".pptx": load_pptx,
    ".docx": load_docx,
    ".xls": load_excel,
    ".xlsx": load_excel,
}


def load_file(file_path: str) -> list[Document]:
    """
    Loads a file using the appropriate loader based on its extension.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    loader_func = FILE_LOADERS.get(extension)
    if not loader_func:
        raise ValueError(f"Unsupported file type: {extension}")

    return loader_func(file_path)
