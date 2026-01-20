from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Not a PDF file")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError("PDF is empty or unreadable")

    return docs
