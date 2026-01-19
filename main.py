from fastapi import FastAPI
from pydantic import BaseModel

from libs.agent import initialize_agent
from libs.summary import summarize_single_pdf

app = FastAPI()
agent = initialize_agent()


class SummaryResponse(BaseModel):
    file_path: str
    summary: str


class FilePathRequest(BaseModel):
    file_path: str


class FolderPathRequest(BaseModel):
    folder_path: str


@app.post("/summarize-file/", response_model=SummaryResponse)
async def summarize_file_endpoint(request: FilePathRequest):
    """
    Summarizes a single PDF file from its path.
    """
    file_path = request.file_path
    summary = summarize_single_pdf(file_path, agent)
    return {"file_path": file_path, "summary": summary}


# @app.post("/summarize-folder/")
# async def summarize_folder_endpoint(request: FolderPathRequest):
#     """
#     Summarizes all PDF files in a given folder path.
#     """
#     summaries = []
#     for filename in os.listdir(request.folder_path):
#         if filename.lower().endswith(".pdf"):
#             file_path = os.path.join(request.folder_path, filename)
#             summary = summarize_pdf(file_path)
#             summaries.append({"file_path": file_path, "summary": summary})
#     return summaries
