from pydantic import BaseModel
from pydantic.alias_generators import to_camel

from app.schemas.rag import IndexFolderResponse
from app.schemas.summarize import SingleSummaryResponse


class SummaryAndIndexRequest(BaseModel):
    folder_path: str
    regenerate: bool = False
    sync: bool = False

    class Config:
        alias_generator = to_camel
        populate_by_name = True


class SummaryAndIndexResponse(BaseModel):
    folder_path: str
    summaries: list[SingleSummaryResponse]
    summary_duration: float
    index: IndexFolderResponse
    duration: float

    class Config:
        alias_generator = to_camel
        populate_by_name = True
