from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class SingleSummaryResponse(BaseModel):
    file_path: str
    summary: str
    duration: float

    class Config:
        alias_generator = to_camel
        populate_by_name = True


class MultipleSummariesResponse(BaseModel):
    summaries: list[SingleSummaryResponse]
    duration: float


class FilePathRequest(BaseModel):
    file_path: str

    class Config:
        alias_generator = to_camel
        populate_by_name = True


class FolderPathRequest(BaseModel):
    folder_path: str

    class Config:
        alias_generator = to_camel
        populate_by_name = True
