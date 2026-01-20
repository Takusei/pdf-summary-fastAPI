from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class SingleSummaryResponse(BaseModel):
    file_path: str
    file_name: str
    file_size: int
    last_modified_time: float
    file_type: str
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
    regenerate: bool = False

    class Config:
        alias_generator = to_camel
        populate_by_name = True
