from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class IndexFolderRequest(BaseModel):
    folder_path: str

    class Config:
        alias_generator = to_camel
        populate_by_name = True


class IndexFolderResponse(BaseModel):
    folder_path: str
    added: int
    updated: int
    skipped: int
    duration: float

    class Config:
        alias_generator = to_camel
        populate_by_name = True
