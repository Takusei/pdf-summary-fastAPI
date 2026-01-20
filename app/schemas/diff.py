from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class DiffRequest(BaseModel):
    folder_path: str

    class Config:
        alias_generator = to_camel
        populate_by_name = True


class DiffResponse(BaseModel):
    changed: bool
