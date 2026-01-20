from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class TreeRequest(BaseModel):
    folder_path: str
    regenerate: bool = False

    class Config:
        alias_generator = to_camel
        populate_by_name = True
