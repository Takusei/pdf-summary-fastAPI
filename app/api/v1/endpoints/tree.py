from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.tree import TreeRequest
from app.services.tree_generator import get_tree

router = APIRouter()


@router.post("")
def read_tree(request: TreeRequest):
    """
    Get the directory tree structure for a given folder path.
    """
    folder_path = Path(request.folder_path)
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid folder path")

    tree = get_tree(folder_path, regenerate=request.regenerate)
    return tree
