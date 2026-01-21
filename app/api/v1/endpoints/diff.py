from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.diff import DiffRequest, DiffResponse
from app.services.diff_check import check_diff

router = APIRouter()


@router.post("", response_model=DiffResponse)
def get_diff_endpoint(request: DiffRequest) -> DiffResponse:
    """
    Checks if there are any file changes in the given folder path
    compared to the cached tree.
    """
    folder_path = Path(request.folder_path)
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid folder path")

    has_changes = check_diff(folder_path)
    return DiffResponse(changed=has_changes)
