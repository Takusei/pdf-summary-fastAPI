from fastapi import APIRouter

from .endpoints import diff, health, summarize, tree

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(summarize.router, prefix="/summarize", tags=["summarize"])
api_router.include_router(tree.router, prefix="/tree", tags=["tree"])
api_router.include_router(diff.router, prefix="/diff", tags=["diff"])
