from fastapi import APIRouter

from .endpoints import health, summarize

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(summarize.router, prefix="/summarize", tags=["summarize"])
