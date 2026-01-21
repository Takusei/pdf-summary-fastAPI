import os

import uvicorn

from app.core.config import settings
from app.main import app as fastapi_app


def main() -> None:
    host = os.getenv("BACKEND_HOST", settings.BACKEND_HOST)
    port = int(os.getenv("BACKEND_PORT", settings.BACKEND_PORT))

    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
