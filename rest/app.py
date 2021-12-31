
import asyncio

import uvicorn
import uvloop
from fastapi import FastAPI

from rest.router.health_check import router as health_check_router
from rest.router.inference import router as inference_router


def create_app():
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    app = FastAPI(
        docs_url="/inference/docs",
        openapi_url="/inference/openapi.json",
    )
    return app


def include_router(application: FastAPI):
    application.include_router(health_check_router)
    application.include_router(inference_router)


app = create_app()
include_router(application=app)


if __name__ == "__main__":
    uvicorn.run("rest.app:app", host="0.0.0.0", port=9999, reload=False)