from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.users import router as users_router
from app.core.exceptions import UserAlreadyExistsError
from app.api.exception_handlers import register_exception_handlers
from app.api.retrieval import router as retrieval_router

app = FastAPI(
    title="Atlas",
    description="Production AI agent platform.",
    version="0.1.0",
)

app.include_router(users_router)
app.include_router(documents_router)
app.include_router(retrieval_router)

register_exception_handlers(app)


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok"}
