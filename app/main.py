from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.users import router as users_router
from app.core.exceptions import UserAlreadyExistsError

app = FastAPI(
    title="Atlas",
    description="Production AI agent platform.",
    version="0.1.0",
)

app.include_router(users_router)


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(UserAlreadyExistsError)
async def user_already_exists_handler(
    request: Request,
    exc: UserAlreadyExistsError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc)},
    )
