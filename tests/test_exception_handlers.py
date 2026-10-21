import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app.api.exception_handlers import register_exception_handlers
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidDocumentError,
    UserAlreadyExistsError,
)


@pytest.fixture
def error_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/user-conflict")
    async def user_conflict():
        raise UserAlreadyExistsError("User already exists.")

    @app.get("/duplicate-document")
    async def duplicate_document():
        raise DuplicateDocumentError("Document already exists.")

    @app.get("/invalid-document")
    async def invalid_document():
        raise InvalidDocumentError("Invalid PDF.")

    @app.get("/missing-document")
    async def missing_document():
        raise DocumentNotFoundError("Document not found.")

    @app.get("/http-error")
    async def http_error():
        raise HTTPException(
            status_code=413,
            detail="File exceeds the upload limit.",
        )

    @app.get("/unexpected-error")
    async def unexpected_error():
        raise RuntimeError("Internal database details.")

    @app.get("/validate/{count}")
    async def validate_count(count: int):
        return {"count": count}

    return app


async def get_response(
    app: FastAPI,
    path: str,
) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=app,
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_user_conflict_handler(error_app):
    response = await get_response(error_app, "/user-conflict")

    assert response.status_code == 409
    assert response.json() == {"detail": "User already exists."}


@pytest.mark.asyncio
async def test_duplicate_document_handler(error_app):
    response = await get_response(error_app, "/duplicate-document")

    assert response.status_code == 409
    assert response.json() == {
        "code": "duplicate_document",
        "detail": "Document already exists.",
    }


@pytest.mark.asyncio
async def test_invalid_document_handler(error_app):
    response = await get_response(error_app, "/invalid-document")

    assert response.status_code == 400
    assert response.json() == {
        "code": "invalid_document",
        "detail": "Invalid PDF.",
    }


@pytest.mark.asyncio
async def test_document_not_found_handler(error_app):
    response = await get_response(error_app, "/missing-document")

    assert response.status_code == 404
    assert response.json() == {
        "code": "document_not_found",
        "detail": "Document not found.",
    }


@pytest.mark.asyncio
async def test_http_exception_handler(error_app):
    response = await get_response(error_app, "/http-error")

    assert response.status_code == 413
    assert response.json() == {
        "code": "upload_too_large",
        "detail": "File exceeds the upload limit.",
    }


@pytest.mark.asyncio
async def test_validation_error_handler(error_app):
    response = await get_response(error_app, "/validate/not-an-integer")

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_error",
        "detail": "Request validation failed.",
    }


@pytest.mark.asyncio
async def test_unexpected_error_handler(error_app):
    response = await get_response(error_app, "/unexpected-error")

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "detail": "An unexpected error occurred.",
    }
    assert "Internal database details." not in response.text
