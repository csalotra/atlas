import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import (
    DuplicateDocumentError,
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    InvalidDocumentError,
    UserAlreadyExistsError,
)

logger = logging.getLogger(__name__)


async def user_already_exists_handler(
    request: Request,
    exc: UserAlreadyExistsError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc)},
    )


async def duplicate_document_handler(
    request: Request,
    exc: DuplicateDocumentError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "code": "duplicate_document",
            "detail": str(exc),
        },
    )


async def invalid_document_handler(
    request: Request,
    exc: InvalidDocumentError,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "code": "invalid_document",
            "detail": str(exc),
        },
    )


async def document_not_found_handler(
    request: Request,
    exc: DocumentNotFoundError,
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "code": "document_not_found",
            "detail": str(exc),
        },
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "code": "validation_error",
            "detail": "Request validation failed.",
        },
    )


async def document_already_exists_handler(
    request: Request,
    exc: DocumentAlreadyExistsError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "code": "document_already_exists",
            "detail": str(exc),
        },
    )


async def http_error_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    codes = {
        400: "bad_request",
        404: "not_found",
        409: "conflict",
        413: "upload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
    }
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": codes.get(exc.status_code, "http_error"),
            "detail": detail,
        },
        headers=exc.headers,
    )


async def unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled API error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "detail": "An unexpected error occurred.",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        UserAlreadyExistsError,
        user_already_exists_handler,
    )
    app.add_exception_handler(
        DuplicateDocumentError,
        duplicate_document_handler,
    )
    app.add_exception_handler(
        InvalidDocumentError,
        invalid_document_handler,
    )
    app.add_exception_handler(
        DocumentNotFoundError,
        document_not_found_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )
    app.add_exception_handler(
        StarletteHTTPException,
        http_error_handler,
    )
    app.add_exception_handler(
        Exception,
        unexpected_error_handler,
    )
    app.add_exception_handler(
        DocumentAlreadyExistsError,
        document_already_exists_handler,
    )
