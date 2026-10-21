from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.core.exceptions import DuplicateDocumentError
from app.dependencies import get_document_ingestion_service


@pytest.mark.asyncio
async def test_upload_rolls_back_when_ingestion_fails(
    test_app,
    db_session,
) -> None:
    service = Mock()
    service.ingest = AsyncMock(side_effect=RuntimeError("Database operation failed"))
    test_app.dependency_overrides[get_document_ingestion_service] = lambda: service

    rollback = AsyncMock()
    original_rollback = db_session.rollback
    db_session.rollback = rollback

    try:
        transport = httpx.ASGITransport(
            app=test_app,
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/documents",
                files={
                    "file": (
                        "report.pdf",
                        b"%PDF-1.7\nsample content",
                        "application/pdf",
                    )
                },
            )
    finally:
        db_session.rollback = original_rollback

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "detail": "An unexpected error occurred.",
    }
    service.ingest.assert_awaited_once()
    rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_duplicate_upload_uses_central_error_response(
    test_app,
    db_session,
) -> None:
    service = Mock()
    service.ingest = AsyncMock(
        side_effect=DuplicateDocumentError("This document has already been ingested.")
    )
    test_app.dependency_overrides[get_document_ingestion_service] = lambda: service

    rollback = AsyncMock()
    original_rollback = db_session.rollback
    db_session.rollback = rollback

    try:
        transport = httpx.ASGITransport(
            app=test_app,
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/documents",
                files={
                    "file": (
                        "report.pdf",
                        b"%PDF-1.7\nsample content",
                        "application/pdf",
                    )
                },
            )
    finally:
        db_session.rollback = original_rollback

    assert response.status_code == 409
    assert response.json() == {
        "code": "duplicate_document",
        "detail": "This document has already been ingested.",
    }
    service.ingest.assert_awaited_once()
    rollback.assert_awaited_once()
