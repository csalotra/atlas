import httpx
import pytest
from sqlalchemy import select

from app.core.exceptions import UserAlreadyExistsError
from app.db.models.user import User
from app.services.user import UserService


@pytest.mark.asyncio
async def test_create_user(db_session) -> None:
    service = UserService(db_session)

    user = await service.create_user("test@example.com")

    assert user.email == "test@example.com"


@pytest.mark.asyncio
async def test_create_duplicate_user(db_session) -> None:
    service = UserService(db_session)

    await service.create_user("duplicate@example.com")

    with pytest.raises(UserAlreadyExistsError):
        await service.create_user("duplicate@example.com")


@pytest.mark.asyncio
async def test_create_user_api(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/users",
            json={"email": "api@example.com"},
        )

    assert response.status_code == 201

    data = response.json()

    assert data["email"] == "api@example.com"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_async_session_directly(db_session) -> None:
    result = await db_session.execute(
        select(User).where(User.email == "direct@example.com")
    )

    user = result.scalar_one_or_none()

    assert user is None


@pytest.mark.asyncio
async def test_create_user_invalid_email(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/users",
            json={"email": "not-an-email"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_duplicate_user_api(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first_response = await client.post(
            "/users",
            json={"email": "api-duplicate@example.com"},
        )

        second_response = await client.post(
            "/users",
            json={"email": "api-duplicate@example.com"},
        )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert second_response.json()["detail"] == ("User with this email already exists")
