import pytest

from app.core.exceptions import UserAlreadyExistsError
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