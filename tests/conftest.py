from collections.abc import AsyncGenerator

import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings
from app.db.session import get_db


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    settings = get_settings()

    database_url = make_url(settings.database_url).set(
        database="atlas_test",
    )

    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
    )

    async with engine.connect() as connection:
        transaction = await connection.begin()

        session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()

    await engine.dispose()


@pytest_asyncio.fixture
async def test_app(db_session: AsyncSession) -> AsyncGenerator[FastAPI]:
    from app.main import app as fastapi_app

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db

    try:
        yield fastapi_app
    finally:
        fastapi_app.dependency_overrides.clear()
