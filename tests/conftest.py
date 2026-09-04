from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import get_settings


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    settings = get_settings()

    database_url = settings.database_url.replace(
        "/atlas",
        "/atlas_test",
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