from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import UserAlreadyExistsError
from app.db.models.user import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, user_id: UUID) -> User | None:
        result = await self.session.execute(select(User).where(User.id == user_id))

        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(select(User).where(User.email == email))

        return result.scalar_one_or_none()

    async def create(self, user: User) -> User:
        self.session.add(user)

        try:
            await self.session.flush()
        except IntegrityError as exc:
            await self.session.rollback()

            raise UserAlreadyExistsError("User with this email already exists") from exc

        return user
