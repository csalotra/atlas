from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import UserAlreadyExistsError
from app.db.models.user import User
from app.repositories.user import UserRepository


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = UserRepository(session)

    async def create_user(self, email: str) -> User:
        existing_user = await self.repository.get_by_email(email)

        if existing_user is not None:
            raise UserAlreadyExistsError("User with this email already exists")

        user = User(email=email)

        user = await self.repository.create(user)

        await self.session.commit()
        await self.session.refresh(user)

        return user
