class AppError(Exception):
    """Base class for application-level errors."""


class UserAlreadyExistsError(AppError):
    """Raised when attempting to create a user that already exists."""
