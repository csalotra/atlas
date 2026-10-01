class AppError(Exception):
    """Base class for application-level errors."""


class UserAlreadyExistsError(AppError):
    """Raised when attempting to create a user that already exists."""


class LLMError(AppError):
    """Base class for LLM-related errors."""

    def __init__(self) -> None:
        super().__init__("An LLM error occurred.")


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""

    def __init__(self) -> None:
        super().__init__("LLM request timed out.")


class LLMRateLimitError(LLMError):
    """Raised when an LLM request is rate limited."""

    def __init__(self) -> None:
        super().__init__("LLM request was rate limited.")


class LLMConnectionError(LLMError):
    """Raised when the LLM service cannot be reached."""

    def __init__(self) -> None:
        super().__init__("Could not connect to the LLM service.")


class LLMServerError(LLMError):
    """Raised when the LLM service returns a server error."""

    def __init__(self) -> None:
        super().__init__("The LLM service returned a server error.")


class LLMAuthenticationError(LLMError):
    """Raised when authentication with the LLM service fails."""

    def __init__(self) -> None:
        super().__init__("Authentication with the LLM service failed.")


class LLMInvalidRequestError(LLMError):
    """Raised when an LLM request is invalid."""

    def __init__(self) -> None:
        super().__init__("The LLM request was invalid.")


class LLMNotFoundError(LLMError):
    """Raised when the requested LLM resource cannot be found."""

    def __init__(self) -> None:
        super().__init__("The requested LLM resource was not found.")
