import openai
import pytest
import httpx2

from unittest.mock import Mock

from app.core.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from app.llm.client import LLMClient


def test_llm_client_timeout(monkeypatch):
    llm = LLMClient()

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(side_effect=openai.APITimeoutError(request=None)),
    )

    with pytest.raises(LLMTimeoutError, match="LLM request timed out.") as exc_info:
        llm.generate("Hello")

    assert isinstance(exc_info.value.__cause__, openai.APITimeoutError)


def test_llm_client_rate_limit(monkeypatch):
    llm = LLMClient()

    request = httpx2.Request("POST", "http://test")
    response = httpx2.Response(429, request=request)

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(
            side_effect=openai.RateLimitError(
                "rate limited",
                response=response,
                body=None,
            )
        ),
    )

    with pytest.raises(
        LLMRateLimitError,
        match="LLM request was rate limited.",
    ) as exc_info:
        llm.generate("Hello")

    assert isinstance(exc_info.value.__cause__, openai.RateLimitError)


def test_llm_client_connection_error(monkeypatch):
    llm = LLMClient()

    request = httpx2.Request("POST", "http://test")

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(
            side_effect=openai.APIConnectionError(
                request=request,
            )
        ),
    )

    with pytest.raises(
        LLMConnectionError,
        match="Could not connect to the LLM service.",
    ) as exc_info:
        llm.generate("Hello")

    assert isinstance(
        exc_info.value.__cause__,
        openai.APIConnectionError,
    )


def test_llm_client_server_error(monkeypatch):
    llm = LLMClient()

    request = httpx2.Request("POST", "http://test")
    response = httpx2.Response(500, request=request)

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(
            side_effect=openai.InternalServerError(
                "server error",
                response=response,
                body=None,
            )
        ),
    )

    with pytest.raises(
        LLMServerError,
        match="The LLM service returned a server error.",
    ) as exc_info:
        llm.generate("Hello")

    assert isinstance(
        exc_info.value.__cause__,
        openai.InternalServerError,
    )


def test_llm_client_authentication_error(monkeypatch):
    llm = LLMClient()

    request = httpx2.Request("POST", "http://test")
    response = httpx2.Response(401, request=request)

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(
            side_effect=openai.AuthenticationError(
                "authentication failed",
                response=response,
                body=None,
            )
        ),
    )

    with pytest.raises(
        LLMAuthenticationError,
        match="Authentication with the LLM service failed.",
    ) as exc_info:
        llm.generate("Hello")

    assert isinstance(
        exc_info.value.__cause__,
        openai.AuthenticationError,
    )


def test_llm_client_invalid_request(monkeypatch):
    llm = LLMClient()

    request = httpx2.Request("POST", "http://test")
    response = httpx2.Response(400, request=request)

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(
            side_effect=openai.BadRequestError(
                "bad request",
                response=response,
                body=None,
            )
        ),
    )

    with pytest.raises(
        LLMInvalidRequestError,
        match="The LLM request was invalid.",
    ) as exc_info:
        llm.generate("Hello")

    assert isinstance(
        exc_info.value.__cause__,
        openai.BadRequestError,
    )


def test_llm_client_not_found(monkeypatch):
    llm = LLMClient()

    request = httpx2.Request("POST", "http://test")
    response = httpx2.Response(404, request=request)

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(
            side_effect=openai.NotFoundError(
                "resource not found",
                response=response,
                body=None,
            )
        ),
    )

    with pytest.raises(
        LLMNotFoundError,
        match="The requested LLM resource was not found.",
    ) as exc_info:
        llm.generate("Hello")

    assert isinstance(
        exc_info.value.__cause__,
        openai.NotFoundError,
    )


def test_llm_client_generic_error(monkeypatch):
    llm = LLMClient()

    request = httpx2.Request("POST", "http://test")

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(
            side_effect=openai.APIError(
                "unexpected API error",
                request=request,
                body=None,
            )
        ),
    )

    with pytest.raises(
        LLMError,
        match="An LLM error occurred.",
    ) as exc_info:
        llm.generate("Hello")

    assert isinstance(exc_info.value.__cause__, openai.APIError)
