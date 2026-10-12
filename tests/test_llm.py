from unittest import result

import openai
import pytest
import httpx2

from unittest.mock import Mock
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk

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
        llm.generate([{"role": "user", "content": "Hello"}])

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
        llm.generate([{"role": "user", "content": "Hello"}])

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
        llm.generate([{"role": "user", "content": "Hello"}])

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
        llm.generate([{"role": "user", "content": "Hello"}])

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
        llm.generate([{"role": "user", "content": "Hello"}])

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
        llm.generate([{"role": "user", "content": "Hello"}])

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
        llm.generate([{"role": "user", "content": "Hello"}])

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
        llm.generate([{"role": "user", "content": "Hello"}])

    assert isinstance(exc_info.value.__cause__, openai.APIError)


def test_llm_client_success(monkeypatch):
    llm = LLMClient()

    response = ChatCompletion(
        id="test-id",
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": ChatCompletionMessage(
                    role="assistant",
                    content="Hello!",
                ),
            }
        ],
        created=0,
        model="atlas-default",
        object="chat.completion",
    )

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        Mock(return_value=response),
    )

    result = llm.generate([{"role": "user", "content": "Hello"}])

    assert result.text == "Hello!"
    assert result.model == "atlas-default"


def test_llm_client_stream(monkeypatch):
    llm = LLMClient()

    chunks = [
        ChatCompletionChunk(
            id="chunk-1",
            choices=[
                {
                    "delta": {
                        "content": "Hello",
                        "role": "assistant",
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            created=0,
            model="atlas-default",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chunk-2",
            choices=[
                {
                    "delta": {
                        "content": " there!",
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            created=0,
            model="atlas-default",
            object="chat.completion.chunk",
        ),
    ]

    mock_create = Mock(return_value=chunks)

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create,
    )

    result = list(llm.stream([{"role": "user", "content": "Hello"}]))

    assert [chunk.text for chunk in result] == [
        "Hello",
        " there!",
    ]

    print(mock_create.call_args)
    mock_create.assert_called_once_with(
        model=llm.settings.litellm_model,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        temperature=0.7,
    )


def test_llm_client_custom_temperature(monkeypatch):
    llm = LLMClient()

    response = ChatCompletion(
        id="test-id",
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
            }
        ],
        created=0,
        model="atlas-default",
        object="chat.completion",
    )

    mock_create = Mock(return_value=response)

    monkeypatch.setattr(
        llm.client.chat.completions,
        "create",
        mock_create,
    )

    llm.generate(
        [{"role": "user", "content": "Hello"}],
        temperature=0.2,
    )

    mock_create.assert_called_once_with(
        model=llm.settings.litellm_model,
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.2,
    )


def test_llm_client_rejects_empty_messages():
    llm = LLMClient()

    with pytest.raises(LLMInvalidRequestError):
        llm.generate([])


def test_llm_client_stream_rejects_empty_messages():
    llm = LLMClient()

    with pytest.raises(LLMInvalidRequestError):
        list(llm.stream([]))
