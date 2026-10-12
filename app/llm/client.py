from openai import OpenAI
import openai
from app.llm.models import LLMResponse, LLMStreamChunk
from collections.abc import Iterator

from app.core.config import get_settings
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


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.litellm_master_key,
            base_url=self.settings.litellm_base_url,
            timeout=self.settings.llm_timeout,
            max_retries=self.settings.llm_max_retries,
        )

    def _create_completion(self, request):
        try:
            return self.client.chat.completions.create(**request)

        except openai.APITimeoutError as exc:
            raise LLMTimeoutError from exc

        except openai.RateLimitError as exc:
            raise LLMRateLimitError from exc

        except openai.APIConnectionError as exc:
            raise LLMConnectionError from exc

        except openai.InternalServerError as exc:
            raise LLMServerError from exc

        except openai.AuthenticationError as exc:
            raise LLMAuthenticationError from exc

        except openai.BadRequestError as exc:
            raise LLMInvalidRequestError from exc

        except openai.NotFoundError as exc:
            raise LLMNotFoundError from exc

        except openai.APIError as exc:
            raise LLMError from exc

    def _build_request(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        stream: bool = False,
    ) -> dict[str, object]:
        request = {
            "model": self.settings.litellm_model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        if stream:
            request["stream"] = True

        return request

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not messages:
            raise LLMInvalidRequestError

        request = self._build_request(
            messages,
            temperature,
            max_tokens,
        )

        response = self._create_completion(request)

        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=(
                response.usage.completion_tokens if response.usage else None
            ),
            total_tokens=response.usage.total_tokens if response.usage else None,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        if not messages:
            raise LLMInvalidRequestError

        request = self._build_request(
            messages,
            temperature,
            max_tokens,
            stream=True,
        )

        response = self._create_completion(request)

        for chunk in response:
            text = chunk.choices[0].delta.content

            if text:
                yield LLMStreamChunk(text=text)
