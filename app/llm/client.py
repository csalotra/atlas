from openai import OpenAI
import openai
from openai.types.chat import ChatCompletion

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

    def generate(self, prompt: str) -> ChatCompletion:
        try:
            return self.client.chat.completions.create(
                model=self.settings.litellm_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
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
