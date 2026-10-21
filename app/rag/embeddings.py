from openai import AsyncOpenAI

from app.core.config import get_settings


class EmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.client = AsyncOpenAI(
            api_key=settings.litellm_master_key,
            base_url=settings.litellm_base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self.model = settings.embedding_model

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding
