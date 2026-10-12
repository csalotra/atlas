from openai import OpenAI

from app.core.config import get_settings


class EmbeddingClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.client = OpenAI(
            api_key=settings.litellm_master_key,
            base_url=settings.litellm_base_url,
        )
        self.model = settings.embedding_model

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding
