import pytest

from app.llm.client import LLMClient


@pytest.mark.integration
def test_llm_real_generation():
    llm = LLMClient()

    response = llm.generate(
        [{"role": "user", "content": "Reply with exactly: integration-ok"}],
        max_tokens=20,
    )

    assert response.text
    assert response.model
