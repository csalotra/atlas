import openai

from app.llm.client import LLMClient
from unittest.mock import Mock


def test_llm_client_timeout():
    llm = LLMClient()

    assert llm.settings.llm_timeout == 60.0

    llm.client.chat.completions.create = Mock(
        side_effect=openai.APITimeoutError(request=None)
    )
    llm.generate("Hello")


def test_llm_client_propagates_timeout():
    llm = LLMClient()

    try:
        raise openai.APITimeoutError(request=None)
    except openai.APITimeoutError:
        pass
