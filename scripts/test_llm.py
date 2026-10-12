from app.llm.client import LLMClient

llm = LLMClient()

response = llm.generate([{"role": "user", "content": "Hello"}])

print(response.text)
