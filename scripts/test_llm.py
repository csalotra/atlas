from app.llm.client import LLMClient

llm = LLMClient()

response = llm.generate("Write a short poem about the ocean.")

print(response.choices[0].message.content)
