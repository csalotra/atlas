import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.environ["LITELLM_MASTER_KEY"],
    base_url="http://localhost:4000/v1",
)


response = client.chat.completions.create(
    model="atlas-default",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one short sentence.",
        }
    ],
)

print(response.choices[0].message.content)