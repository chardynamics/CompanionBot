from openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()
import os

client = OpenRouter(
    api_key=os.getenv("HACKCLUB_API_KEY"),
    server_url="https://ai.hackclub.com/proxy/v1",
)

response = client.chat.send(
    model="qwen/qwen3-32b",
    messages=[
        {"role": "user", "content": "Tell me a joke."}
    ],
    stream=False,
)

print(response.choices[0].message.content)