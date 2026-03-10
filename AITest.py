from openrouter import OpenRouter

client = OpenRouter(
    api_key="sk-hc-v1-f2f57a42ad75421bb33f2a09ce1b7c81dfd59237319a42b2b3f4bbd6544d2cf9",
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