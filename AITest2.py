from openrouter import OpenRouter
import os
with OpenRouter(
    api_key=os.getenv("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")
) as client:
    response = client.chat.send(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(response.choices[0].message.content)