import base64
import requests
from dotenv import load_dotenv
import os

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_data = encode_image("foto.jpg")

load_dotenv()

response = requests.post(
    "https://ai.hackclub.com/proxy/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-5.4-image-2",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Make a concise joke involving the two people in the photo"},
                {
                    "type": "image_url",
                    "image_url": {
                       "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }]
        "modalities": ["image", "text"],
        "size": "320x240"
    }
)

result = response.json()

if result.get("choices"):
    message = result["choices"][0]["message"]
    if message.get("images"):
        image_url = message["images"][0]["image_url"]["url"]
        
        # Handle data URI prefix
        if "," in image_url:
            base64_data = image_url.split(",")[1]
        else:
            base64_data = image_url

        image_bytes = base64.b64decode(base64_data)
        with open("output_image.jpg", "wb") as f:
            f.write(image_bytes)