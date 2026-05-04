import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import io
import os

load_dotenv()

response = requests.post(
    "https://ai.hackclub.com/proxy/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-5.4-image-2",
        #"model": "google/gemini-3-pro-image-preview",
        "messages": [
            {
                "role": "user",
                "content": "Make a picture of Kylian Mbappe taking a selfie with the UCL trophy at Real Madrid"
            }
        ],
        "modalities": ["image", "text"],
        "size": "1536x1024"
    }
)

result = response.json()

if result.get("choices"):
    message = result["choices"][0]["message"]

    if message.get("images"):
        image_url = message["images"][0]["image_url"]["url"]

        # Handle data URI prefix
        base64_data = image_url.split(",")[1] if "," in image_url else image_url
        image_bytes = base64.b64decode(base64_data)

        # Downscale to 320x240 using Pillow
        img = Image.open(io.BytesIO(image_bytes))
        img_resized = img.resize((320, 240), Image.LANCZOS)
        img_resized.save("output_image.jpg")

        print(f"Saved! Original: {img.size} → Resized: {img_resized.size}")