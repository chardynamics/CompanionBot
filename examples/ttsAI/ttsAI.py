#import replicate
import os
from dotenv import load_dotenv
import requests

load_dotenv()
AUDIO_FILE_PATH = "Ryan.mp3"

with open("recording.wav", "rb") as f:

    model = "qwen/qwen3-tts"
    url = f"https://ai.hackclub.com/proxy/v1/replicate/models/{model}/predictions"
    headers = {
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    data = {
        "input": {
            "mode": "custom_voice",
            "text": "Hello, I'm Aiden and it's very nice to meet you",
            "speaker": "Uncle_fu",
            "language": "auto"
        }
    }
    req = requests.post(url, headers=headers, json=data)
    req_json = req.json()
    # Step 2: Extract output (usually a URL or list of URLs)
    output = req_json.get("output")

    # Handle cases where output is a list
    if isinstance(output, list):
        audio_url = output[0]
    else:
        audio_url = output

    # Step 3: Download the audio file
    audio_resp = requests.get(audio_url)
    audio_resp.raise_for_status()

    # Step 4: Save to disk
    with open(AUDIO_FILE_PATH, "wb") as f:
        f.write(audio_resp.content)

    print(f"Audio file successfully saved to {AUDIO_FILE_PATH}")