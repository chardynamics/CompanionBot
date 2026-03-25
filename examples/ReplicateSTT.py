#import replicate
import os
from dotenv import load_dotenv
import requests
import base64

load_dotenv()

with open("recording.wav", "rb") as f:
    binary_file_data = f.read()
    base64_encoded_data = base64.b64encode(binary_file_data)
    base64_output = base64_encoded_data.decode('utf-8')

    audio = f"data:audio/wav;base64,{base64_output}"
    model = "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"
    url = f"https://ai.hackclub.com/proxy/v1/replicate/models/{model}/predictions"
    headers = {
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    data = {
        "input": {
            "type": 'input_text', 
            "task": "transcribe",
            "audio": audio,
            "return_timestamps": True,
        }
    }
    req = requests.post(url, headers=headers, json=data)
    chunks = req.json()["output"]["chunks"]
    print(chunks)

#Example return of chunks
#[{'text': ' 1, 2, 3, this is the test', 'timestamp': [0, 3.12]}, {'text': " This is probably the normal voice that you're talking to", 'timestamp': [3.12, 6.4]}]

"""
os.environ["REPLICATE_API_URL"] = "https://ai.hackclub.com/proxy/v1/replicate"
os.environ["REPLICATE_API_TOKEN"] = os.getenv("HACKCLUB_API_KEY")

recording = open("recording.wav", "rb")
output = replicate.run(
    "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
    input={
        "task": "transcribe",
        "audio": recording,
        "language": "None", # Use 'None' for auto-detection
        "timestamp": "chunk",
        "batch_size": 64,
        "diarise_audio": False
    }
)
print(output)
"""