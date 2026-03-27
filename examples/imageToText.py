from picamera2 import Picamera2
import base64
import requests
from dotenv import load_dotenv
import wave
from piper.voice import PiperVoice
import os
import simpleaudio as sa

load_dotenv()
model_path = os.path.expanduser("en_US-amy-medium.onnx")
voice = PiperVoice.load(model_path)

picam2 = Picamera2()

picam2.start()
picam2.capture_file("foto.jpg")
picam2.stop()

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_data = encode_image("foto.jpg")

response = requests.post(
    "https://ai.hackclub.com/proxy/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "google/gemini-2.5-flash",
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
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])

text = result["choices"][0]["message"]["content"]
if isinstance(text, list):
    text = "".join(part.get("text", "") for part in text)
else:
    text = text

audio_bytes = voice.synthesize(text)
output_filename = "output.wav"
sample_rate = 22050

with wave.open(output_filename, 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono audio
    wav_file.setsampwidth(2)  # 16-bit sample width (S16_LE)
    wav_file.setframerate(sample_rate)

    # Now process your long text
    for audio_chunk in voice.synthesize(text):
        # Use the pre-formatted bytes attribute we found
        wav_file.writeframes(audio_chunk.audio_int16_bytes)

print(f"Audio saved to {output_filename}")

wave_obj = sa.WaveObject.from_wave_file("output.wav")
play_obj = wave_obj.play()
play_obj.wait_done()