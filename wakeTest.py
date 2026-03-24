import pyaudio
import numpy as np
import webrtcvad
import wave
import collections
import time
import requests
from openwakeword.model import Model
import os
from dotenv import load_dotenv
import requests
import base64
import soxr

load_dotenv()

# -----------------------
# Config
# -----------------------
RATE = 44100
CHUNK = 400  # 20ms (required for webrtcvad)
CHANNELS = 1

PREBUFFER_SECONDS = 1
MAX_RECORD_SECONDS = 8

VAD_MODE = 2
SILENCE_LIMIT = 30  # ~0.6 sec

WAKE_THRESHOLD = 0.5

# -----------------------
# Init
# -----------------------
vad = webrtcvad.Vad(VAD_MODE)
model = Model()

pa = pyaudio.PyAudio()
stream = pa.open(
    rate=RATE,
    channels=CHANNELS,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=CHUNK
)

# rolling pre-buffer
prebuffer = collections.deque(maxlen=int((RATE / CHUNK) * PREBUFFER_SECONDS))


# -----------------------
# Save audio
# -----------------------
def save_audio(frames, filename="command.wav"):
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


"""
from vosk import Model, KaldiRecognizer
import json

vosk_model = Model("vosk-model")  # path to your folder


def transcribe_audio(filename):
    wf = wave.open(filename, "rb")

    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    
    results.append(json.loads(rec.FinalResult()))

    # Combine text
    text = " ".join([r.get("text", "") for r in results]).strip()

    print("Transcript:", text)
    return text
"""

# -----------------------
# Send to cloud
# -----------------------
def transcribe_audio(filename):
    print("Sending to cloud...")
    with open(filename, "rb") as f:
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
        print("Transcript:", req.get("text"))
        return req.get("text")

def model_return(text):
    print("Sending to cloud...")
    model = "google/gemini-2.5-flash-lite-preview-09-2025"
    url = "https://ai.hackclub.com/proxy/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    data = {
        "model": model,
        "input": [
            {
                "type": 'message', 
                "role": "user",
                "content": [
                    {
                    "type": "input_text",
                    "text": text,
                    }
                ],
            }
        ],
    }
    req = requests.post(url, headers=headers, json=data)
    print("Transcript:", req.get("text"))
    return req.get("text")

# -----------------------
# Record with VAD + timeout
# -----------------------
def record_until_silence():
    frames = list(prebuffer)  # include pre-buffer
    silence_counter = 0
    start_time = time.time()

    print("Recording...")

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        is_speech = vad.is_speech(data, RATE)

        if is_speech:
            silence_counter = 0
        else:
            silence_counter += 1

        # stop on silence
        if silence_counter > SILENCE_LIMIT:
            break

        # stop on timeout
        if time.time() - start_time > MAX_RECORD_SECONDS:
            print("Timeout reached")
            break

    print("Finished recording")

    filename = "command.wav"
    save_audio(frames, filename)

    return filename


# -----------------------
# Main loop
# -----------------------
print("Listening for wake word...")

while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio = np.frombuffer(data, dtype=np.int16)

    # add to pre-buffer
    prebuffer.append(data)

    prediction = model.predict(audio)
    score = list(prediction.values())[0]

    if score > WAKE_THRESHOLD:
        print("Wake word detected!")

        filename = record_until_silence()
        text = transcribe_audio(filename)

        model_return = api_call(text)

        print("Back to listening...\n")