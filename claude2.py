#!/usr/bin/env python3
"""
Wake word listener for Raspberry Pi 5
- Captures mic audio at 44100 Hz
- Resamples to 16000 Hz using soxr
- Buffers into 400-sample chunks
- Feeds into openWakeWord for detection
- On detection: records until silence, saves as WAV

Dependencies:
    pip install pyaudio soxr openwakeword numpy
    sudo apt-get install portaudio19-dev
"""
import base64
import requests
import pyaudio
import soxr
import numpy as np
import wave
import time
import os
from dotenv import load_dotenv
from collections import deque
from openwakeword.model import Model

load_dotenv()
# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MIC_RATE            = 44100     # Hardware capture rate (Hz)
TARGET_RATE         = 16000     # openWakeWord expected rate (Hz)
OWW_CHUNK           = 400       # Samples per chunk at 16kHz (25 ms)
CHANNELS            = 1
FORMAT              = pyaudio.paInt16

RESAMPLE_RATIO      = MIC_RATE / TARGET_RATE
MIC_CHUNK           = int(np.ceil(OWW_CHUNK * RESAMPLE_RATIO))  # ≈ 1103 samples

DETECTION_THRESHOLD = 0.5

# ── Recording settings ─────────────────────────
# How long to record after wake word (seconds)
MAX_RECORD_SECONDS  = 10
# Stop early if silence lasts this long (seconds)
SILENCE_TIMEOUT     = 2.0
# RMS amplitude below this = silence (tune to your mic/room)
SILENCE_THRESHOLD   = 300
# Where to save recordings
OUTPUT_DIR          = "recordings"

# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────
resampler = soxr.ResampleStream(
    in_rate=MIC_RATE,
    out_rate=TARGET_RATE,
    num_channels=CHANNELS,
    quality="HQ",
    dtype="int16",
)

sample_buffer: deque[int] = deque()

# Shared state between callback and main thread
state = {
    "recording":        False,
    "recorded_frames":  [],     # raw MIC_RATE int16 samples (for WAV)
    "silence_start":    None,
    "record_start":     None,
    "wake_word":        "unknown",
    "stream":    None,   # set after stream is created
    "pa":        None,
    "oww_model": None,
    "needs_processing": False,  # ← add this
    "pending_filename": None,   # ← and this
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def rms(audio: np.ndarray) -> float:
    """Root-mean-square amplitude of an int16 chunk."""
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def save_recording(frames: list, wake_word: str) -> str:
    """Save recorded frames (at MIC_RATE) to a timestamped WAV file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, "recording.wav")

    audio = np.concatenate(frames).astype(np.int16)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # int16 = 2 bytes
        wf.setframerate(MIC_RATE)   # save at original rate — easier to play back
        wf.writeframes(audio.tobytes())

    print(f"[SAVED] {filename}  ({len(audio)/MIC_RATE:.1f}s)")
    return filename


# ──────────────────────────────────────────────
# openWakeWord inference
# ──────────────────────────────────────────────
def drain_buffer(model: Model) -> None:
    """Pull OWW_CHUNK-sized windows from the buffer and run inference."""
    while len(sample_buffer) >= OWW_CHUNK:
        chunk = np.array(
            [sample_buffer.popleft() for _ in range(OWW_CHUNK)],
            dtype=np.int16,
        )
        predictions = model.predict(chunk)
        for wake_word, score in predictions.items():
            if score >= DETECTION_THRESHOLD and not state["recording"]:
                print(f"[DETECTED] '{wake_word}'  score={score:.3f}")
                start_recording(wake_word)


def start_recording(wake_word: str) -> None:
    state["recording"]       = True
    state["recorded_frames"] = []
    state["silence_start"]   = None
    state["record_start"]    = time.time()
    state["wake_word"]       = wake_word
    print(f"[RECORDING] Listening for up to {MAX_RECORD_SECONDS}s …")


# ──────────────────────────────────────────────
# PyAudio callback
# ──────────────────────────────────────────────
def audio_callback(
    in_data, frame_count, time_info, status_flags, *, oww_model: Model
):
    if status_flags:
        print(f"[WARNING] PyAudio status: {status_flags}")

    mic_audio = np.frombuffer(in_data, dtype=np.int16).copy()

    # ── If recording, capture raw MIC_RATE audio ──
    if state["recording"]:
        state["recorded_frames"].append(mic_audio)
        now     = time.time()
        elapsed = now - state["record_start"]
        level   = rms(mic_audio)
        print(f"[RMS] {level:.0f}")  # ← add this temporarily

        # Silence detection
        if level < SILENCE_THRESHOLD:
            if state["silence_start"] is None:
                state["silence_start"] = now
            elif now - state["silence_start"] >= SILENCE_TIMEOUT:
                print(f"[SILENCE] Stopping after {elapsed:.1f}s")
                finish_recording()
                return (None, pyaudio.paContinue)
        else:
            state["silence_start"] = None  # reset on non-silence

        # Hard time limit
        if elapsed >= MAX_RECORD_SECONDS:
            print(f"[TIMEOUT] Max recording time reached ({MAX_RECORD_SECONDS}s)")
            finish_recording()
            return (None, pyaudio.paContinue)

    # ── Otherwise feed into openWakeWord ──
    else:
        resampled = resampler.resample_chunk(mic_audio, last=False)
        sample_buffer.extend(resampled.tolist())
        drain_buffer(oww_model)

    return (None, pyaudio.paContinue)

# -----------------------
# Send to cloud
# -----------------------
def transcribe_audio(filename):
    print("Sending to cloud...")
    with open(filename, "rb") as f:
        binary_file_data = f.read()
    
    base64_output = base64.b64encode(binary_file_data).decode('utf-8')
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

    print("Transcript:", req.json().get("text"))
    return req.json().get("text")

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
    print("Transcript:", req.json().get("text"))
    return req.json().get("text")

def finish_recording() -> None:
    frames    = state["recorded_frames"]
    wake_word = state["wake_word"]
    state["recording"]       = False
    state["recorded_frames"] = []
    state["silence_start"]   = None
    state["record_start"]    = None
    state["wake_word"]       = "unknown"

    if frames:
        filename = save_recording(frames, wake_word)
        state["pending_filename"] = filename
        state["needs_processing"] = True

    print("[PROCESSING] Will transcribe shortly …\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("Loading openWakeWord model …")
    oww_model = Model()
    print(f"Loaded models: {list(oww_model.models.keys())}")

    pa = pyaudio.PyAudio()

    print("\nAvailable input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}")

    stream = pa.open(
        rate=MIC_RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        input_device_index=1,
        frames_per_buffer=MIC_CHUNK,
        stream_callback=lambda in_data, frame_count, time_info, status:
            audio_callback(
                in_data, frame_count, time_info, status, oww_model=oww_model
            ),
    )
    state["stream"] = stream

    print(
        f"\nListening …  "
        f"(mic={MIC_RATE} Hz → target={TARGET_RATE} Hz, "
        f"chunk={OWW_CHUNK} samples = {OWW_CHUNK/TARGET_RATE*1000:.1f} ms)"
    )
    print(f"Recordings saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("Press Ctrl+C to stop.\n")

    try:
        stream.start_stream()
        while True:
            if state["needs_processing"]:
                state["needs_processing"] = False

                stream.stop_stream()

                print("Transcribing …")
                text = transcribe_audio(state["pending_filename"])

                print("Getting response …")
                response = model_return(text)
                print("[RESPONSE]", response)

                print("[LISTENING] Waiting for wake word …\n")
                stream.start_stream()

            time.sleep(0.05)  # small sleep to avoid busy-waiting
    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("Done.")


if __name__ == "__main__":
    main()