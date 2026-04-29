import json
import requests
import os
import time
import board
import neopixel
import threading
import pygame
import pyaudio
import soxr
import numpy as np
import wave
from collections import deque
from openwakeword.model import Model
import subprocess
import base64
import io
from PIL import Image

from dotenv import load_dotenv
#from ina219 import INA219
from adafruit_servokit import ServoKit
from picamera2 import Picamera2
from src.modules.ai_camera import IMX500Detector
from piper.voice import PiperVoice

MIC_RATE            = 44100     # Hardware capture rate (Hz)
TARGET_RATE         = 16000     # openWakeWord expected rate (Hz)
OWW_CHUNK           = 400       # Samples per chunk at 16kHz (25 ms)
CHANNELS            = 1
FORMAT              = pyaudio.paInt16

RESAMPLE_RATIO      = MIC_RATE / TARGET_RATE
MIC_CHUNK           = int(np.ceil(OWW_CHUNK * RESAMPLE_RATIO))  # ≈ 1103 samples

DETECTION_THRESHOLD = 0.5

SAMPLE_RATE = 22050
OUTPUT_FILENAME = os.path.expanduser("~/CompanionBot/recordings/output.wav")

# ── Recording settings ─────────────────────────
# How long to record after wake word (seconds)
MAX_RECORD_SECONDS  = 10
# Stop early if silence lasts this long (seconds)
SILENCE_TIMEOUT     = 2.0
# RMS amplitude below this = silence (tune to your mic/room)
SILENCE_THRESHOLD   = 3000
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
    "oww_model": None,
    "needs_processing": False,
    "pending_filename": None,
    "processing": False,
    "skip_chunks": 0,
}

model_path = os.path.expanduser("~/CompanionBot/src/models/en_US-amy-medium.onnx")
voice = PiperVoice.load(model_path)

load_dotenv()
kit = ServoKit(channels=16)

#ina = INA219(addr=0x41)
#readings = ina.getReadings()
defaultThrottle = 0.2

detector = IMX500Detector()
detector.start()
pygame.mixer.init()

# Choose an open pin connected to the Data In of the NeoPixel strip, i.e. board.D18
# NeoPixels must be connected to D10, D12, D18 or D21 to work.
pixel_pin = board.D12
num_pixels = 5

# The order of the pixel colors - RGB or GRB. Some NeoPixels have red and green reversed!
# For RGBW NeoPixels, simply change the ORDER to RGBW or GRBW.
ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
    pixel_pin, num_pixels, brightness=0.2, auto_write=False, pixel_order=ORDER
)

_lightshow_thread = None
_lightshow_stop = threading.Event()

def measure_noise_floor(pa, duration=2.0):
    """Record a few seconds of silence at startup to calibrate."""
    print("Calibrating mic — please be quiet …")
    
    cal_stream = pa.open(
        rate=MIC_RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        input_device_index=1,
        frames_per_buffer=MIC_CHUNK,
    )
    
    samples = []
    start = time.time()
    while time.time() - start < duration:
        data = cal_stream.read(MIC_CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        samples.append(rms(audio))
    
    cal_stream.stop_stream()
    cal_stream.close()
    
    floor = np.mean(samples)
    threshold = floor * 2.5
    print(f"[CALIBRATION] Noise floor: {floor:.0f} → threshold: {threshold:.0f}")
    return threshold

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
    while len(sample_buffer) >= OWW_CHUNK:
        chunk = np.array(
            [sample_buffer.popleft() for _ in range(OWW_CHUNK)],
            dtype=np.int16,
        )
        if state["skip_chunks"] > 0:
            state["skip_chunks"] -= 1
            model.predict(chunk)
            continue

        predictions = model.predict(chunk)
        
        # Only log the highest scoring model this chunk
        best_word, best_score = max(predictions.items(), key=lambda x: x[1])
        if best_score > 0.1:  # ignore near-zero noise
            print(f"[PRED] {best_word}: {best_score:.3f}")

        for wake_word, score in predictions.items():
            if (score >= DETECTION_THRESHOLD
                    and not state["recording"]
                    and not state["processing"]
                    and not state["needs_processing"]):
                print(f"[DETECTED] '{wake_word}' score={score:.3f}")
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
        if len(sample_buffer) > OWW_CHUNK * 3:
            print(f"[BUFFER WARNING] size={len(sample_buffer)}")
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
    req = requests.post(url, headers=headers, json=data, timeout=60)

    result = req.json()
    
    text = result.get("output", {}).get("text")
    print("Transcript:", text)
    return text
    

def play_audio(response):
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    with wave.open(OUTPUT_FILENAME, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        for audio_chunk in voice.synthesize(response):
            wav_file.writeframes(audio_chunk.audio_int16_bytes)

    print(f"Audio saved to {OUTPUT_FILENAME}")
    subprocess.run(["aplay", OUTPUT_FILENAME], check=True)

def _lightshow_loop():
    """Runs in background thread until stop event is set."""
    while not _lightshow_stop.is_set():
        rainbow_cycle(0.001)

def wheel(pos):
    # Input a value 0 to 255 to get a color value.
    # The colours are a transition r - g - b - back to r.
    if pos < 0 or pos > 255:
        r = g = b = 0
    elif pos < 85:
        r = int(pos * 3)
        g = int(255 - pos * 3)
        b = 0
    elif pos < 170:
        pos -= 85
        r = int(255 - pos * 3)
        g = 0
        b = int(pos * 3)
    else:
        pos -= 170
        r = 0
        g = int(pos * 3)
        b = int(255 - pos * 3)
    return (r, g, b) if ORDER in {neopixel.RGB, neopixel.GRB} else (r, g, b, 0)


def rainbow_cycle(wait):
    for j in range(255):
        for i in range(num_pixels):
            pixel_index = (i * 256 // num_pixels) + j
            pixels[i] = wheel(pixel_index & 255)
        pixels.show()
        time.sleep(wait)


def take_picture(prompt: str):
    image_b64 = detector.capture_image_b64()
    url = "https://ai.hackclub.com/proxy/v1/responses"
    headers={
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "google/gemini-2.5-flash-lite-preview-09-2025",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_text",
                        "text": "When responding, don't use any special characters that aren't words or punctuation, your response will be played through TTS. Also make responses roughly 2-3 sentences long.",
                    },
                ],
            }
        ],
    }
    req = requests.post(url, headers=headers, json=data, timeout=30)
    result = req.json()
    print(f"Model raw response: {result}")
    play_audio(result["output"][0]["content"][0]["text"])

def imageGen(prompt: str, image: bool = False):
    url = "https://ai.hackclub.com/proxy/v1/responses"
    headers={
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json",
    }
    if image:
        image_b64 = detector.capture_image_b64()
        json={
            "model": "openai/gpt-5.4-image-2",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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
    else:
        json={
            "model": "openai/gpt-5.4-image-2",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }]
            "modalities": ["image", "text"],
            "size": "320x240"
        }
    req = requests.post(url, headers=headers, json=data, timeout=30)
    result = req.json()
    print(f"Model raw response: {result}")
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

def get_objects_detected():
    play_audio(detector.get_objects_detected())

def get_battery():
    play_audio(f"Battery level is at {readings['percent']:.1f}% with {readings['load_voltage']:.2f} volts, {readings['current']:.6f} amps, and {readings['power']:.3f} watts.")

def Movement(turn: str, offset: float = 0.0):
    if (turn == "left"):
        kit.continuous_servo[0].throttle = defaultThrottle + offset
        kit.continuous_servo[1].throttle = defaultThrottle + offset
        kit.continuous_servo[14].throttle = defaultThrottle + offset
        kit.continuous_servo[15].throttle = defaultThrottle + offset
    elif (turn == "right"):
        kit.continuous_servo[0].throttle = defaultThrottle - offset
        kit.continuous_servo[1].throttle = defaultThrottle - offset
        kit.continuous_servo[14].throttle = defaultThrottle - offset
        kit.continuous_servo[15].throttle = defaultThrottle - offset
    elif (turn == "forward"):
        kit.continuous_servo[0].throttle = defaultThrottle + offset
        kit.continuous_servo[1].throttle = defaultThrottle + offset
        kit.continuous_servo[14].throttle = defaultThrottle - offset
        kit.continuous_servo[15].throttle = defaultThrottle - offset
    elif (turn == "backward"):
        kit.continuous_servo[0].throttle = defaultThrottle - offset
        kit.continuous_servo[1].throttle = defaultThrottle - offset
        kit.continuous_servo[14].throttle = defaultThrottle + offset
        kit.continuous_servo[15].throttle = defaultThrottle + offset

def spin():
    Movement(turn="right", offset=0.2)
    time.sleep(2)
    Movement(turn="right", offset=0.0)

def move(direction: str, seconds: float):
    if direction == "forward":
        Movement(turn="forward", offset=0.2)
    elif direction == "backward":
        Movement(turn="backward", offset=0.2)
    time.sleep(seconds)
    Movement(turn="left", offset=0.0)

def turn(direction: str, seconds: float):
    if direction == "left":
        Movement(turn="left", offset=0.2)
    elif direction == "right":
        Movement(turn="right", offset=0.2)
    time.sleep(seconds)
    Movement(turn="left", offset=0.0)

def tail_lightshow():
    global _lightshow_thread
    
    if _lightshow_thread and _lightshow_thread.is_alive():
        # Already running — turn it off
        _lightshow_stop.set()
        _lightshow_thread.join()
        _lightshow_thread = None
        pixels.fill((0, 0, 0))
        pixels.show()
        play_audio("Tail lightshow stopped.")
    else:
        # Not running — turn it on
        _lightshow_stop.clear()
        _lightshow_thread = threading.Thread(target=_lightshow_loop, daemon=True)
        _lightshow_thread.start()
        play_audio("Tail lightshow started.")

def play_music(song: str):
    res = requests.get(
        "https://api.deezer.com/search",
        params={"q": song, "limit": 1},
        timeout=10
    )
    data = res.json()

    if not data.get("data"):
        return f"No results found for '{song}'."

    track = data["data"][0]
    preview_url = track["preview"]
    title = track["title"]
    artist = track["artist"]["name"]

    # Stream the preview MP3 into pygame
    audio = requests.get(preview_url, timeout=10)
    audio_buffer = io.BytesIO(audio.content)

    play_audio(f"Now playing: {title} by {artist} (30s preview).")
    pygame.mixer.music.load(audio_buffer, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def follow_person():
    # PID constants — tune these for your robot
    Kp = 0.003
    Ki = 0.0001
    Kd = 0.001

    screen_width = 640  # adjust to your camera resolution
    screen_center_x = screen_width / 2
    deadzone = 30  # pixels — don't correct if already close enough

    integral = 0.0
    last_error = 0.0
    last_seen = time.time()
    timeout = 10.0  # stop after 10s of no person detected

    play_audio("Following person.")

    while True:
        detections = detector.get_detections()
        labels = detector.get_labels()

        # Find first person detection
        person = next(
            (d for d in detections if labels[int(d.category)].lower() == "person"),
            None
        )

        if person is None:
            Movement(turn="left", offset=0.0)  # stop
            if time.time() - last_seen > timeout:
                play_audio("Lost person, stopping.")
                break
            time.sleep(0.05)
            continue

        last_seen = time.time()

        # Get horizontal center of bounding box
        x, y, w, h = person.box
        person_center_x = x + w / 2
        error = person_center_x - screen_center_x

        if abs(error) < deadzone:
            Movement(turn="left", offset=0.0)  # centered, stop turning
            time.sleep(0.05)
            continue

        # PID calculation
        integral += error
        derivative = error - last_error
        last_error = error

        correction = Kp * error + Ki * integral + Kd * derivative
        correction = max(-0.2, min(0.2, correction))  # clamp to safe throttle range

        # Positive error = person is to the right, turn right
        if error > 0:
            kit.continuous_servo[0].throttle = defaultThrottle - correction
            kit.continuous_servo[1].throttle = defaultThrottle - correction
            kit.continuous_servo[14].throttle = defaultThrottle + correction
            kit.continuous_servo[15].throttle = defaultThrottle + correction
        else:
            kit.continuous_servo[0].throttle = defaultThrottle + abs(correction)
            kit.continuous_servo[1].throttle = defaultThrottle + abs(correction)
            kit.continuous_servo[14].throttle = defaultThrottle - abs(correction)
            kit.continuous_servo[15].throttle = defaultThrottle - abs(correction)

        time.sleep(0.05)

    Movement(turn="left", offset=0.0)  # ensure stopped on exit
    return "Stopped following person."

def predictive_driving(prompt: str):
    driving_system_prompt = """
You are controlling a 4-wheel robot. Based on the image and the user's goal, output ONLY a JSON object. No extra text.

Robot characteristics:
- turn(direction, seconds): "left" or "right". 1 second ≈ 90 degree turn.
- move(direction, seconds): "forward" or "backward". 1 second ≈ 30cm of travel.
- Estimate distances and angles from the image as best you can.

Return format:
{
  "done": false,
  "commands": [
    {"fn": "turn", "args": {"direction": "left", "seconds": 0.5}},
    {"fn": "move", "args": {"direction": "forward", "seconds": 1.2}}
  ]
}

When the goal is achieved, return:
{ "done": true, "commands": [] }
"""

    max_iterations = 10  # safety cap so it doesn't loop forever
    all_executed = []

    for i in range(max_iterations):
        image_b64 = detector.capture_image_b64()

        data = {
            "model": "google/gemini-2.5-flash-lite-preview-09-2025",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": driving_system_prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                        {"type": "input_text", "text": f"Goal: {prompt}"},
                    ],
                }
            ],
        }

        req = requests.post(
            "https://ai.hackclub.com/proxy/v1/responses",
            headers={
                "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=30
        )

        raw = req.json()["output"][0]["content"][0]["text"]
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)

        if parsed.get("done"):
            play_audio("Goal reached.")
            break

        for cmd in parsed.get("commands", []):
            fn_name = cmd["fn"]
            args = cmd["args"]
            if fn_name == "move":
                move(**args)
            elif fn_name == "turn":
                turn(**args)
            all_executed.append(f"{fn_name}({args})")

    else:
        # Hit max_iterations without done=true
        play_audio("Reached maximum attempts without completing the goal.")

    summary = "Executed: " + ", ".join(all_executed)
    return summary

# --- Map of available functions ---
FUNCTIONS = {
    "get_battery": get_battery,
    "take_picture": take_picture,
    "spin": spin,
    "turn": turn,
    "play_music": play_music,
    "tail_lightshow": tail_lightshow,
    "follow_person": follow_person,
    "get_objects_detected": get_objects_detected,
    "predictive_driving": predictive_driving,
    "move": move,
    "imageGen": imageGen
}

# --- Describe tools to the model ---
TOOLS_DESCRIPTION = """
You are a helpful assistant on a robot dog. When the user asks something that would use these tools, respond ONLY with a JSON object (no extra text, no markdown) in this format:
{
  "tool": "<tool_name>",
  "args": { "<arg_name>": <value>, ... }
}

Available tools:
- get_battery() — returns a string describing the current battery level and readings from the INA219 sensor
- take_picture(prompt: str) — uses the camera to take a photo and uses the prompt argument and photo to respond to the user with whatever extra they asked about what the camera sees (e.g. "what's in front of me?" or "make a joke about what you see")
- spin() - makes the robot spin in a circle
- turn(direction: str, seconds: float) - turns the robot in the specified direction ("left" or "right") for a specified number of seconds
- move(direction: str, seconds: float) - moves the robot forward by direction ("forward" or "backward") for a specified number of seconds
- predictive_driving(prompt: str) - takes a photo using the camera and uses the prompt and photo to move a series of turns
- get_objects_detected() - returns name of objects currently detected in front of the robot, only use when there isn't anything else in the prompt that would rather be another prompt for take_picture()
- follow_person() - uses the camera to identify and follow a person in front of the robot
- tail_lightshow() - running it flips it on or off
- play_music(song: str) - plays a song through the robot's speakers
- imageGen(prompt: str, image: bool) - generates an image based on the prompt that will be displayed on the robot's screen, can optionally take into account to include the camera as input if image=true

If no tool applies, use:
{ "tool": "none", "args": {}, "response": "your plain text answer here" }
"""

messages = [
    {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": TOOLS_DESCRIPTION,
            }
        ],
    }
]

def model_return(text):
    if not text:
        print("[ERROR] No text to send to model")
        return None
    messages.append({
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": text,
            }
        ],
    })

    print("Sending to model...")
    model = "google/gemini-2.5-flash-lite-preview-09-2025"
    url = "https://ai.hackclub.com/proxy/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('HACKCLUB_API_KEY')}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    data = {
        "model": model,
        "input": messages,
    }
    req = requests.post(url, headers=headers, json=data, timeout=30)
    result = req.json()

    # Extract text from the /responses endpoint format
    raw = result["output"][0]["content"][0]["text"]

    # Append assistant reply to history
    messages.append({
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": raw}],
    })

    # Strip markdown fences in case the model wraps JSON in ```
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    parsed = json.loads(raw)

    if parsed["tool"] == "none":
        return parsed["response"]

    # Look up and call the real function
    tool_name = parsed["tool"]
    args = parsed["args"]

    if tool_name not in FUNCTIONS:
        return f"Unknown tool: {tool_name}"

    result = FUNCTIONS[tool_name](**args)
    return str(result)


def finish_recording() -> None:
    if not state["recording"]:
        return
    
    # Reset model state
    for model_obj in state["oww_model"].models.values():
        if hasattr(model_obj, 'reset'):
            model_obj.reset()
    if hasattr(state["oww_model"], 'prediction_buffer'):
        for key in state["oww_model"].prediction_buffer:
            state["oww_model"].prediction_buffer[key] = [0.0] * len(state["oww_model"].prediction_buffer[key])
    
    frames    = state["recorded_frames"]
    wake_word = state["wake_word"]
    state["recording"]       = False
    state["recorded_frames"] = []
    state["silence_start"]   = None
    state["record_start"]    = None
    state["wake_word"]       = "unknown"
    state["processing"]      = True
    sample_buffer.clear()
    global resampler
    resampler = soxr.ResampleStream(
        in_rate=MIC_RATE,
        out_rate=TARGET_RATE,
        num_channels=CHANNELS,
        quality="HQ",
        dtype="int16",
    )

    if frames:
        filename = save_recording(frames, wake_word)
        state["pending_filename"] = filename
        state["needs_processing"] = True

    print("[PROCESSING] Will transcribe shortly …\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    global SILENCE_THRESHOLD

    print("Loading openWakeWord model …")
    oww_model = Model()
    state["oww_model"] = oww_model  # ← add this
    print(f"Loaded models: {list(oww_model.models.keys())}")

    pa = pyaudio.PyAudio()
    SILENCE_THRESHOLD = measure_noise_floor(pa)  # ← before pa.open(callback...)

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

    print(
        f"\nListening …  "
        f"(mic={MIC_RATE} Hz → target={TARGET_RATE} Hz, "
        f"chunk={OWW_CHUNK} samples = {OWW_CHUNK/TARGET_RATE*1000:.1f} ms)"
    )
    print("Press Ctrl+C to stop.\n")

    try:
        stream.start_stream()
        while True:
            if state["needs_processing"]:
                state["needs_processing"] = False

                stream.stop_stream()

                print("Transcribing …")
                text = transcribe_audio(state["pending_filename"])

                if not text or not text.strip():
                    print("[SKIPPING] Empty transcript")
                    sample_buffer.clear()
                    state["processing"] = False
                    stream.start_stream()
                    continue

                print("Getting response …")
                response = model_return(text)
                print("[RESPONSE]", response)

                sample_buffer.clear()
                state["skip_chunks"] = int(np.ceil(15.0 * TARGET_RATE / OWW_CHUNK))
                state["processing"] = False
                stream.start_stream()  # now safe — skip_chunks is already set

                time.sleep(0.5)
                state["skip_chunks"] = 50

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("Done.")


if __name__ == "__main__":
    main()