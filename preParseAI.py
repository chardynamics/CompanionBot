import json
import requests
import os
import time
import wave
import io
import board
import neopixel
import threading
import pygame
import subprocess

from dotenv import load_dotenv
#from ina219 import INA219
#from adafruit_servokit import ServoKit
from src.modules.ai_camera import IMX500Detector
from piper.voice import PiperVoice


current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILENAME = os.path.join(current_dir, 'output.wav')
model_path = os.path.join(current_dir, 'src', 'models', 'en_US-amy-medium.onnx')
voice = PiperVoice.load(model_path)
SAMPLE_RATE = 22050
load_dotenv()
#kit = ServoKit(channels=16)

#ina = INA219(addr=0x41)
#readings = ina.getReadings()
defaultThrottle = 0.2

detector = IMX500Detector()
detector.start()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

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

PULSES_PER_REV = 20        # replace with your calibrated value
WHEEL_DIAMETER_CM = 6.4    # measure your actual wheel
WHEELBASE_CM = 26.5        # measure left wheel to right wheel

WHEEL_CIRCUMFERENCE_CM = 3.14159 * WHEEL_DIAMETER_CM
CM_PER_PULSE = WHEEL_CIRCUMFERENCE_CM / PULSES_PER_REV

_lightshow_thread = None
_lightshow_stop = threading.Event()

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
                        "text": "When responding, don't use any special characters that aren't words or punctuation, your response will be played through TTS.",
                    },
                ],
            }
        ],
    }
    req = requests.post(url, headers=headers, json=data, timeout=30)
    result = req.json()
    print(f"Model raw response: {result}")
    play_audio(result["output"][0]["content"][0]["text"])

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

def move(direction: str, distance_cm: float):
    # Reset counts
    for key in encoder_counts:
        encoder_counts[key] = 0

    target_pulses = distance_cm / CM_PER_PULSE

    if direction == "forward":
        Movement(turn="forward", offset=0.2)
    elif direction == "backward":
        Movement(turn="backward", offset=0.2)

    # Wait until average pulse count hits target
    while True:
        avg = sum(encoder_counts.values()) / len(encoder_counts)
        if avg >= target_pulses:
            break
        time.sleep(0.005)

    Movement(turn="left", offset=0.0)  # stop

def turn(direction: str, degrees: float):
    for key in encoder_counts:
        encoder_counts[key] = 0

    # Arc length the outer wheels need to travel for the given angle
    arc_cm = (degrees / 360.0) * 3.14159 * WHEELBASE_CM
    target_pulses = arc_cm / CM_PER_PULSE

    if direction == "left":
        Movement(turn="left", offset=0.2)
    elif direction == "right":
        Movement(turn="right", offset=0.2)

    while True:
        # Use the faster (outer) wheels as the reference
        if direction == "left":
            avg = (encoder_counts["fl"] + encoder_counts["rl"]) / 2
        else:
            avg = (encoder_counts["fr"] + encoder_counts["rr"]) / 2
        if avg >= target_pulses:
            break
        time.sleep(0.005)

    Movement(turn="left", offset=0.0)  # stop

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

    play_audio(f"Now playing: {title} by {artist} (30 second preview).")
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
    print(f"Model raw response: {raw}")

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

print(model_return("Make a joke with the things you see"))