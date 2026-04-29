import time
import atexit
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit

# --- Servo setup ---
kit = ServoKit(channels=16)
defaultThrottle = 0.2

# --- Encoder constants (replace with your calibrated values) ---
ENCODER_PINS = {
    "fl": 6,
    "fr": 13,
    "rl": 23,
    "rr": 22,
}
encoder_counts = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}

PULSES_PER_REV = 20         # replace with your calibrated value
WHEEL_DIAMETER_CM = 6.5     # measure your wheel
WHEELBASE_CM = 15.0         # measure left wheel to right wheel

WHEEL_CIRCUMFERENCE_CM = 3.14159 * WHEEL_DIAMETER_CM
CM_PER_PULSE = WHEEL_CIRCUMFERENCE_CM / PULSES_PER_REV

# --- Encoder callback (must be defined before GPIO setup) ---
def _encoder_callback(channel, name):
    encoder_counts[name] += 1

# --- GPIO setup ---
GPIO.setmode(GPIO.BCM)
for name, pin in ENCODER_PINS.items():
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(
        pin, GPIO.RISING,
        callback=lambda channel, n=name: _encoder_callback(channel, n)
    )

atexit.register(GPIO.cleanup)

# --- Low-level movement ---
def Movement(turn: str, offset: float = 0.0):
    if turn == "forward":
        kit.continuous_servo[0].throttle = defaultThrottle + offset
        kit.continuous_servo[1].throttle = defaultThrottle + offset
        kit.continuous_servo[14].throttle = defaultThrottle - offset
        kit.continuous_servo[15].throttle = defaultThrottle - offset
    elif turn == "backward":
        kit.continuous_servo[0].throttle = defaultThrottle - offset
        kit.continuous_servo[1].throttle = defaultThrottle + offset
        kit.continuous_servo[14].throttle = defaultThrottle + offset
        kit.continuous_servo[15].throttle = defaultThrottle - offset
    elif turn == "left":
        kit.continuous_servo[0].throttle = defaultThrottle + offset
        kit.continuous_servo[1].throttle = defaultThrottle + offset
        kit.continuous_servo[14].throttle = defaultThrottle + offset
        kit.continuous_servo[15].throttle = defaultThrottle + offset
    elif turn == "right":
        kit.continuous_servo[0].throttle = defaultThrottle - offset
        kit.continuous_servo[1].throttle = defaultThrottle - offset
        kit.continuous_servo[14].throttle = defaultThrottle - offset
        kit.continuous_servo[15].throttle = defaultThrottle - offset

def killThrottle():
    for ch in [0, 1, 14, 15]:
        kit.continuous_servo[ch].throttle = defaultThrottle

# --- Encoder-based move ---
def move(direction: str, distance_cm: float):
    for key in encoder_counts:
        encoder_counts[key] = 0

    target_pulses = distance_cm / CM_PER_PULSE
    print(f"move({direction}, {distance_cm}cm) — target: {target_pulses:.1f} pulses")

    Movement(turn=direction, offset=0.2)

    while True:
        avg = sum(encoder_counts.values()) / len(encoder_counts)
        if avg >= target_pulses:
            break
        time.sleep(0.005)

    killThrottle()
    print(f"  done — counts: {encoder_counts}")

# --- Encoder-based turn ---
def turn(direction: str, degrees: float):
    for key in encoder_counts:
        encoder_counts[key] = 0

    arc_cm = (degrees / 360.0) * 3.14159 * WHEELBASE_CM
    target_pulses = arc_cm / CM_PER_PULSE
    print(f"turn({direction}, {degrees}°) — target: {target_pulses:.1f} pulses")

    Movement(turn=direction, offset=0.2)

    while True:
        # Use the outer wheels as reference
        if direction == "left":
            avg = (encoder_counts["fl"] + encoder_counts["rl"]) / 2
        else:
            avg = (encoder_counts["fr"] + encoder_counts["rr"]) / 2
        if avg >= target_pulses:
            break
        time.sleep(0.005)

    killThrottle()
    print(f"  done — counts: {encoder_counts}")

# --- Test sequence ---
if __name__ == "__main__":
    print("Starting movement test...")
    time.sleep(1)

    print("\n[1] Forward 50cm")
    move("forward", 50)
    time.sleep(1)

    print("\n[2] Turn left 90 degrees")
    turn("left", 90)
    time.sleep(1)

    print("\n[3] Forward 30cm")
    move("forward", 30)
    time.sleep(1)

    print("\n[4] Turn right 90 degrees")
    turn("right", 90)
    time.sleep(1)

    print("\n[5] Backward 50cm")
    move("backward", 50)
    time.sleep(1)

    print("\nTest complete.")