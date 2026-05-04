import time
import atexit
import lgpio
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

PULSES_PER_REV = 20
WHEEL_DIAMETER_CM = 6.5
WHEELBASE_CM = 15.0
WHEEL_CIRCUMFERENCE_CM = 3.14159 * WHEEL_DIAMETER_CM
CM_PER_PULSE = WHEEL_CIRCUMFERENCE_CM / PULSES_PER_REV

# --- GPIO / lgpio setup ---
CHIP = lgpio.gpiochip_open(0)

def _encoder_callback(chip, gpio, level, tick, name):
    encoder_counts[name] += 1

for name, pin in ENCODER_PINS.items():
    lgpio.gpio_claim_input(CHIP, pin, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_alert(CHIP, pin, lgpio.RISING_EDGE)
    lgpio.callback(CHIP, pin, lgpio.RISING_EDGE,
                   lambda chip, gpio, level, tick, n=name: _encoder_callback(chip, gpio, level, tick, n))

atexit.register(lgpio.gpiochip_close, CHIP)

# --- Low-level movement ---
def Movement(turn: str, offset: float = 0.0):
    if turn == "forward":
        kit.continuous_servo[0].throttle  =  defaultThrottle + offset
        kit.continuous_servo[1].throttle  =  defaultThrottle + offset
        kit.continuous_servo[14].throttle =  defaultThrottle - offset
        kit.continuous_servo[15].throttle =  defaultThrottle - offset
    elif turn == "backward":
        kit.continuous_servo[0].throttle  =  defaultThrottle - offset
        kit.continuous_servo[1].throttle  =  defaultThrottle + offset
        kit.continuous_servo[14].throttle =  defaultThrottle + offset
        kit.continuous_servo[15].throttle =  defaultThrottle - offset
    elif turn == "left":
        kit.continuous_servo[0].throttle  =  defaultThrottle + offset
        kit.continuous_servo[1].throttle  =  defaultThrottle + offset
        kit.continuous_servo[14].throttle =  defaultThrottle + offset
        kit.continuous_servo[15].throttle =  defaultThrottle + offset
    elif turn == "right":
        kit.continuous_servo[0].throttle  =  defaultThrottle - offset
        kit.continuous_servo[1].throttle  =  defaultThrottle - offset
        kit.continuous_servo[14].throttle =  defaultThrottle - offset
        kit.continuous_servo[15].throttle =  defaultThrottle - offset

def killThrottle():
    for ch in [0, 1, 14, 15]:
        kit.continuous_servo[ch].throttle = defaultThrottle

def move(direction: str, distance_cm: float):
    for key in encoder_counts:
        encoder_counts[key] = 0
    time.sleep(0.05)
    for key in encoder_counts:
        encoder_counts[key] = 0

    target_pulses = distance_cm / CM_PER_PULSE
    print(f"move({direction}, {distance_cm}cm) — target: {target_pulses:.1f} pulses")

    Movement(turn=direction, offset=0.2)

    # Wait until we've seen at least a few fresh pulses before checking target
    time.sleep(0.1)

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

    print("\n[1] Forward 10cm")
    move("forward", 10)
    time.sleep(1)

    print("\nTest complete.")