import lgpio
import time
from adafruit_servokit import ServoKit

# --- Servo setup ---
kit = ServoKit(channels=16)
defaultThrottle = 0.2

# --- Feedback pin mapping ---
FEEDBACK_PINS = {
    "fl": 6,
    "fr": 13,
    "rl": 23,
    "rr": 22,
}

# --- Physical constants (measure these with a ruler) ---
WHEEL_DIAMETER_CM = 6.5
WHEELBASE_CM = 25
WHEEL_CIRCUMFERENCE_CM = 3.14159 * WHEEL_DIAMETER_CM

# --- lgpio setup ---
h = lgpio.gpiochip_open(0)
for name, pin in FEEDBACK_PINS.items():
    lgpio.gpio_claim_input(h, pin)

# --- Low level movement ---
def Movement(turn: str, offset: float = 0.0):
    if turn == "forward":
        kit.continuous_servo[0].throttle = defaultThrottle - offset   # fl flipped
        kit.continuous_servo[1].throttle = defaultThrottle - offset   # rl flipped
        kit.continuous_servo[13].throttle = defaultThrottle - offset  # rr
        kit.continuous_servo[15].throttle = defaultThrottle - offset  # fr
    elif turn == "backward":
        kit.continuous_servo[0].throttle = defaultThrottle + offset   # fl flipped
        kit.continuous_servo[1].throttle = defaultThrottle + offset   # rl flipped
        kit.continuous_servo[13].throttle = defaultThrottle + offset  # rr
        kit.continuous_servo[15].throttle = defaultThrottle + offset  # fr
    elif turn == "left":
        kit.continuous_servo[0].throttle = defaultThrottle + offset   # fl flipped = forward
        kit.continuous_servo[1].throttle = defaultThrottle + offset   # rl flipped = forward
        kit.continuous_servo[13].throttle = defaultThrottle - offset  # rr forward (opposite)
        kit.continuous_servo[15].throttle = defaultThrottle - offset  # fr forward (opposite)
    elif turn == "right":
        kit.continuous_servo[0].throttle = defaultThrottle - offset   # fl flipped = backward
        kit.continuous_servo[1].throttle = defaultThrottle - offset   # rl flipped = backward
        kit.continuous_servo[13].throttle = defaultThrottle + offset  # rr backward (opposite)
        kit.continuous_servo[15].throttle = defaultThrottle + offset  # fr backward (opposite)

def killThrottle():
    for ch in [0, 1, 13, 15]:
        kit.continuous_servo[ch].throttle = defaultThrottle

# --- Read current angle from PWM feedback ---
def get_angle(pin):
    TIMEOUT = 0.1  # seconds before giving up
    
    deadline = time.monotonic() + TIMEOUT
    while lgpio.gpio_read(h, pin) == 1:
        if time.monotonic() > deadline:
            return None
    
    deadline = time.monotonic() + TIMEOUT
    while lgpio.gpio_read(h, pin) == 0:
        if time.monotonic() > deadline:
            return None

    high_start = time.monotonic_ns()
    
    deadline = time.monotonic() + TIMEOUT
    while lgpio.gpio_read(h, pin) == 1:
        if time.monotonic() > deadline:
            return None
    high_time = time.monotonic_ns() - high_start

    deadline = time.monotonic() + TIMEOUT
    while lgpio.gpio_read(h, pin) == 0:
        if time.monotonic() > deadline:
            return None
    total_time = time.monotonic_ns() - high_start

    dc = (high_time / total_time) * 100
    return (dc / 100) * 360

# --- Move exact distance in cm ---
def move(direction: str, distance_cm: float):
    rotations_needed = distance_cm / WHEEL_CIRCUMFERENCE_CM
    print(f"move({direction}, {distance_cm}cm) — needs {rotations_needed:.2f} rotations")

    prev_fl = get_angle(FEEDBACK_PINS["fl"])
    prev_fr = get_angle(FEEDBACK_PINS["fr"])

    Movement(turn=direction, offset=0.2)

    total_rotated = 0.0
    while total_rotated < rotations_needed:
        curr_fl = get_angle(FEEDBACK_PINS["fl"])
        curr_fr = get_angle(FEEDBACK_PINS["fr"])

        if curr_fl is None or curr_fr is None:
            continue

        delta_fl = (curr_fl - prev_fl) % 360
        delta_fr = (curr_fr - prev_fr) % 360

        # Ignore huge jumps — likely a bad reading
        if delta_fl < 180 and delta_fr < 180:
            avg_delta = ((delta_fl + delta_fr) / 2) / 360
            total_rotated += avg_delta

        prev_fl = curr_fl
        prev_fr = curr_fr

    killThrottle()
    print(f"  done — total rotations: {total_rotated:.2f}")

# --- Turn exact degrees ---
def turn(direction: str, degrees: float):
    arc_cm = (degrees / 360.0) * 3.14159 * WHEELBASE_CM
    rotations_needed = arc_cm / WHEEL_CIRCUMFERENCE_CM
    print(f"turn({direction}, {degrees}°) — needs {rotations_needed:.2f} rotations")

    if direction == "left":
        pin = FEEDBACK_PINS["fl"]
    else:
        pin = FEEDBACK_PINS["fr"]

    prev_angle = get_angle(pin)
    Movement(turn=direction, offset=0.2)

    total_rotated = 0.0
    while total_rotated < rotations_needed:
        curr_angle = get_angle(pin)
        if curr_angle is None:
            continue
            
        delta = (curr_angle - prev_angle) % 360

        # debug
        print(f"  curr={curr_angle:.1f} prev={prev_angle:.1f} delta={delta:.1f} total={total_rotated:.3f}")

        # ignore noise — only count meaningful movement
        if 0 < delta < 180:
            total_rotated += delta / 360

        prev_angle = curr_angle

    killThrottle()
    print(f"  done — total rotations: {total_rotated:.2f}")

if __name__ == "__main__":
    turn("left", 90)