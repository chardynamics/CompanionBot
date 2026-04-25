import RPi.GPIO as GPIO

# --- Encoder globals ---
ENCODER_PINS = {
    "fl": 14,   # front left  - servo 0
    "fr": 6,  # front right - servo 1
    "rl": 14,   # rear left   - servo 14
    "rr": 4,  # rear right  - servo 15
}

encoder_counts = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}

PULSES_PER_REV = 20       # measure this — spin wheel one full turn, count pulses
WHEEL_DIAMETER_CM = 6.4   # measure your wheel diameter

WHEEL_CIRCUMFERENCE_CM = 3.14159 * WHEEL_DIAMETER_CM
CM_PER_PULSE = WHEEL_CIRCUMFERENCE_CM / PULSES_PER_REV

GPIO.setmode(GPIO.BCM)
for name, pin in ENCODER_PINS.items():
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(
        pin, GPIO.RISING,
        callback=lambda channel, n=name: _encoder_callback(channel, n)
    )

def _encoder_callback(channel, name):
    encoder_counts[name] += 1