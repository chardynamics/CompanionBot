import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

ENCODER_PINS = {
    "fl": 6,
    "fr": 13, 
    "rl": 23,
    "rr": 22,
}

#encoder_counts = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}
encoder_counts = {"fr": 0, "rl": 0, "rr": 0}

PULSES_PER_REV = 20
WHEEL_DIAMETER_CM = 6.4
WHEEL_CIRCUMFERENCE_CM = 3.14159 * WHEEL_DIAMETER_CM
CM_PER_PULSE = WHEEL_CIRCUMFERENCE_CM / PULSES_PER_REV

def _encoder_callback(channel, name):
    encoder_counts[name] += 1

for name, pin in ENCODER_PINS.items():
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(
        pin, GPIO.RISING,
        callback=lambda channel, n=name: _encoder_callback(channel, n)
    )