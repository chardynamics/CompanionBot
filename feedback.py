import lgpio
import time

FEEDBACK_PINS = {
    "fl": 6,
    "fr": 13,
    "rl": 23,
    "rr": 22,
}

h = lgpio.gpiochip_open(0)

for name, pin in FEEDBACK_PINS.items():
    lgpio.gpio_claim_input(h, pin)

def read_duty_cycle(pin):
    # Measure high and low pulse widths
    while lgpio.gpio_read(h, pin) == 1:
        pass
    while lgpio.gpio_read(h, pin) == 0:
        pass
    high_start = time.monotonic_ns()
    while lgpio.gpio_read(h, pin) == 1:
        pass
    high_time = time.monotonic_ns() - high_start
    while lgpio.gpio_read(h, pin) == 0:
        pass
    total_time = time.monotonic_ns() - high_start
    return (high_time / total_time) * 100

while True:
    for name, pin in FEEDBACK_PINS.items():
        dc = read_duty_cycle(pin)
        angle = (dc / 100) * 360
        print(f"{name}: duty={dc:.1f}% angle={angle:.1f}°")
    time.sleep(0.1)