"""
Robotic Dog - Parallax Feedback 360 Servo Controller
=====================================================
Hardware : Raspberry Pi 4 + Adafruit PCA9685 PWM HAT
Libraries: adafruit-circuitpython-servokit, pigpio

Wiring per servo:
  Red    -> 5V
  Black  -> GND
  White  -> PCA9685 PWM channel  (speed control)
  Yellow -> Raspberry Pi GPIO    (position feedback)

Setup (run once):
  pip install adafruit-circuitpython-servokit pigpio
  sudo pigpiod
"""

import math
import threading
import time

import pigpio
from adafruit_servokit import ServoKit


# ---------------------------------------------------------------------------
# Wiring — edit to match your physical setup
# ---------------------------------------------------------------------------

# (PCA9685 channel, inverted?, odo_inverted?)
SERVO_CHANNELS = {
    "front_left":  (0,  False, False),
    "front_right": (14, True,  True),
    "rear_left":   (1,  False, True),
    "rear_right":  (13, True,  False),
}

# BCM GPIO pins for the yellow feedback wires
FEEDBACK_PINS = {
    "front_left":  6,
    "front_right": 13,
    "rear_left":   22,
    "rear_right":  23,
}

# ---------------------------------------------------------------------------
# Per-servo neutral throttle
# Run dog.calibrate_neutral() once to find the right value for each servo,
# then hard-code the results here.
# ---------------------------------------------------------------------------
SERVO_NEUTRAL = {
    "front_left":  0.20,
    "front_right": 0.20,
    "rear_left":   0.20,
    "rear_right":  0.20,
}

# ---------------------------------------------------------------------------
# Physical dimensions — measure your robot and update these
# ---------------------------------------------------------------------------

WHEEL_DIAMETER_MM      = 65.0
TRACK_WIDTH_MM         = 245.0
WHEEL_CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM

# ---------------------------------------------------------------------------
# Parallax feedback signal constants (from datasheet)
# ---------------------------------------------------------------------------

DUTY_CYCLE_MIN   = 2.9
DUTY_CYCLE_MAX   = 97.1
UNITS_PER_CIRCLE = 360

PULSE_MIN_US = 1000
PULSE_MAX_US = 2000

# ---------------------------------------------------------------------------
# Motion tuning
# ---------------------------------------------------------------------------

RAMP_UP_DELAY_S = 0.3


# ---------------------------------------------------------------------------
# FeedbackReader
# ---------------------------------------------------------------------------

class FeedbackReader:
    """
    Reads the Parallax 360 feedback PWM on the yellow wire.

    The servo outputs a ~910 Hz square wave whose duty cycle encodes
    absolute wheel position within one revolution (0-360 deg).

    Properties
    ----------
    angle         : absolute position within the current revolution (0-360 deg)
    total_degrees : cumulative signed degrees turned since reset_odometry()
    distance_mm   : total_degrees converted to linear wheel travel (mm)
    """

    def __init__(self, pi: pigpio.pi, pin: int):
        self._pi         = pi
        self.pin         = pin
        self._lock       = threading.Lock()
        self._duty_cycle = 0.0
        self._prev_angle = None
        self._total_deg  = 0.0
        self._last_rise  = None
        self._last_fall  = None

        pi.set_mode(pin, pigpio.INPUT)
        self._cb = pi.callback(pin, pigpio.EITHER_EDGE, self._on_edge)

    def _on_edge(self, gpio, level, tick):
        if level == 2:
            return

        if level == 1:  # rising edge — new cycle started
            if self._last_rise is not None and self._last_fall is not None:
                t_cycle = pigpio.tickDiff(self._last_rise, tick)
                t_high  = pigpio.tickDiff(self._last_rise, self._last_fall)
                if t_cycle > 0:
                    dc        = 100.0 * t_high / t_cycle
                    new_angle = ((dc - DUTY_CYCLE_MIN) /
                                 (DUTY_CYCLE_MAX - DUTY_CYCLE_MIN + 1.0) *
                                 UNITS_PER_CIRCLE)
                    with self._lock:
                        self._duty_cycle = dc
                        if self._prev_angle is not None:
                            delta = new_angle - self._prev_angle
                            half  = UNITS_PER_CIRCLE / 2.0
                            if delta >  half: delta -= UNITS_PER_CIRCLE
                            if delta < -half: delta += UNITS_PER_CIRCLE
                            self._total_deg += delta
                        self._prev_angle = new_angle
            self._last_rise = tick

        else:  # falling edge
            self._last_fall = tick

    @property
    def angle(self) -> float:
        with self._lock:
            dc = max(DUTY_CYCLE_MIN, min(DUTY_CYCLE_MAX, self._duty_cycle))
        return ((dc - DUTY_CYCLE_MIN) /
                (DUTY_CYCLE_MAX - DUTY_CYCLE_MIN + 1.0) * UNITS_PER_CIRCLE)

    @property
    def total_degrees(self) -> float:
        with self._lock:
            return self._total_deg

    @property
    def distance_mm(self) -> float:
        return (self.total_degrees / 360.0) * WHEEL_CIRCUMFERENCE_MM

    def reset_odometry(self):
        with self._lock:
            self._total_deg = 0.0

    def cleanup(self):
        self._cb.cancel()


# ---------------------------------------------------------------------------
# Servo360
# ---------------------------------------------------------------------------

class Servo360:
    """
    One Parallax Feedback 360 wheel servo.

    set_speed(+1.0) always means "forward" regardless of which side the
    servo is mounted on — the inverted flag handles the physical flip.
    """

    def __init__(self, kit: ServoKit, channel: int, feedback_pin: int,
                 pi: pigpio.pi, name: str = "servo",
                 inverted: bool = False, odo_inverted: bool = False,
                 neutral: float = 0.0):
        self.name      = name
        self._inverted = inverted
        self._odo_sign = -1.0 if odo_inverted else 1.0
        self._neutral  = neutral
        self._servo    = kit.continuous_servo[channel]
        self._servo.set_pulse_width_range(PULSE_MIN_US, PULSE_MAX_US)
        self._feedback = FeedbackReader(pi, feedback_pin)

    def _speed_to_throttle(self, speed: float) -> float:
        """
        Convert logical speed (-1.0 to +1.0) to a raw throttle value.

        Order of operations:
          1. Clamp to [-1, 1]
          2. Flip sign if servo is mounted inverted
          3. Shift output so speed=0 maps to neutral, and +-1 still reach +-1
        """
        speed = max(-1.0, min(1.0, speed))

        if self._inverted:
            speed = -speed

        if speed >= 0.0:
            throttle = self._neutral + speed * (1.0 - self._neutral)
        else:
            throttle = self._neutral + speed * (1.0 + self._neutral)

        return max(-1.0, min(1.0, throttle))

    def set_speed(self, speed: float):
        """speed: -1.0 = full reverse, 0.0 = stop, +1.0 = full forward"""
        self._servo.throttle = self._speed_to_throttle(speed)

    def stop(self):
        """Send the neutral throttle to halt the wheel."""
        self._servo.throttle = self._neutral
        time.sleep(0.05)

    @property
    def angle(self) -> float:
        return self._feedback.angle

    @property
    def total_degrees(self) -> float:
        return self._feedback.total_degrees * self._odo_sign

    @property
    def distance_mm(self) -> float:
        return self._feedback.distance_mm * self._odo_sign

    def reset_odometry(self):
        self._feedback.reset_odometry()

    def cleanup(self):
        self.stop()
        self._feedback.cleanup()


# ---------------------------------------------------------------------------
# RoboticDog
# ---------------------------------------------------------------------------

class RoboticDog:
    """
    Four-wheel drive robot with odometry-based motion control.

    Quick reference
    ---------------
        dog.calibrate_neutral()        # find stop throttle per servo (run once)
        dog.travel_distance_mm(500)    # forward 50 cm
        dog.travel_distance_mm(-200)   # backward 20 cm
        dog.turn_degrees(90)           # right 90 deg
        dog.turn_degrees(-45)          # left 45 deg
        dog.print_status()             # show all wheel odometers
    """

    def __init__(self):
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError(
                "Cannot connect to pigpiod.\n"
                "Start it with:  sudo pigpiod")

        self.kit = ServoKit(channels=16)

        self.servos: dict[str, Servo360] = {}
        for name, (channel, inverted, odo_inverted) in SERVO_CHANNELS.items():
            pin = FEEDBACK_PINS[name]
            self.servos[name] = Servo360(
                self.kit, channel, pin,
                pi=self._pi, name=name,
                inverted=inverted,
                odo_inverted=odo_inverted,
                neutral=SERVO_NEUTRAL[name],
            )
            print(f"  {name:>12s}  channel={channel}  GPIO={pin}  "
                  f"inverted={inverted}  odo_inverted={odo_inverted}  "
                  f"neutral={SERVO_NEUTRAL[name]:.3f}")

        time.sleep(0.2)
        print("RoboticDog ready.\n")

    # -- Helpers -------------------------------------------------------------

    def stop_all(self):
        for s in self.servos.values():
            s.stop()

    def _reset_all_odometry(self):
        for s in self.servos.values():
            s.reset_odometry()

    def print_status(self):
        print("Wheel odometry:")
        for name, s in self.servos.items():
            print(f"  {name:>12s}  "
                  f"angle={s.angle:6.1f} deg  "
                  f"distance={s.distance_mm:8.1f} mm  "
                  f"({s.total_degrees:.1f} deg total)")

    # -- Calibration ---------------------------------------------------------

    def calibrate_neutral(self):
        """
        Interactive helper — finds the true stop throttle for each servo.

        For each servo:
          +   increases throttle by 0.01
          -   decreases throttle by 0.01
          q   accepts current value and moves to the next servo

        Copy the printed SERVO_NEUTRAL values into the constant at the
        top of this file, then comment out calibrate_neutral() in main().
        """
        print("=== Neutral calibration ===")
        print("Commands:  + (increase 0.01)   - (decrease 0.01)   q (accept & next)\n")
        results = {}
        for name, servo in self.servos.items():
            print(f"Calibrating '{name}' ...")
            throttle = servo._neutral
            while True:
                servo._servo.throttle = throttle
                cmd = input(
                    f"  throttle={throttle:.3f} | still moving? [+/-/q] "
                ).strip()
                if cmd == "+":
                    throttle = round(throttle + 0.01, 3)
                elif cmd == "-":
                    throttle = round(throttle - 0.01, 3)
                elif cmd == "q":
                    results[name] = throttle
                    servo._servo.throttle = throttle
                    print(f"  Accepted {throttle:.3f} for '{name}'\n")
                    break
                else:
                    print("  Unknown command — use +, -, or q.")

        print("\nCalibration complete. Update SERVO_NEUTRAL with these values:")
        print("SERVO_NEUTRAL = {")
        for name, val in results.items():
            print(f'    "{name}": {val:.3f},')
        print("}")

    # -- Motion --------------------------------------------------------------

    def travel_distance_mm(self, distance_mm: float, speed: float = 0.4,
                           timeout_s: float = 30.0):
        """
        Drive straight forward or backward for a precise distance.

        Parameters
        ----------
        distance_mm : distance to travel in mm (negative = reverse)
        speed       : drive speed 0.0-1.0 (default 0.4)
        timeout_s   : safety cutoff in seconds
        """
        direction = 1.0 if distance_mm >= 0 else -1.0
        target_mm = abs(distance_mm)
        label     = "forward" if direction > 0 else "backward"
        print(f"Travelling {label} {target_mm:.0f} mm ...")

        for s in self.servos.values():
            s.set_speed(direction * abs(speed))

        # Let wheels spin up, then reset so ramp distance is not counted
        time.sleep(RAMP_UP_DELAY_S)
        self._reset_all_odometry()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            avg_mm = sum(abs(s.distance_mm)
                         for s in self.servos.values()) / 4.0
            if avg_mm >= target_mm:
                break
            time.sleep(0.005)
        else:
            print("  Warning: timed out before reaching target distance.")

        self.stop_all()
        actual_mm = sum(abs(s.distance_mm)
                        for s in self.servos.values()) / 4.0
        print(f"  Stopped at {actual_mm:.1f} mm  (target {target_mm:.1f} mm)")

    def turn_degrees(self, degrees: float, speed: float = 0.35,
                     timeout_s: float = 15.0):
        """
        Turn in place by a precise number of degrees.

        Parameters
        ----------
        degrees   : degrees to turn (positive = right/CW, negative = left/CCW)
        speed     : drive speed 0.0-1.0 (default 0.35)
        timeout_s : safety cutoff in seconds
        """
        direction = 1.0 if degrees >= 0 else -1.0
        arc_mm    = math.pi * TRACK_WIDTH_MM * (abs(degrees) / 360.0)
        label     = "right" if direction > 0 else "left"
        print(f"Turning {label} {abs(degrees):.0f} deg  "
              f"(each side travels {arc_mm:.1f} mm) ...")

        left  = [self.servos["front_left"],  self.servos["rear_left"]]
        right = [self.servos["front_right"], self.servos["rear_right"]]

        # Left side forward, right side backward (or vice versa)
        for s in left:  s.set_speed( direction * abs(speed))
        for s in right: s.set_speed(-direction * abs(speed))

        # Let wheels spin up, then reset so ramp distance is not counted
        time.sleep(RAMP_UP_DELAY_S)
        self._reset_all_odometry()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            left_avg  = sum(abs(s.distance_mm) for s in left)  / 2.0
            right_avg = sum(abs(s.distance_mm) for s in right) / 2.0
            # Wait for BOTH sides to reach the arc — not just the average
            if left_avg >= arc_mm and right_avg >= arc_mm:
                break
            time.sleep(0.005)
        else:
            print("  Warning: timed out before reaching target angle.")

        self.stop_all()
        actual_arc = sum(abs(s.distance_mm)
                         for s in self.servos.values()) / 4.0
        actual_deg = (actual_arc / (math.pi * TRACK_WIDTH_MM)) * 360.0
        print(f"  Stopped at {actual_deg:.1f} deg  (target {abs(degrees):.1f} deg)")

    # -- Cleanup -------------------------------------------------------------

    def cleanup(self):
        self.stop_all()
        time.sleep(0.1)
        for s in self.servos.values():
            s.cleanup()
        self._pi.stop()
        print("Cleanup complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dog = RoboticDog()
    try:
        dog.turn_degrees(90, speed=0.4)
        dog.print_status()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        dog.cleanup()


if __name__ == "__main__":
    main()
