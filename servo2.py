"""
Robotic Dog - Parallax Feedback 360 High-Speed Servo Controller
===============================================================
Hardware: Raspberry Pi 4 + Adafruit PCA9685 PWM HAT/breakout
Library:  adafruit-circuitpython-servokit

Wiring per servo:
  Red   -> 5V
  Black -> GND
  White -> PCA9685 PWM channel (control signal)
  Yellow-> Raspberry Pi GPIO pin (feedback signal, 3.3V logic)

Feedback signal spec (from Parallax datasheet):
  - Period (tCycle): 1/910 Hz ≈ 1.1 ms  (+/- 5%)
  - High pulse (tHigh): 3.3V pulse whose width encodes position
  - Duty Cycle = 100% x (tHigh / tCycle)
  - Duty Cycle Min = 2.9%  (origin / 0°)
  - Duty Cycle Max = 97.1% (just before one full CW revolution)
  - Angular position = (DC - DC_min) / (DC_max - DC_min + 1) * units_per_circle

Install dependencies:
  pip3 install adafruit-circuitpython-servokit RPi.GPIO
"""

import time
import threading
import board
import busio
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# PCA9685 channels assigned to each leg servo
# Right-side servos are mounted opposite, so their throttle is inverted.
# Format: { name: (channel, inverted) }
SERVO_CHANNELS = {
    "front_left":  (1, False),
    "front_right": (14, True),
    "rear_left":   (0, False),
    "rear_right":  (13, True),
}

# GPIO BCM pin numbers connected to the yellow feedback wires
FEEDBACK_PINS = {
    "front_left":  6,
    "front_right": 13,
    "rear_left":   23,
    "rear_right":  22,
}
# Feedback signal constants (from datasheet)
DUTY_CYCLE_MIN = 2.9        # % at origin (0 units)
DUTY_CYCLE_MAX = 97.1       # % approaching 1 full CW revolution
UNITS_PER_CIRCLE = 360      # degrees; change to 64 for encoder-tick mode

# Continuous servo speed range for ServoKit (microseconds)
# Parallax 360 uses standard RC pulse: 1000 µs = full CW, 2000 µs = full CCW,
# ~1500 µs = stop.  ServoKit ContinuousServo throttle: -1.0 to +1.0
PULSE_MIN_US = 1000
PULSE_MAX_US = 2000
PULSE_STOP_US = 1500

# ---------------------------------------------------------------------------
# Physical robot dimensions  <-- MEASURE YOUR ROBOT AND SET THESE
# ---------------------------------------------------------------------------

WHEEL_DIAMETER_MM = 60.0   # outer diameter of your wheel in millimetres
TRACK_WIDTH_MM    = 120.0  # distance between left and right wheel centres (mm)

import math
WHEEL_CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM  # mm per full revolution

# ---------------------------------------------------------------------------
# Feedback reader (measures duty cycle on yellow wire using GPIO pulse timing)
# ---------------------------------------------------------------------------

class FeedbackReader:
    """
    Reads the PWM feedback signal from a Parallax Feedback 360 servo.
    Uses RPi.GPIO edge detection to measure tHigh and tCycle durations,
    then computes duty cycle and angular position.
    """

    def __init__(self, pin: int, units_per_circle: int = UNITS_PER_CIRCLE):
        self.pin = pin
        self.units_per_circle = units_per_circle
        self._duty_cycle = 0.0          # most recent duty cycle %
        self._lock = threading.Lock()
        self._last_rise = None
        self._last_fall = None
        self._t_high = None
        self._t_cycle = None

        # Cumulative rotation tracking
        self._total_degrees = 0.0       # signed degrees travelled since reset
        self._prev_angle = None         # last known angle reading

        GPIO.setup(pin, GPIO.IN)
        try:
            GPIO.remove_event_detect(pin)
        except RuntimeError:
            pass  # wasn't set, that's fine

        print(f"DEBUG: About to add edge detect on pin {pin}")  # ← add this
        GPIO.add_event_detect(pin, GPIO.BOTH, callback=self._edge_callback)
        print(f"DEBUG: Successfully added edge detect on pin {pin}")  # ← and this

    def _edge_callback(self, channel):
        now = time.perf_counter()
        if GPIO.input(channel) == GPIO.HIGH:
            # Rising edge: start of tHigh, end of tCycle low portion
            if self._last_rise is not None and self._last_fall is not None:
                t_cycle = now - self._last_rise
                t_high  = self._last_fall - self._last_rise
                if t_cycle > 0:
                    dc = 100.0 * t_high / t_cycle
                    new_angle = ((dc - DUTY_CYCLE_MIN) /
                                 (DUTY_CYCLE_MAX - DUTY_CYCLE_MIN + 1.0) *
                                 self.units_per_circle)
                    with self._lock:
                        self._duty_cycle = dc
                        self._t_high  = t_high * 1e6
                        self._t_cycle = t_cycle * 1e6
                        # Accumulate degrees with wrap-around detection
                        if self._prev_angle is not None:
                            delta = new_angle - self._prev_angle
                            # Handle 0/360 wrap: pick the shorter arc
                            half = self.units_per_circle / 2.0
                            if delta > half:
                                delta -= self.units_per_circle
                            elif delta < -half:
                                delta += self.units_per_circle
                            self._total_degrees += delta
                        self._prev_angle = new_angle
            self._last_rise = now
        else:
            # Falling edge: end of tHigh
            self._last_fall = now

    @property
    def duty_cycle(self) -> float:
        """Return most recent duty cycle percentage (0–100)."""
        with self._lock:
            return self._duty_cycle

    @property
    def total_degrees(self) -> float:
        """Cumulative signed degrees rotated since last reset_odometry()."""
        with self._lock:
            return self._total_degrees

    def reset_odometry(self):
        """Zero the cumulative rotation counter."""
        with self._lock:
            self._total_degrees = 0.0

    @property
    def angle(self) -> float:
        """
        Return angular position in chosen units (default: degrees 0–359.x).
        Formula from datasheet:
          position = (DC - DC_min) / (DC_max - DC_min + 1) * units_per_circle
        """
        dc = self.duty_cycle
        dc_clamped = max(DUTY_CYCLE_MIN, min(DUTY_CYCLE_MAX, dc))
        position = ((dc_clamped - DUTY_CYCLE_MIN) /
                    (DUTY_CYCLE_MAX - DUTY_CYCLE_MIN + 1.0) *
                    self.units_per_circle)
        return position

    def cleanup(self):
        GPIO.remove_event_detect(self.pin)


# ---------------------------------------------------------------------------
# Servo controller wrapper
# ---------------------------------------------------------------------------

class Servo360:
    """
    Wraps a single Parallax Feedback 360 servo channel on the PCA9685.
    Provides speed control and position feedback.
    """

    def __init__(self, kit: ServoKit, channel: int, feedback_pin: int,
                 name: str = "servo", inverted: bool = False):
        self.name = name
        self._channel = channel
        self._inverted = inverted
        self._servo = kit.continuous_servo[channel]
        # Configure pulse width range to match Parallax 360 spec
        self._servo.set_pulse_width_range(PULSE_MIN_US, PULSE_MAX_US)
        self._feedback = FeedbackReader(feedback_pin)

    # -- Speed control -------------------------------------------------------

    def set_speed(self, speed: float):
        """
        Set rotation speed.
          speed =  1.0 : full clockwise
          speed = -1.0 : full counter-clockwise
          speed =  0.0 : stop
        """
        speed = max(-1.0, min(1.0, speed))
        self._servo.throttle = -speed if self._inverted else speed

    def stop(self):
        self._servo.throttle = 0.0

    # -- Feedback ------------------------------------------------------------

    @property
    def angle(self) -> float:
        """Current angular position in degrees (0–359.x)."""
        return self._feedback.angle

    @property
    def duty_cycle(self) -> float:
        """Current feedback duty cycle (%)."""
        return self._feedback.duty_cycle

    @property
    def total_degrees(self) -> float:
        """Cumulative signed degrees rotated since last reset_odometry()."""
        return self._feedback.total_degrees

    @property
    def distance_mm(self) -> float:
        """Cumulative distance travelled in mm since last reset_odometry()."""
        return (self._feedback.total_degrees / 360.0) * WHEEL_CIRCUMFERENCE_MM

    def reset_odometry(self):
        """Zero this servo's cumulative rotation counter."""
        self._feedback.reset_odometry()

    # -- High-level helpers --------------------------------------------------

    def rotate_to_angle(self, target_deg: float, speed: float = 0.3,
                        tolerance_deg: float = 3.0, timeout_s: float = 5.0):
        """
        Rotate servo until feedback angle is within tolerance of target_deg.
        Uses simple bang-bang control; replace with PID for smoother motion.

        Parameters
        ----------
        target_deg   : desired angle in degrees (0–360)
        speed        : magnitude of drive speed (0.0–1.0)
        tolerance_deg: acceptable error in degrees
        timeout_s    : maximum time to attempt the move
        """
        speed = abs(speed)
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            current = self.angle
            error = target_deg - current

            # Handle wrap-around (shortest path)
            if error > 180:
                error -= 360
            elif error < -180:
                error += 360

            if abs(error) <= tolerance_deg:
                self.stop()
                return True

            direction = 1.0 if error > 0 else -1.0
            self.set_speed(direction * speed)
            time.sleep(0.005)

        self.stop()
        print(f"[{self.name}] rotate_to_angle timed out "
              f"(target={target_deg:.1f}°, current={self.angle:.1f}°)")
        return False

    def cleanup(self):
        self.stop()
        self._feedback.cleanup()


# ---------------------------------------------------------------------------
# RoboticDog — four-servo manager
# ---------------------------------------------------------------------------

class RoboticDog:
    """
    High-level controller for a four-servo robotic dog.
    Initialises the PCA9685 via I2C and creates one Servo360 per leg.
    """

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Initialise PCA9685 (I2C address 0x40 by default)
        self.kit = ServoKit(channels=16)

        self.servos: dict[str, Servo360] = {}
        for name, (channel, inverted) in SERVO_CHANNELS.items():
            fb_pin = FEEDBACK_PINS[name]
            self.servos[name] = Servo360(self.kit, channel, fb_pin,
                                         name=name, inverted=inverted)
            print(f"  Initialised servo '{name}' on channel {channel}, "
                  f"feedback GPIO {fb_pin}, inverted={inverted}")

        # Allow feedback readers to settle
        time.sleep(0.2)
        print("RoboticDog ready.\n")

    # -- Status --------------------------------------------------------------

    def print_positions(self):
        print("Current servo positions:")
        for name, servo in self.servos.items():
            print(f"  {name:>12s}: {servo.angle:6.1f}°  "
                  f"(duty cycle {servo.duty_cycle:.1f}%)")

    # -- Basic gaits ---------------------------------------------------------

    def stop_all(self):
        for servo in self.servos.values():
            servo.stop()

    def spin_all(self, speed: float = 0.3, duration_s: float = 1.0):
        """Spin all servos at given speed for duration_s seconds."""
        for servo in self.servos.values():
            servo.set_speed(speed)
        time.sleep(duration_s)
        self.stop_all()

    def center_all(self, speed: float = 0.3, timeout_s: float = 5.0):
        """Move all servos to the 180° (center) position."""
        print("Centering all servos to 180°...")
        threads = []
        for servo in self.servos.values():
            t = threading.Thread(
                target=servo.rotate_to_angle,
                args=(180.0,),
                kwargs={"speed": speed, "timeout_s": timeout_s},
                daemon=True,
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        self.print_positions()

    def walk_forward(self, steps: int = 4, step_speed: float = 0.4):
        """
        Minimal placeholder gait: alternate leg pairs forward.
        Replace this with your actual inverse-kinematics gait planner.
        """
        print(f"Walking forward {steps} steps...")
        front_pair = [self.servos["front_left"], self.servos["front_right"]]
        rear_pair  = [self.servos["rear_left"],  self.servos["rear_right"]]

        for step in range(steps):
            active = front_pair if step % 2 == 0 else rear_pair
            for s in active:
                s.set_speed(step_speed)
            time.sleep(0.3)
            for s in active:
                s.stop()
            time.sleep(0.05)

        self.stop_all()

    def _reset_all_odometry(self):
        for servo in self.servos.values():
            servo.reset_odometry()

    def travel_distance_mm(self, distance_mm: float, speed: float = 0.4,
                           timeout_s: float = 30.0):
        """
        Drive straight forward (positive) or backward (negative) for
        exactly distance_mm millimetres, measured via wheel feedback.

        Uses the average of all four wheel odometers to handle minor
        slip differences between sides.
        """
        direction = 1.0 if distance_mm >= 0 else -1.0
        target_mm = abs(distance_mm)
        print(f"Travelling {'forward' if direction > 0 else 'backward'} "
              f"{target_mm:.1f} mm...")

        self._reset_all_odometry()
        for servo in self.servos.values():
            servo.set_speed(direction * abs(speed))

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # Average absolute distance across all four wheels
            travelled = sum(abs(s.distance_mm)
                            for s in self.servos.values()) / 4.0
            if travelled >= target_mm:
                break
            time.sleep(0.005)
        else:
            print("  Warning: travel_distance_mm timed out.")

        self.stop_all()
        actual = sum(abs(s.distance_mm)
                     for s in self.servos.values()) / 4.0
        print(f"  Stopped at ~{actual:.1f} mm (target {target_mm:.1f} mm)")

    def turn_degrees(self, degrees: float, speed: float = 0.35,
                     timeout_s: float = 15.0):
        """
        Turn the robot in place by the given number of degrees.
          Positive = clockwise (right turn)
          Negative = counter-clockwise (left turn)

        Each wheel traces an arc of:
          arc = pi * TRACK_WIDTH_MM * (|degrees| / 360)
        Left wheels go forward for a CW turn; right wheels go backward.
        """
        direction = 1.0 if degrees >= 0 else -1.0
        arc_mm = math.pi * TRACK_WIDTH_MM * (abs(degrees) / 360.0)
        print(f"Turning {'right (CW)' if direction > 0 else 'left (CCW)'} "
              f"{abs(degrees):.1f}° (arc per side = {arc_mm:.1f} mm)...")

        self._reset_all_odometry()

        left_servos  = [self.servos["front_left"],  self.servos["rear_left"]]
        right_servos = [self.servos["front_right"], self.servos["rear_right"]]

        # CW turn: left wheels forward (+), right wheels backward (-)
        for s in left_servos:
            s.set_speed( direction * abs(speed))
        for s in right_servos:
            s.set_speed(-direction * abs(speed))

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            left_dist  = sum(abs(s.distance_mm) for s in left_servos)  / 2.0
            right_dist = sum(abs(s.distance_mm) for s in right_servos) / 2.0
            avg_arc = (left_dist + right_dist) / 2.0
            if avg_arc >= arc_mm:
                break
            time.sleep(0.005)
        else:
            print("  Warning: turn_degrees timed out.")

        self.stop_all()
        # Report actual degrees turned from odometry
        avg_arc_actual = sum(abs(s.distance_mm)
                             for s in self.servos.values()) / 4.0
        actual_deg = (avg_arc_actual / (math.pi * TRACK_WIDTH_MM)) * 360.0
        print(f"  Stopped at ~{actual_deg:.1f}° (target {abs(degrees):.1f}°)")

    # -- Cleanup -------------------------------------------------------------

    def cleanup(self):
        self.stop_all()
        for servo in self.servos.values():
            servo.cleanup()
        GPIO.cleanup()
        print("Cleanup complete.")


# ---------------------------------------------------------------------------
# Demo / entry point
# ---------------------------------------------------------------------------

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)   # cleanup resets mode, so set it again
    GPIO.setwarnings(False)
    dog = RoboticDog()
    try:
        # Print live feedback positions
        dog.print_positions()

        # Center all servos
        dog.center_all(speed=0.3)

        # Short walk demo
        dog.walk_forward(steps=4)

        # Final position report
        dog.print_positions()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        dog.cleanup()


if __name__ == "__main__":
    main()