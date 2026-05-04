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

# (PCA9685 channel, inverted?)
# Right-side servos are mounted mirrored so their direction is flipped.
SERVO_CHANNELS = {
    "front_left":  (1, False),
    "front_right": (14, True),
    "rear_left":   (0, False),
    "rear_right":  (13, True),
}

# BCM GPIO pins for the yellow feedback wires
FEEDBACK_PINS = {
    "front_left":  6,
    "front_right": 13,
    "rear_left":   22,
    "rear_right":  23,
}

# ---------------------------------------------------------------------------
# Physical dimensions — measure your robot and update these
# ---------------------------------------------------------------------------

WHEEL_DIAMETER_MM = 60.0    # outer diameter of one wheel (mm)
TRACK_WIDTH_MM    = 120.0   # centre-to-centre distance between left and right wheels (mm)


WHEEL_CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM

# ---------------------------------------------------------------------------
# Parallax feedback signal constants (from datasheet)
# ---------------------------------------------------------------------------

DUTY_CYCLE_MIN   = 2.9    # % — sensor output at the 0° origin
DUTY_CYCLE_MAX   = 97.1   # % — sensor output just before one full CW revolution
UNITS_PER_CIRCLE = 360    # work in degrees

PULSE_MIN_US = 1000       # µs — full clockwise
PULSE_MAX_US = 2000       # µs — full counter-clockwise


# ---------------------------------------------------------------------------
# FeedbackReader
# ---------------------------------------------------------------------------

class FeedbackReader:
    """
    Reads the Parallax 360 feedback PWM on the yellow wire.

    The servo outputs a ~910 Hz square wave whose duty cycle encodes
    absolute wheel position within one revolution (0-360 deg).

    pigpio times the edges in hardware so all four servos can be read
    simultaneously with ~1 us accuracy regardless of CPU load.

    Properties
    ----------
    angle         : absolute position within the current revolution (0-360 deg)
    total_degrees : cumulative signed degrees turned since reset_odometry()
                    this is what travel_distance and turn use
    distance_mm   : total_degrees converted to linear wheel travel (mm)
    """

    def __init__(self, pi: pigpio.pi, pin: int):
        self._pi   = pi
        self.pin   = pin

        self._lock       = threading.Lock()
        self._duty_cycle = 0.0
        self._prev_angle = None
        self._total_deg  = 0.0

        self._last_rise = None  # pigpio tick (us) of the last rising edge
        self._last_fall = None  # pigpio tick (us) of the last falling edge

        pi.set_mode(pin, pigpio.INPUT)
        self._cb = pi.callback(pin, pigpio.EITHER_EDGE, self._on_edge)

    def _on_edge(self, gpio, level, tick):
        if level == 2:          # pigpio watchdog timeout - ignore
            return

        if level == 1:          # rising edge - a new cycle just started
            if self._last_rise is not None and self._last_fall is not None:
                t_cycle = pigpio.tickDiff(self._last_rise, tick)
                t_high  = pigpio.tickDiff(self._last_rise, self._last_fall)
                if t_cycle > 0:
                    dc = 100.0 * t_high / t_cycle
                    new_angle = ((dc - DUTY_CYCLE_MIN) /
                                 (DUTY_CYCLE_MAX - DUTY_CYCLE_MIN + 1.0) *
                                 UNITS_PER_CIRCLE)
                    with self._lock:
                        self._duty_cycle = dc
                        if self._prev_angle is not None:
                            # Shortest-arc delta handles the 359->0 wrap
                            delta = new_angle - self._prev_angle
                            half  = UNITS_PER_CIRCLE / 2.0
                            if delta >  half: delta -= UNITS_PER_CIRCLE
                            if delta < -half: delta += UNITS_PER_CIRCLE
                            self._total_deg += delta
                        self._prev_angle = new_angle
            self._last_rise = tick

        else:                   # falling edge - end of the high pulse
            self._last_fall = tick

    @property
    def angle(self) -> float:
        """Absolute wheel position within the current revolution (0-360 deg)."""
        with self._lock:
            dc = max(DUTY_CYCLE_MIN, min(DUTY_CYCLE_MAX, self._duty_cycle))
        return ((dc - DUTY_CYCLE_MIN) /
                (DUTY_CYCLE_MAX - DUTY_CYCLE_MIN + 1.0) * UNITS_PER_CIRCLE)

    @property
    def total_degrees(self) -> float:
        """Cumulative signed degrees turned since last reset_odometry()."""
        with self._lock:
            return self._total_deg

    @property
    def distance_mm(self) -> float:
        """Cumulative linear distance travelled since last reset_odometry()."""
        return (self.total_degrees / 360.0) * WHEEL_CIRCUMFERENCE_MM

    def reset_odometry(self):
        """Zero the cumulative rotation counter."""
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
    servo is mounted on - the inverted flag handles the physical flip.
    """

    def __init__(self, kit: ServoKit, channel: int, feedback_pin: int,
                 pi: pigpio.pi, name: str = "servo", inverted: bool = False):
        self.name      = name
        self._inverted = inverted
        # Inverted servos spin physically backwards for a given throttle,
        # so their feedback counts down when we command "forward".
        # odo_sign flips the reading back so distance_mm is always positive
        # when the robot moves forward.
        self._odo_sign = -1.0 if inverted else 1.0
        self._servo    = kit.continuous_servo[channel]
        self._servo.set_pulse_width_range(PULSE_MIN_US, PULSE_MAX_US)
        self._feedback = FeedbackReader(pi, feedback_pin)
    def set_speed(self, speed: float):
        """speed: -1.0 = full reverse, 0.0 = stop, +1.0 = full forward"""
        speed = max(-1.0, min(1.0, speed))
        self._servo.throttle = -speed if self._inverted else speed

    def stop(self):
        self._servo.throttle = 0.0

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

    How distance works
    ------------------
    Each wheel's yellow feedback wire reports its absolute angle within
    one revolution. FeedbackReader accumulates those tiny angle deltas
    (including 359->0 wrap-around) into total_degrees, then converts to
    mm using the wheel circumference. travel_distance_mm() drives all four
    wheels and stops when the average odometer hits the target.

    How turning works
    -----------------
    To turn in place, left wheels spin forward and right wheels spin
    backward (or vice versa). The arc each side must travel for a given
    robot body rotation is:
        arc_mm = pi * track_width_mm * (turn_degrees / 360)

    Quick reference
    ---------------
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
        for name, (channel, inverted) in SERVO_CHANNELS.items():
            pin = FEEDBACK_PINS[name]
            self.servos[name] = Servo360(
                self.kit, channel, pin,
                pi=self._pi, name=name, inverted=inverted,
            )
            print(f"  {name:>12s}  channel={channel}  "
                  f"GPIO={pin}  inverted={inverted}")

        time.sleep(0.2)   # let feedback readers collect a first reading
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

        self._reset_all_odometry()
        for s in self.servos.values():
            s.set_speed(direction * abs(speed))

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

        self._reset_all_odometry()

        left  = [self.servos["front_left"],  self.servos["rear_left"]]
        right = [self.servos["front_right"], self.servos["rear_right"]]

        # CW turn: left wheels forward, right wheels backward
        for s in left:  s.set_speed( direction * abs(speed))
        for s in right: s.set_speed(-direction * abs(speed))

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            left_avg  = sum(abs(s.distance_mm) for s in left)  / 2.0
            right_avg = sum(abs(s.distance_mm) for s in right) / 2.0
            if (left_avg + right_avg) / 2.0 >= arc_mm:
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
        # Send stop pulse to all servos
        self.stop_all()
        # Wait long enough for the PCA9685 to latch the stop pulse
        # (one PWM cycle at 50 Hz = 20 ms; 100 ms is a safe margin)
        time.sleep(0.1)
        # Disable every PCA9685 channel so it outputs no pulse at all,
        # rather than holding the last value after Python exits
        for channel in range(16):
            self.kit.continuous_servo[channel].fraction = None
        for s in self.servos.values():
            s.cleanup()
        self._pi.stop()
        print("Cleanup complete.")


# ---------------------------------------------------------------------------
# Entry point — edit main() to define your robot's behaviour
# ---------------------------------------------------------------------------

def main():
    dog = RoboticDog()
    try:
        # Drive forward 50 cm
        dog.travel_distance_mm(500, speed=0.4)

        # Turn right 90 degrees in place
        dog.turn_degrees(90, speed=0.35)

        # Drive forward another 30 cm
        dog.travel_distance_mm(300, speed=0.4)

        # Turn left 180 degrees
        dog.turn_degrees(-180, speed=0.35)

        dog.print_status()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        dog.cleanup()


if __name__ == "__main__":
    main()