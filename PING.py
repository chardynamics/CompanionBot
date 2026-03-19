import sys
import RPi.GPIO as GPIO
import time

class UltrasonicParallaxPING(object):

    def __init__(self, signal):
        self.signal = signal

        self.timeout = 0.05

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

    def distance(self):
        arrivalTime = 0
        startTime = 0

        GPIO.setup(self.signal, GPIO.OUT) # Set to low
        GPIO.output(self.signal, False)

        time.sleep(0.01)

        GPIO.output(self.signal, True) # Set high

        time.sleep(0.00001)

        GPIO.output(self.signal, False) # Set low
        GPIO.setup(self.signal, GPIO.IN) # Set to input

        timeout_start = time.time()

        # Count microseconds that SIG was high
        while GPIO.input(self.signal) == 0:
            startTime = time.time()

            if startTime - timeout_start > self.timeout:
                return -1

        while GPIO.input(self.signal) == 1:
            arrivalTime = time.time()
        
            if startTime - timeout_start > self.timeout:
                return -1

        if startTime != 0 and arrivalTime != 0:
            pulse_duration = arrivalTime - startTime
            # The speed of sound is 340 m/s or 29 microseconds per centimeter.
            # The ping travels out and back, so to find the distance of the
            # object we take half of the distance travelled.
            # distance = duration / 29 / 2
            #distance = pulse_duration * 100 * 343.0 / 2
            distance = (pulse_duration * 34300) / 2

            #print('start = %s'%startTime,)
            #print('end = %s'%arrivalTime)
            if distance >= 0:
                return distance
            else:
                return -1
        else :
            #print('start = %s'%startTime,)
            #print('end = %s'%arrivalTime)
            return -1

    def speed(self):
        start_time = time.time()

        start_distance = self.distance() * 0.01     # to m conversion
        end_distance = self.distance() * 0.01       # to m conversion

        end_time = time.time()

        speed = (end_distance - start_distance) / 1.0   # m/s

        return speed