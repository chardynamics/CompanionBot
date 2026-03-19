import PINGClass
import time

PINGsensor = PINGClass.UltrasonicParallaxPING(17)

# Main loop to print distance
try:
    while True:
        dist_cm = PINGsensor.read_distance()
        print(f"Distance to object is {dist_cm:.2f} cm")
        time.sleep(1) # Wait before the next measurement to avoid echo interference

except KeyboardInterrupt:
    pass