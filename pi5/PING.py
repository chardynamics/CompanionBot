import time
from gpiozero import OutputDevice, InputDevice
from signal import pause

# Define the single GPIO pin connected to the PING sensor's SIG pin (using BCM numbering)
PING_PIN = 17

def read_distance():
    # 1. Send the trigger pulse
    # Temporarily treat the pin as an output to send the pulse
    trigger = OutputDevice(PING_PIN, initial_value=False)
    trigger.on()
    time.sleep(0.000005) # 5 microsecond pulse as per PING specs
    trigger.off()
    trigger.close() # Release the pin so it can be used as an input

    # 2. Listen for the echo pulse
    # Now treat the pin as an input to measure the pulse duration
    echo = InputDevice(PING_PIN)
    while echo.is_active == False:
        starttime = time.time()
    while echo.is_active == True:
        endtime = time.time()
    echo.close() # Release the pin

    duration = endtime - starttime

    # 3. Calculate the distance
    # Speed of sound is approximately 34300 cm/s
    # Distance is (duration * speed_of_sound) / 2 (accounting for the round trip)
    distance = duration * 17150
    return distance

# Main loop to print distance
try:
    while True:
        dist_cm = read_distance()
        print(f"Distance to object is {dist_cm:.2f} cm")
        time.sleep(1) # Wait before the next measurement to avoid echo interference

except KeyboardInterrupt:
    pass