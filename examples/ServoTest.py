# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

"""Simple test for a standard servo on channel 0 and a continuous rotation servo on channel 1."""

import time

from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=16)
##Never use 15

while True:
    speed = float(input("Enter the speed (0.2 is nothing): "))
    for i in range(11, 15):
        kit.continuous_servo[i].throttle = speed