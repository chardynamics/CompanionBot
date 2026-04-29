import os
import pygame
import time
import math

os.environ['SDL_VIDEODRIVER'] = 'kmsdrm'
os.environ['SDL_FBDEV'] = '/dev/fb0'
os.environ['SDL_VIDEO_FBCON_ROTATION'] = '0'

pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((320, 240), pygame.FULLSCREEN, 16)
clock = pygame.time.Clock()

BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
BLUE   = (100, 180, 255)
DARK   = (20, 20, 40)

def draw_eye(surface, cx, cy, open_amount):
    # open_amount: 1.0 = fully open, 0.0 = fully closed
    eye_w = 60
    eye_h = int(50 * open_amount)

    # white of eye
    pygame.draw.ellipse(surface, WHITE, (cx - eye_w//2, cy - eye_h//2, eye_w, max(eye_h, 2)))
    # iris
    if open_amount > 0.1:
        iris_h = int(36 * open_amount)
        pygame.draw.ellipse(surface, BLUE, (cx - 18, cy - iris_h//2, 36, iris_h))
        # pupil
        pupil_h = int(22 * open_amount)
        pygame.draw.ellipse(surface, BLACK, (cx - 11, cy - pupil_h//2, 22, pupil_h))
        # shine
        pygame.draw.circle(surface, WHITE, (cx + 6, cy - int(8 * open_amount)), 5)

def blink_sequence():
    # returns a list of open_amount values for a blink
    steps = []
    for i in range(5):   # closing
        steps.append(1.0 - i * 0.2)
    for i in range(5):   # opening
        steps.append(i * 0.2)
    return steps

running = True
blink_timer = 0
blink_frames = []
blink_index = 0
open_amount = 1.0
next_blink = 180  # frames until next blink (~3 seconds at 60fps)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            running = False

    screen.fill(DARK)

    # handle blinking
    blink_timer += 1
    if blink_timer >= next_blink and not blink_frames:
        blink_frames = blink_sequence()
        blink_index = 0
        next_blink = 180 + int(120 * (time.time() % 1))  # randomish interval

    if blink_frames:
        open_amount = blink_frames[blink_index]
        blink_index += 1
        if blink_index >= len(blink_frames):
            blink_frames = []
            open_amount = 1.0

    # draw both eyes
    draw_eye(screen, 100, 120, open_amount)
    draw_eye(screen, 220, 120, open_amount)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()