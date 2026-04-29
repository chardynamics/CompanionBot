#!/usr/bin/env python3
import os
import sys
import time
import math
import random
import pygame

# --- Run on the framebuffer, no desktop needed ---
os.environ['SDL_VIDEODRIVER'] = 'kmsdrm'
os.environ['SDL_FBDEV'] = '/dev/fb0'

# --- Configuration ---
SCREEN_W, SCREEN_H = 320, 240
FPS = 30

# Colors
BG        = (10,  10,  30)
EYE_WHITE = (220, 230, 255)
IRIS      = (80,  160, 255)
PUPIL     = (0,   0,   0)
SHINE     = (255, 255, 255)
BAT_GREEN = (80,  220, 100)
BAT_WARN  = (255, 200, 50)
BAT_LOW   = (255, 60,  60)
TEXT_COL  = (180, 200, 255)

# ------------------------------------------------------------------ #
#  BATTERY — replace this function with real sensor reading if needed #
# ------------------------------------------------------------------ #
def get_battery_percent():
    # TODO: read from your battery sensor here
    # e.g. INA219, a UPS HAT, or a GPIO ADC
    # For now returns a fake value that ticks down for demo purposes
    return max(0, 85 - int(time.time() / 10) % 100)


# ------------------------------------------------------------------ #
#  EYE DRAWING                                                        #
# ------------------------------------------------------------------ #
def draw_eye(surface, cx, cy, open_ratio, look_x=0, look_y=0):
    """
    cx, cy      — center of eye
    open_ratio  — 1.0 fully open, 0.0 fully closed
    look_x/y    — pupil offset in pixels (-10 to 10)
    """
    EW, EH_MAX = 72, 58
    eye_h = int(EH_MAX * open_ratio)

    if eye_h < 2:
        # Draw closed line
        pygame.draw.line(surface, EYE_WHITE,
                         (cx - EW//2, cy), (cx + EW//2, cy), 2)
        return

    # Sclera (white)
    pygame.draw.ellipse(surface, EYE_WHITE,
                        (cx - EW//2, cy - eye_h//2, EW, eye_h))

    # Iris
    iris_r = int(22 * open_ratio)
    if iris_r > 1:
        pygame.draw.ellipse(surface, IRIS,
                            (cx - iris_r + look_x,
                             cy - iris_r + look_y,
                             iris_r * 2, iris_r * 2))
        # Pupil
        p = int(13 * open_ratio)
        pygame.draw.ellipse(surface, PUPIL,
                            (cx - p + look_x,
                             cy - p + look_y,
                             p * 2, p * 2))
        # Shine
        pygame.draw.circle(surface, SHINE,
                           (cx + iris_r//2 + look_x,
                            cy - iris_r//2 + look_y), 4)

    # Eyelid clip — draw filled rect over top/bottom to square off the ellipse
    clip_h = max(0, (EH_MAX - eye_h) // 2)
    if clip_h:
        pygame.draw.rect(surface, BG,
                         (cx - EW//2 - 2, cy - EH_MAX//2, EW + 4, clip_h))
        pygame.draw.rect(surface, BG,
                         (cx - EW//2 - 2, cy + eye_h//2, EW + 4, clip_h + 2))


# ------------------------------------------------------------------ #
#  BATTERY BAR                                                        #
# ------------------------------------------------------------------ #
def draw_battery(surface, font, x, y, pct):
    pct = max(0, min(100, pct))
    color = BAT_GREEN if pct > 50 else (BAT_WARN if pct > 20 else BAT_LOW)

    # Outline
    bar_w, bar_h = 44, 16
    pygame.draw.rect(surface, TEXT_COL, (x, y, bar_w, bar_h), 2)
    # Nub
    pygame.draw.rect(surface, TEXT_COL, (x + bar_w, y + 4, 4, 8))
    # Fill
    fill_w = int((bar_w - 4) * pct / 100)
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x + 2, y + 2, fill_w, bar_h - 4))

    # Percentage text
    label = font.render(f'{pct}%', True, color)
    surface.blit(label, (x + bar_w + 8, y))


# ------------------------------------------------------------------ #
#  MAIN                                                               #
# ------------------------------------------------------------------ #
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN, 16)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    font_small = pygame.font.SysFont(None, 20)
    font_med   = pygame.font.SysFont(None, 26)

    # Eye positions (landscape, centered vertically with room for status bar)
    EYE_Y   = 115
    LEFT_X  = 95
    RIGHT_X = 225

    # Blink state
    open_ratio   = 1.0
    blink_frames = []
    blink_index  = 0
    frame_count  = 0
    next_blink   = random.randint(120, 300)

    # Look direction (slow drift)
    look_x, look_y = 0, 0
    look_target_x, look_target_y = 0, 0
    look_timer = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                running = False

        frame_count += 1

        # --- Blink logic ---
        if frame_count >= next_blink and not blink_frames:
            # Build blink: close over 4 frames, open over 6
            blink_frames = (
                [1.0, 0.6, 0.2, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0]
            )
            blink_index = 0
            next_blink = frame_count + random.randint(100, 300)

        if blink_frames:
            open_ratio = blink_frames[blink_index]
            blink_index += 1
            if blink_index >= len(blink_frames):
                blink_frames = []
                open_ratio = 1.0

        # --- Look drift ---
        look_timer -= 1
        if look_timer <= 0:
            look_target_x = random.randint(-8, 8)
            look_target_y = random.randint(-4, 4)
            look_timer = random.randint(60, 180)

        look_x += (look_target_x - look_x) * 0.1
        look_y += (look_target_y - look_y) * 0.1

        # --- Draw ---
        screen.fill(BG)

        draw_eye(screen, LEFT_X,  EYE_Y, open_ratio, int(look_x), int(look_y))
        draw_eye(screen, RIGHT_X, EYE_Y, open_ratio, int(look_x), int(look_y))

        # Status bar at bottom
        pygame.draw.line(screen, (40, 40, 80), (0, 205), (320, 205), 1)

        bat = get_battery_percent()
        draw_battery(screen, font_small, 8, 212, bat)

        # IP or status message — replace with whatever you want
        status = font_small.render('robot-pi  |  ready', True, TEXT_COL)
        screen.blit(status, (80, 214))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()