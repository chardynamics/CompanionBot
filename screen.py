#!/usr/bin/env python3
import os
import sys
import time
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- Screen config ---
SCREEN_W, SCREEN_H = 320, 240
FB_DEV = '/dev/fb0'
FPS = 20

# --- Colors (RGB) ---
BG        = (10,  10,  30)
EYE_WHITE = (220, 230, 255)
IRIS      = (80,  160, 255)
PUPIL     = (0,   0,   0)
SHINE     = (255, 255, 255)
BAT_GREEN = (80,  220, 100)
BAT_WARN  = (255, 200, 50)
BAT_LOW   = (255, 60,  60)
TEXT_COL  = (180, 200, 255)
LINE_COL  = (40,  40,  80)


def rgb_to_rgb565_bytes(img):
    arr = np.array(img, dtype=np.uint16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return rgb565.astype('<u2').tobytes()


def get_battery_percent():
    # TODO: replace with real sensor read
    return 85


def draw_eye(draw, cx, cy, open_ratio, look_x=0, look_y=0):
    EW, EH_MAX = 72, 58
    eye_h = int(EH_MAX * open_ratio)

    if eye_h < 3:
        draw.line([(cx - EW//2, cy), (cx + EW//2, cy)], fill=EYE_WHITE, width=2)
        return

    draw.ellipse([cx - EW//2, cy - eye_h//2,
                  cx + EW//2, cy + eye_h//2], fill=EYE_WHITE)

    iris_r = int(22 * open_ratio)
    if iris_r > 1:
        ix, iy = cx + look_x, cy + look_y
        draw.ellipse([ix - iris_r, iy - iris_r,
                      ix + iris_r, iy + iris_r], fill=IRIS)
        p = int(13 * open_ratio)
        draw.ellipse([ix - p, iy - p, ix + p, iy + p], fill=PUPIL)
        sx = ix + iris_r // 2
        sy = iy - iris_r // 2
        draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=SHINE)

    clip = max(0, (EH_MAX - eye_h) // 2)
    if clip > 0:
        draw.rectangle([cx - EW//2 - 2, cy - EH_MAX//2,
                        cx + EW//2 + 2, cy - eye_h//2], fill=BG)
        draw.rectangle([cx - EW//2 - 2, cy + eye_h//2,
                        cx + EW//2 + 2, cy + EH_MAX//2 + 2], fill=BG)


def draw_battery(draw, font, x, y, pct):
    pct = max(0, min(100, pct))
    color = BAT_GREEN if pct > 50 else (BAT_WARN if pct > 20 else BAT_LOW)
    bar_w, bar_h = 44, 16
    draw.rectangle([x, y, x + bar_w, y + bar_h], outline=TEXT_COL)
    draw.rectangle([x + bar_w, y + 5, x + bar_w + 4, y + bar_h - 5], fill=TEXT_COL)
    fill_w = int((bar_w - 4) * pct / 100)
    if fill_w > 0:
        draw.rectangle([x + 2, y + 2, x + 2 + fill_w, y + bar_h - 2], fill=color)
    draw.text((x + bar_w + 8, y + 1), f'{pct}%', font=font, fill=color)


def main():
    fb = open(FB_DEV, 'wb')

    try:
        font_small = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:
        font_small = ImageFont.load_default()

    EYE_Y   = 112
    LEFT_X  = 95
    RIGHT_X = 225

    open_ratio = 1.0
    blink_seq  = []
    blink_idx  = 0
    frame      = 0
    next_blink = random.randint(60, 150)

    look_x, look_y   = 0.0, 0.0
    target_x, target_y = 0.0, 0.0
    look_timer = 0
    frame_delay = 1.0 / FPS

    print("Robot face running. Ctrl+C to stop.")

    try:
        while True:
            t0 = time.time()
            frame += 1

            # Blink
            if frame >= next_blink and not blink_seq:
                blink_seq = [1.0, 0.6, 0.2, 0.0, 0.0,
                             0.2, 0.5, 0.85, 1.0, 1.0]
                blink_idx  = 0
                next_blink = frame + random.randint(60, 200)

            if blink_seq:
                open_ratio = blink_seq[blink_idx]
                blink_idx += 1
                if blink_idx >= len(blink_seq):
                    blink_seq  = []
                    open_ratio = 1.0

            # Look drift
            look_timer -= 1
            if look_timer <= 0:
                target_x   = random.uniform(-8, 8)
                target_y   = random.uniform(-4, 4)
                look_timer = random.randint(40, 120)

            look_x += (target_x - look_x) * 0.15
            look_y += (target_y - look_y) * 0.15

            # Draw
            img  = Image.new('RGB', (SCREEN_W, SCREEN_H), BG)
            draw = ImageDraw.Draw(img)

            draw_eye(draw, LEFT_X,  EYE_Y, open_ratio, int(look_x), int(look_y))
            draw_eye(draw, RIGHT_X, EYE_Y, open_ratio, int(look_x), int(look_y))

            draw.line([(0, 205), (SCREEN_W, 205)], fill=LINE_COL, width=1)
            bat = get_battery_percent()
            draw_battery(draw, font_small, 8, 210, bat)
            draw.text((82, 212), 'CompanionBot  |  ready',
                      font=font_small, fill=TEXT_COL)

            fb.seek(0)
            fb.write(rgb_to_rgb565_bytes(img))
            fb.flush()

            elapsed = time.time() - t0
            sleep   = frame_delay - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        fb.close()


if __name__ == '__main__':
    main()