#!/usr/bin/env python3
import os
import sys
import time
import math
import random
import pygame
import json
from src.UPS import INA219
import subprocess

SCREEN_STATE_FILE = "/tmp/screen_state.json"

os.environ['SDL_VIDEODRIVER'] = 'kmsdrm'
os.environ['SDL_FBDEV'] = '/dev/fb0'

SCREEN_W, SCREEN_H = 320, 240
FPS = 60

ina = INA219(addr=0x40)

BG        = (10,  10,  30)
EYE_WHITE = (220, 230, 255)
IRIS      = (80,  160, 255)
PUPIL     = (0,   0,   0)
SHINE     = (255, 255, 255)
BAT_GREEN = (80,  220, 100)
BAT_WARN  = (255, 200, 50)
BAT_LOW   = (255, 60,  60)
TEXT_COL  = (180, 200, 255)
SONG_BG   = (15,  15,  40)
SONG_COL  = (100, 200, 255)
ARTIST_COL= (160, 160, 200)

# --- Screen mode state ---
screen_mode = "eyes"         # "eyes" | "now_playing" | "show_image" | "listening" | "processing"
mode_data   = {}
mode_until  = 0              # monotonic time when we return to eyes (0 = forever)
last_ts     = None
current_image = None         # loaded pygame Surface for show_image mode


def get_battery_percent():
    readings = ina.getReadings()
    return readings['percent']

def get_wifi_name():
    try:
        result = subprocess.run(["iwgetid", "-r"], capture_output=True, text=True, timeout=3)
        ssid = result.stdout.strip()
        return ssid if ssid else "Not connected"
    except Exception:
        return "Unknown"


# ------------------------------------------------------------------ #
#  POLL SHARED STATE FILE                                             #
# ------------------------------------------------------------------ #
def poll_screen_state():
    global screen_mode, mode_data, mode_until, last_ts, current_image
    try:
        with open(SCREEN_STATE_FILE) as f:
            state = json.load(f)

        ts = state.get("timestamp")
        if ts == last_ts:
            return  # nothing new
        last_ts = ts

        event = state.get("event")
        data  = state.get("data", {})

        if event == "now_playing":
            screen_mode = "now_playing"
            mode_data   = data
            mode_until  = 0

            # Download album art
            cover_url = data.get("cover_url")
            if cover_url:
                try:
                    import requests, io
                    img_data = requests.get(cover_url, timeout=5).content
                    img = pygame.image.load(io.BytesIO(img_data)).convert()
                    mode_data["cover_surface"] = pygame.transform.scale(img, (100, 100))
                except Exception as e:
                    print(f"[SCREEN] Could not load cover: {e}")
                    mode_data["cover_surface"] = None

        elif event == "show_image":
            path = data.get("path", "output_image.jpg")
            try:
                img = pygame.image.load(path).convert()
                current_image = pygame.transform.scale(img, (SCREEN_W, SCREEN_H))
                screen_mode = "show_image"
                mode_data   = data
                mode_until  = 0
            except Exception as e:
                print(f"[SCREEN] Could not load image: {e}")

        elif event == "listening":
            screen_mode = "eyes"
            mode_data   = {}
            mode_until  = 0
            current_image = None

        elif event == "processing":
            screen_mode = "processing"
            mode_data   = {}
            mode_until  = 0
            current_image = None

        elif event == "speaking":
            screen_mode = "speaking"
            mode_data   = {}
            mode_until  = 0
            current_image = None

    except (FileNotFoundError, json.JSONDecodeError):
        pass  # not written yet, or mid-write — just skip


# ------------------------------------------------------------------ #
#  EYE DRAWING                                                        #
# ------------------------------------------------------------------ #
def draw_eye(surface, cx, cy, open_ratio, look_x=0, look_y=0):
    EW, EH_MAX = 72, 58
    eye_h = int(EH_MAX * open_ratio)

    if eye_h < 2:
        pygame.draw.line(surface, EYE_WHITE, (cx - EW//2, cy), (cx + EW//2, cy), 2)
        return

    pygame.draw.ellipse(surface, EYE_WHITE, (cx - EW//2, cy - eye_h//2, EW, eye_h))

    iris_r = int(22 * open_ratio)
    if iris_r > 1:
        pygame.draw.ellipse(surface, IRIS,
                            (cx - iris_r + look_x, cy - iris_r + look_y, iris_r * 2, iris_r * 2))
        p = int(13 * open_ratio)
        pygame.draw.ellipse(surface, PUPIL,
                            (cx - p + look_x, cy - p + look_y, p * 2, p * 2))
        pygame.draw.circle(surface, SHINE,
                           (cx + iris_r//2 + look_x, cy - iris_r//2 + look_y), 4)

    clip_h = max(0, (EH_MAX - eye_h) // 2)
    if clip_h:
        pygame.draw.rect(surface, BG, (cx - EW//2 - 2, cy - EH_MAX//2, EW + 4, clip_h))
        pygame.draw.rect(surface, BG, (cx - EW//2 - 2, cy + eye_h//2, EW + 4, clip_h + 2))


# ------------------------------------------------------------------ #
#  BATTERY BAR                                                        #
# ------------------------------------------------------------------ #
def draw_battery(surface, font, x, y, pct):
    pct = max(0, min(100, pct))
    color = BAT_GREEN if pct > 50 else (BAT_WARN if pct > 20 else BAT_LOW)
    bar_w, bar_h = 44, 16
    pygame.draw.rect(surface, TEXT_COL, (x, y, bar_w, bar_h), 2)
    pygame.draw.rect(surface, TEXT_COL, (x + bar_w, y + 4, 4, 8))
    fill_w = int((bar_w - 4) * pct / 100)
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x + 2, y + 2, fill_w, bar_h - 4))
    label = font.render(f'{pct}%', True, color)
    surface.blit(label, (x + bar_w + 8, y))


# ------------------------------------------------------------------ #
#  MODE RENDERERS                                                     #
# ------------------------------------------------------------------ #
def draw_now_playing(surface, font_med, font_small, frame_count):
    surface.fill(SONG_BG)

    cover = mode_data.get("cover_surface")
    if cover:
        surface.blit(cover, (SCREEN_W // 2 - 50, 20))  # centered, 100x100
        title_y  = 130
        artist_y = 158
    else:
        # fallback to the music note if no cover loaded
        note = font_med.render("♪", True, SONG_COL)
        surface.blit(note, (SCREEN_W // 2 - note.get_width() // 2, 30))
        title_y  = 95
        artist_y = 128

    title  = mode_data.get("title",  "Unknown")
    artist = mode_data.get("artist", "Unknown")

    # Truncate if too long
    if len(title) > 22:
        title = title[:20] + "…"
    if len(artist) > 26:
        artist = artist[:24] + "…"

    t_surf = font_med.render(title,  True, SONG_COL)
    a_surf = font_small.render(f"by {artist}", True, ARTIST_COL)

    surface.blit(t_surf,  (SCREEN_W // 2 - t_surf.get_width() // 2,  95))
    surface.blit(a_surf,  (SCREEN_W // 2 - a_surf.get_width() // 2, 128))

    # Scrolling equalizer bars at the bottom
    bar_count = 12
    bar_max_h = 30
    bar_w = 14
    gap = 4
    total_w = bar_count * (bar_w + gap) - gap
    start_x = (SCREEN_W - total_w) // 2
    for i in range(bar_count):
        h = int(bar_max_h * abs(math.sin(frame_count * 0.12 + i * 0.6)))
        h = max(4, h)
        bx = start_x + i * (bar_w + gap)
        pygame.draw.rect(surface, SONG_COL, (bx, 185 - h, bar_w, h), border_radius=3)


def draw_show_image(surface):
    if current_image:
        surface.blit(current_image, (0, 0))
    else:
        surface.fill(BG)


def draw_processing(surface, font_med, font_small, frame_count):
    surface.fill(BG)
    dots = "." * (1 + (frame_count // 20) % 3)
    text = font_med.render(f"Thinking{dots}", True, TEXT_COL)
    surface.blit(text, (SCREEN_W // 2 - text.get_width() // 2, SCREEN_H // 2 - 20))

    # Spinning arc
    cx, cy, r = SCREEN_W // 2, 80, 28
    angle = (frame_count * 4) % 360
    for i in range(0, 270, 10):
        a = math.radians(angle + i)
        alpha = int(255 * i / 270)
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        col = (int(IRIS[0] * alpha / 255), int(IRIS[1] * alpha / 255), int(IRIS[2] * alpha / 255))
        pygame.draw.circle(surface, col, (px, py), 3)


def draw_speaking(surface, font_med, font_small, frame_count):
    surface.fill(BG)
    text = font_med.render("Speaking…", True, TEXT_COL)
    surface.blit(text, (SCREEN_W // 2 - text.get_width() // 2, SCREEN_H // 2 - 20))


# ------------------------------------------------------------------ #
#  STATUS BAR (drawn on top of everything except show_image)         #
# ------------------------------------------------------------------ #
def draw_status_bar(surface, font_small, wifi_name, bat):
    if screen_mode == "show_image":
        return  # let the image breathe
    pygame.draw.line(surface, (40, 40, 80), (0, 205), (320, 205), 1)
    draw_battery(surface, font_small, 8, 212, bat)
    ssid_surface = font_small.render(
        wifi_name, True,
        BAT_GREEN if "Not connected" not in wifi_name else BAT_LOW
    )
    surface.blit(ssid_surface, (100, 212))


# ------------------------------------------------------------------ #
#  MAIN                                                               #
# ------------------------------------------------------------------ #
def main():
    global screen_mode, mode_data, mode_until, current_image  # ← add this line
    
    try:
        with open(SCREEN_STATE_FILE, "w") as f:
            json.dump({"event": "listening", "data": {}, "timestamp": time.time()}, f)
    except Exception:
        pass
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN, 16)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    font_small = pygame.font.SysFont(None, 20)
    font_med   = pygame.font.SysFont(None, 26)

    EYE_Y   = 115
    LEFT_X  = 95
    RIGHT_X = 225

    wifi_name   = get_wifi_name()
    last_update = time.monotonic()
    POLL_INTERVAL = 60

    # Blink state
    open_ratio   = 1.0
    blink_frames = []
    blink_index  = 0
    frame_count  = 0
    next_blink   = random.randint(120, 300)

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

        # Poll shared state every 6 frames (~10Hz at 60fps) — cheap enough
        if frame_count % 6 == 0:
            poll_screen_state()

            if screen_mode == "show_image":
                if mode_until != 0 and time.monotonic() > mode_until:
                    screen_mode = "eyes"
                    current_image = None

        now = time.monotonic()
        if now - last_update >= POLL_INTERVAL:
            wifi_name   = get_wifi_name()
            last_update = now

        bat = get_battery_percent()

        # --- Blink logic (only when showing eyes) ---
        if screen_mode == "eyes":
            if frame_count >= next_blink and not blink_frames:
                blink_frames = [1.0, 0.6, 0.2, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0]
                blink_index  = 0
                next_blink   = frame_count + random.randint(100, 300)

            if blink_frames:
                open_ratio  = blink_frames[blink_index]
                blink_index += 1
                if blink_index >= len(blink_frames):
                    blink_frames = []
                    open_ratio   = 1.0

            look_timer -= 1
            if look_timer <= 0:
                look_target_x = random.randint(-8, 8)
                look_target_y = random.randint(-4, 4)
                look_timer    = random.randint(60, 180)

            look_x += (look_target_x - look_x) * 0.1
            look_y += (look_target_y - look_y) * 0.1

        # --- Draw ---
        screen.fill(BG)

        if screen_mode == "eyes":
            draw_eye(screen, LEFT_X,  EYE_Y, open_ratio, int(look_x), int(look_y))
            draw_eye(screen, RIGHT_X, EYE_Y, open_ratio, int(look_x), int(look_y))

        elif screen_mode == "now_playing":
            draw_now_playing(screen, font_med, font_small, frame_count)

        elif screen_mode == "show_image":
            draw_show_image(screen)

        elif screen_mode == "processing":
            draw_processing(screen, font_med, font_small, frame_count)

        elif screen_mode == "speaking":
            draw_speaking(screen, font_med, font_small, frame_count)

        draw_status_bar(screen, font_small, wifi_name, bat)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()