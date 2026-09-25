import os
import sys
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLES_DIR

def generate_photorealistic_background(width=640, height=480) -> np.ndarray:
    """Generates a rich textured indoor scene with walls, flooring, paintings, and furniture."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Wall gradient
    for y in range(int(height * 0.65)):
        ratio = y / (height * 0.65)
        canvas[y, :, 0] = int(50 + ratio * 20)
        canvas[y, :, 1] = int(60 + ratio * 25)
        canvas[y, :, 2] = int(80 + ratio * 35)

    # Wooden floor gradient
    for y in range(int(height * 0.65), height):
        ratio = (y - height * 0.65) / (height * 0.35)
        canvas[y, :, 0] = int(140 - ratio * 40)
        canvas[y, :, 1] = int(95 - ratio * 30)
        canvas[y, :, 2] = int(60 - ratio * 20)

    # Floor planks
    for x in range(0, width, 60):
        canvas[int(height * 0.65):, x:x+2] = [40, 25, 15]

    # Art on wall
    art_x1, art_y1, art_x2, art_y2 = int(width * 0.1), int(height * 0.15), int(width * 0.4), int(height * 0.45)
    canvas[art_y1:art_y2, art_x1:art_x2] = [210, 180, 140]
    canvas[art_y1+5:art_y2-5, art_x1+5:art_x2-5] = [80, 140, 180]

    # Window
    win_x1, win_y1, win_x2, win_y2 = int(width * 0.65), int(height * 0.1), int(width * 0.9), int(height * 0.55)
    canvas[win_y1:win_y2, win_x1:win_x2] = [180, 220, 255]
    canvas[win_y1:win_y2, win_x1:win_x1+4] = [240, 240, 245]
    canvas[win_y1:win_y2, win_x2-4:win_x2] = [240, 240, 245]
    canvas[win_y1:win_y1+4, win_x1:win_x2] = [240, 240, 245]
    canvas[win_y2-4:win_y2, win_x1:win_x2] = [240, 240, 245]

    return np.ascontiguousarray(canvas, dtype=np.uint8)

def create_sample_clips():
    """Generates benchmark moving-camera test clips and sample test images with exact 640x480 dimensions."""
    width, height = 640, 480
    bg_wide = generate_photorealistic_background(width + 200, height + 40)
    fps = 30
    total_frames = 45  # 1.5 seconds smooth loop

    # 1. Generate sample_moving_person.gif
    print("Generating sample 1: sample_moving_person.gif ...")
    frames_pil_1 = []
    for t in range(total_frames):
        cam_shift_x = int(50 + 40 * math.sin(t / 10.0))
        cam_shift_y = int(10 + 5 * math.cos(t / 10.0))

        frame_arr = np.ascontiguousarray(
            bg_wide[cam_shift_y:cam_shift_y+height, cam_shift_x:cam_shift_x+width, :].copy(),
            dtype=np.uint8
        )
        pil_img = Image.fromarray(frame_arr).resize((width, height), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(pil_img)

        person_x = int(width * 0.5 + 15 * math.sin(t / 15.0))
        person_y = int(height * 0.55)

        # Head
        draw.ellipse([person_x - 28, person_y - 108, person_x + 28, person_y - 52], fill=(230, 190, 160))
        # Hair
        draw.chord([person_x - 28, person_y - 110, person_x + 28, person_y - 80], start=180, end=360, fill=(40, 30, 25))
        # Torso
        draw.rectangle([person_x - 38, person_y - 50, person_x + 38, person_y + 60], fill=(35, 75, 150))
        # Legs
        draw.rectangle([person_x - 30, person_y + 60, person_x - 5, person_y + 150], fill=(45, 45, 55))
        draw.rectangle([person_x + 5, person_y + 60, person_x + 30, person_y + 150], fill=(45, 45, 55))

        frames_pil_1.append(pil_img)

    frames_pil_1[0].save(
        str(SAMPLES_DIR / "sample_moving_person.gif"),
        save_all=True,
        append_images=frames_pil_1[1:],
        duration=40,
        loop=0
    )
    print(f"Sample 1 saved to: {SAMPLES_DIR / 'sample_moving_person.gif'}")

    # 2. Generate sample_red_cloak.gif
    print("Generating sample 2: sample_red_cloak.gif ...")
    frames_pil_2 = []

    for t in range(total_frames):
        cam_shift_x = int(50 + 35 * math.sin(t / 12.0))
        cam_shift_y = int(10)

        frame_arr = np.ascontiguousarray(
            bg_wide[cam_shift_y:cam_shift_y+height, cam_shift_x:cam_shift_x+width, :].copy(),
            dtype=np.uint8
        )
        pil_img = Image.fromarray(frame_arr).resize((width, height), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(pil_img)

        person_x = int(width * 0.5)
        person_y = int(height * 0.52)

        # Head
        draw.ellipse([person_x - 28, person_y - 108, person_x + 28, person_y - 52], fill=(230, 190, 160))
        # Red cloak polygon
        cloak_poly = [
            (person_x - 65, person_y - 45),
            (person_x + 65, person_y - 45),
            (person_x + 85, person_y + 130),
            (person_x - 85, person_y + 130)
        ]
        draw.polygon(cloak_poly, fill=(225, 25, 30))
        draw.line([(person_x - 20, person_y - 40), (person_x - 35, person_y + 120)], fill=(255, 75, 75), width=3)
        draw.line([(person_x + 20, person_y - 40), (person_x + 35, person_y + 120)], fill=(255, 75, 75), width=3)

        frames_pil_2.append(pil_img)

    frames_pil_2[0].save(
        str(SAMPLES_DIR / "sample_red_cloak.gif"),
        save_all=True,
        append_images=frames_pil_2[1:],
        duration=40,
        loop=0
    )
    print(f"Sample 2 saved to: {SAMPLES_DIR / 'sample_red_cloak.gif'}")

    # 3. Generate static high-res sample desk image
    sample_img_path = SAMPLES_DIR / "sample_desk_object.png"
    img_arr = np.ascontiguousarray(bg_wide[10:10+height, 50:50+width, :].copy(), dtype=np.uint8)
    pil_desk = Image.fromarray(img_arr).resize((width, height), Image.Resampling.NEAREST)
    draw_desk = ImageDraw.Draw(pil_desk)
    # Coffee mug
    draw_desk.rectangle([280, 270, 340, 340], fill=(235, 235, 240))
    draw_desk.ellipse([280, 260, 340, 280], fill=(110, 60, 30))
    draw_desk.arc([335, 285, 365, 325], start=270, end=90, fill=(235, 235, 240), width=4)
    pil_desk.save(str(sample_img_path))
    print(f"Sample image saved to: {sample_img_path}")

if __name__ == "__main__":
    create_sample_clips()
