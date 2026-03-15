"""Generate numbered cube face textures."""

import os
from PIL import Image, ImageDraw, ImageFont


def generate_face_textures(output_dir="assets", size=256):
    os.makedirs(output_dir, exist_ok=True)

    # Color scheme for each face
    colors = {
        1: ("#FF4444", "#FFFFFF"),  # Red bg, white text
        2: ("#4444FF", "#FFFFFF"),  # Blue bg, white text
        3: ("#44CC44", "#FFFFFF"),  # Green bg, white text
        4: ("#FFAA00", "#000000"),  # Orange bg, black text
        5: ("#CC44CC", "#FFFFFF"),  # Purple bg, white text
        6: ("#44CCCC", "#000000"),  # Cyan bg, black text
    }

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 160
        )
    except OSError:
        font = ImageFont.load_default()

    for face_num in range(1, 7):
        bg_color, text_color = colors[face_num]
        img = Image.new("RGB", (size, size), bg_color)
        draw = ImageDraw.Draw(img)

        text = str(face_num)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (size - w) / 2 - bbox[0]
        y = (size - h) / 2 - bbox[1]
        draw.text((x, y), text, fill=text_color, font=font)

        path = os.path.join(output_dir, f"face_{face_num}.png")
        img.save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    generate_face_textures()
