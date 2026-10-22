"""Generate the PWA icons from the same mark the app header uses.

    python scripts/make_icons.py

Writes static/icons/{icon-192,icon-512,maskable-512,apple-touch-icon,favicon-64}.png
Rerun after changing brand colours; the shapes are drawn, not traced, so they stay
crisp at every size.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "static" / "icons"
PURPLE = (139, 123, 255)
INDIGO = (75, 63, 212)


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def rounded_mask(size: int, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def draw_icon(size: int, *, maskable: bool = False) -> Image.Image:
    """Diagonal purple gradient, white C arc, white dot — the header logo."""
    scale = 4                                    # supersample, then downscale
    s = size * scale
    img = Image.new("RGB", (s, s), INDIGO)
    px = img.load()
    for y in range(s):
        for x in range(0, s, 4):                 # 4px stripes are plenty for a gradient
            t = (x / s + y / s) / 2
            c = lerp(PURPLE, INDIGO, t)
            for k in range(min(4, s - x)):
                px[x + k, y] = c

    icon = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    if maskable:
        icon.paste(img, (0, 0))                  # full bleed: the OS applies its own mask
        inset = 0.20
    else:
        radius = s * 0.24
        icon.paste(img, (0, 0), rounded_mask(s, radius))
        inset = 0.14

    # The mark: a thick ring with a wedge cut out of its right side, tips rounded —
    # the same C the header draws with a stroked SVG path.
    cx = cy = s / 2
    ring_r = s * (0.5 - inset) - s * 0.055
    stroke = s * 0.088
    gap = 46                                   # degrees of opening, centred on 0
    mask = Image.new("L", (s, s), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse((cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
               outline=255, width=round(stroke))
    reach = s
    wedge = [(cx, cy),
             (cx + reach * math.cos(math.radians(-gap / 2)), cy + reach * math.sin(math.radians(-gap / 2))),
             (cx + reach * math.cos(math.radians(gap / 2)), cy + reach * math.sin(math.radians(gap / 2)))]
    md.polygon(wedge, fill=0)
    for deg in (-gap / 2, gap / 2):            # rounded stroke caps
        ang = math.radians(deg)
        px_, py_ = cx + ring_r * math.cos(ang), cy + ring_r * math.sin(ang)
        r = stroke / 2
        md.ellipse((px_ - r, py_ - r, px_ + r, py_ + r), fill=255)

    white = Image.new("RGBA", (s, s), (255, 255, 255, 255))
    icon.paste(white, (0, 0), mask)

    dot = ImageDraw.Draw(icon)
    r = s * 0.072
    dot.ellipse((cx - r, cy - r, cx + r, cy + r), fill="white")

    return icon.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("icon-192.png", 192, False),
        ("icon-512.png", 512, False),
        ("maskable-512.png", 512, True),
        ("apple-touch-icon.png", 180, False),
        ("favicon-64.png", 64, False),
    ]
    for name, size, maskable in jobs:
        draw_icon(size, maskable=maskable).save(OUT / name, optimize=True)
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
