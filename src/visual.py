"""Visual stimulus rendering + counting.

We render simple scenes (N coloured blobs on white) with PIL, then count them
back with a flood-fill connected-components pass on a thresholded alpha
channel. This gives us:

  - `render(visual_id) -> PIL.Image` for the tutor UI
  - `count_blobs(img) -> int` for visual grounding / self-check

Why no OWLVit / YOLO?  The brief caps total on-device footprint at 75 MB.
OWLVit-tiny alone is ~500 MB. A connected-components counter is exact on
synthetic stimuli, <1 ms to run, and gives the child's answer a ground-truth
check the tutor can reason about.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image, ImageDraw

CANVAS = (360, 240)       # demo-friendly size
BG     = (255, 255, 255)
PALETTE = [
    (220,  40,  40),       # red
    (240, 180,  40),       # yellow
    ( 40, 160,  90),       # green
    ( 40, 110, 200),       # blue
    (180,  90, 200),       # purple
    (220, 110,  40),       # orange
]

_VISUAL_RE = re.compile(r"^(?P<name>[a-z_]+)_(?P<n>\d+)$")
_ABSTRACT_EXPR_RE = re.compile(r"^abstract_(?P<a>\d+)_(?P<op>plus|minus)_(?P<b>\d+)$")


def parse_visual(visual_id: str) -> tuple[str, int] | None:
    """`'apples_3' → ('apples', 3)` ; returns None if the id has no count."""
    m = _VISUAL_RE.match(visual_id)
    if not m:
        return None
    return m.group("name"), int(m.group("n"))


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────
def _seed_rng(visual_id: str) -> int:
    """Stable per-id seed so the same visual renders identically every time."""
    return int(hashlib.sha256(visual_id.encode()).hexdigest(), 16) & 0xFFFF


def _jitter(rng: "random.Random", x: int, y: int, r: int) -> tuple[int, int]:
    dx = rng.randint(-r // 3, r // 3)
    dy = rng.randint(-r // 3, r // 3)
    return x + dx, y + dy


def render(visual_id: str | None, n: int | None = None, *,
           radius: int = 22, color: tuple[int, int, int] | None = None,
           ) -> Image.Image:
    """Render a visual stimulus.

    If `visual_id` parses as `<name>_<count>` (e.g. `'apples_3'`), we draw
    that many blobs. Otherwise `n` must be supplied. For abstract items like
    `'abstract_27_plus_14'` we draw no blobs — the child reads the stem
    instead. `visual_id=None` is treated as an abstract item (blank canvas).
    """
    import random
    if not visual_id:
        visual_id = "blank"
    rng = random.Random(_seed_rng(visual_id))

    parsed = parse_visual(visual_id)
    if parsed is not None:
        n = parsed[1] if n is None else n
    elif n is None:
        n = 0  # abstract stimulus

    img = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(img)

    if n <= 0:
        _render_abstract_hint(draw, visual_id)
        return img

    # Lay out in an evenly-spaced grid, then jitter so it looks hand-placed.
    # Cap cols so neighbouring blobs can't overlap after jitter:
    # need cell_w ≥ 2*radius + slack
    max_cols = max(2, (CANVAS[0] - radius * 2) // (radius * 2 + 10))
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    cell_w = CANVAS[0] // (cols + 1)
    cell_h = CANVAS[1] // (rows + 1)

    clr = color or rng.choice(PALETTE)
    drawn = 0
    for r in range(rows):
        for c in range(cols):
            if drawn >= n:
                break
            cx = cell_w * (c + 1)
            cy = cell_h * (r + 1)
            cx, cy = _jitter(rng, cx, cy, radius)
            draw.ellipse(
                [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                fill=clr, outline=(0, 0, 0), width=2,
            )
            drawn += 1
    return img


def _render_abstract_hint(draw: ImageDraw.ImageDraw, visual_id: str) -> None:
    """Draw a lightweight math hint for abstract items instead of blank white."""
    m = _ABSTRACT_EXPR_RE.match(visual_id or "")
    if m:
        a = m.group("a")
        b = m.group("b")
        op = "+" if m.group("op") == "plus" else "-"
        text = f"{a} {op} {b} = ?"
    else:
        # Generic fallback for any other abstract id.
        text = "Solve the equation"

    # Soft card background to improve readability for early learners.
    card_w, card_h = 230, 92
    x0 = (CANVAS[0] - card_w) // 2
    y0 = (CANVAS[1] - card_h) // 2
    x1 = x0 + card_w
    y1 = y0 + card_h
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=16,
                           fill=(245, 248, 255), outline=(40, 110, 200), width=3)

    # Center text using PIL default font for portability.
    bbox = draw.textbbox((0, 0), text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (CANVAS[0] - tw) // 2
    ty = (CANVAS[1] - th) // 2
    draw.text((tx, ty), text, fill=(20, 20, 20))


# ──────────────────────────────────────────────────────────────────────────
# Counting — connected-components on a dark-pixel mask
# ──────────────────────────────────────────────────────────────────────────
def count_blobs(img: Image.Image, *, min_size: int = 80) -> int:
    """Count connected dark regions in an RGB image.

    Uses iterative flood-fill so it works without scipy. `min_size`
    filters out antialiasing specks.
    """
    gray = img.convert("L")
    w, h = gray.size
    px = gray.load()

    visited = [[False] * w for _ in range(h)]
    count = 0
    THRESH = 200                       # below = "blob pixel"

    for y in range(h):
        for x in range(w):
            if visited[y][x] or px[x, y] >= THRESH:
                continue
            # BFS flood-fill
            stack = [(x, y)]
            size = 0
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cy < 0 or cx >= w or cy >= h:
                    continue
                if visited[cy][cx] or px[cx, cy] >= THRESH:
                    continue
                visited[cy][cx] = True
                size += 1
                stack.extend([(cx + 1, cy), (cx - 1, cy),
                              (cx, cy + 1), (cx, cy - 1)])
            if size >= min_size:
                count += 1
    return count


def save_stimulus(visual_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = render(visual_id)
    path = out_dir / f"{visual_id}.png"
    img.save(path)
    return path
