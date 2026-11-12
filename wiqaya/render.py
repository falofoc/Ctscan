"""Key images: every reported number is shown on the pixels it was measured from."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

COLORS = {"red": (255, 64, 64), "yellow": (255, 214, 10), "cyan": (60, 200, 255), "green": (60, 220, 120),
          "magenta": (255, 80, 220), "orange": (255, 150, 30)}


def window(a: np.ndarray, level: float, width: float) -> np.ndarray:
    lo = level - width / 2
    return (np.clip((a.astype(np.float32) - lo) / width, 0, 1) * 255).astype(np.uint8)


def axial(arr3: np.ndarray, z: int) -> np.ndarray:
    """Radiological convention: anterior up, patient right on image left."""
    return arr3[:, :, z].T[::-1, ::-1]


def sagittal(arr3: np.ndarray, x: int) -> np.ndarray:
    """Superior up, anterior on image left."""
    return arr3[x, :, :].T[::-1, ::-1]


def coronal(arr3: np.ndarray, y: int) -> np.ndarray:
    return arr3[:, y, :].T[::-1, ::-1]


def compose(gray: np.ndarray, layers: list[tuple[np.ndarray, str, str]], aspect: float = 1.0,
            size: int = 420, label: str | None = None) -> bytes:
    """layers: (mask, colour, 'fill' | 'edge'). `aspect` = row spacing / column spacing."""
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    for mask, colour, mode in layers:
        if mask is None or not mask.any():
            continue
        c = np.array(COLORS[colour], dtype=np.float32)
        if mode == "edge":
            edge = ndi.binary_dilation(mask, iterations=1) & ~ndi.binary_erosion(mask, iterations=1)
            rgb[edge] = c
        else:
            rgb[mask] = 0.45 * rgb[mask] + 0.55 * c
    img = Image.fromarray(rgb.astype(np.uint8))
    h, w = gray.shape
    scale = size / max(w, h * aspect)
    img = img.resize((max(int(w * scale), 1), max(int(h * aspect * scale), 1)), Image.BILINEAR)
    if label:
        ImageDraw.Draw(img).text((6, 4), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def crop_around(shape2d, centre_rc, half: int):
    r, c = centre_rc
    r0, r1 = max(r - half, 0), min(r + half, shape2d[0])
    c0, c1 = max(c - half, 0), min(c + half, shape2d[1])
    return slice(r0, r1), slice(c0, c1)
