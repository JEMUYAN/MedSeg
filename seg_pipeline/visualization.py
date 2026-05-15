from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def overlay_mask_on_image(
    image_rgb: Image.Image,
    mask_l: Image.Image,
    *,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> Image.Image:
    img = image_rgb.convert("RGBA")
    m = mask_l.convert("L")

    if m.size != img.size:
        m = m.resize(img.size, resample=Image.NEAREST)

    a = m.point(lambda p: int((p / 255.0) * 255 * float(alpha)))
    overlay = Image.new("RGBA", img.size, color + (0,))
    overlay.putalpha(a)
    return Image.alpha_composite(img, overlay).convert("RGB")


def overlay_from_meta(
    meta_path: str,
    *,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
    output_path: Optional[str] = None,
    save: bool = True,
) -> Image.Image:
    meta = _read_json(meta_path)

    query_image_path = meta.get("query_image_path")
    mask_path = meta.get("output_mask_path")

    if not query_image_path:
        frames = (
            meta.get("sam3_meta", {})
            .get("synthetic_video", {})
            .get("frames", [])
        )
        if isinstance(frames, list) and len(frames) > 0:
            query_image_path = frames[-1].get("source_path")

    if not mask_path:
        raise ValueError(f"meta 缺少 output_mask_path: {meta_path}")
    if not query_image_path:
        raise ValueError(f"meta 缺少 query_image_path: {meta_path}")

    img = Image.open(query_image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    vis = overlay_mask_on_image(img, mask, color=color, alpha=alpha)

    if save:
        out = output_path
        if not out:
            p = Path(mask_path)
            out = str(p.with_name(f"{p.stem}_overlay.png"))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        vis.save(out, format="PNG")

    return vis

