"""
parking_core.py
Shared helpers used by the Streamlit app (app.py).

The class-mapping logic here is IDENTICAL to the one in the Colab notebook
(cell "Class mapping"), so training and inference agree on which class ids
mean EMPTY and which mean OCCUPIED.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------
EMPTY_KEYWORDS = ("empty", "free", "vacant", "available", "unoccupied")
OCCUPIED_KEYWORDS = ("occupied", "car", "vehicle", "taken", "full", "busy", "parked")

EMPTY_COLOR = (46, 158, 91)      # green
OCCUPIED_COLOR = (209, 69, 59)   # red
OTHER_COLOR = (31, 95, 173)      # blue (classes that are neither)


def classify_name(name: str) -> str:
    """Return 'empty', 'occupied' or 'other' for a class name."""
    n = str(name).lower().replace("_", "-")
    # check EMPTY first: "unoccupied" contains "occupied"
    if any(k in n for k in EMPTY_KEYWORDS):
        return "empty"
    if any(k in n for k in OCCUPIED_KEYWORDS):
        return "occupied"
    return "other"


def build_class_map(names: dict | list) -> dict:
    """names -> {'empty': [ids], 'occupied': [ids], 'other': [ids], 'mode': ...}

    mode:
      'occupancy'     model predicts both empty and occupied spaces (best case)
      'vehicle_only'  model only predicts vehicles -> empty needs total bays
      'unknown'       class names could not be interpreted
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    cmap = {"empty": [], "occupied": [], "other": []}
    for cid, nm in names.items():
        cmap[classify_name(nm)].append(int(cid))
    if cmap["empty"] and cmap["occupied"]:
        mode = "occupancy"
    elif cmap["occupied"]:
        mode = "vehicle_only"
    else:
        mode = "unknown"
    cmap["mode"] = mode
    return cmap


# ---------------------------------------------------------------------------
# Model info written by the notebook
# ---------------------------------------------------------------------------
def load_model_info(model_dir: Path) -> dict:
    """Read model_info.json produced by the notebook (if present)."""
    p = Path(model_dir) / "model_info.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    cls_id: int
    name: str
    group: str          # empty / occupied / other
    conf: float
    box: tuple          # x1, y1, x2, y2 (pixels)


@dataclass
class ParkingResult:
    task: str                         # 'detect' or 'classify'
    detections: list = field(default_factory=list)
    top_label: str | None = None      # classification only
    top_group: str | None = None
    top_conf: float | None = None
    probs: dict = field(default_factory=dict)

    @property
    def n_empty(self) -> int:
        return sum(d.group == "empty" for d in self.detections)

    @property
    def n_occupied(self) -> int:
        return sum(d.group == "occupied" for d in self.detections)

    @property
    def n_other(self) -> int:
        return sum(d.group == "other" for d in self.detections)


def parse_ultralytics_result(res, names: dict, cmap: dict) -> ParkingResult:
    """Convert one ultralytics Results object into a ParkingResult."""
    group_of = {}
    for g in ("empty", "occupied", "other"):
        for cid in cmap.get(g, []):
            group_of[int(cid)] = g

    # Classification model
    if getattr(res, "probs", None) is not None and res.boxes is None:
        probs = res.probs.data.cpu().numpy()
        top = int(probs.argmax())
        return ParkingResult(
            task="classify",
            top_label=str(names[top]),
            top_group=group_of.get(top, "other"),
            top_conf=float(probs[top]),
            probs={str(names[i]): float(p) for i, p in enumerate(probs)},
        )

    # Detection model
    out = ParkingResult(task="detect")
    if res.boxes is None or len(res.boxes) == 0:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    for b, c, k in zip(xyxy, confs, clss):
        out.detections.append(
            Detection(
                cls_id=int(k),
                name=str(names[int(k)]),
                group=group_of.get(int(k), "other"),
                conf=float(c),
                box=tuple(float(v) for v in b),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_detections(img: Image.Image, result: ParkingResult, show_labels: bool = True) -> Image.Image:
    """Draw green (empty) / red (occupied) boxes on a copy of the image."""
    out = img.convert("RGB").copy()
    if result.task != "detect" or not result.detections:
        return out
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    thick = max(2, round(min(w, h) / 300))
    font = _font(max(10, round(min(w, h) / 55)))
    colors = {"empty": EMPTY_COLOR, "occupied": OCCUPIED_COLOR, "other": OTHER_COLOR}
    for d in result.detections:
        col = colors[d.group]
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], fill=col + (45,), outline=col + (255,), width=thick)
        if show_labels:
            txt = f"{d.name} {d.conf:.2f}"
            tb = draw.textbbox((0, 0), txt, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = y1 - th - 4 if y1 - th - 4 > 0 else y1
            draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=col + (230,))
            draw.text((x1 + 3, ty + 1), txt, fill=(255, 255, 255), font=font)
    return out


def to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))
