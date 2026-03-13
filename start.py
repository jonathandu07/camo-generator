# -*- coding: utf-8 -*-
"""
start.py
Interface Kivy moderne, asynchrone, redimensionnable, stylée selon une direction :
- glassmorphisme
- neumorphisme
- bento design
- flat design
- liquid glass (simulation visuelle)

Fonctions :
- génération séquentielle stricte via main.py
- prise en compte de log.py en direct
- affichage live du diagnostic des rejets
- barre de chargement stylée
- aperçu principal + projection silhouette
- galerie des camouflages générés
- monitoring CPU / RAM / processus
- prévention de la mise en veille sous Windows
- réglage léger de l'intensité machine
- boutons Commencer / Arrêter
- préflight tests automatiques sur test_main.py et test_start.py
"""

from __future__ import annotations

import asyncio
import csv
import ctypes
import io
import json
import math
import os
import time
import platform
import shutil
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

try:
    import psutil
except Exception:
    psutil = None

os.environ.setdefault("KIVY_NO_ARGS", "1")

try:
    import log as camo_log
except Exception:
    camo_log = None

sys.modules.setdefault("start", sys.modules[__name__])

# ---------------- Kivy config avant imports UI ----------------
from kivy.config import Config

Config.set("graphics", "fullscreen", "0")
Config.set("graphics", "resizable", "1")
Config.set("graphics", "minimum_width", "1280")
Config.set("graphics", "minimum_height", "780")
Config.set("graphics", "width", "1680")
Config.set("graphics", "height", "980")
Config.set("input", "mouse", "mouse,multitouch_on_demand")

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.graphics import Color, Line, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.properties import NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# ---------------- moteur principal async ----------------
import main as camo


# ============================================================
# PALETTE STRICTE — UNIQUEMENT TES COULEURS
# ============================================================

PALETTE_HEX = {
    "BL": "#f4fefe",
    "GW": "#f7f7ff",
    "BG": "#e5e5e5",
    "GF": "#d9d9d9",
    "GAXD": "#707070",
    "VG": "#6B6C66",
    "TDF": "#81613C",
    "BT": "#644E41",
    "BC": "#9F8670",
    "JA": "#FFCB60",
    "JB": "#E8D630",
    "JO": "#EFD807",
    "JV": "#FFC600",
    "NG": "#3E5349",
    "VDG": "#95A595",
    "VO": "#424530",
    "BF": "#051440",
    "BA": "#81A1B8",
    "BM": "#03224C",
    "BFW": "#091226",
    "GS": "#0A0B0A",
    "NF": "#1E1E1E",
    "NA": "#303030",
    "RS": "#75161E",
    "RV": "#FF0000",
    "RF": "#EC1920",
    "Vermeil": "#FF0921",
    "RP": "#8B0000",
    "black": "#000000",
    "white": "#ffffff",
}


def hex_rgba(name: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    hx = PALETTE_HEX[name].lstrip("#")
    return (
        int(hx[0:2], 16) / 255.0,
        int(hx[2:4], 16) / 255.0,
        int(hx[4:6], 16) / 255.0,
        alpha,
    )


C = {
    "bg_root": hex_rgba("BFW", 1.0),
    "bg_panel": hex_rgba("BM", 0.82),
    "bg_panel_soft": hex_rgba("BF", 0.78),
    "bg_panel_inner": hex_rgba("BF", 0.92),
    "bg_input": hex_rgba("NA", 0.90),
    "bg_input_soft": hex_rgba("BF", 0.96),
    "stroke": hex_rgba("BA", 0.42),
    "stroke_soft": hex_rgba("VDG", 0.18),
    "stroke_strong": hex_rgba("BL", 0.14),
    "text_main": hex_rgba("BL", 1.0),
    "text_soft": hex_rgba("GF", 1.0),
    "text_muted": hex_rgba("BA", 1.0),
    "success": hex_rgba("NG", 1.0),
    "warning": hex_rgba("JA", 1.0),
    "danger": hex_rgba("RF", 1.0),
    "accent": hex_rgba("BA", 1.0),
    "accent_soft": hex_rgba("BA", 0.24),
    "progress_bg": hex_rgba("NA", 0.96),
    "progress_fill": hex_rgba("BA", 0.96),
    "progress_glow": hex_rgba("BL", 0.18),
    "thumb_bg": hex_rgba("BM", 0.90),
    "thumb_border": hex_rgba("BA", 0.22),
    "shadow": hex_rgba("GS", 0.42),
    "shadow_soft": hex_rgba("GS", 0.22),
    "glass_top": hex_rgba("BL", 0.06),
    "glass_mid": hex_rgba("BL", 0.035),
    "glass_bottom": hex_rgba("BFW", 0.10),
    "btn_launch": hex_rgba("NG", 1.0),
    "btn_launch_down": hex_rgba("VO", 1.0),
    "btn_stop": hex_rgba("RS", 1.0),
    "btn_stop_down": hex_rgba("RP", 1.0),
    "btn_neutral": hex_rgba("BT", 1.0),
    "btn_disabled": hex_rgba("VG", 0.55),
}


# ============================================================
# CONSTANTES UI / EXPORT
# ============================================================

APP_TITLE = "Camouflage Armée Fédérale Europe"
BEST_DIR_NAME = "best_of"
REPORT_NAME = "rapport_camouflages_v3.csv"
DEFAULT_OUTPUT_DIR = Path("camouflages_federale_europe")
DEFAULT_TARGET_COUNT = 100
DEFAULT_TOP_K = 20

DEFAULT_PREFLIGHT_MODULES = ("test_main", "test_start")
DEFAULT_PREFLIGHT_TIMEOUT_S = None

RUN_MODE_BLOCKING = "blocking"
RUN_MODE_NON_BLOCKING = "non_blocking"
RUN_MODE_SKIP_TESTS = "skip_tests"

WEIGHT_RATIO = 0.28
WEIGHT_SILHOUETTE = 0.30
WEIGHT_CONTOUR = 0.24
WEIGHT_MAIN_METRICS = 0.18

MIN_SILHOUETTE_COLOR_DIVERSITY = 0.62
MIN_CONTOUR_BREAK_SCORE = 0.44
MIN_OUTLINE_BAND_DIVERSITY = 0.58
MIN_SMALL_SCALE_STRUCTURAL_SCORE = 0.42

THUMB_SIZE = (240, 150)
GALLERY_COLUMNS = 3

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


# ============================================================
# STRUCTURES
# ============================================================

@dataclass
class CandidateRecord:
    index: int
    seed: int
    local_attempt: int
    global_attempt: int
    image_path: Path
    score_final: float
    score_ratio: float
    score_silhouette: float
    score_contour: float
    score_main: float
    silhouette_color_diversity: float
    contour_break_score: float
    outline_band_diversity: float
    small_scale_structural_score: float
    rs: np.ndarray
    metrics: Dict[str, float]


# ============================================================
# EVENT LOOP ASYNC DÉDIÉ
# ============================================================

class AsyncioThreadRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="kivy-async-loop")
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ============================================================
# OUTILS SYSTÈME
# ============================================================

def open_folder(path: Path) -> None:
    path = path.resolve()
    system = platform.system().lower()
    try:
        if system == "windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def prevent_sleep(enable: bool) -> None:
    if platform.system().lower() != "windows":
        return
    try:
        if enable:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def pil_to_coreimage(pil_img: PILImage.Image) -> CoreImage:
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)
    return CoreImage(bio, ext="png")


def make_thumbnail(pil_img: PILImage.Image, size: Tuple[int, int]) -> PILImage.Image:
    img = pil_img.copy()
    img.thumbnail(size, PILImage.Resampling.LANCZOS)
    canvas = PILImage.new("RGB", size, tuple(int(v * 255) for v in C["bg_root"][:3]))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def palette_map() -> Dict[Tuple[int, int, int], int]:
    return {
        tuple(camo.RGB[camo.IDX_COYOTE].tolist()): camo.IDX_COYOTE,
        tuple(camo.RGB[camo.IDX_OLIVE].tolist()): camo.IDX_OLIVE,
        tuple(camo.RGB[camo.IDX_TERRE].tolist()): camo.IDX_TERRE,
        tuple(camo.RGB[camo.IDX_GRIS].tolist()): camo.IDX_GRIS,
    }


PALETTE_TO_INDEX = palette_map()


def rgb_image_to_index_canvas(img: PILImage.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for rgb, idx in PALETTE_TO_INDEX.items():
        mask = np.all(arr == np.array(rgb, dtype=np.uint8), axis=-1)
        out[mask] = idx
    return out


def downsample_nearest(canvas: np.ndarray, factor: int) -> np.ndarray:
    return canvas[::factor, ::factor]


def boundary_mask(canvas: np.ndarray) -> np.ndarray:
    diff = np.zeros_like(canvas, dtype=bool)
    diff[1:, :] |= canvas[1:, :] != canvas[:-1, :]
    diff[:-1, :] |= canvas[:-1, :] != canvas[1:, :]
    diff[:, 1:] |= canvas[:, 1:] != canvas[:, :-1]
    diff[:, :-1] |= canvas[:, :-1] != canvas[:, 1:]
    return diff


def largest_component_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape
    total = int(mask.sum())
    if total == 0:
        return 0.0

    visited = np.zeros_like(mask, dtype=bool)
    best = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            size = 0

            while stack:
                cy, cx = stack.pop()
                size += 1

                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy < h - 1 and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx < w - 1 and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    stack.append((cy, cx + 1))

            if size > best:
                best = size

    return best / total


# ============================================================
# SILHOUETTE PSEUDO-HUMAINE
# ============================================================

def build_silhouette_mask(width: int, height: int) -> np.ndarray:
    img = PILImage.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    head_w = int(width * 0.18)
    head_h = int(height * 0.12)
    head_x1 = (width - head_w) // 2
    head_y1 = int(height * 0.05)
    draw.ellipse([head_x1, head_y1, head_x1 + head_w, head_y1 + head_h], fill=255)

    torso_w = int(width * 0.34)
    torso_h = int(height * 0.38)
    torso_x1 = (width - torso_w) // 2
    torso_y1 = int(height * 0.16)
    draw.rounded_rectangle(
        [torso_x1, torso_y1, torso_x1 + torso_w, torso_y1 + torso_h],
        radius=int(width * 0.03),
        fill=255,
    )

    shoulder_w = int(width * 0.58)
    shoulder_h = int(height * 0.10)
    shoulder_x1 = (width - shoulder_w) // 2
    shoulder_y1 = int(height * 0.14)
    draw.rounded_rectangle(
        [shoulder_x1, shoulder_y1, shoulder_x1 + shoulder_w, shoulder_y1 + shoulder_h],
        radius=int(width * 0.025),
        fill=255,
    )

    arm_w = int(width * 0.11)
    arm_h = int(height * 0.32)
    left_arm_x1 = int(width * 0.15)
    right_arm_x1 = width - left_arm_x1 - arm_w
    arm_y1 = int(height * 0.20)
    draw.rounded_rectangle(
        [left_arm_x1, arm_y1, left_arm_x1 + arm_w, arm_y1 + arm_h],
        radius=int(width * 0.02),
        fill=255,
    )
    draw.rounded_rectangle(
        [right_arm_x1, arm_y1, right_arm_x1 + arm_w, arm_y1 + arm_h],
        radius=int(width * 0.02),
        fill=255,
    )

    pelvis_w = int(width * 0.30)
    pelvis_h = int(height * 0.10)
    pelvis_x1 = (width - pelvis_w) // 2
    pelvis_y1 = int(height * 0.51)
    draw.rounded_rectangle(
        [pelvis_x1, pelvis_y1, pelvis_x1 + pelvis_w, pelvis_y1 + pelvis_h],
        radius=int(width * 0.02),
        fill=255,
    )

    leg_w = int(width * 0.12)
    leg_h = int(height * 0.32)
    leg_gap = int(width * 0.04)
    left_leg_x1 = (width // 2) - leg_gap // 2 - leg_w
    right_leg_x1 = (width // 2) + leg_gap // 2
    leg_y1 = int(height * 0.58)
    draw.rounded_rectangle(
        [left_leg_x1, leg_y1, left_leg_x1 + leg_w, leg_y1 + leg_h],
        radius=int(width * 0.018),
        fill=255,
    )
    draw.rounded_rectangle(
        [right_leg_x1, leg_y1, right_leg_x1 + leg_w, leg_y1 + leg_h],
        radius=int(width * 0.018),
        fill=255,
    )

    return np.array(img, dtype=np.uint8) > 0


def silhouette_boundary(mask: np.ndarray) -> np.ndarray:
    b = np.zeros_like(mask, dtype=bool)
    b[1:, :] |= mask[1:, :] != mask[:-1, :]
    b[:-1, :] |= mask[:-1, :] != mask[1:, :]
    b[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return b & mask


def dilate_bool(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y1 = max(0, dy)
            y2 = min(h, h + dy)
            x1 = max(0, dx)
            x2 = min(w, w + dx)

            sy1 = max(0, -dy)
            sy2 = min(h, h - dy)
            sx1 = max(0, -dx)
            sx2 = min(w, w - dx)

            out[y1:y2, x1:x2] |= mask[sy1:sy2, sx1:sx2]
    return out


def silhouette_projection_image(index_canvas: np.ndarray) -> PILImage.Image:
    h, w = index_canvas.shape
    sil = build_silhouette_mask(w, h)

    background = np.full(
        (h, w, 3),
        (
            int(hex_rgba("BFW")[0] * 255),
            int(hex_rgba("BFW")[1] * 255),
            int(hex_rgba("BFW")[2] * 255),
        ),
        dtype=np.uint8,
    )
    rgb = camo.RGB[index_canvas]
    background[sil] = rgb[sil]

    bound = silhouette_boundary(sil)
    background[bound] = np.array(
        [
            int(hex_rgba("BL")[0] * 255),
            int(hex_rgba("BL")[1] * 255),
            int(hex_rgba("BL")[2] * 255),
        ],
        dtype=np.uint8,
    )

    return PILImage.fromarray(background, "RGB")


def silhouette_color_diversity_score(index_canvas: np.ndarray) -> float:
    sil = build_silhouette_mask(index_canvas.shape[1], index_canvas.shape[0])
    data = index_canvas[sil]
    if data.size == 0:
        return 0.0
    unique_count = len(np.unique(data))
    hist = np.bincount(data, minlength=4).astype(float)
    hist /= hist.sum()
    entropy = -np.sum([p * math.log(p + 1e-12) for p in hist if p > 0])
    entropy /= math.log(4.0)
    return clamp01((unique_count / 4.0) * 0.45 + entropy * 0.55)


def contour_break_score(index_canvas: np.ndarray) -> Tuple[float, float]:
    h, w = index_canvas.shape
    sil = build_silhouette_mask(w, h)
    bound = silhouette_boundary(sil)

    band = dilate_bool(bound, radius=5) & sil
    vals = index_canvas[band]
    if vals.size == 0:
        return 0.0, 0.0

    hist = np.bincount(vals, minlength=4).astype(float)
    hist /= hist.sum()
    entropy = -np.sum([p * math.log(p + 1e-12) for p in hist if p > 0])
    entropy /= math.log(4.0)

    ys, xs = np.where(bound)
    varied = 0
    total = len(xs)

    for y, x in zip(ys, xs):
        y1, y2 = max(0, y - 3), min(h, y + 4)
        x1, x2 = max(0, x - 3), min(w, x + 4)
        neighborhood = index_canvas[y1:y2, x1:x2]
        if len(np.unique(neighborhood)) >= 2:
            varied += 1

    local_variation = varied / total if total else 0.0
    score = clamp01(local_variation * 0.62 + entropy * 0.38)
    return score, float(entropy)


def small_scale_structural_score(index_canvas: np.ndarray) -> float:
    small = downsample_nearest(index_canvas, 4)
    olive_ratio = largest_component_ratio(small == camo.IDX_OLIVE)
    bd = float(np.mean(boundary_mask(small)))

    s1 = clamp01((olive_ratio - 0.08) / 0.18)
    s2 = 1.0 - min(1.0, abs(bd - 0.11) / 0.12)
    return clamp01(0.58 * s1 + 0.42 * s2)


# ============================================================
# SCORING GLOBAL
# ============================================================

def ratio_score(rs: np.ndarray) -> float:
    err = np.abs(rs - camo.TARGET)
    mae = float(np.mean(err))
    return clamp01(1.0 - mae / 0.05)


def main_metrics_score(metrics: Dict[str, float]) -> float:
    parts = [
        clamp01((metrics["largest_olive_component_ratio"] - 0.12) / 0.18),
        clamp01(1.0 - metrics["center_empty_ratio"] / 0.60),
        clamp01(1.0 - metrics["mirror_similarity"] / 0.90),
        clamp01((metrics["olive_multizone_share"] - 0.25) / 0.45),
        clamp01(1.0 - abs(metrics["boundary_density"] - 0.14) / 0.12),
        clamp01((metrics["vert_olive_macro_share"] - 0.45) / 0.30),
        clamp01((metrics["terre_de_france_transition_share"] - 0.20) / 0.30),
        clamp01((metrics["vert_de_gris_micro_share"] - 0.50) / 0.25),
    ]
    return float(np.mean(parts))


def evaluate_candidate_v3(
    pil_img: PILImage.Image,
    rs: np.ndarray,
    metrics: Dict[str, float],
) -> Tuple[Dict[str, float], bool]:
    index_canvas = rgb_image_to_index_canvas(pil_img)

    sil_div = silhouette_color_diversity_score(index_canvas)
    contour_score, outline_band_div = contour_break_score(index_canvas)
    small_scale_score = small_scale_structural_score(index_canvas)

    s_ratio = ratio_score(rs)
    s_main = main_metrics_score(metrics)
    s_sil = sil_div
    s_contour = clamp01(0.65 * contour_score + 0.35 * outline_band_div)

    final_score = (
        WEIGHT_RATIO * s_ratio
        + WEIGHT_SILHOUETTE * s_sil
        + WEIGHT_CONTOUR * s_contour
        + WEIGHT_MAIN_METRICS * s_main
    )

    is_valid_v3 = (
        sil_div >= MIN_SILHOUETTE_COLOR_DIVERSITY
        and contour_score >= MIN_CONTOUR_BREAK_SCORE
        and outline_band_div >= MIN_OUTLINE_BAND_DIVERSITY
        and small_scale_score >= MIN_SMALL_SCALE_STRUCTURAL_SCORE
    )

    return {
        "score_final": final_score,
        "score_ratio": s_ratio,
        "score_silhouette": s_sil,
        "score_contour": s_contour,
        "score_main": s_main,
        "silhouette_color_diversity": sil_div,
        "contour_break_score": contour_score,
        "outline_band_diversity": outline_band_div,
        "small_scale_structural_score": small_scale_score,
    }, is_valid_v3


async def async_evaluate_candidate_v3(
    pil_img: PILImage.Image,
    rs: np.ndarray,
    metrics: Dict[str, float],
) -> Tuple[Dict[str, float], bool]:
    return await asyncio.to_thread(evaluate_candidate_v3, pil_img, rs, metrics)


# ============================================================
# WIDGETS STYLE
# ============================================================

class GlassProgressBar(Widget):
    max_value = NumericProperty(100.0)
    value = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(26)
        self.bind(pos=self._redraw, size=self._redraw, value=self._redraw, max_value=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        radius = dp(13)
        pad = dp(1.5)

        progress = 0.0 if self.max_value <= 0 else max(0.0, min(1.0, self.value / self.max_value))
        fill_w = max(dp(8), (self.width - pad * 2) * progress)

        with self.canvas.before:
            Color(*C["shadow_soft"])
            RoundedRectangle(
                pos=(self.x, self.y - dp(1)),
                size=self.size,
                radius=[radius] * 4,
            )

            Color(*C["progress_bg"])
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[radius] * 4,
            )

            Color(*C["glass_top"])
            RoundedRectangle(
                pos=(self.x + pad, self.y + self.height * 0.54),
                size=(self.width - pad * 2, self.height * 0.34),
                radius=[radius] * 4,
            )

            Color(*C["progress_fill"])
            RoundedRectangle(
                pos=(self.x + pad, self.y + pad),
                size=(fill_w, self.height - pad * 2),
                radius=[radius] * 4,
            )

            Color(*C["progress_glow"])
            RoundedRectangle(
                pos=(self.x + pad + dp(2), self.y + self.height * 0.56),
                size=(max(dp(4), fill_w - dp(4)), self.height * 0.18),
                radius=[radius] * 4,
            )

            Color(*C["stroke_soft"])
            Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, radius),
                width=1.0,
            )


class GlassCard(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self._redraw, size=self._redraw)
        self.padding = dp(18)
        self.spacing = dp(12)

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(24)
        with self.canvas.before:
            Color(*C["shadow"])
            RoundedRectangle(
                pos=(self.x, self.y - dp(3)),
                size=self.size,
                radius=[r] * 4,
            )

            Color(*C["bg_panel"])
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[r] * 4,
            )

            Color(*C["glass_mid"])
            RoundedRectangle(
                pos=(self.x + dp(1), self.y + self.height * 0.42),
                size=(self.width - dp(2), self.height * 0.46),
                radius=[r] * 4,
            )

            Color(*C["glass_top"])
            RoundedRectangle(
                pos=(self.x + dp(2), self.y + self.height * 0.62),
                size=(self.width - dp(4), self.height * 0.22),
                radius=[r] * 4,
            )

            Color(*C["stroke"])
            Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, r),
                width=1.0,
            )


class SoftPane(BoxLayout):
    def __init__(self, radius: float = 18, **kwargs):
        super().__init__(**kwargs)
        self._radius = radius
        self.padding = dp(8)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(self._radius)
        with self.canvas.before:
            Color(*C["bg_panel_inner"])
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[r] * 4,
            )
            Color(*C["stroke_soft"])
            Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, r),
                width=1.0,
            )


class SoftTextInput(TextInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_active = ""
        self.background_color = (0, 0, 0, 0)
        self.foreground_color = C["text_main"]
        self.cursor_color = C["text_main"]
        self.padding = [dp(14), dp(14), dp(14), dp(14)]
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(16)
        with self.canvas.before:
            Color(*C["shadow_soft"])
            RoundedRectangle(
                pos=(self.x, self.y - dp(1)),
                size=self.size,
                radius=[r] * 4,
            )
            Color(*C["bg_input_soft"])
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[r] * 4,
            )
            Color(*C["glass_top"])
            RoundedRectangle(
                pos=(self.x + dp(2), self.y + self.height * 0.56),
                size=(self.width - dp(4), self.height * 0.22),
                radius=[r] * 4,
            )
            Color(*C["stroke_soft"])
            Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, r),
                width=1.0,
            )


class ImageStage(Widget):
    def __init__(self, image_widget: Image, **kwargs):
        super().__init__(**kwargs)
        self.image_widget = image_widget
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(18)
        pad = dp(2)
        with self.canvas.before:
            Color(*C["bg_panel_inner"])
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[r] * 4,
            )

            Color(*C["glass_top"])
            RoundedRectangle(
                pos=(self.x + pad, self.y + self.height * 0.68),
                size=(self.width - pad * 2, self.height * 0.18),
                radius=[r] * 4,
            )

            Color(*C["stroke_strong"])
            Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, r),
                width=1.0,
            )


class SoftButton(Button):
    def __init__(self, role: str = "neutral", **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = C["text_main"]
        self.size_hint_y = None
        self.height = dp(58)
        self.bold = True
        self.bind(pos=self._redraw, size=self._redraw, state=self._redraw, disabled=self._redraw)

    def _palette(self):
        if self.disabled:
            return C["btn_disabled"], C["btn_disabled"]

        if self.role == "launch":
            return C["btn_launch"], C["btn_launch_down"]
        if self.role == "stop":
            return C["btn_stop"], C["btn_stop_down"]
        return C["btn_neutral"], C["btn_neutral"]

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(18)
        up, down = self._palette()
        bg = down if self.state == "down" else up

        with self.canvas.before:
            Color(*C["shadow_soft"])
            RoundedRectangle(
                pos=(self.x, self.y - dp(1.5)),
                size=self.size,
                radius=[r] * 4,
            )

            Color(*bg)
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[r] * 4,
            )

            Color(*C["glass_top"])
            RoundedRectangle(
                pos=(self.x + dp(2), self.y + self.height * 0.58),
                size=(self.width - dp(4), self.height * 0.18),
                radius=[r] * 4,
            )

            Color(*C["stroke_soft"])
            Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, r),
                width=1.0,
            )


class LogView(ScrollView):
    text = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_scroll_x = False
        self.bar_width = dp(8)

        self.label = Label(
            text="",
            markup=False,
            size_hint_y=None,
            text_size=(0, None),
            halign="left",
            valign="top",
            font_size=sp(13),
            color=C["text_soft"],
        )
        self.label.bind(texture_size=self._update_label_height, width=self._update_text_width)
        self.add_widget(self.label)

    def _update_label_height(self, *_):
        self.label.height = self.label.texture_size[1] + dp(16)

    def _update_text_width(self, *_):
        self.label.text_size = (self.width - dp(24), None)

    def append(self, line: str) -> None:
        lines = self.label.text.splitlines() if self.label.text else []
        lines.append(line)
        if len(lines) > 600:
            lines = lines[-600:]
        self.label.text = "\n".join(lines)
        Clock.schedule_once(lambda dt: setattr(self, "scroll_y", 0), 0.05)


class GalleryThumb(Button):
    def __init__(self, app_ref: "CamouflageApp", image_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.app_ref = app_ref
        self.image_path = image_path
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.size_hint_y = None
        self.height = dp(184)

        self.container = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))
        self.add_widget(self.container)

        self.stage = SoftPane(orientation="vertical", size_hint_y=1)
        self.thumb = Image()
        self.stage.add_widget(self.thumb)

        self.caption = Label(
            text=image_path.name,
            size_hint_y=None,
            height=dp(24),
            font_size=sp(11),
            color=C["text_soft"],
            halign="center",
            valign="middle",
        )
        self.caption.bind(size=lambda *a: setattr(self.caption, "text_size", self.caption.size))

        self.container.add_widget(self.stage)
        self.container.add_widget(self.caption)

        self.bind(on_release=self._open_preview)
        self.bind(pos=self._redraw, size=self._redraw, state=self._redraw)
        self.load_thumbnail()

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(20)
        y = self.y - dp(1.5 if self.state == "normal" else 0.5)
        with self.canvas.before:
            Color(*C["shadow_soft"])
            RoundedRectangle(pos=(self.x, y), size=self.size, radius=[r] * 4)
            Color(*C["thumb_bg"])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[r] * 4)
            Color(*C["thumb_border"])
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, r), width=1.0)

    def load_thumbnail(self):
        try:
            img = PILImage.open(self.image_path).convert("RGB")
            thumb = make_thumbnail(img, THUMB_SIZE)
            self.thumb.texture = pil_to_coreimage(thumb).texture
        except Exception:
            pass

    def _open_preview(self, *_):
        try:
            pil_img = PILImage.open(self.image_path).convert("RGB")
            idx = rgb_image_to_index_canvas(pil_img)
            sil = silhouette_projection_image(idx)
            self.app_ref.update_preview(pil_img, sil)
            self.app_ref.log(f"Aperçu galerie : {self.image_path.name}")
        except Exception as exc:
            self.app_ref.log(f"Impossible d'ouvrir {self.image_path.name} : {exc}")


# ============================================================
# APPLICATION
# ============================================================

class CamouflageApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.async_runner = AsyncioThreadRunner()
        self.current_future: Optional[Future] = None
        self.preflight_future: Optional[Future] = None

        self.stop_flag = False
        self.running = False
        self.stopping = False

        self.current_output_dir = DEFAULT_OUTPUT_DIR
        self.best_records: List[CandidateRecord] = []

        self.current_preview_img: Optional[PILImage.Image] = None
        self.current_silhouette_preview: Optional[PILImage.Image] = None

        self.accepted_count = 0
        self.total_attempts = 0

        self.process = psutil.Process() if psutil else None
        self.machine_intensity = 85.0

        self.tests_ran = False
        self.tests_ok = False
        self.tests_summary = "Tests non lancés."

        self.run_mode = RUN_MODE_BLOCKING

        self.preflight_running = False
        self.preflight_pending_start = False

        self.run_mode_label: Optional[Label] = None
        self.mode_blocking_btn: Optional[SoftButton] = None
        self.mode_non_blocking_btn: Optional[SoftButton] = None
        self.mode_skip_tests_btn: Optional[SoftButton] = None

        self.title_label: Optional[Label] = None
        self.status_label: Optional[Label] = None
        self.count_input: Optional[SoftTextInput] = None
        self.start_btn: Optional[SoftButton] = None
        self.stop_btn: Optional[SoftButton] = None
        self.progress_bar: Optional[GlassProgressBar] = None
        self.progress_text: Optional[Label] = None
        self.attempt_text: Optional[Label] = None
        self.intensity_slider: Optional[Slider] = None
        self.intensity_label: Optional[Label] = None
        self.resource_text: Optional[Label] = None
        self.resource_hint: Optional[Label] = None
        self.score_text: Optional[Label] = None
        self.color_text: Optional[Label] = None
        self.extra_text: Optional[Label] = None
        self.gallery_scroll: Optional[ScrollView] = None
        self.gallery_grid: Optional[GridLayout] = None
        self.preview_img: Optional[Image] = None
        self.preview_silhouette: Optional[Image] = None
        self.log_view: Optional[LogView] = None

        self.tests_label: Optional[Label] = None
        self.runtime_enabled_label: Optional[Label] = None
        self.runtime_last_label: Optional[Label] = None
        self._runtime_subscription_active = False
        self._runtime_subscriber_callback = None
        self._runtime_buffer_limit = 800

        self.diag_enabled_label: Optional[Label] = None
        self.diag_summary_label: Optional[Label] = None
        self.diag_top_rules_label: Optional[Label] = None
        self.diag_last_fail_label: Optional[Label] = None
        self.diag_log_view: Optional[LogView] = None

        self.diag_total = 0
        self.diag_accepts = 0
        self.diag_rejects = 0
        self.diag_rule_counter: Counter[str] = Counter()
        self.diag_last_rules: List[str] = []

    def build(self):
        Window.clearcolor = C["bg_root"]
        root = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        header = GlassCard(orientation="horizontal", size_hint_y=None, height=dp(132))

        title_box = BoxLayout(orientation="vertical", spacing=dp(2))
        tiny = Label(
            text="Image 000 | essai 0000",
            size_hint_y=None,
            height=dp(20),
            font_size=sp(11),
            color=C["text_muted"],
            halign="left",
            valign="middle",
        )
        tiny.bind(size=lambda *a: setattr(tiny, "text_size", tiny.size))
        self.attempt_text = tiny

        self.title_label = Label(
            text=APP_TITLE,
            font_size=sp(18),
            bold=True,
            color=C["text_main"],
            halign="left",
            valign="middle",
        )
        self.title_label.bind(size=lambda *a: setattr(self.title_label, "text_size", self.title_label.size))

        title_box.add_widget(self.attempt_text)
        title_box.add_widget(self.title_label)

        self.status_label = Label(
            text="Prêt",
            font_size=sp(15),
            color=C["text_muted"],
            size_hint_x=0.18,
            halign="right",
            valign="middle",
        )
        self.status_label.bind(size=lambda *a: setattr(self.status_label, "text_size", self.status_label.size))

        header.add_widget(title_box)
        header.add_widget(self.status_label)
        root.add_widget(header)

        body = BoxLayout(spacing=dp(10))

        left = BoxLayout(orientation="vertical", size_hint_x=0.34, spacing=dp(10))

        left_scroll = ScrollView(do_scroll_x=False, bar_width=dp(8))
        left_content = BoxLayout(orientation="vertical", spacing=dp(10), size_hint_y=None)
        left_content.bind(minimum_height=left_content.setter("height"))

        controls = GlassCard(
            orientation="vertical",
            size_hint_y=None,
        )
        controls.bind(minimum_height=controls.setter("height"))

        controls.add_widget(self._label("Paramètres"))

        self.count_input = SoftTextInput(
            text=str(DEFAULT_TARGET_COUNT),
            multiline=False,
            input_filter="int",
            size_hint_y=None,
            height=dp(50),
        )
        controls.add_widget(self.count_input)

        btn_row = BoxLayout(size_hint_y=None, height=dp(64), spacing=dp(12))
        self.start_btn = self._styled_button("Commencer", "launch", self.start_generation)
        self.stop_btn = self._styled_button("Arrêter", "stop", self.stop_generation)
        btn_row.add_widget(self.start_btn)
        btn_row.add_widget(self.stop_btn)
        controls.add_widget(btn_row)

        controls.add_widget(self._label("Préflight tests"))
        self.tests_label = self._small_label("Tests non lancés.")
        controls.add_widget(self.tests_label)

        controls.add_widget(self._label("Mode de démarrage"))
        mode_grid = GridLayout(cols=1, size_hint_y=None, spacing=dp(8), height=dp(194))
        self.mode_blocking_btn = self._styled_button("● Tests bloquants", "launch", lambda *_: self._set_run_mode(RUN_MODE_BLOCKING))
        self.mode_non_blocking_btn = self._styled_button("○ Tests non bloquants", "neutral", lambda *_: self._set_run_mode(RUN_MODE_NON_BLOCKING))
        self.mode_skip_tests_btn = self._styled_button("○ Sans tests", "neutral", lambda *_: self._set_run_mode(RUN_MODE_SKIP_TESTS))
        mode_grid.add_widget(self.mode_blocking_btn)
        mode_grid.add_widget(self.mode_non_blocking_btn)
        mode_grid.add_widget(self.mode_skip_tests_btn)
        controls.add_widget(mode_grid)
        self.run_mode_label = self._small_label("Mode actuel : tests bloquants")
        controls.add_widget(self.run_mode_label)

        controls.add_widget(self._label("Runtime live"))
        self.runtime_enabled_label = self._small_label(
            "Flux runtime : actif" if camo_log is not None else "Flux runtime : indisponible",
            color=C["success"] if camo_log is not None else C["warning"],
        )
        self.runtime_last_label = self._small_label("Dernier runtime : --", color=C["text_muted"])
        controls.add_widget(self.runtime_enabled_label)
        controls.add_widget(self.runtime_last_label)

        controls.add_widget(self._label("Chargement"))
        self.progress_bar = GlassProgressBar()
        controls.add_widget(self.progress_bar)

        self.progress_text = self._small_label("0 / 0 validé(s)")
        controls.add_widget(self.progress_text)

        controls.add_widget(self._label("Intensité machine"))
        intensity_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(10))
        self.intensity_slider = Slider(min=25, max=100, value=85)
        self.intensity_label = self._small_label("85 %", size_hint_x=0.22)
        self.intensity_slider.bind(value=self._on_intensity_change)
        intensity_row.add_widget(self.intensity_slider)
        intensity_row.add_widget(self.intensity_label)
        controls.add_widget(intensity_row)

        controls.add_widget(self._label("Ressources"))
        self.resource_text = self._small_label("CPU -- | RAM -- | Disque -- | Processus --")
        self.resource_hint = self._small_label("La cadence s’adapte légèrement quand la charge monte.")
        controls.add_widget(self.resource_text)
        controls.add_widget(self.resource_hint)

        controls.add_widget(self._label("Scores et métriques"))
        self.score_text = self._small_label("Score -- | ratio -- | silhouette -- | contour --")
        self.color_text = self._small_label("C -- | O -- | T -- | G --")
        self.extra_text = self._small_label("Olive conn. -- | centre -- | limites -- | miroir --")
        controls.add_widget(self.score_text)
        controls.add_widget(self.color_text)
        controls.add_widget(self.extra_text)

        controls.add_widget(self._label("Diagnostic live"))
        self.diag_enabled_label = self._small_label(
            "Analyse log.py : active" if camo_log is not None else "Analyse log.py : indisponible",
            color=C["warning"] if camo_log is None else C["success"],
        )
        self.diag_summary_label = self._small_label("Tentatives 0 | acceptés 0 | rejetés 0 | taux 0.00%")
        self.diag_top_rules_label = self._small_label("Top règles : --")
        self.diag_last_fail_label = self._small_label("Dernier rejet : --")
        controls.add_widget(self.diag_enabled_label)
        controls.add_widget(self.diag_summary_label)
        controls.add_widget(self.diag_top_rules_label)
        controls.add_widget(self.diag_last_fail_label)

        left_content.add_widget(controls)

        gallery_card = GlassCard(orientation="vertical", size_hint_y=None, height=dp(520))
        gallery_card.add_widget(self._label("Galerie"))
        self.gallery_scroll = ScrollView(do_scroll_x=False)
        self.gallery_grid = GridLayout(
            cols=GALLERY_COLUMNS,
            spacing=dp(10),
            padding=dp(2),
            size_hint_y=None,
        )
        self.gallery_grid.bind(minimum_height=self.gallery_grid.setter("height"))
        self.gallery_scroll.add_widget(self.gallery_grid)
        gallery_card.add_widget(self.gallery_scroll)
        left_content.add_widget(gallery_card)

        left_scroll.add_widget(left_content)
        left.add_widget(left_scroll)

        right = BoxLayout(orientation="vertical", spacing=dp(10))

        previews = BoxLayout(spacing=dp(10), size_hint_y=0.46)

        self.preview_img = Image()
        self.preview_silhouette = Image()

        previews.add_widget(self._carded_view("Camouflage courant / validé", self.preview_img))
        previews.add_widget(self._carded_view("Projection silhouette", self.preview_silhouette))
        right.add_widget(previews)

        bottom_split = BoxLayout(spacing=dp(10), size_hint_y=0.54)

        log_card = GlassCard(orientation="vertical")
        log_card.add_widget(self._label("Journal"))
        self.log_view = LogView()
        log_card.add_widget(self.log_view)

        diag_card = GlassCard(orientation="vertical")
        diag_card.add_widget(self._label("Logs diagnostic live"))
        self.diag_log_view = LogView()
        diag_card.add_widget(self.diag_log_view)

        bottom_split.add_widget(log_card)
        bottom_split.add_widget(diag_card)

        right.add_widget(bottom_split)

        body.add_widget(left)
        body.add_widget(right)
        root.add_widget(body)

        self._refresh_controls_state()
        self._refresh_run_mode_buttons()
        self.reload_gallery()
        self._reset_live_diagnostics()
        self._update_preflight_label(self.tests_summary, ok=None)
        Clock.schedule_interval(self._update_resource_monitor, 1.0)
        Clock.schedule_interval(self._refresh_gallery_periodic, 3.0)

        return root

    def on_start(self):
        try:
            Window.maximize()
        except Exception:
            pass

        self._subscribe_runtime_feed()
        self._bootstrap_runtime_feed()
        self._emit_runtime("INFO", "start", "Interface Kivy démarrée")

    # ---------------- runtime live ----------------

    def _format_runtime_event_line(self, event: Any) -> str:
        try:
            formatter = getattr(event, "format_line", None)
            if callable(formatter):
                return str(formatter())
        except Exception:
            pass

        ts = getattr(event, "ts", time.time())
        level = str(getattr(event, "level", "INFO")).upper()
        source = str(getattr(event, "source", "runtime"))
        message = str(getattr(event, "message", ""))
        payload = getattr(event, "payload", {}) or {}
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
        if payload:
            try:
                payload_txt = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                payload_txt = str(payload)
            return f"{stamp} | {level:<8} | {source} | {message} | {payload_txt}"
        return f"{stamp} | {level:<8} | {source} | {message}"

    def _emit_runtime(self, level: str, source: str, message: str, **payload: Any) -> None:
        if camo_log is None or not hasattr(camo_log, "log_event"):
            return
        try:
            camo_log.log_event(level, source, message, **payload)
        except Exception:
            pass

    def _runtime_event_to_diag(self, event: Any) -> bool:
        source = str(getattr(event, "source", "")).lower()
        return source in {
            "diagnostic",
            "summary",
            "tests",
            "generate_diagnostics",
            "async_generate_diagnostics",
            "export",
            "start_worker",
        }

    @mainthread
    def _append_runtime_line(self, line: str, to_diag: bool) -> None:
        if self.log_view is not None:
            self.log_view.append(line)
        if to_diag and self.diag_log_view is not None:
            self.diag_log_view.append(line)
        if self.runtime_last_label is not None:
            shortened = line if len(line) <= 140 else line[:137] + "..."
            self.runtime_last_label.text = f"Dernier runtime : {shortened}"

    def _on_runtime_event(self, event: Any) -> None:
        line = self._format_runtime_event_line(event)
        self._append_runtime_line(line, self._runtime_event_to_diag(event))

    def _subscribe_runtime_feed(self) -> None:
        if camo_log is None:
            return
        if self._runtime_subscription_active:
            return

        callback = self._on_runtime_event
        try:
            manager = getattr(camo_log, "LOG_MANAGER", None)
            if manager is not None and hasattr(manager, "subscribe"):
                manager.subscribe(callback)
                self._runtime_subscription_active = True
                self._runtime_subscriber_callback = callback
        except Exception as exc:
            self.log(f"Abonnement runtime impossible : {exc}")

    def _unsubscribe_runtime_feed(self) -> None:
        if camo_log is None or not self._runtime_subscription_active:
            return
        callback = self._runtime_subscriber_callback
        try:
            manager = getattr(camo_log, "LOG_MANAGER", None)
            if manager is not None and hasattr(manager, "unsubscribe") and callback is not None:
                manager.unsubscribe(callback)
        except Exception:
            pass
        self._runtime_subscription_active = False
        self._runtime_subscriber_callback = None

    def _bootstrap_runtime_feed(self) -> None:
        if camo_log is None:
            return
        try:
            lines = camo_log.get_recent_runtime_lines(25)
        except Exception:
            lines = []
        for line in lines[-25:]:
            self._append_runtime_line(str(line), True)

    # ---------------- style helpers ----------------

    def _styled_button(self, text: str, role: str, on_release_callback) -> SoftButton:
        btn = SoftButton(text=text, role=role)
        btn.bind(on_release=on_release_callback)
        return btn

    def _carded_view(self, title: str, image_widget: Image) -> Widget:
        box = GlassCard(orientation="vertical")
        box.add_widget(self._label(title))

        stage_container = SoftPane(orientation="vertical")
        overlay = BoxLayout(orientation="vertical", padding=dp(8))
        overlay.add_widget(image_widget)
        stage_container.add_widget(overlay)
        box.add_widget(stage_container)
        return box

    def _label(self, text: str, **kwargs) -> Label:
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(26))
        kwargs.setdefault("font_size", sp(15))
        kwargs.setdefault("color", C["text_main"])
        kwargs.setdefault("halign", "left")
        kwargs.setdefault("valign", "middle")
        lbl = Label(text=text, **kwargs)
        lbl.bind(size=lambda *a: setattr(lbl, "text_size", lbl.size))
        return lbl

    def _small_label(self, text: str, **kwargs) -> Label:
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(22))
        kwargs.setdefault("font_size", sp(13))
        kwargs.setdefault("color", C["text_soft"])
        kwargs.setdefault("halign", "left")
        kwargs.setdefault("valign", "middle")
        lbl = Label(text=text, **kwargs)
        lbl.bind(size=lambda *a: setattr(lbl, "text_size", lbl.size))
        return lbl

    def _run_mode_text(self, mode: Optional[str] = None) -> str:
        mode = self.run_mode if mode is None else mode
        if mode == RUN_MODE_NON_BLOCKING:
            return "tests non bloquants"
        if mode == RUN_MODE_SKIP_TESTS:
            return "sans tests"
        return "tests bloquants"

    @mainthread
    def _refresh_run_mode_buttons(self):
        items = [
            (self.mode_blocking_btn, RUN_MODE_BLOCKING, "Tests bloquants"),
            (self.mode_non_blocking_btn, RUN_MODE_NON_BLOCKING, "Tests non bloquants"),
            (self.mode_skip_tests_btn, RUN_MODE_SKIP_TESTS, "Sans tests"),
        ]
        for btn, mode, label in items:
            if btn is None:
                continue
            selected = self.run_mode == mode
            btn.text = f"{'●' if selected else '○'} {label}"
            btn.role = "launch" if selected else "neutral"
            btn.disabled = self.running or self.preflight_running
            btn._redraw()

        if self.run_mode_label is not None:
            self.run_mode_label.text = f"Mode actuel : {self._run_mode_text()}"

    def _set_run_mode(self, mode: str):
        if mode not in {RUN_MODE_BLOCKING, RUN_MODE_NON_BLOCKING, RUN_MODE_SKIP_TESTS}:
            return
        if self.running or self.preflight_running:
            self.log("Impossible de changer le mode pendant une exécution ou un préflight.")
            return
        self.run_mode = mode
        self._refresh_run_mode_buttons()
        self.log(f"Mode sélectionné : {self._run_mode_text(mode)}.")
        self._emit_runtime("INFO", "start_mode", "Mode de démarrage mis à jour", mode=mode)

    # ---------------- préflight tests ----------------

    @mainthread
    def _update_preflight_label(self, text: str, ok: Optional[bool] = None):
        if self.tests_label is None:
            return
        self.tests_label.text = text
        if ok is True:
            self.tests_label.color = C["success"]
        elif ok is False:
            self.tests_label.color = C["danger"]
        else:
            self.tests_label.color = C["text_soft"]

    async def _async_run_preflight_via_log(self) -> Tuple[bool, str]:
        if camo_log is None:
            return False, "log.py indisponible : impossible de lancer le préflight."

        try:
            summary = await camo_log.async_run_preflight_tests(
                module_names=DEFAULT_PREFLIGHT_MODULES,
                output_dir=Path("logs_generation"),
                timeout_s=DEFAULT_PREFLIGHT_TIMEOUT_S,
            )

            if hasattr(summary, "short_text"):
                text = str(summary.short_text())
                ok = bool(getattr(summary, "ok", False))
                return ok, text

            if isinstance(summary, dict):
                ok = bool(summary.get("ok", False))
                total = int(summary.get("total", 0))
                failures = int(summary.get("failures", 0))
                errors = int(summary.get("errors", 0))
                if ok:
                    return True, f"{total} tests OK"
                return False, f"{total} tests exécutés | {failures} échec(s) | {errors} erreur(s)"

            return False, "Préflight : format de réponse inattendu."
        except Exception as exc:
            return False, f"Impossible d'exécuter les tests via log.py : {type(exc).__name__}: {exc}"

    def _ensure_preflight_tests(self, pending_start: bool = True) -> bool:
        if self.tests_ran and self.tests_ok:
            return True

        if self.preflight_running:
            self.log("Préflight déjà en cours…")
            return False

        mode_text = "bloquant" if pending_start else "non bloquant"

        self.preflight_running = True
        self.preflight_pending_start = pending_start
        self.status("Préflight en cours…", ok=True)
        self.log(f"Lancement des tests de préflight via log.py ({mode_text})…")
        self.diag_log(f"Préflight lancé via log.py ({mode_text}).")
        self._emit_runtime("INFO", "start_preflight", "Préflight lancé depuis le front", mode=mode_text, modules=list(DEFAULT_PREFLIGHT_MODULES), timeout_s=DEFAULT_PREFLIGHT_TIMEOUT_S)
        self._update_preflight_label(f"Préflight en cours ({mode_text})…", ok=None)
        self._refresh_controls_state()

        fut = self.async_runner.submit(self._async_run_preflight_via_log())
        self.preflight_future = fut
        fut.add_done_callback(self._on_preflight_future_done)
        return False

    def _on_preflight_future_done(self, fut: Future):
        try:
            ok, summary = fut.result()
        except Exception as exc:
            ok, summary = False, f"Préflight interrompu : {type(exc).__name__}: {exc}"

        Clock.schedule_once(lambda dt: self._on_preflight_finished(ok, summary), 0)

    @mainthread
    def _on_preflight_finished(self, ok: bool, summary: str):
        pending_start = self.preflight_pending_start

        self.preflight_running = False
        self.preflight_future = None
        self.tests_ran = True
        self.tests_ok = ok
        self.tests_summary = summary
        self.preflight_pending_start = False

        self._update_preflight_label(summary, ok=ok)

        if ok:
            self.log(f"Préflight OK : {summary}")
            self.diag_log(f"Préflight OK : {summary}")
            if not self.running:
                self.status("Préflight terminé", ok=True)
            self._emit_runtime("INFO", "start_preflight", "Préflight terminé", ok=True, summary=summary)
        else:
            self.log(f"Préflight KO : {summary}")
            self.diag_log(f"Préflight KO : {summary}")
            if pending_start:
                self.status("Tests KO", ok=False)
            elif not self.running:
                self.status("Tests KO (tolérés)", ok=False)
            self._emit_runtime("ERROR", "start_preflight", "Préflight en échec", ok=False, summary=summary, blocking=pending_start)
            if not pending_start:
                self.log("Préflight non bloquant : la génération continue malgré l'échec des tests.")
                self.diag_log("Préflight non bloquant : poursuite malgré échec / timeout / incomplétude.")

        self._refresh_controls_state()

        if pending_start and ok:
            self._start_generation_after_preflight()

    # ---------------- diagnostic live ----------------

    def _reset_live_diagnostics(self):
        self.diag_total = 0
        self.diag_accepts = 0
        self.diag_rejects = 0
        self.diag_rule_counter = Counter()
        self.diag_last_rules = []
        self._refresh_live_diag_labels()
        if self.diag_log_view is not None:
            self.diag_log_view.label.text = ""

    async def _extract_failure_rules_async(
        self,
        candidate: camo.CandidateResult,
        target_index: int,
        local_attempt: int,
    ) -> List[str]:
        if camo_log is None:
            return []

        try:
            if hasattr(camo_log, "async_analyze_candidate"):
                diagnostic = await camo_log.async_analyze_candidate(
                    candidate, target_index=target_index, local_attempt=local_attempt
                )
            elif hasattr(camo_log, "analyze_candidate"):
                diagnostic = await asyncio.to_thread(
                    camo_log.analyze_candidate,
                    candidate,
                    target_index,
                    local_attempt,
                )
            else:
                return []

            failures = getattr(diagnostic, "failures", [])
            rules: List[str] = []
            for failure in failures:
                rule_name = getattr(failure, "rule", None)
                if rule_name:
                    rules.append(str(rule_name))
            return rules

        except Exception as exc:
            self.diag_log(f"Diagnostic indisponible pour seed={candidate.seed} : {exc}")
            return []

    @mainthread
    def _refresh_live_diag_labels(self):
        if self.diag_summary_label is not None:
            rate = (self.diag_accepts / self.diag_total) if self.diag_total else 0.0
            self.diag_summary_label.text = (
                f"Tentatives {self.diag_total} | acceptés {self.diag_accepts} | "
                f"rejetés {self.diag_rejects} | taux {rate:.2%}"
            )

        if self.diag_top_rules_label is not None:
            if self.diag_rule_counter:
                top = self.diag_rule_counter.most_common(3)
                text = " | ".join(f"{name}:{count}" for name, count in top)
                self.diag_top_rules_label.text = f"Top règles : {text}"
            else:
                self.diag_top_rules_label.text = "Top règles : --"

        if self.diag_last_fail_label is not None:
            if self.diag_last_rules:
                self.diag_last_fail_label.text = "Dernier rejet : " + " | ".join(self.diag_last_rules[:3])
            else:
                self.diag_last_fail_label.text = "Dernier rejet : --"

    @mainthread
    def diag_log(self, line: str):
        if self.diag_log_view is not None:
            self.diag_log_view.append(line)

    async def _register_live_diagnostic_async(
        self,
        candidate: camo.CandidateResult,
        target_index: int,
        local_attempt: int,
        accepted: bool,
    ):
        self.diag_total += 1

        if accepted:
            self.diag_accepts += 1
            self.diag_last_rules = []
            self.diag_log(
                f"[img={target_index:03d} essai={local_attempt:04d}] accepté | seed={candidate.seed}"
            )
            self._refresh_live_diag_labels()
            return

        self.diag_rejects += 1
        rules = await self._extract_failure_rules_async(candidate, target_index, local_attempt)
        self.diag_last_rules = rules[:]

        if rules:
            for rule in rules:
                self.diag_rule_counter[rule] += 1
            joined = " | ".join(rules[:6])
            self.diag_log(
                f"[img={target_index:03d} essai={local_attempt:04d}] rejet | seed={candidate.seed} | règles: {joined}"
            )
        else:
            self.diag_log(
                f"[img={target_index:03d} essai={local_attempt:04d}] rejet | seed={candidate.seed} | règles: non disponibles"
            )

        self._refresh_live_diag_labels()

    # ---------------- controls / gallery ----------------

    def _on_intensity_change(self, _slider, value):
        self.machine_intensity = float(value)
        if self.intensity_label is not None:
            self.intensity_label.text = f"{int(value)} %"

    @mainthread
    def _refresh_controls_state(self):
        if self.start_btn is not None:
            self.start_btn.disabled = self.running or self.stopping or self.preflight_running
        if self.stop_btn is not None:
            self.stop_btn.disabled = (not self.running) and (not self.stopping) and (not self.preflight_running)
        self._refresh_run_mode_buttons()

    @mainthread
    def reload_gallery(self):
        if self.gallery_grid is None:
            return

        self.gallery_grid.clear_widgets()

        if not self.current_output_dir.exists():
            return

        files = sorted(self.current_output_dir.glob("camouflage_*.png"))
        for p in files:
            self.gallery_grid.add_widget(GalleryThumb(self, p))

    def _refresh_gallery_periodic(self, _dt):
        self.reload_gallery()

    # ---------------- monitoring ----------------

    def _update_resource_monitor(self, _dt):
        if self.resource_text is None:
            return

        if psutil is None:
            self.resource_text.text = "Installer psutil pour voir CPU / RAM / disque."
            return

        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            anchor = self.current_output_dir.resolve().anchor or "C:\\"
            disk = psutil.disk_usage(anchor).percent

            proc_cpu = self.process.cpu_percent(interval=None) if self.process else 0.0
            proc_mem = self.process.memory_info().rss / (1024 ** 3) if self.process else 0.0

            self.resource_text.text = (
                f"CPU {cpu:.0f}% | RAM {ram:.0f}% | Disque {disk:.0f}% | "
                f"Processus {proc_cpu:.0f}% / {proc_mem:.2f} Go"
            )
        except Exception:
            self.resource_text.text = "Monitoring indisponible."

    async def _adaptive_pause(self):
        base = (100.0 - self.machine_intensity) / 1000.0

        if psutil is None:
            if base > 0:
                await asyncio.sleep(base)
            return

        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            extra = 0.0
            if cpu >= 95 or ram >= 92:
                extra = 0.09
            elif cpu >= 88 or ram >= 88:
                extra = 0.05
            elif cpu >= 80 or ram >= 84:
                extra = 0.02

            pause = base + extra
            if pause > 0:
                await asyncio.sleep(pause)
        except Exception:
            if base > 0:
                await asyncio.sleep(base)

    # ---------------- future binding ----------------

    def _bind_future(self, fut: Future):
        self.current_future = fut
        fut.add_done_callback(self._on_future_done)

    def _on_future_done(self, fut: Future):
        try:
            _ = fut.result()
        except Exception as exc:
            self._handle_future_exception(exc)
        finally:
            self._clear_current_future_if_same(fut)

    @mainthread
    def _handle_future_exception(self, exc: BaseException):
        if self.running or self.stopping:
            self.running = False
            self.stopping = False
            self.stop_flag = False
            prevent_sleep(False)
            self.status("Erreur", ok=False)
            self.log(f"Erreur non capturée : {exc}")
            self.diag_log(f"Erreur diagnostic : {exc}")
            self._refresh_controls_state()

    @mainthread
    def _clear_current_future_if_same(self, fut: Future):
        if self.current_future is fut:
            self.current_future = None
        self._refresh_controls_state()

    # ---------------- génération ----------------

    def start_generation(self, *_):
        if self.running or self.stopping or self.preflight_running:
            return

        if self.current_future is not None and not self.current_future.done():
            self.log("Une génération est déjà en cours.")
            return

        if self.count_input is None or self.progress_bar is None:
            self.log("Interface incomplète.")
            return

        if self.run_mode == RUN_MODE_SKIP_TESTS:
            self.tests_summary = "Tests ignorés (mode sans tests)."
            self._update_preflight_label(self.tests_summary, ok=None)
            self.log("Mode sans tests : démarrage direct de la génération.")
            self.diag_log("Préflight ignoré : démarrage sans tests.")
            self._emit_runtime("WARNING", "start_preflight", "Préflight ignoré", mode="sans_tests")
            self._start_generation_after_preflight()
            return

        if self.run_mode == RUN_MODE_NON_BLOCKING:
            if self.tests_ran and self.tests_ok:
                self.log("Préflight déjà validé : démarrage direct en mode non bloquant.")
                self._start_generation_after_preflight()
                return

            if camo_log is None:
                self.log("log.py indisponible : démarrage direct sans préflight non bloquant.")
                self._start_generation_after_preflight()
                return

            self._ensure_preflight_tests(pending_start=False)
            self._start_generation_after_preflight(allow_during_preflight=True)
            return

        if self.tests_ran and self.tests_ok:
            self._start_generation_after_preflight()
            return

        self._ensure_preflight_tests(pending_start=True)

    def _start_generation_after_preflight(self, allow_during_preflight: bool = False):
        if self.running or self.stopping or (self.preflight_running and not allow_during_preflight):
            return

        if self.current_future is not None and not self.current_future.done():
            self.log("Une génération est déjà en cours.")
            return

        if self.count_input is None or self.progress_bar is None:
            self.log("Interface incomplète.")
            return

        try:
            count = int(self.count_input.text.strip())
            if count <= 0:
                raise ValueError
        except Exception:
            self.log("Nombre de camouflages invalide.")
            return

        self.current_output_dir = DEFAULT_OUTPUT_DIR
        self.current_output_dir.mkdir(parents=True, exist_ok=True)

        self.best_records.clear()
        self.stop_flag = False
        self.stopping = False
        self.running = True
        self.accepted_count = 0
        self.total_attempts = 0

        self.progress_bar.max_value = count
        self.progress_bar.value = 0

        self._reset_live_diagnostics()
        prevent_sleep(True)

        self.status("Génération en cours…", ok=True)
        self.log(f"Démarrage : {count} camouflage(s) demandé(s).")
        self.log(f"Mode : {self._run_mode_text()}.")
        self.log(f"Dossier : {self.current_output_dir.resolve()}")
        self.diag_log("Diagnostic live initialisé.")
        self._emit_runtime("INFO", "start_worker", "Génération démarrée", target_count=count, output_dir=str(self.current_output_dir.resolve()), run_mode=self.run_mode)
        self._refresh_controls_state()

        fut = self.async_runner.submit(self._async_worker_generate(count))
        self._bind_future(fut)

    def stop_generation(self, *_):
        if self.running and not self.stopping:
            self.stop_flag = True
            self.stopping = True
            self.status("Arrêt demandé…", ok=False)
            self.log("Arrêt demandé. Fin de la tentative courante puis arrêt.")
            if self.preflight_running:
                self.log("Le préflight en arrière-plan continue jusqu'à sa fin, mais la génération s'arrête.")
            self.diag_log("Arrêt demandé par l'utilisateur.")
            self._emit_runtime("WARNING", "start_worker", "Arrêt demandé", accepted_count=self.accepted_count, total_attempts=self.total_attempts)
            self._refresh_controls_state()
            return

        if self.preflight_running:
            self.preflight_pending_start = False
            self.status("Préflight en cours…", ok=False)
            self.log("Arrêt demandé pendant le préflight. La génération ne démarrera pas.")
            self.diag_log("Démarrage annulé pendant le préflight.")
            self._refresh_controls_state()
            return

        if not self.running or self.stopping:
            return

    async def _async_should_stop(self) -> bool:
        return self.stop_flag

    async def _async_worker_generate(self, target_count: int):
        rows: List[dict] = []
        total_attempts = 0

        try:
            for target_index in range(1, target_count + 1):
                local_attempt = 0

                while True:
                    if await self._async_should_stop():
                        await self._async_finish_stopped(rows)
                        return

                    total_attempts += 1
                    local_attempt += 1
                    self.total_attempts = total_attempts

                    seed = camo.build_seed(target_index, local_attempt, base_seed=camo.DEFAULT_BASE_SEED)
                    self._emit_runtime("INFO", "start_worker", "Tentative en cours", target_index=target_index, local_attempt=local_attempt, global_attempt=total_attempts, seed=seed)
                    candidate = await camo.async_generate_candidate_from_seed(seed)

                    extra_scores, valid_v3 = await async_evaluate_candidate_v3(
                        candidate.image,
                        candidate.ratios,
                        candidate.metrics,
                    )
                    valid_main = await camo.async_validate_candidate_result(candidate)
                    full_valid = valid_main and valid_v3

                    idx_canvas = await asyncio.to_thread(rgb_image_to_index_canvas, candidate.image)
                    silhouette_img = await asyncio.to_thread(silhouette_projection_image, idx_canvas)

                    self.update_preview(candidate.image, silhouette_img)
                    self.update_attempt_status(
                        target_index=target_index,
                        attempt_idx=local_attempt,
                        target_total=target_count,
                        accepted_count=len(rows),
                        rs=candidate.ratios,
                        extra_scores=extra_scores,
                        metrics=candidate.metrics,
                    )

                    await self._register_live_diagnostic_async(
                        candidate=candidate,
                        target_index=target_index,
                        local_attempt=local_attempt,
                        accepted=full_valid,
                    )

                    if not full_valid:
                        self.log(
                            f"[img={target_index:03d} essai={local_attempt:04d}] rejeté | "
                            f"SF={extra_scores['score_final']:.3f} | "
                            f"C={candidate.ratios[camo.IDX_COYOTE]*100:.1f} "
                            f"O={candidate.ratios[camo.IDX_OLIVE]*100:.1f} "
                            f"T={candidate.ratios[camo.IDX_TERRE]*100:.1f} "
                            f"G={candidate.ratios[camo.IDX_GRIS]*100:.1f}"
                        )
                        self._emit_runtime("WARNING", "start_worker", "Candidat rejeté", target_index=target_index, local_attempt=local_attempt, global_attempt=total_attempts, seed=candidate.seed, score_final=round(float(extra_scores['score_final']), 5))
                        await self._adaptive_pause()
                        await asyncio.sleep(0)
                        continue

                    filename = self.current_output_dir / f"camouflage_{target_index:03d}.png"
                    await camo.async_save_candidate_image(candidate, filename)

                    record = CandidateRecord(
                        index=target_index,
                        seed=candidate.seed,
                        local_attempt=local_attempt,
                        global_attempt=total_attempts,
                        image_path=filename,
                        score_final=extra_scores["score_final"],
                        score_ratio=extra_scores["score_ratio"],
                        score_silhouette=extra_scores["score_silhouette"],
                        score_contour=extra_scores["score_contour"],
                        score_main=extra_scores["score_main"],
                        silhouette_color_diversity=extra_scores["silhouette_color_diversity"],
                        contour_break_score=extra_scores["contour_break_score"],
                        outline_band_diversity=extra_scores["outline_band_diversity"],
                        small_scale_structural_score=extra_scores["small_scale_structural_score"],
                        rs=candidate.ratios.copy(),
                        metrics=dict(candidate.metrics),
                    )
                    self.best_records.append(record)
                    self.best_records.sort(key=lambda r: r.score_final, reverse=True)

                    rows.append(
                        {
                            "index": target_index,
                            "seed": candidate.seed,
                            "attempts_for_this_image": local_attempt,
                            "global_attempt": total_attempts,
                            "coyote_brown_pct": round(float(candidate.ratios[camo.IDX_COYOTE] * 100), 2),
                            "vert_olive_pct": round(float(candidate.ratios[camo.IDX_OLIVE] * 100), 2),
                            "terre_de_france_pct": round(float(candidate.ratios[camo.IDX_TERRE] * 100), 2),
                            "vert_de_gris_pct": round(float(candidate.ratios[camo.IDX_GRIS] * 100), 2),
                            "score_final": round(extra_scores["score_final"], 5),
                            "score_ratio": round(extra_scores["score_ratio"], 5),
                            "score_silhouette": round(extra_scores["score_silhouette"], 5),
                            "score_contour": round(extra_scores["score_contour"], 5),
                            "score_main": round(extra_scores["score_main"], 5),
                            "silhouette_color_diversity": round(extra_scores["silhouette_color_diversity"], 5),
                            "contour_break_score": round(extra_scores["contour_break_score"], 5),
                            "outline_band_diversity": round(extra_scores["outline_band_diversity"], 5),
                            "small_scale_structural_score": round(extra_scores["small_scale_structural_score"], 5),
                            "largest_olive_component_ratio": round(candidate.metrics["largest_olive_component_ratio"], 5),
                            "largest_olive_component_ratio_small": round(candidate.metrics["largest_olive_component_ratio_small"], 5),
                            "olive_multizone_share": round(candidate.metrics["olive_multizone_share"], 5),
                            "center_empty_ratio": round(candidate.metrics["center_empty_ratio"], 5),
                            "center_empty_ratio_small": round(candidate.metrics["center_empty_ratio_small"], 5),
                            "boundary_density": round(candidate.metrics["boundary_density"], 5),
                            "boundary_density_small": round(candidate.metrics["boundary_density_small"], 5),
                            "boundary_density_tiny": round(candidate.metrics["boundary_density_tiny"], 5),
                            "mirror_similarity": round(candidate.metrics["mirror_similarity"], 5),
                            "oblique_share": round(candidate.metrics["oblique_share"], 5),
                            "vertical_share": round(candidate.metrics["vertical_share"], 5),
                            "angle_dominance_ratio": round(candidate.metrics["angle_dominance_ratio"], 5),
                            "olive_macro_share": round(candidate.metrics["vert_olive_macro_share"], 5),
                            "terre_transition_share": round(candidate.metrics["terre_de_france_transition_share"], 5),
                            "gris_micro_share": round(candidate.metrics["vert_de_gris_micro_share"], 5),
                            "gris_macro_share": round(candidate.metrics["vert_de_gris_macro_share"], 5),
                            "angles": " ".join(map(str, candidate.profile.allowed_angles)),
                        }
                    )

                    self.accepted_count = len(rows)
                    self.update_progress(len(rows), target_count)
                    self.log(
                        f"[img={target_index:03d}] accepté -> {filename.name} | "
                        f"SF={extra_scores['score_final']:.3f} | "
                        f"silhouette={extra_scores['score_silhouette']:.3f} | "
                        f"contour={extra_scores['contour_break_score']:.3f}"
                    )
                    self._emit_runtime("INFO", "start_worker", "Candidat accepté", target_index=target_index, local_attempt=local_attempt, global_attempt=total_attempts, seed=candidate.seed, image_path=str(filename.resolve()), score_final=round(float(extra_scores['score_final']), 5))
                    self.reload_gallery()

                    await self._adaptive_pause()
                    await asyncio.sleep(0)
                    break

            await self._async_finish_success(rows)

        except asyncio.CancelledError:
            await self._async_finish_stopped(rows)
            raise
        except Exception as e:
            await self._async_finish_error(str(e))

    # ---------------- fin de traitement ----------------

    async def _async_write_report(self, rows: List[dict]) -> Path:
        report_path = self.current_output_dir / REPORT_NAME
        await asyncio.to_thread(self._write_report_sync, rows, report_path)
        return report_path

    def _write_report_sync(self, rows: List[dict], report_path: Path) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            report_path.write_text("", encoding="utf-8")
            return

        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    async def _async_export_best_of(self, top_k: int) -> Path:
        best_dir = self.current_output_dir / BEST_DIR_NAME
        best_dir.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(self._clear_best_dir_sync, best_dir)

        best_subset = self.best_records[:top_k]
        rows = []

        for rank, rec in enumerate(best_subset, start=1):
            dst = best_dir / f"best_{rank:03d}_camouflage_{rec.index:03d}.png"
            await asyncio.to_thread(shutil.copy2, rec.image_path, dst)

            rows.append(
                {
                    "rank": rank,
                    "source_index": rec.index,
                    "seed": rec.seed,
                    "global_attempt": rec.global_attempt,
                    "attempts_for_this_image": rec.local_attempt,
                    "score_final": round(rec.score_final, 5),
                    "score_ratio": round(rec.score_ratio, 5),
                    "score_silhouette": round(rec.score_silhouette, 5),
                    "score_contour": round(rec.score_contour, 5),
                    "score_main": round(rec.score_main, 5),
                    "silhouette_color_diversity": round(rec.silhouette_color_diversity, 5),
                    "contour_break_score": round(rec.contour_break_score, 5),
                    "outline_band_diversity": round(rec.outline_band_diversity, 5),
                    "small_scale_structural_score": round(rec.small_scale_structural_score, 5),
                    "coyote_brown_pct": round(float(rec.rs[camo.IDX_COYOTE] * 100), 2),
                    "vert_olive_pct": round(float(rec.rs[camo.IDX_OLIVE] * 100), 2),
                    "terre_de_france_pct": round(float(rec.rs[camo.IDX_TERRE] * 100), 2),
                    "vert_de_gris_pct": round(float(rec.rs[camo.IDX_GRIS] * 100), 2),
                }
            )

        if rows:
            await asyncio.to_thread(self._write_best_of_csv_sync, best_dir, rows)

        return best_dir

    def _clear_best_dir_sync(self, best_dir: Path) -> None:
        for child in best_dir.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except Exception:
                    pass

    def _write_best_of_csv_sync(self, best_dir: Path, rows: List[dict]) -> None:
        with (best_dir / "best_of.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    async def _async_finish_success(self, rows: List[dict]):
        report_path = await self._async_write_report(rows)
        best_dir = await self._async_export_best_of(DEFAULT_TOP_K)
        prevent_sleep(False)

        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Terminé", ok=True)
        self.log(f"Rapport écrit : {report_path}")
        self.log(f"Best-of exporté : {best_dir}")
        self.log("Génération terminée avec succès.")
        self.diag_log("Diagnostic live terminé avec succès.")
        self._emit_runtime("INFO", "start_worker", "Génération terminée avec succès", accepted_count=len(rows), total_attempts=self.total_attempts)
        self._refresh_controls_state()
        self._refresh_run_mode_buttons()
        self.reload_gallery()

    async def _async_finish_stopped(self, rows: List[dict]):
        report_path = await self._async_write_report(rows)
        if rows:
            best_dir = await self._async_export_best_of(min(DEFAULT_TOP_K, len(rows)))
            self.log(f"Rapport partiel écrit : {report_path}")
            self.log(f"Best-of partiel exporté : {best_dir}")
        else:
            self.log(f"Rapport vide écrit : {report_path}")

        prevent_sleep(False)

        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Arrêté", ok=False)
        self.log("Génération arrêtée proprement.")
        self.diag_log("Diagnostic live arrêté proprement.")
        self._emit_runtime("WARNING", "start_worker", "Génération arrêtée", accepted_count=len(rows), total_attempts=self.total_attempts)
        self._refresh_controls_state()
        self._refresh_run_mode_buttons()
        self.reload_gallery()

    async def _async_finish_error(self, message: str):
        prevent_sleep(False)

        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Erreur", ok=False)
        self.log(f"Erreur : {message}")
        self.diag_log(f"Erreur diagnostic : {message}")
        self._emit_runtime("ERROR", "start_worker", "Erreur de génération", message=message, accepted_count=self.accepted_count, total_attempts=self.total_attempts)
        self._refresh_controls_state()

    # ---------------- UI updates ----------------

    @mainthread
    def status(self, text: str, ok: bool = True):
        if self.status_label is None:
            return
        self.status_label.text = text
        self.status_label.color = C["success"] if ok else C["danger"]

    @mainthread
    def log(self, line: str):
        if self.log_view is not None:
            self.log_view.append(line)

    @mainthread
    def update_progress(self, current: int, total: int):
        if self.progress_bar is None:
            return
        self.progress_bar.max_value = total
        self.progress_bar.value = current

    @mainthread
    def update_attempt_status(
        self,
        target_index: int,
        attempt_idx: int,
        target_total: int,
        accepted_count: int,
        rs: np.ndarray,
        extra_scores: Dict[str, float],
        metrics: Dict[str, float],
    ):
        if self.progress_text is not None:
            self.progress_text.text = f"{accepted_count} / {target_total} validé(s)"
        if self.attempt_text is not None:
            self.attempt_text.text = f"Image {target_index:03d} | essai {attempt_idx:04d}"

        if self.color_text is not None:
            self.color_text.text = (
                f"C {rs[camo.IDX_COYOTE]*100:.2f}% | "
                f"O {rs[camo.IDX_OLIVE]*100:.2f}% | "
                f"T {rs[camo.IDX_TERRE]*100:.2f}% | "
                f"G {rs[camo.IDX_GRIS]*100:.2f}%"
            )

        if self.score_text is not None:
            self.score_text.text = (
                f"Score {extra_scores['score_final']:.3f} | "
                f"ratio {extra_scores['score_ratio']:.3f} | "
                f"silhouette {extra_scores['score_silhouette']:.3f} | "
                f"contour {extra_scores['score_contour']:.3f}"
            )

        if self.extra_text is not None:
            self.extra_text.text = (
                f"Olive conn. {metrics['largest_olive_component_ratio']:.3f} | "
                f"centre {metrics['center_empty_ratio']:.3f} | "
                f"limites {metrics['boundary_density']:.3f} | "
                f"miroir {metrics['mirror_similarity']:.3f}"
            )

    @mainthread
    def update_preview(self, pil_img: PILImage.Image, silhouette_img: PILImage.Image):
        self.current_preview_img = pil_img
        self.current_silhouette_preview = silhouette_img
        if self.preview_img is not None:
            self.preview_img.texture = pil_to_coreimage(pil_img).texture
        if self.preview_silhouette is not None:
            self.preview_silhouette.texture = pil_to_coreimage(silhouette_img).texture

    def on_stop(self):
        self.stop_flag = True
        self.stopping = True
        self.preflight_pending_start = False
        prevent_sleep(False)
        self._emit_runtime("INFO", "start", "Arrêt de l'interface Kivy")
        self._unsubscribe_runtime_feed()

        fut = self.current_future
        if fut is not None and not fut.done():
            fut.add_done_callback(lambda _f: self.async_runner.stop())
        else:
            try:
                self.async_runner.stop()
            except Exception:
                pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    CamouflageApp().run()