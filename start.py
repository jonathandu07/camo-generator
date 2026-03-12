# -*- coding: utf-8 -*-
"""
start.py
Interface Kivy moderne, asynchrone, resizable, orientée production,
pour piloter main.py comme module.

Fonctions :
- interface épurée : lancer / arrêter / nombre de camouflages
- génération séquentielle stricte
- orchestration asynchrone
- barre de progression
- aperçu principal + projection silhouette
- galerie des camouflages déjà générés
- monitoring ressources (CPU / RAM / disque / process)
- prévention de la mise en veille pendant la génération
- adaptation légère de cadence si la machine est trop chargée

Pré-requis :
    pip install kivy pillow numpy psutil

Arborescence attendue :
    .
    ├── main.py
    └── start.py
"""

from __future__ import annotations

import asyncio
import csv
import ctypes
import io
import math
import os
import platform
import shutil
import subprocess
import threading
import time
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

# ---------------- Kivy config avant imports UI ----------------
from kivy.config import Config

Config.set("graphics", "fullscreen", "0")
Config.set("graphics", "resizable", "1")
Config.set("graphics", "minimum_width", "1200")
Config.set("graphics", "minimum_height", "760")
Config.set("graphics", "width", "1600")
Config.set("graphics", "height", "980")
Config.set("input", "mouse", "mouse,multitouch_on_demand")

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# ---------------- moteur principal async ----------------
import main as camo


# ============================================================
# CONSTANTES UI / EXPORT
# ============================================================

APP_TITLE = "Camouflage Armée Fédérale Europe"
REPORT_NAME = "rapport_camouflages_v3.csv"
BEST_DIR_NAME = "best_of"
DEFAULT_OUTPUT_DIR = Path("camouflages_federale_europe")
DEFAULT_TARGET_COUNT = 100
DEFAULT_TOP_K = 20

WEIGHT_RATIO = 0.28
WEIGHT_SILHOUETTE = 0.30
WEIGHT_CONTOUR = 0.24
WEIGHT_MAIN_METRICS = 0.18

MIN_SILHOUETTE_COLOR_DIVERSITY = 0.62
MIN_CONTOUR_BREAK_SCORE = 0.44
MIN_OUTLINE_BAND_DIVERSITY = 0.58
MIN_SMALL_SCALE_STRUCTURAL_SCORE = 0.42

THUMB_SIZE = (220, 140)
GALLERY_COLUMNS = 3

# Anti-veille Windows
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
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
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
    canvas = PILImage.new("RGB", size, (22, 24, 28))
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

    background = np.full((h, w, 3), 24, dtype=np.uint8)
    rgb = camo.RGB[index_canvas]
    background[sil] = rgb[sil]

    bound = silhouette_boundary(sil)
    background[bound] = np.array([230, 230, 230], dtype=np.uint8)
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
# WIDGETS
# ============================================================

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
            color=(0.93, 0.95, 0.98, 1),
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
        if len(lines) > 400:
            lines = lines[-400:]
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
        self.height = dp(170)
        self.padding = dp(8)
        self.bind(on_release=self._open_preview)

        self.container = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(6))
        self.add_widget(self.container)

        self.thumb = Image(size_hint_y=None, height=dp(128))
        self.caption = Label(
            text=image_path.name,
            size_hint_y=None,
            height=dp(24),
            font_size=sp(11),
            color=(0.90, 0.92, 0.95, 1),
            halign="center",
            valign="middle",
        )
        self.caption.bind(size=lambda *a: setattr(self.caption, "text_size", self.caption.size))

        self.container.add_widget(self.thumb)
        self.container.add_widget(self.caption)

        with self.canvas.before:
            from kivy.graphics import Color, RoundedRectangle
            Color(0.16, 0.18, 0.22, 0.82)
            self._bg = RoundedRectangle(radius=[dp(18)] * 4, pos=self.pos, size=self.size)
        self.bind(pos=lambda inst, val: setattr(self._bg, "pos", val))
        self.bind(size=lambda inst, val: setattr(self._bg, "size", val))

        self.load_thumbnail()

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

        self.stop_flag = False
        self.running = False
        self.stopping = False

        self.current_output_dir = DEFAULT_OUTPUT_DIR
        self.best_records: List[CandidateRecord] = []

        self.current_preview_img: Optional[PILImage.Image] = None
        self.current_silhouette_preview: Optional[PILImage.Image] = None

        self.accepted_count = 0
        self.total_attempts = 0
        self.last_resource_poll = 0.0
        self.process = psutil.Process() if psutil else None

    def build(self):
        Window.clearcolor = (0.05, 0.06, 0.08, 1)

        root = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        header = self._card(BoxLayout(orientation="horizontal", spacing=dp(10), padding=dp(16)), min_h=dp(74))
        self.title_label = Label(
            text=APP_TITLE,
            font_size=sp(24),
            bold=True,
            color=(0.97, 0.98, 1.0, 1),
            halign="left",
            valign="middle",
        )
        self.title_label.bind(size=lambda *a: setattr(self.title_label, "text_size", self.title_label.size))

        self.status_label = Label(
            text="Prêt",
            font_size=sp(15),
            color=(0.76, 0.92, 0.78, 1),
            size_hint_x=0.25,
            halign="right",
            valign="middle",
        )
        self.status_label.bind(size=lambda *a: setattr(self.status_label, "text_size", self.status_label.size))

        header.add_widget(self.title_label)
        header.add_widget(self.status_label)
        root.add_widget(header)

        body = BoxLayout(spacing=dp(10))

        # Colonne gauche
        left = BoxLayout(orientation="vertical", size_hint_x=0.33, spacing=dp(10))

        control_card = self._card(BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(14)))
        control_card.add_widget(self._label("Nombre de camouflages à générer"))

        self.count_input = TextInput(
            text=str(DEFAULT_TARGET_COUNT),
            multiline=False,
            input_filter="int",
            size_hint_y=None,
            height=dp(46),
            background_normal="",
            background_active="",
            background_color=(0.14, 0.16, 0.20, 0.9),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1),
            padding=[dp(12), dp(12), dp(12), dp(12)],
        )
        control_card.add_widget(self.count_input)

        btn_row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        self.start_btn = Button(
            text="Lancer",
            background_normal="",
            background_down="",
            background_color=(0.18, 0.56, 0.34, 1),
            color=(1, 1, 1, 1),
        )
        self.stop_btn = Button(
            text="Arrêter",
            background_normal="",
            background_down="",
            background_color=(0.62, 0.20, 0.20, 1),
            color=(1, 1, 1, 1),
        )
        self.start_btn.bind(on_release=self.start_generation)
        self.stop_btn.bind(on_release=self.stop_generation)
        btn_row.add_widget(self.start_btn)
        btn_row.add_widget(self.stop_btn)
        control_card.add_widget(btn_row)

        control_card.add_widget(self._label("Progression globale"))
        self.progress_global = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(18))
        control_card.add_widget(self.progress_global)

        self.progress_text = self._small_label("0 / 0 validé")
        self.attempt_text = self._small_label("Image 000 | essai 0000")
        control_card.add_widget(self.progress_text)
        control_card.add_widget(self.attempt_text)

        self.resource_title = self._label("Ressources système")
        self.resource_text = self._small_label("CPU -- | RAM -- | Disque -- | Processus --")
        self.resource_hint = self._small_label("Régulation automatique active si la charge monte.")
        control_card.add_widget(self.resource_title)
        control_card.add_widget(self.resource_text)
        control_card.add_widget(self.resource_hint)

        self.score_text = self._small_label("Score final -- | ratio -- | silhouette -- | contour --")
        self.color_text = self._small_label("C -- | O -- | T -- | G --")
        self.extra_text = self._small_label("Olive conn. -- | centre -- | limites -- | miroir --")
        control_card.add_widget(self.score_text)
        control_card.add_widget(self.color_text)
        control_card.add_widget(self.extra_text)

        left.add_widget(control_card)

        gallery_card = self._card(BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(12)))
        gallery_card.add_widget(self._label("Galerie des camouflages déjà générés"))

        self.gallery_scroll = ScrollView(do_scroll_x=False)
        self.gallery_grid = GridLayout(
            cols=GALLERY_COLUMNS,
            spacing=dp(10),
            padding=dp(4),
            size_hint_y=None,
        )
        self.gallery_grid.bind(minimum_height=self.gallery_grid.setter("height"))
        self.gallery_scroll.add_widget(self.gallery_grid)
        gallery_card.add_widget(self.gallery_scroll)
        left.add_widget(gallery_card)

        # Colonne droite
        right = BoxLayout(orientation="vertical", spacing=dp(10))

        preview_row = BoxLayout(spacing=dp(10), size_hint_y=0.58)
        self.preview_img = Image()
        self.preview_silhouette = Image()

        preview_row.add_widget(self._carded_view("Camouflage courant / validé", self.preview_img))
        preview_row.add_widget(self._carded_view("Projection silhouette", self.preview_silhouette))
        right.add_widget(preview_row)

        log_card = self._card(BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(12)))
        log_card.add_widget(self._label("Journal"))
        self.log_view = LogView()
        log_card.add_widget(self.log_view)
        right.add_widget(log_card)

        body.add_widget(left)
        body.add_widget(right)

        root.add_widget(body)

        self._refresh_controls_state()
        self.reload_gallery()
        Clock.schedule_interval(self._update_resource_monitor, 1.0)
        Clock.schedule_interval(self._refresh_gallery_periodic, 3.0)

        return root

    def on_start(self):
        try:
            Window.maximize()
        except Exception:
            pass

    # ---------------- style helpers ----------------

    def _card(self, widget: Widget, min_h: Optional[float] = None) -> Widget:
        if min_h is not None:
            widget.size_hint_y = None
            widget.height = min_h

        with widget.canvas.before:
            from kivy.graphics import Color, RoundedRectangle
            Color(0.13, 0.15, 0.19, 0.86)
            widget._bg = RoundedRectangle(radius=[dp(22)] * 4, pos=widget.pos, size=widget.size)
        widget.bind(pos=lambda inst, val: setattr(widget._bg, "pos", val))
        widget.bind(size=lambda inst, val: setattr(widget._bg, "size", val))
        return widget

    def _carded_view(self, title: str, view: Widget) -> Widget:
        box = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(12))
        box.add_widget(self._label(title))
        box.add_widget(view)
        return self._card(box)

    def _label(self, text: str, **kwargs) -> Label:
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(24))
        kwargs.setdefault("font_size", sp(15))
        kwargs.setdefault("color", (0.95, 0.97, 1.0, 1))
        kwargs.setdefault("halign", "left")
        kwargs.setdefault("valign", "middle")
        lbl = Label(text=text, **kwargs)
        lbl.bind(size=lambda *a: setattr(lbl, "text_size", lbl.size))
        return lbl

    def _small_label(self, text: str, **kwargs) -> Label:
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("height", dp(22))
        kwargs.setdefault("font_size", sp(13))
        kwargs.setdefault("color", (0.83, 0.88, 0.94, 1))
        kwargs.setdefault("halign", "left")
        kwargs.setdefault("valign", "middle")
        lbl = Label(text=text, **kwargs)
        lbl.bind(size=lambda *a: setattr(lbl, "text_size", lbl.size))
        return lbl

    # ---------------- controls / gallery ----------------

    @mainthread
    def _refresh_controls_state(self):
        self.start_btn.disabled = self.running or self.stopping
        self.stop_btn.disabled = (not self.running) and (not self.stopping)

    @mainthread
    def reload_gallery(self):
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
        if psutil is None:
            self.resource_text.text = "Installer psutil pour voir CPU / RAM / disque."
            return

        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage(str(self.current_output_dir.resolve().anchor or "C:\\")).percent

            proc_cpu = self.process.cpu_percent(interval=None) if self.process else 0.0
            proc_mem = self.process.memory_info().rss / (1024 ** 3) if self.process else 0.0

            self.resource_text.text = (
                f"CPU {cpu:.0f}% | RAM {ram:.0f}% | Disque {disk:.0f}% | "
                f"Processus {proc_cpu:.0f}% / {proc_mem:.2f} Go"
            )
        except Exception:
            self.resource_text.text = "Monitoring indisponible."

    async def _adaptive_pause(self):
        if psutil is None:
            return
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            if cpu >= 95 or ram >= 92:
                await asyncio.sleep(0.08)
            elif cpu >= 88 or ram >= 88:
                await asyncio.sleep(0.04)
            elif cpu >= 80 or ram >= 84:
                await asyncio.sleep(0.015)
        except Exception:
            return

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
            self._refresh_controls_state()

    @mainthread
    def _clear_current_future_if_same(self, fut: Future):
        if self.current_future is fut:
            self.current_future = None
        self._refresh_controls_state()

    # ---------------- génération ----------------

    def start_generation(self, *_):
        if self.running or self.stopping:
            return

        if self.current_future is not None and not self.current_future.done():
            self.log("Une génération est déjà en cours.")
            return

        try:
            count = int(self.count_input.text.strip())
            if count <= 0:
                raise ValueError
        except Exception:
            self.log("Nombre d'images invalide.")
            return

        self.current_output_dir = DEFAULT_OUTPUT_DIR
        self.current_output_dir.mkdir(parents=True, exist_ok=True)

        self.best_records.clear()
        self.stop_flag = False
        self.stopping = False
        self.running = True
        self.accepted_count = 0
        self.total_attempts = 0
        self.progress_global.max = count
        self.progress_global.value = 0

        prevent_sleep(True)

        self.status("Génération en cours…", ok=True)
        self.log(f"Démarrage : {count} camouflages à produire.")
        self.log(f"Dossier : {self.current_output_dir.resolve()}")
        self._refresh_controls_state()

        fut = self.async_runner.submit(self._async_worker_generate(count))
        self._bind_future(fut)

    def stop_generation(self, *_):
        if not self.running or self.stopping:
            return

        self.stop_flag = True
        self.stopping = True
        self.status("Arrêt demandé…", ok=False)
        self.log("Arrêt demandé. Fin de la tentative courante puis arrêt.")
        self._refresh_controls_state()

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

                    if not full_valid:
                        self.log(
                            f"[img={target_index:03d} essai={local_attempt:04d}] rejeté | "
                            f"SF={extra_scores['score_final']:.3f} | "
                            f"C={candidate.ratios[camo.IDX_COYOTE]*100:.1f} "
                            f"O={candidate.ratios[camo.IDX_OLIVE]*100:.1f} "
                            f"T={candidate.ratios[camo.IDX_TERRE]*100:.1f} "
                            f"G={candidate.ratios[camo.IDX_GRIS]*100:.1f}"
                        )
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

                    rows.append({
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
                    })

                    self.accepted_count = len(rows)
                    self.update_progress(target_index, target_count)
                    self.log(
                        f"[img={target_index:03d}] accepté -> {filename.name} | "
                        f"SF={extra_scores['score_final']:.3f} | "
                        f"silhouette={extra_scores['score_silhouette']:.3f} | "
                        f"contour={extra_scores['contour_break_score']:.3f}"
                    )
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

            rows.append({
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
            })

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
        self._refresh_controls_state()
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
        self._refresh_controls_state()
        self.reload_gallery()

    async def _async_finish_error(self, message: str):
        prevent_sleep(False)

        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Erreur", ok=False)
        self.log(f"Erreur : {message}")
        self._refresh_controls_state()

    # ---------------- UI updates ----------------

    @mainthread
    def status(self, text: str, ok: bool = True):
        self.status_label.text = text
        self.status_label.color = (0.78, 0.92, 0.78, 1) if ok else (0.95, 0.68, 0.68, 1)

    @mainthread
    def log(self, line: str):
        self.log_view.append(line)

    @mainthread
    def update_progress(self, current: int, total: int):
        self.progress_global.max = total
        self.progress_global.value = current

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
        self.progress_text.text = f"{accepted_count} / {target_total} validé(s)"
        self.attempt_text.text = f"Image {target_index:03d} | essai {attempt_idx:04d}"

        self.color_text.text = (
            f"C {rs[camo.IDX_COYOTE]*100:.2f}% | "
            f"O {rs[camo.IDX_OLIVE]*100:.2f}% | "
            f"T {rs[camo.IDX_TERRE]*100:.2f}% | "
            f"G {rs[camo.IDX_GRIS]*100:.2f}%"
        )

        self.score_text.text = (
            f"Score {extra_scores['score_final']:.3f} | "
            f"ratio {extra_scores['score_ratio']:.3f} | "
            f"silhouette {extra_scores['score_silhouette']:.3f} | "
            f"contour {extra_scores['score_contour']:.3f}"
        )

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
        self.preview_img.texture = pil_to_coreimage(pil_img).texture
        self.preview_silhouette.texture = pil_to_coreimage(silhouette_img).texture

    def on_stop(self):
        self.stop_flag = True
        self.stopping = True
        prevent_sleep(False)

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