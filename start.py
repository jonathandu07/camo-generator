# -*- coding: utf-8 -*-
"""
start.py
Front-end Kivy nettoyé et réaligné avec main_v2_camouflage.py / main.py.

Objectifs :
- supprimer les doublons de scoring déjà présents dans le backend ;
- retirer les métriques obsolètes côté front ;
- conserver la génération séquentielle stricte avec aperçu live ;
- afficher les nouvelles métriques utiles au camouflage militaire ;
- garder une interface moderne et lisible.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

try:
    import psutil
except Exception:
    psutil = None

os.environ.setdefault("KIVY_NO_ARGS", "1")
sys.modules.setdefault("start", sys.modules[__name__])

try:
    import log as camo_log
except Exception:
    camo_log = None

try:
    import main_v2_camouflage as camo
except Exception:
    import main as camo

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
from kivy.properties import NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget


# ============================================================
# PALETTE UI
# ============================================================

PALETTE_HEX = {
    "BL": "#f4fefe",
    "GF": "#d9d9d9",
    "BA": "#81A1B8",
    "BF": "#051440",
    "BM": "#03224C",
    "BFW": "#091226",
    "GS": "#0A0B0A",
    "NA": "#303030",
    "RS": "#75161E",
    "RP": "#8B0000",
    "NG": "#3E5349",
    "VO": "#424530",
    "JA": "#FFCB60",
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
    "bg_panel": hex_rgba("BM", 0.84),
    "bg_panel_inner": hex_rgba("BF", 0.94),
    "bg_input": hex_rgba("NA", 0.90),
    "stroke": hex_rgba("BA", 0.36),
    "stroke_soft": hex_rgba("BA", 0.18),
    "text_main": hex_rgba("BL", 1.0),
    "text_soft": hex_rgba("GF", 1.0),
    "text_muted": hex_rgba("BA", 1.0),
    "success": hex_rgba("NG", 1.0),
    "warning": hex_rgba("JA", 1.0),
    "danger": hex_rgba("RS", 1.0),
    "shadow": hex_rgba("GS", 0.35),
    "glass": hex_rgba("BL", 0.05),
    "btn_launch": hex_rgba("NG", 1.0),
    "btn_launch_down": hex_rgba("VO", 1.0),
    "btn_stop": hex_rgba("RS", 1.0),
    "btn_stop_down": hex_rgba("RP", 1.0),
    "btn_neutral": hex_rgba("BA", 0.90),
    "btn_disabled": hex_rgba("NA", 0.55),
    "progress_bg": hex_rgba("NA", 0.92),
    "progress_fill": hex_rgba("BA", 0.95),
}


# ============================================================
# CONSTANTES
# ============================================================

APP_TITLE = "Camouflage Armée Fédérale Europe"
DEFAULT_OUTPUT_DIR = Path("camouflages_federale_europe")
DEFAULT_TARGET_COUNT = 100
DEFAULT_TOP_K = 20
DEFAULT_PREFLIGHT_MODULES = ("test_main", "test_start")

RUN_MODE_BLOCKING = "blocking"
RUN_MODE_NON_BLOCKING = "non_blocking"
RUN_MODE_SKIP_TESTS = "skip_tests"

THUMB_SIZE = (240, 150)
GALLERY_COLUMNS = 3
REPORT_NAME = "rapport_camouflages_front.csv"
BEST_DIR_NAME = "best_of"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


# ============================================================
# DONNÉES / OUTILS
# ============================================================

@dataclass
class CandidateRecord:
    index: int
    seed: int
    local_attempt: int
    global_attempt: int
    image_path: Path
    metrics: Dict[str, float]
    ratios: np.ndarray


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
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


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
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    for rgb, idx in PALETTE_TO_INDEX.items():
        mask = np.all(arr == np.array(rgb, dtype=np.uint8), axis=-1)
        out[mask] = idx
    return out


# ============================================================
# SILHOUETTE (garde l'aperçu, pas le scoring)
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
    draw.rounded_rectangle([torso_x1, torso_y1, torso_x1 + torso_w, torso_y1 + torso_h], radius=int(width * 0.03), fill=255)

    shoulder_w = int(width * 0.58)
    shoulder_h = int(height * 0.10)
    shoulder_x1 = (width - shoulder_w) // 2
    shoulder_y1 = int(height * 0.14)
    draw.rounded_rectangle([shoulder_x1, shoulder_y1, shoulder_x1 + shoulder_w, shoulder_y1 + shoulder_h], radius=int(width * 0.025), fill=255)

    arm_w = int(width * 0.11)
    arm_h = int(height * 0.32)
    left_arm_x1 = int(width * 0.15)
    right_arm_x1 = width - left_arm_x1 - arm_w
    arm_y1 = int(height * 0.20)
    draw.rounded_rectangle([left_arm_x1, arm_y1, left_arm_x1 + arm_w, arm_y1 + arm_h], radius=int(width * 0.02), fill=255)
    draw.rounded_rectangle([right_arm_x1, arm_y1, right_arm_x1 + arm_w, arm_y1 + arm_h], radius=int(width * 0.02), fill=255)

    pelvis_w = int(width * 0.30)
    pelvis_h = int(height * 0.10)
    pelvis_x1 = (width - pelvis_w) // 2
    pelvis_y1 = int(height * 0.51)
    draw.rounded_rectangle([pelvis_x1, pelvis_y1, pelvis_x1 + pelvis_w, pelvis_y1 + pelvis_h], radius=int(width * 0.02), fill=255)

    leg_w = int(width * 0.12)
    leg_h = int(height * 0.32)
    leg_gap = int(width * 0.04)
    left_leg_x1 = (width // 2) - leg_gap // 2 - leg_w
    right_leg_x1 = (width // 2) + leg_gap // 2
    leg_y1 = int(height * 0.58)
    draw.rounded_rectangle([left_leg_x1, leg_y1, left_leg_x1 + leg_w, leg_y1 + leg_h], radius=int(width * 0.018), fill=255)
    draw.rounded_rectangle([right_leg_x1, leg_y1, right_leg_x1 + leg_w, leg_y1 + leg_h], radius=int(width * 0.018), fill=255)

    return np.array(img, dtype=np.uint8) > 0


def silhouette_boundary(mask: np.ndarray) -> np.ndarray:
    b = np.zeros_like(mask, dtype=bool)
    b[1:, :] |= mask[1:, :] != mask[:-1, :]
    b[:-1, :] |= mask[:-1, :] != mask[1:, :]
    b[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return b & mask


def silhouette_projection_image(index_canvas: np.ndarray) -> PILImage.Image:
    h, w = index_canvas.shape
    sil = build_silhouette_mask(w, h)
    background = np.full((h, w, 3), tuple(int(v * 255) for v in C["bg_root"][:3]), dtype=np.uint8)
    rgb = camo.RGB[index_canvas]
    background[sil] = rgb[sil]
    background[silhouette_boundary(sil)] = np.array([255, 255, 255], dtype=np.uint8)
    return PILImage.fromarray(background, "RGB")


# ============================================================
# EXTRACTION DES SCORES BACKEND
# ============================================================

def extract_backend_scores(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "score_final": float(metrics.get("visual_score_final", 0.0)),
        "score_ratio": float(metrics.get("visual_score_ratio", 0.0)),
        "score_silhouette": float(metrics.get("visual_score_silhouette", 0.0)),
        "score_contour": float(metrics.get("visual_score_contour", 0.0)),
        "score_main": float(metrics.get("visual_score_main", 0.0)),
        "silhouette_color_diversity": float(metrics.get("visual_silhouette_color_diversity", 0.0)),
        "contour_break_score": float(metrics.get("visual_contour_break_score", 0.0)),
        "outline_band_diversity": float(metrics.get("visual_outline_band_diversity", 0.0)),
        "small_scale_structural_score": float(metrics.get("visual_small_scale_structural_score", 0.0)),
        "military_score": float(metrics.get("visual_military_score", 0.0)),
    }


# ============================================================
# WIDGETS UI
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
            Color(*C["shadow"])
            RoundedRectangle(pos=(self.x, self.y - dp(1)), size=self.size, radius=[radius] * 4)
            Color(*C["progress_bg"])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[radius] * 4)
            Color(*C["progress_fill"])
            RoundedRectangle(pos=(self.x + pad, self.y + pad), size=(fill_w, self.height - pad * 2), radius=[radius] * 4)
            Color(*C["stroke_soft"])
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, radius), width=1.0)


class GlassCard(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding = dp(16)
        self.spacing = dp(10)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(22)
        with self.canvas.before:
            Color(*C["shadow"])
            RoundedRectangle(pos=(self.x, self.y - dp(2)), size=self.size, radius=[r] * 4)
            Color(*C["bg_panel"])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[r] * 4)
            Color(*C["glass"])
            RoundedRectangle(pos=(self.x + dp(1), self.y + self.height * 0.55), size=(self.width - dp(2), self.height * 0.22), radius=[r] * 4)
            Color(*C["stroke"])
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, r), width=1.0)


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
            RoundedRectangle(pos=self.pos, size=self.size, radius=[r] * 4)
            Color(*C["stroke_soft"])
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, r), width=1.0)


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
            Color(*C["bg_input"])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[r] * 4)
            Color(*C["stroke_soft"])
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, r), width=1.0)


class SoftButton(Button):
    def __init__(self, role: str = "neutral", **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = C["text_main"]
        self.bold = True
        self.size_hint_y = None
        self.height = dp(54)
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
            Color(*C["shadow"])
            RoundedRectangle(pos=(self.x, self.y - dp(1)), size=self.size, radius=[r] * 4)
            Color(*bg)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[r] * 4)
            Color(*C["stroke_soft"])
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, r), width=1.0)


class LogView(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_scroll_x = False
        self.bar_width = dp(8)
        self.label = Label(text="", size_hint_y=None, text_size=(0, None), halign="left", valign="top", font_size=sp(13), color=C["text_soft"])
        self.label.bind(texture_size=self._update_label_height, width=self._update_text_width)
        self.add_widget(self.label)

    def _update_label_height(self, *_):
        self.label.height = self.label.texture_size[1] + dp(16)

    def _update_text_width(self, *_):
        self.label.text_size = (self.width - dp(24), None)

    def append(self, line: str):
        lines = self.label.text.splitlines() if self.label.text else []
        lines.append(line)
        if len(lines) > 700:
            lines = lines[-700:]
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
        self.caption = Label(text=image_path.name, size_hint_y=None, height=dp(22), font_size=sp(11), color=C["text_soft"], halign="center", valign="middle")
        self.caption.bind(size=lambda *a: setattr(self.caption, "text_size", self.caption.size))
        self.container.add_widget(self.stage)
        self.container.add_widget(self.caption)
        self.bind(on_release=self._open_preview)
        self.bind(pos=self._redraw, size=self._redraw, state=self._redraw)
        self.load_thumbnail()

    def _redraw(self, *_):
        self.canvas.before.clear()
        r = dp(20)
        with self.canvas.before:
            Color(*C["shadow"])
            RoundedRectangle(pos=(self.x, self.y - dp(1)), size=self.size, radius=[r] * 4)
            Color(*C["bg_panel_inner"])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[r] * 4)
            Color(*C["stroke_soft"])
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, r), width=1.0)

    def load_thumbnail(self):
        try:
            img = PILImage.open(self.image_path).convert("RGB")
            self.thumb.texture = pil_to_coreimage(make_thumbnail(img, THUMB_SIZE)).texture
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
        self.preflight_running = False
        self.preflight_pending_start = False
        self.current_output_dir = DEFAULT_OUTPUT_DIR
        self.best_records: List[CandidateRecord] = []
        self.accepted_count = 0
        self.total_attempts = 0
        self.machine_intensity = 100.0
        self.process = psutil.Process() if psutil else None
        self.tests_ran = False
        self.tests_ok = False
        self.tests_summary = "Tests non lancés."
        self.run_mode = RUN_MODE_BLOCKING
        self.diag_total = 0
        self.diag_accepts = 0
        self.diag_rejects = 0
        self.diag_rule_counter: Counter[str] = Counter()
        self.diag_last_rules: List[str] = []
        self._runtime_subscription_active = False
        self._runtime_subscriber_callback = None

        self.status_label: Optional[Label] = None
        self.attempt_text: Optional[Label] = None
        self.count_input: Optional[SoftTextInput] = None
        self.start_btn: Optional[SoftButton] = None
        self.stop_btn: Optional[SoftButton] = None
        self.open_btn: Optional[SoftButton] = None
        self.progress_bar: Optional[GlassProgressBar] = None
        self.progress_text: Optional[Label] = None
        self.intensity_slider: Optional[Slider] = None
        self.intensity_label: Optional[Label] = None
        self.resource_text: Optional[Label] = None
        self.tests_label: Optional[Label] = None
        self.run_mode_label: Optional[Label] = None
        self.score_text: Optional[Label] = None
        self.color_text: Optional[Label] = None
        self.extra_text: Optional[Label] = None
        self.struct_text: Optional[Label] = None
        self.diag_summary_label: Optional[Label] = None
        self.diag_top_rules_label: Optional[Label] = None
        self.diag_last_fail_label: Optional[Label] = None
        self.runtime_last_label: Optional[Label] = None
        self.preview_img: Optional[Image] = None
        self.preview_silhouette: Optional[Image] = None
        self.live_preview_img: Optional[Image] = None
        self.live_stage_label: Optional[Label] = None
        self.live_counts_label: Optional[Label] = None
        self.live_meta_label: Optional[Label] = None
        self.gallery_grid: Optional[GridLayout] = None
        self.log_view: Optional[LogView] = None
        self.diag_log_view: Optional[LogView] = None
        self.mode_blocking_btn: Optional[SoftButton] = None
        self.mode_non_blocking_btn: Optional[SoftButton] = None
        self.mode_skip_tests_btn: Optional[SoftButton] = None

    def build(self):
        Window.clearcolor = C["bg_root"]
        root = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        header = GlassCard(orientation="horizontal", size_hint_y=None, height=dp(120))
        title_box = BoxLayout(orientation="vertical", spacing=dp(4))
        self.attempt_text = self._small_label("Image 000 | essai 0000 | total 000000 | seed --")
        title = self._label(APP_TITLE, font_size=sp(19), height=dp(28))
        title.bold = True
        title_box.add_widget(self.attempt_text)
        title_box.add_widget(title)
        self.status_label = self._label("Prêt", size_hint_x=0.20, halign="right")
        header.add_widget(title_box)
        header.add_widget(self.status_label)
        root.add_widget(header)

        body = BoxLayout(spacing=dp(10))
        left = BoxLayout(orientation="vertical", spacing=dp(10), size_hint_x=0.36)
        right = BoxLayout(orientation="vertical", spacing=dp(10))

        # Panneau contrôle
        control_scroll = ScrollView(do_scroll_x=False, bar_width=dp(8))
        control_content = BoxLayout(orientation="vertical", spacing=dp(10), size_hint_y=None)
        control_content.bind(minimum_height=control_content.setter("height"))

        controls = GlassCard(orientation="vertical", size_hint_y=None)
        controls.bind(minimum_height=controls.setter("height"))
        controls.add_widget(self._label("Paramètres"))
        self.count_input = SoftTextInput(text=str(DEFAULT_TARGET_COUNT), multiline=False, input_filter="int", size_hint_y=None, height=dp(50))
        controls.add_widget(self.count_input)

        row = BoxLayout(size_hint_y=None, height=dp(58), spacing=dp(10))
        self.start_btn = self._button("Commencer", "launch", self.start_generation)
        self.stop_btn = self._button("Arrêter", "stop", self.stop_generation)
        self.open_btn = self._button("Ouvrir le dossier", "neutral", lambda *_: open_folder(self.current_output_dir))
        row.add_widget(self.start_btn)
        row.add_widget(self.stop_btn)
        controls.add_widget(row)
        controls.add_widget(self.open_btn)

        controls.add_widget(self._label("Préflight"))
        self.tests_label = self._small_label(self.tests_summary)
        controls.add_widget(self.tests_label)

        controls.add_widget(self._label("Mode de démarrage"))
        mode_grid = GridLayout(cols=1, size_hint_y=None, spacing=dp(8), height=dp(194))
        self.mode_blocking_btn = self._button("● Tests bloquants", "launch", lambda *_: self._set_run_mode(RUN_MODE_BLOCKING))
        self.mode_non_blocking_btn = self._button("○ Tests non bloquants", "neutral", lambda *_: self._set_run_mode(RUN_MODE_NON_BLOCKING))
        self.mode_skip_tests_btn = self._button("○ Sans tests", "neutral", lambda *_: self._set_run_mode(RUN_MODE_SKIP_TESTS))
        mode_grid.add_widget(self.mode_blocking_btn)
        mode_grid.add_widget(self.mode_non_blocking_btn)
        mode_grid.add_widget(self.mode_skip_tests_btn)
        controls.add_widget(mode_grid)
        self.run_mode_label = self._small_label("Mode actuel : tests bloquants")
        controls.add_widget(self.run_mode_label)

        controls.add_widget(self._label("Chargement"))
        self.progress_bar = GlassProgressBar()
        self.progress_text = self._small_label("0 / 0 validé(s) | tentatives 0 | rejetés 0")
        controls.add_widget(self.progress_bar)
        controls.add_widget(self.progress_text)

        controls.add_widget(self._label("Intensité machine"))
        intensity_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8))
        self.intensity_slider = Slider(min=25, max=100, value=100)
        self.intensity_label = self._small_label("100 %", size_hint_x=0.22)
        self.intensity_slider.bind(value=self._on_intensity_change)
        intensity_row.add_widget(self.intensity_slider)
        intensity_row.add_widget(self.intensity_label)
        controls.add_widget(intensity_row)

        controls.add_widget(self._label("Monitoring"))
        self.resource_text = self._small_label("CPU -- | RAM -- | Disque -- | Processus --")
        controls.add_widget(self.resource_text)

        controls.add_widget(self._label("Scores backend"))
        self.score_text = self._small_label("Score -- | ratio -- | silhouette -- | contour --")
        self.color_text = self._small_label("C -- | O -- | T -- | G --")
        self.extra_text = self._small_label("olive conn. -- | centre -- | limites -- | miroir --")
        self.struct_text = self._small_label("terre macro -- | gris macro -- | continuité brune -- | militaire --")
        controls.add_widget(self.score_text)
        controls.add_widget(self.color_text)
        controls.add_widget(self.extra_text)
        controls.add_widget(self.struct_text)

        controls.add_widget(self._label("Runtime / diagnostic"))
        self.runtime_last_label = self._small_label("Dernier runtime : --")
        self.diag_summary_label = self._small_label("Tentatives 0 | acceptés 0 | rejetés 0 | taux 0.00%")
        self.diag_top_rules_label = self._small_label("Top règles : --")
        self.diag_last_fail_label = self._small_label("Dernier rejet : --")
        controls.add_widget(self.runtime_last_label)
        controls.add_widget(self.diag_summary_label)
        controls.add_widget(self.diag_top_rules_label)
        controls.add_widget(self.diag_last_fail_label)

        control_content.add_widget(controls)

        gallery_card = GlassCard(orientation="vertical", size_hint_y=None, height=dp(500))
        gallery_card.add_widget(self._label("Galerie"))
        gallery_scroll = ScrollView(do_scroll_x=False)
        self.gallery_grid = GridLayout(cols=GALLERY_COLUMNS, spacing=dp(10), padding=dp(2), size_hint_y=None)
        self.gallery_grid.bind(minimum_height=self.gallery_grid.setter("height"))
        gallery_scroll.add_widget(self.gallery_grid)
        gallery_card.add_widget(gallery_scroll)
        control_content.add_widget(gallery_card)
        control_scroll.add_widget(control_content)
        left.add_widget(control_scroll)

        previews = GridLayout(cols=3, spacing=dp(10), size_hint_y=0.46)
        self.preview_img = Image()
        self.preview_silhouette = Image()
        self.live_preview_img = Image()
        previews.add_widget(self._carded_view("Camouflage courant", self.preview_img))
        previews.add_widget(self._carded_view("Projection silhouette", self.preview_silhouette))

        live_card = GlassCard(orientation="vertical")
        live_card.add_widget(self._label("Suivi direct de construction"))
        self.live_stage_label = self._small_label("Étape : attente")
        self.live_counts_label = self._small_label("Macros -- | transitions -- | micros --")
        self.live_meta_label = self._small_label("Image -- | essai -- | seed --")
        live_card.add_widget(self.live_stage_label)
        live_card.add_widget(self.live_counts_label)
        live_card.add_widget(self.live_meta_label)
        live_pane = SoftPane(orientation="vertical")
        live_pane.add_widget(self.live_preview_img)
        live_card.add_widget(live_pane)
        previews.add_widget(live_card)
        right.add_widget(previews)

        bottom = BoxLayout(spacing=dp(10), size_hint_y=0.54)
        log_card = GlassCard(orientation="vertical")
        log_card.add_widget(self._label("Journal"))
        self.log_view = LogView()
        log_card.add_widget(self.log_view)
        diag_card = GlassCard(orientation="vertical")
        diag_card.add_widget(self._label("Diagnostic live"))
        self.diag_log_view = LogView()
        diag_card.add_widget(self.diag_log_view)
        bottom.add_widget(log_card)
        bottom.add_widget(diag_card)
        right.add_widget(bottom)

        body.add_widget(left)
        body.add_widget(right)
        root.add_widget(body)

        self._refresh_controls_state()
        self._refresh_run_mode_buttons()
        self.reload_gallery()
        self._refresh_diag_labels()
        Clock.schedule_interval(self._update_resource_monitor, 1.0)
        Clock.schedule_interval(lambda dt: self.reload_gallery(), 3.0)
        return root

    # ---------- small UI helpers ----------
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

    def _button(self, text: str, role: str, callback) -> SoftButton:
        btn = SoftButton(text=text, role=role)
        btn.bind(on_release=callback)
        return btn

    def _carded_view(self, title: str, image_widget: Image) -> Widget:
        box = GlassCard(orientation="vertical")
        box.add_widget(self._label(title))
        pane = SoftPane(orientation="vertical")
        pane.add_widget(image_widget)
        box.add_widget(pane)
        return box

    # ---------- runtime ----------
    def on_start(self):
        try:
            Window.maximize()
        except Exception:
            pass
        self._subscribe_runtime_feed()
        self._emit_runtime("INFO", "start", "Interface Kivy démarrée")

    def _emit_runtime(self, level: str, source: str, message: str, **payload: Any):
        if camo_log is None or not hasattr(camo_log, "log_event"):
            return
        try:
            camo_log.log_event(level, source, message, **payload)
        except Exception:
            pass

    def _format_runtime_event(self, event: Any) -> str:
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

    def _on_runtime_event(self, event: Any):
        line = self._format_runtime_event(event)
        self._append_runtime_line(line)
        self._maybe_handle_live_runtime_payload(event)

    @mainthread
    def _append_runtime_line(self, line: str):
        if self.log_view is not None:
            self.log_view.append(line)
        if self.runtime_last_label is not None:
            short = line if len(line) <= 140 else line[:137] + "..."
            self.runtime_last_label.text = f"Dernier runtime : {short}"

    def _subscribe_runtime_feed(self):
        if camo_log is None or self._runtime_subscription_active:
            return
        try:
            manager = getattr(camo_log, "LOG_MANAGER", None)
            if manager is not None and hasattr(manager, "subscribe"):
                manager.subscribe(self._on_runtime_event)
                self._runtime_subscription_active = True
                self._runtime_subscriber_callback = self._on_runtime_event
        except Exception:
            pass

    def _unsubscribe_runtime_feed(self):
        if camo_log is None or not self._runtime_subscription_active:
            return
        try:
            manager = getattr(camo_log, "LOG_MANAGER", None)
            if manager is not None and hasattr(manager, "unsubscribe"):
                manager.unsubscribe(self._runtime_subscriber_callback)
        except Exception:
            pass
        self._runtime_subscription_active = False
        self._runtime_subscriber_callback = None

    # ---------- log / status ----------
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
    def diag_log(self, line: str):
        if self.diag_log_view is not None:
            self.diag_log_view.append(line)

    @mainthread
    def update_progress(self, current: int, total: int):
        if self.progress_bar is not None:
            self.progress_bar.max_value = total
            self.progress_bar.value = current

    @mainthread
    def update_preview(self, pil_img: PILImage.Image, silhouette_img: PILImage.Image):
        if self.preview_img is not None:
            self.preview_img.texture = pil_to_coreimage(pil_img).texture
        if self.preview_silhouette is not None:
            self.preview_silhouette.texture = pil_to_coreimage(silhouette_img).texture

    @mainthread
    def update_live_stage(
        self,
        stage: str,
        target_index: Optional[int] = None,
        local_attempt: Optional[int] = None,
        seed: Optional[int] = None,
        macro_count: Optional[int] = None,
        transition_count: Optional[int] = None,
        micro_count: Optional[int] = None,
        pil_img: Optional[PILImage.Image] = None,
        preview_path: Optional[str] = None,
    ):
        if self.live_stage_label is not None:
            self.live_stage_label.text = f"Étape : {stage}"
        if self.live_counts_label is not None:
            m = "--" if macro_count is None else str(macro_count)
            t = "--" if transition_count is None else str(transition_count)
            mi = "--" if micro_count is None else str(micro_count)
            self.live_counts_label.text = f"Macros {m} | transitions {t} | micros {mi}"
        if self.live_meta_label is not None:
            ti = "--" if target_index is None else f"{target_index:03d}"
            la = "--" if local_attempt is None else f"{local_attempt:04d}"
            sd = "--" if seed is None else str(seed)
            self.live_meta_label.text = f"Image {ti} | essai {la} | seed {sd}"
        try:
            if pil_img is not None and self.live_preview_img is not None:
                self.live_preview_img.texture = pil_to_coreimage(pil_img).texture
            elif preview_path and self.live_preview_img is not None and Path(preview_path).exists():
                img = PILImage.open(preview_path).convert("RGB")
                self.live_preview_img.texture = pil_to_coreimage(img).texture
        except Exception:
            pass

    def _maybe_handle_live_runtime_payload(self, event: Any):
        payload = getattr(event, "payload", {}) or {}
        if not isinstance(payload, dict):
            return
        stage = payload.get("stage") or payload.get("live_stage") or payload.get("phase")
        preview_path = payload.get("preview_path") or payload.get("snapshot_path") or payload.get("frame_path")
        macro_count = payload.get("macro_count")
        transition_count = payload.get("transition_count")
        micro_count = payload.get("micro_count")
        target_index = payload.get("target_index")
        local_attempt = payload.get("local_attempt")
        seed = payload.get("seed")
        if stage or preview_path or macro_count is not None or transition_count is not None or micro_count is not None:
            self.update_live_stage(
                stage=str(stage or "construction"),
                target_index=int(target_index) if target_index is not None else None,
                local_attempt=int(local_attempt) if local_attempt is not None else None,
                seed=int(seed) if seed is not None else None,
                macro_count=int(macro_count) if macro_count is not None else None,
                transition_count=int(transition_count) if transition_count is not None else None,
                micro_count=int(micro_count) if micro_count is not None else None,
                preview_path=str(preview_path) if preview_path else None,
            )

    def _refresh_controls_state(self):
        if self.start_btn is not None:
            self.start_btn.disabled = self.running or self.stopping or self.preflight_running
        if self.stop_btn is not None:
            self.stop_btn.disabled = not (self.running or self.preflight_running or self.stopping)
        if self.open_btn is not None:
            self.open_btn.disabled = False
        self._refresh_run_mode_buttons()

    def _on_intensity_change(self, _slider, value):
        if self.intensity_label is not None:
            self.intensity_label.text = f"{int(value)} %"
        self.machine_intensity = float(value)

    def _run_mode_text(self, mode: Optional[str] = None) -> str:
        mode = self.run_mode if mode is None else mode
        if mode == RUN_MODE_NON_BLOCKING:
            return "tests non bloquants"
        if mode == RUN_MODE_SKIP_TESTS:
            return "sans tests"
        return "tests bloquants"

    def _set_run_mode(self, mode: str):
        if self.running or self.preflight_running:
            return
        self.run_mode = mode
        self._refresh_run_mode_buttons()
        self.log(f"Mode sélectionné : {self._run_mode_text(mode)}")

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

    @mainthread
    def _refresh_diag_labels(self):
        if self.diag_summary_label is not None:
            rate = (self.diag_accepts / self.diag_total) if self.diag_total else 0.0
            self.diag_summary_label.text = f"Tentatives {self.diag_total} | acceptés {self.diag_accepts} | rejetés {self.diag_rejects} | taux {rate:.2%}"
        if self.diag_top_rules_label is not None:
            if self.diag_rule_counter:
                top = " | ".join(f"{n}:{c}" for n, c in self.diag_rule_counter.most_common(3))
                self.diag_top_rules_label.text = f"Top règles : {top}"
            else:
                self.diag_top_rules_label.text = "Top règles : --"
        if self.diag_last_fail_label is not None:
            self.diag_last_fail_label.text = "Dernier rejet : " + (" | ".join(self.diag_last_rules[:4]) if self.diag_last_rules else "--")

    # ---------- gallery ----------
    @mainthread
    def reload_gallery(self):
        if self.gallery_grid is None:
            return
        self.gallery_grid.clear_widgets()
        if not self.current_output_dir.exists():
            return
        for p in sorted(self.current_output_dir.glob("camouflage_*.png")):
            self.gallery_grid.add_widget(GalleryThumb(self, p))

    # ---------- preflight ----------
    @mainthread
    def _update_preflight_label(self, text: str, ok: Optional[bool] = None):
        if self.tests_label is None:
            return
        self.tests_label.text = text
        self.tests_label.color = C["success"] if ok is True else C["danger"] if ok is False else C["text_soft"]

    async def _async_run_preflight(self) -> Tuple[bool, str]:
        if camo_log is None or not hasattr(camo_log, "async_run_preflight_tests"):
            return False, "log.py indisponible : impossible de lancer le préflight."
        try:
            summary = await camo_log.async_run_preflight_tests(module_names=DEFAULT_PREFLIGHT_MODULES, output_dir=Path("logs_generation"), timeout_s=None)
            if hasattr(summary, "short_text"):
                return bool(getattr(summary, "ok", False)), str(summary.short_text())
            if isinstance(summary, dict):
                ok = bool(summary.get("ok", False))
                total = int(summary.get("total", 0))
                failures = int(summary.get("failures", 0))
                errors = int(summary.get("errors", 0))
                return ok, f"{total} tests | {failures} échec(s) | {errors} erreur(s)"
            return False, "Préflight : réponse inattendue."
        except Exception as exc:
            return False, f"Préflight impossible : {exc}"

    def _ensure_preflight(self, pending_start: bool = True) -> bool:
        if self.tests_ran and self.tests_ok:
            return True
        if self.preflight_running:
            return False
        self.preflight_running = True
        self.preflight_pending_start = pending_start
        self._update_preflight_label("Préflight en cours…", ok=None)
        self.status("Préflight en cours…", ok=True)
        self._refresh_controls_state()
        fut = self.async_runner.submit(self._async_run_preflight())
        self.preflight_future = fut
        fut.add_done_callback(self._on_preflight_done)
        return False

    def _on_preflight_done(self, fut: Future):
        try:
            ok, summary = fut.result()
        except Exception as exc:
            ok, summary = False, str(exc)
        Clock.schedule_once(lambda dt: self._finish_preflight(ok, summary), 0)

    @mainthread
    def _finish_preflight(self, ok: bool, summary: str):
        pending_start = self.preflight_pending_start
        self.preflight_running = False
        self.preflight_pending_start = False
        self.tests_ran = True
        self.tests_ok = ok
        self.tests_summary = summary
        self._update_preflight_label(summary, ok=ok)
        self.log(f"Préflight {'OK' if ok else 'KO'} : {summary}")
        self._refresh_controls_state()
        if pending_start and ok:
            self._start_generation_after_preflight()
        elif pending_start and not ok:
            self.status("Tests KO", ok=False)

    # ---------- diagnostics ----------
    async def _extract_failure_rules(self, candidate: camo.CandidateResult, target_index: int, local_attempt: int) -> List[str]:
        try:
            failures = await asyncio.to_thread(camo.extract_rejection_failures, candidate, target_index, local_attempt)
            return [str(item.get("rule", "")) for item in failures if item.get("rule")]
        except Exception:
            return []

    async def _register_live_diag(self, candidate: camo.CandidateResult, target_index: int, local_attempt: int, accepted: bool):
        self.diag_total += 1
        if accepted:
            self.diag_accepts += 1
            self.diag_last_rules = []
            self.diag_log(f"[img={target_index:03d} essai={local_attempt:04d}] accepté | seed={candidate.seed}")
            self._refresh_diag_labels()
            return
        self.diag_rejects += 1
        rules = await self._extract_failure_rules(candidate, target_index, local_attempt)
        self.diag_last_rules = rules[:]
        for rule in rules:
            self.diag_rule_counter[rule] += 1
        joined = " | ".join(rules[:6]) if rules else "non disponibles"
        self.diag_log(f"[img={target_index:03d} essai={local_attempt:04d}] rejet | seed={candidate.seed} | règles: {joined}")
        self._refresh_diag_labels()

    # ---------- generation ----------
    def start_generation(self, *_):
        if self.running or self.preflight_running:
            return
        if self.current_future is not None and not self.current_future.done():
            return

        if self.run_mode == RUN_MODE_SKIP_TESTS:
            self.tests_summary = "Tests ignorés (mode sans tests)."
            self._update_preflight_label(self.tests_summary, ok=None)
            self._start_generation_after_preflight()
            return

        if self.run_mode == RUN_MODE_NON_BLOCKING:
            if not (self.tests_ran and self.tests_ok):
                self._ensure_preflight(pending_start=False)
            self._start_generation_after_preflight(allow_during_preflight=True)
            return

        if self.tests_ran and self.tests_ok:
            self._start_generation_after_preflight()
            return

        self._ensure_preflight(pending_start=True)

    def _start_generation_after_preflight(self, allow_during_preflight: bool = False):
        if self.running or self.stopping or (self.preflight_running and not allow_during_preflight):
            return
        try:
            count = int((self.count_input.text if self.count_input is not None else "0").strip())
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
        self.diag_total = self.diag_accepts = self.diag_rejects = 0
        self.diag_rule_counter = Counter()
        self.diag_last_rules = []
        self.update_progress(0, count)
        self._refresh_diag_labels()
        prevent_sleep(True)
        self.status("Génération en cours…", ok=True)
        self.log(f"Démarrage : {count} camouflage(s)")
        self._refresh_controls_state()
        fut = self.async_runner.submit(self._async_worker_generate(count))
        self.current_future = fut
        fut.add_done_callback(self._on_generation_done)

    def _on_generation_done(self, fut: Future):
        try:
            fut.result()
        except Exception as exc:
            Clock.schedule_once(lambda dt: self._finish_error(str(exc)), 0)
        finally:
            Clock.schedule_once(lambda dt: self._clear_future(fut), 0)

    @mainthread
    def _clear_future(self, fut: Future):
        if self.current_future is fut:
            self.current_future = None
        self._refresh_controls_state()

    def stop_generation(self, *_):
        if self.running:
            self.stop_flag = True
            self.stopping = True
            self.status("Arrêt demandé…", ok=False)
            self.log("Arrêt demandé. Fin de la tentative courante puis arrêt.")
            self._refresh_controls_state()
            return
        if self.preflight_running:
            self.preflight_pending_start = False
            self.log("Démarrage annulé pendant le préflight.")
            return

    async def _async_should_stop(self) -> bool:
        return self.stop_flag

    async def _adaptive_pause(self):
        base = 0.0 if self.machine_intensity >= 90 else (100.0 - self.machine_intensity) / 5000.0
        extra = 0.0
        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                if cpu >= 98 or ram >= 95:
                    extra = 0.03
                elif cpu >= 95 or ram >= 92:
                    extra = 0.015
            except Exception:
                pass
        if base + extra > 0:
            await asyncio.sleep(base + extra)

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
                    self.update_live_stage("génération backend", target_index, local_attempt, seed)
                    candidate = await camo.async_generate_candidate_from_seed(seed)
                    self.update_live_stage("validation", target_index, local_attempt, seed, pil_img=candidate.image)
                    valid = await camo.async_validate_candidate_result(candidate)
                    scores = extract_backend_scores(candidate.metrics)
                    silhouette_img = await asyncio.to_thread(silhouette_projection_image, rgb_image_to_index_canvas(candidate.image))
                    self.update_preview(candidate.image, silhouette_img)
                    await self._register_live_diag(candidate, target_index, local_attempt, valid)
                    self._update_attempt_status(target_index, local_attempt, total_attempts, seed, target_count, len(rows) + (1 if valid else 0), self.diag_rejects, valid, candidate.ratios, scores, candidate.metrics)

                    if not valid:
                        self.update_live_stage("rejeté", target_index, local_attempt, seed, pil_img=candidate.image)
                        self.log(f"[img={target_index:03d} essai={local_attempt:04d}] rejeté | SF={scores['score_final']:.3f}")
                        await self._adaptive_pause()
                        await asyncio.sleep(0)
                        continue

                    filename = self.current_output_dir / f"camouflage_{target_index:03d}.png"
                    self.update_live_stage("export image", target_index, local_attempt, seed, pil_img=candidate.image)
                    await camo.async_save_candidate_image(candidate, filename)
                    record = CandidateRecord(index=target_index, seed=candidate.seed, local_attempt=local_attempt, global_attempt=total_attempts, image_path=filename, metrics=dict(candidate.metrics), ratios=candidate.ratios.copy())
                    self.best_records.append(record)
                    self.best_records.sort(key=lambda r: float(r.metrics.get("visual_score_final", 0.0)), reverse=True)
                    row = camo.candidate_row(target_index, local_attempt, total_attempts, candidate)
                    row["image_path"] = str(filename)
                    rows.append(row)
                    self.accepted_count = len(rows)
                    self.update_progress(len(rows), target_count)
                    self.update_live_stage("accepté", target_index, local_attempt, seed, pil_img=candidate.image)
                    self.log(f"[img={target_index:03d}] accepté -> {filename.name} | SF={scores['score_final']:.3f} | militaire={scores['military_score']:.3f}")
                    self.reload_gallery()
                    await self._adaptive_pause()
                    await asyncio.sleep(0)
                    break
            await self._async_finish_success(rows)
        except asyncio.CancelledError:
            await self._async_finish_stopped(rows)
            raise
        except Exception as exc:
            await self._async_finish_error(str(exc))

    @mainthread
    def _update_attempt_status(self, target_index: int, attempt_idx: int, global_attempt: int, seed: int, target_total: int, accepted_count: int, rejected_count: int, accepted: bool, rs: np.ndarray, scores: Dict[str, float], metrics: Dict[str, float]):
        if self.progress_text is not None:
            self.progress_text.text = f"{accepted_count} / {target_total} validé(s) | tentatives {global_attempt} | rejetés {rejected_count}"
        if self.attempt_text is not None:
            verdict = "accepté" if accepted else "rejeté"
            self.attempt_text.text = f"Image {target_index:03d} | essai {attempt_idx:04d} | total {global_attempt:06d} | seed {seed} | {verdict}"
        if self.color_text is not None:
            self.color_text.text = f"C {rs[camo.IDX_COYOTE]*100:.2f}% | O {rs[camo.IDX_OLIVE]*100:.2f}% | T {rs[camo.IDX_TERRE]*100:.2f}% | G {rs[camo.IDX_GRIS]*100:.2f}%"
        if self.score_text is not None:
            self.score_text.text = f"Score {scores['score_final']:.3f} | ratio {scores['score_ratio']:.3f} | silhouette {scores['score_silhouette']:.3f} | contour {scores['score_contour']:.3f}"
        if self.extra_text is not None:
            self.extra_text.text = f"olive conn. {metrics.get('largest_olive_component_ratio', 0.0):.3f} | centre {metrics.get('center_empty_ratio', 0.0):.3f} | limites {metrics.get('boundary_density', 0.0):.3f} | miroir {metrics.get('mirror_similarity', 0.0):.3f}"
        if self.struct_text is not None:
            self.struct_text.text = (
                f"terre macro {metrics.get('macro_terre_visible_ratio', 0.0):.3f} | "
                f"gris macro {metrics.get('macro_gris_visible_ratio', 0.0):.3f} | "
                f"continuité brune {metrics.get('central_brown_continuity', 0.0):.3f} | "
                f"militaire {metrics.get('visual_military_score', 0.0):.3f}"
            )

    async def _async_write_report(self, rows: List[dict]) -> Path:
        return await camo.async_write_report(rows, self.current_output_dir, filename=REPORT_NAME)

    async def _async_export_best_of(self, top_k: int) -> Path:
        best_dir = self.current_output_dir / BEST_DIR_NAME
        best_dir.mkdir(parents=True, exist_ok=True)
        for child in best_dir.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except Exception:
                    pass
        rows: List[dict] = []
        for rank, rec in enumerate(self.best_records[:top_k], start=1):
            dst = best_dir / f"best_{rank:03d}_camouflage_{rec.index:03d}.png"
            await asyncio.to_thread(shutil.copy2, rec.image_path, dst)
            rows.append({
                "rank": rank,
                "source_index": rec.index,
                "seed": rec.seed,
                "global_attempt": rec.global_attempt,
                "attempts_for_this_image": rec.local_attempt,
                "score_final": round(float(rec.metrics.get("visual_score_final", 0.0)), 5),
                "score_ratio": round(float(rec.metrics.get("visual_score_ratio", 0.0)), 5),
                "score_silhouette": round(float(rec.metrics.get("visual_score_silhouette", 0.0)), 5),
                "score_contour": round(float(rec.metrics.get("visual_score_contour", 0.0)), 5),
                "score_main": round(float(rec.metrics.get("visual_score_main", 0.0)), 5),
                "military_score": round(float(rec.metrics.get("visual_military_score", 0.0)), 5),
                "coyote_brown_pct": round(float(rec.ratios[camo.IDX_COYOTE] * 100), 2),
                "vert_olive_pct": round(float(rec.ratios[camo.IDX_OLIVE] * 100), 2),
                "terre_de_france_pct": round(float(rec.ratios[camo.IDX_TERRE] * 100), 2),
                "vert_de_gris_pct": round(float(rec.ratios[camo.IDX_GRIS] * 100), 2),
            })
        if rows:
            await asyncio.to_thread(self._write_csv_sync, best_dir / "best_of.csv", rows)
        return best_dir

    def _write_csv_sync(self, path: Path, rows: List[dict]):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            import csv
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
        self.reload_gallery()
        self._refresh_controls_state()

    async def _async_finish_stopped(self, rows: List[dict]):
        report_path = await self._async_write_report(rows)
        if rows:
            await self._async_export_best_of(min(DEFAULT_TOP_K, len(rows)))
        prevent_sleep(False)
        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Arrêté", ok=False)
        self.log(f"Rapport partiel : {report_path}")
        self.reload_gallery()
        self._refresh_controls_state()

    async def _async_finish_error(self, message: str):
        prevent_sleep(False)
        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Erreur", ok=False)
        self.log(f"Erreur : {message}")
        self.diag_log(f"Erreur diagnostic : {message}")
        self._refresh_controls_state()

    @mainthread
    def _finish_error(self, message: str):
        prevent_sleep(False)
        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Erreur", ok=False)
        self.log(f"Erreur : {message}")
        self._refresh_controls_state()

    # ---------- monitoring ----------
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
            self.resource_text.text = f"CPU {cpu:.0f}% | RAM {ram:.0f}% | Disque {disk:.0f}% | Processus {proc_cpu:.0f}% / {proc_mem:.2f} Go"
        except Exception:
            self.resource_text.text = "Monitoring indisponible."

    def on_stop(self):
        self.stop_flag = True
        self.stopping = True
        prevent_sleep(False)
        self._emit_runtime("INFO", "start", "Arrêt de l'interface Kivy")
        self._unsubscribe_runtime_feed()
        fut = self.current_future
        if fut is not None and not fut.done():
            fut.add_done_callback(lambda _f: self.async_runner.stop())
        else:
            self.async_runner.stop()


if __name__ == "__main__":
    CamouflageApp().run()
