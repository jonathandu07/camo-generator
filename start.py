# -*- coding: utf-8 -*-
"""
start.py
Interface Kivy plein écran pour piloter main.py comme module.

Fonctions :
- génération séquentielle stricte
- progression temps réel
- aperçu du camouflage accepté
- projection sur silhouette pseudo-humaine
- score de rupture des contours externes
- tri automatique des meilleurs motifs validés
- export CSV enrichi
- arrêt propre de la génération

Pré-requis :
    pip install kivy pillow numpy

Arborescence attendue :
    .
    ├── main.py
    └── start.py
"""

from __future__ import annotations

import csv
import io
import math
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

# ---------------- Kivy config avant imports UI ----------------
from kivy.config import Config

Config.set("graphics", "fullscreen", "auto")
Config.set("graphics", "resizable", "1")
Config.set("graphics", "minimum_width", "1200")
Config.set("graphics", "minimum_height", "800")
Config.set("input", "mouse", "mouse,multitouch_on_demand")

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# ---------------- moteur principal ----------------
import main as camo


# ============================================================
# CONSTANTES UI / EXPORT
# ============================================================

APP_TITLE = "Camouflage Armée Fédérale Europe — Générateur V3"
BEST_DIR_NAME = "best_of"
REPORT_NAME = "rapport_camouflages_v3.csv"
DEFAULT_OUTPUT_DIR = Path("camouflages_federale_europe")
DEFAULT_TOP_K = 20

# Score final : pondérations
WEIGHT_RATIO = 0.28
WEIGHT_SILHOUETTE = 0.30
WEIGHT_CONTOUR = 0.24
WEIGHT_MAIN_METRICS = 0.18

# Seuils V3 supplémentaires
MIN_SILHOUETTE_COLOR_DIVERSITY = 0.62
MIN_CONTOUR_BREAK_SCORE = 0.44
MIN_OUTLINE_BAND_DIVERSITY = 0.58
MIN_SMALL_SCALE_STRUCTURAL_SCORE = 0.42


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
# OUTILS GÉNÉRAUX
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


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def pil_to_coreimage(pil_img: PILImage.Image) -> CoreImage:
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)
    return CoreImage(bio, ext="png")


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

    # image issue directement du moteur : couleurs exactes
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
    """
    Silhouette pseudo-humaine simplifiée :
    tête, épaules, torse, bras, hanches, jambes.
    """
    img = PILImage.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    # tête
    head_w = int(width * 0.18)
    head_h = int(height * 0.12)
    head_x1 = (width - head_w) // 2
    head_y1 = int(height * 0.05)
    draw.ellipse([head_x1, head_y1, head_x1 + head_w, head_y1 + head_h], fill=255)

    # torse principal
    torso_w = int(width * 0.34)
    torso_h = int(height * 0.38)
    torso_x1 = (width - torso_w) // 2
    torso_y1 = int(height * 0.16)
    draw.rounded_rectangle(
        [torso_x1, torso_y1, torso_x1 + torso_w, torso_y1 + torso_h],
        radius=int(width * 0.03),
        fill=255,
    )

    # épaules
    shoulder_w = int(width * 0.58)
    shoulder_h = int(height * 0.10)
    shoulder_x1 = (width - shoulder_w) // 2
    shoulder_y1 = int(height * 0.14)
    draw.rounded_rectangle(
        [shoulder_x1, shoulder_y1, shoulder_x1 + shoulder_w, shoulder_y1 + shoulder_h],
        radius=int(width * 0.025),
        fill=255,
    )

    # bras
    arm_w = int(width * 0.11)
    arm_h = int(height * 0.32)
    left_arm_x1 = int(width * 0.15)
    right_arm_x1 = width - left_arm_x1 - arm_w
    arm_y1 = int(height * 0.20)
    draw.rounded_rectangle([left_arm_x1, arm_y1, left_arm_x1 + arm_w, arm_y1 + arm_h], radius=int(width * 0.02), fill=255)
    draw.rounded_rectangle([right_arm_x1, arm_y1, right_arm_x1 + arm_w, arm_y1 + arm_h], radius=int(width * 0.02), fill=255)

    # bassin
    pelvis_w = int(width * 0.30)
    pelvis_h = int(height * 0.10)
    pelvis_x1 = (width - pelvis_w) // 2
    pelvis_y1 = int(height * 0.51)
    draw.rounded_rectangle([pelvis_x1, pelvis_y1, pelvis_x1 + pelvis_w, pelvis_y1 + pelvis_h], radius=int(width * 0.02), fill=255)

    # jambes
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
    """
    Crée un aperçu du camouflage projeté sur silhouette sur fond neutre.
    """
    h, w = index_canvas.shape
    sil = build_silhouette_mask(w, h)

    background = np.full((h, w, 3), 22, dtype=np.uint8)
    rgb = camo.RGB[index_canvas]
    background[sil] = rgb[sil]

    # contour léger
    bound = silhouette_boundary(sil)
    background[bound] = np.array([230, 230, 230], dtype=np.uint8)
    return PILImage.fromarray(background, "RGB")


def silhouette_color_diversity_score(index_canvas: np.ndarray) -> float:
    sil = build_silhouette_mask(index_canvas.shape[1], index_canvas.shape[0])
    data = index_canvas[sil]
    if data.size == 0:
        return 0.0
    unique_count = len(np.unique(data))
    # 4 couleurs attendues -> score max si 4 présentes, puis pondération par répartition
    hist = np.bincount(data, minlength=4).astype(float)
    hist /= hist.sum()
    entropy = -np.sum([p * math.log(p + 1e-12) for p in hist if p > 0])
    entropy /= math.log(4.0)
    return clamp01((unique_count / 4.0) * 0.45 + entropy * 0.55)


def contour_break_score(index_canvas: np.ndarray) -> Tuple[float, float]:
    """
    Évalue la rupture des contours externes sur une silhouette pseudo-humaine.

    Retour :
    - contour_break_score
    - outline_band_diversity
    """
    h, w = index_canvas.shape
    sil = build_silhouette_mask(w, h)
    bound = silhouette_boundary(sil)

    # bande intérieure autour du contour
    band = dilate_bool(bound, radius=5) & sil
    vals = index_canvas[band]
    if vals.size == 0:
        return 0.0, 0.0

    hist = np.bincount(vals, minlength=4).astype(float)
    hist /= hist.sum()
    entropy = -np.sum([p * math.log(p + 1e-12) for p in hist if p > 0])
    entropy /= math.log(4.0)

    # fraction de pixels du contour dont le voisinage contient au moins 2 couleurs
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

    # score final : rupture perçue du contour
    score = clamp01(local_variation * 0.62 + entropy * 0.38)
    return score, float(entropy)


def small_scale_structural_score(index_canvas: np.ndarray) -> float:
    """
    Vérifie qu'à petite échelle il reste des masses lisibles,
    et pas juste du bruit.
    """
    small = downsample_nearest(index_canvas, 4)
    olive_ratio = largest_component_ratio(small == camo.IDX_OLIVE)
    bd = float(np.mean(boundary_mask(small)))

    # olive connectée + frontières ni trop faibles ni trop isotropes
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
    parts = []

    parts.append(clamp01((metrics["largest_olive_component_ratio"] - 0.12) / 0.18))
    parts.append(clamp01(1.0 - metrics["center_empty_ratio"] / 0.60))
    parts.append(clamp01(1.0 - metrics["mirror_similarity"] / 0.90))
    parts.append(clamp01((metrics["olive_multizone_share"] - 0.25) / 0.45))
    parts.append(clamp01(1.0 - abs(metrics["boundary_density"] - 0.14) / 0.12))
    parts.append(clamp01((metrics["vert_olive_macro_share"] - 0.45) / 0.30))
    parts.append(clamp01((metrics["terre_de_france_transition_share"] - 0.20) / 0.30))
    parts.append(clamp01((metrics["vert_de_gris_micro_share"] - 0.50) / 0.25))

    return float(np.mean(parts))


def evaluate_candidate_v3(pil_img: PILImage.Image, rs: np.ndarray, metrics: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
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


# ============================================================
# WIDGETS
# ============================================================

class StatLabel(Label):
    pass


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
            color=(0.93, 0.93, 0.95, 1),
        )
        self.label.bind(texture_size=self._update_label_height, width=self._update_text_width)
        self.add_widget(self.label)

    def _update_label_height(self, *_):
        self.label.height = self.label.texture_size[1] + dp(12)

    def _update_text_width(self, *_):
        self.label.text_size = (self.width - dp(24), None)

    def append(self, line: str) -> None:
        lines = self.label.text.splitlines() if self.label.text else []
        lines.append(line)
        if len(lines) > 350:
            lines = lines[-350:]
        self.label.text = "\n".join(lines)
        Clock.schedule_once(lambda dt: setattr(self, "scroll_y", 0), 0.05)


# ============================================================
# APPLICATION
# ============================================================

class CamouflageApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_requested = False
        self.running = False

        self.current_output_dir = DEFAULT_OUTPUT_DIR
        self.best_records: List[CandidateRecord] = []

        self.current_preview_img: Optional[PILImage.Image] = None
        self.current_silhouette_preview: Optional[PILImage.Image] = None

    def build(self):
        Window.clearcolor = (0.07, 0.08, 0.10, 1)

        root = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(70), spacing=dp(10))
        self.title_label = Label(
            text=APP_TITLE,
            bold=True,
            font_size=sp(24),
            color=(0.96, 0.96, 0.98, 1),
            halign="left",
            valign="middle",
        )
        self.title_label.bind(size=lambda *a: setattr(self.title_label, "text_size", self.title_label.size))

        self.status_label = Label(
            text="Prêt",
            font_size=sp(16),
            color=(0.78, 0.90, 0.78, 1),
            size_hint_x=0.28,
            halign="right",
            valign="middle",
        )
        self.status_label.bind(size=lambda *a: setattr(self.status_label, "text_size", self.status_label.size))

        header.add_widget(self.title_label)
        header.add_widget(self.status_label)
        root.add_widget(header)

        # Body
        body = BoxLayout(spacing=dp(10))

        # Colonne gauche : commandes
        left = BoxLayout(orientation="vertical", size_hint_x=0.32, spacing=dp(10))

        controls = GridLayout(cols=1, spacing=dp(8), size_hint_y=None, padding=dp(10))
        controls.bind(minimum_height=controls.setter("height"))

        controls.add_widget(self._label("Dossier de sortie"))
        self.output_input = TextInput(
            text=str(DEFAULT_OUTPUT_DIR),
            multiline=False,
            size_hint_y=None,
            height=dp(40),
            background_color=(0.13, 0.14, 0.17, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1),
        )
        controls.add_widget(self.output_input)

        controls.add_widget(self._label("Nombre d'images validées"))
        self.count_input = TextInput(
            text="100",
            multiline=False,
            input_filter="int",
            size_hint_y=None,
            height=dp(40),
            background_color=(0.13, 0.14, 0.17, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1),
        )
        controls.add_widget(self.count_input)

        controls.add_widget(self._label("Top motifs à copier dans best_of"))
        self.topk_input = TextInput(
            text=str(DEFAULT_TOP_K),
            multiline=False,
            input_filter="int",
            size_hint_y=None,
            height=dp(40),
            background_color=(0.13, 0.14, 0.17, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1),
        )
        controls.add_widget(self.topk_input)

        row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(10))
        row.add_widget(self._label("Aperçu silhouette", size_hint_x=0.7))
        self.silhouette_checkbox = CheckBox(active=True)
        row.add_widget(self.silhouette_checkbox)
        controls.add_widget(row)

        score_box = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(120), spacing=dp(6))
        self.score_target_label = self._small_label("Validation V3 activée : silhouette + contour + tri automatique")
        score_box.add_widget(self.score_target_label)

        controls.add_widget(score_box)

        button_row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))
        self.start_btn = Button(
            text="Démarrer",
            background_normal="",
            background_color=(0.16, 0.48, 0.28, 1),
            color=(1, 1, 1, 1),
        )
        self.start_btn.bind(on_release=self.start_generation)

        self.stop_btn = Button(
            text="Arrêter",
            background_normal="",
            background_color=(0.55, 0.18, 0.18, 1),
            color=(1, 1, 1, 1),
        )
        self.stop_btn.bind(on_release=self.stop_generation)

        self.open_btn = Button(
            text="Ouvrir dossier",
            background_normal="",
            background_color=(0.22, 0.28, 0.42, 1),
            color=(1, 1, 1, 1),
        )
        self.open_btn.bind(on_release=lambda *_: open_folder(Path(self.output_input.text.strip() or ".")))

        button_row.add_widget(self.start_btn)
        button_row.add_widget(self.stop_btn)
        button_row.add_widget(self.open_btn)
        controls.add_widget(button_row)

        # Progression
        controls.add_widget(self._label("Progression globale"))
        self.progress_global = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(18))
        controls.add_widget(self.progress_global)

        controls.add_widget(self._label("Tentative en cours"))
        self.progress_attempt_label = self._small_label("Image 000 | essai 0000")
        controls.add_widget(self.progress_attempt_label)

        # Statistiques
        stats_box = GridLayout(cols=1, spacing=dp(4), size_hint_y=None)
        stats_box.bind(minimum_height=stats_box.setter("height"))

        self.stat_color = self._small_label("C=0.00 O=0.00 T=0.00 G=0.00")
        self.stat_score = self._small_label("Score final = 0.000")
        self.stat_main = self._small_label("Silhouette=0.000 | Contour=0.000 | Main=0.000")
        self.stat_extra = self._small_label("Contour break=0.000 | Outline=0.000 | Small-scale=0.000")

        stats_box.add_widget(self.stat_color)
        stats_box.add_widget(self.stat_score)
        stats_box.add_widget(self.stat_main)
        stats_box.add_widget(self.stat_extra)
        controls.add_widget(stats_box)

        left_scroll = ScrollView()
        left_scroll.add_widget(controls)
        left.add_widget(left_scroll)

        # Colonne droite : aperçu + logs
        right = BoxLayout(orientation="vertical", spacing=dp(10))

        previews = BoxLayout(spacing=dp(10), size_hint_y=0.62)

        self.preview_img = Image(allow_stretch=True, keep_ratio=True)
        self.preview_silhouette = Image(allow_stretch=True, keep_ratio=True)

        previews.add_widget(self._framed_widget("Motif validé / courant", self.preview_img))
        previews.add_widget(self._framed_widget("Projection silhouette", self.preview_silhouette))

        right.add_widget(previews)

        self.log_view = LogView()
        right.add_widget(self._framed_widget("Journal", self.log_view))

        body.add_widget(left)
        body.add_widget(right)
        root.add_widget(body)

        return root

    # ---------------- UI helpers ----------------

    def _label(self, text: str, **kwargs) -> Label:
        lbl = Label(
            text=text,
            size_hint_y=None,
            height=dp(24),
            font_size=sp(15),
            color=(0.94, 0.94, 0.96, 1),
            halign="left",
            valign="middle",
            **kwargs,
        )
        lbl.bind(size=lambda *a: setattr(lbl, "text_size", lbl.size))
        return lbl

    def _small_label(self, text: str, **kwargs) -> Label:
        lbl = Label(
            text=text,
            size_hint_y=None,
            height=dp(22),
            font_size=sp(13),
            color=(0.82, 0.86, 0.90, 1),
            halign="left",
            valign="middle",
            **kwargs,
        )
        lbl.bind(size=lambda *a: setattr(lbl, "text_size", lbl.size))
        return lbl

    def _framed_widget(self, title: str, widget: Widget) -> BoxLayout:
        box = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))
        with box.canvas.before:
            from kivy.graphics import Color, RoundedRectangle
            Color(0.11, 0.12, 0.15, 1)
            box._bg = RoundedRectangle(radius=[dp(12)] * 4, pos=box.pos, size=box.size)
        box.bind(pos=lambda inst, val: setattr(box._bg, "pos", val))
        box.bind(size=lambda inst, val: setattr(box._bg, "size", val))

        title_lbl = self._label(title, size_hint_y=None, height=dp(26))
        box.add_widget(title_lbl)
        box.add_widget(widget)
        return box

    # ---------------- génération ----------------

    def start_generation(self, *_):
        if self.running:
            return

        try:
            count = int(self.count_input.text.strip())
            if count <= 0:
                raise ValueError
        except Exception:
            self.log("Nombre d'images invalide.")
            return

        try:
            top_k = int(self.topk_input.text.strip())
            if top_k <= 0:
                raise ValueError
        except Exception:
            self.log("Nombre top_k invalide.")
            return

        self.current_output_dir = Path(self.output_input.text.strip() or DEFAULT_OUTPUT_DIR)
        self.current_output_dir.mkdir(parents=True, exist_ok=True)

        self.best_records.clear()
        self.stop_requested = False
        self.running = True
        self.progress_global.max = count
        self.progress_global.value = 0
        self.status("Génération en cours...", ok=True)
        self.log(f"Démarrage : {count} images validées à produire.")
        self.log(f"Dossier de sortie : {self.current_output_dir.resolve()}")

        self.worker_thread = threading.Thread(
            target=self._worker_generate,
            args=(count, top_k),
            daemon=True,
        )
        self.worker_thread.start()

    def stop_generation(self, *_):
        if not self.running:
            return
        self.stop_requested = True
        self.status("Arrêt demandé...", ok=False)
        self.log("Arrêt demandé par l'utilisateur.")

    def _worker_generate(self, target_count: int, top_k: int):
        rows = []
        total_attempts = 0

        try:
            for target_index in range(1, target_count + 1):
                local_attempt = 0

                while True:
                    if self.stop_requested:
                        self._finish_stopped(rows, top_k)
                        return

                    total_attempts += 1
                    local_attempt += 1

                    seed = 202603120000 + target_index * 100000 + local_attempt
                    profile = camo.make_profile(seed)

                    pil_img, rs, metrics = camo.generate_one_variant(profile)

                    extra_scores, valid_v3 = evaluate_candidate_v3(pil_img, rs, metrics)
                    full_valid = camo.variant_is_valid(rs, metrics) and valid_v3

                    silhouette_img = silhouette_projection_image(rgb_image_to_index_canvas(pil_img))

                    self.update_preview(pil_img, silhouette_img if self.silhouette_checkbox.active else pil_img)
                    self.update_attempt_status(target_index, local_attempt, rs, extra_scores)

                    if not full_valid:
                        self.log(
                            f"[img={target_index:03d} essai={local_attempt:04d}] rejeté | "
                            f"C={rs[camo.IDX_COYOTE]*100:.1f} "
                            f"O={rs[camo.IDX_OLIVE]*100:.1f} "
                            f"T={rs[camo.IDX_TERRE]*100:.1f} "
                            f"G={rs[camo.IDX_GRIS]*100:.1f} | "
                            f"SF={extra_scores['score_final']:.3f}"
                        )
                        continue

                    filename = self.current_output_dir / f"camouflage_{target_index:03d}.png"
                    pil_img.save(filename)

                    record = CandidateRecord(
                        index=target_index,
                        seed=profile.seed,
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
                        rs=rs.copy(),
                        metrics=dict(metrics),
                    )
                    self.best_records.append(record)
                    self.best_records.sort(key=lambda r: r.score_final, reverse=True)

                    rows.append({
                        "index": target_index,
                        "seed": profile.seed,
                        "attempts_for_this_image": local_attempt,
                        "global_attempt": total_attempts,
                        "coyote_brown_pct": round(float(rs[camo.IDX_COYOTE] * 100), 2),
                        "vert_olive_pct": round(float(rs[camo.IDX_OLIVE] * 100), 2),
                        "terre_de_france_pct": round(float(rs[camo.IDX_TERRE] * 100), 2),
                        "vert_de_gris_pct": round(float(rs[camo.IDX_GRIS] * 100), 2),
                        "score_final": round(extra_scores["score_final"], 5),
                        "score_ratio": round(extra_scores["score_ratio"], 5),
                        "score_silhouette": round(extra_scores["score_silhouette"], 5),
                        "score_contour": round(extra_scores["score_contour"], 5),
                        "score_main": round(extra_scores["score_main"], 5),
                        "silhouette_color_diversity": round(extra_scores["silhouette_color_diversity"], 5),
                        "contour_break_score": round(extra_scores["contour_break_score"], 5),
                        "outline_band_diversity": round(extra_scores["outline_band_diversity"], 5),
                        "small_scale_structural_score": round(extra_scores["small_scale_structural_score"], 5),
                        "largest_olive_component_ratio": round(metrics["largest_olive_component_ratio"], 5),
                        "largest_olive_component_ratio_small": round(metrics["largest_olive_component_ratio_small"], 5),
                        "olive_multizone_share": round(metrics["olive_multizone_share"], 5),
                        "center_empty_ratio": round(metrics["center_empty_ratio"], 5),
                        "center_empty_ratio_small": round(metrics["center_empty_ratio_small"], 5),
                        "boundary_density": round(metrics["boundary_density"], 5),
                        "boundary_density_small": round(metrics["boundary_density_small"], 5),
                        "boundary_density_tiny": round(metrics["boundary_density_tiny"], 5),
                        "mirror_similarity": round(metrics["mirror_similarity"], 5),
                        "oblique_share": round(metrics["oblique_share"], 5),
                        "vertical_share": round(metrics["vertical_share"], 5),
                        "angle_dominance_ratio": round(metrics["angle_dominance_ratio"], 5),
                        "olive_macro_share": round(metrics["vert_olive_macro_share"], 5),
                        "terre_transition_share": round(metrics["terre_de_france_transition_share"], 5),
                        "gris_micro_share": round(metrics["vert_de_gris_micro_share"], 5),
                        "gris_macro_share": round(metrics["vert_de_gris_macro_share"], 5),
                        "angles": " ".join(map(str, profile.allowed_angles)),
                    })

                    self.update_progress(target_index, target_count)
                    self.log(
                        f"[img={target_index:03d}] accepté -> {filename.name} | "
                        f"SF={extra_scores['score_final']:.3f} "
                        f"| silhouette={extra_scores['silhouette_color_diversity']:.3f} "
                        f"| contour={extra_scores['contour_break_score']:.3f}"
                    )

                    break

            self._finish_success(rows, top_k)

        except Exception as e:
            self._finish_error(str(e))

    # ---------------- fin de traitement ----------------

    def _write_report(self, rows: List[dict]) -> Path:
        report_path = self.current_output_dir / REPORT_NAME
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return report_path

    def _export_best_of(self, top_k: int) -> Path:
        best_dir = self.current_output_dir / BEST_DIR_NAME
        best_dir.mkdir(parents=True, exist_ok=True)

        # on vide d'abord les anciens exports png/jpg/csv
        for child in best_dir.iterdir():
            if child.is_file():
                try:
                    child.unlink()
                except Exception:
                    pass

        best_subset = self.best_records[:top_k]

        rows = []
        for rank, rec in enumerate(best_subset, start=1):
            dst = best_dir / f"best_{rank:03d}_camouflage_{rec.index:03d}.png"
            shutil.copy2(rec.image_path, dst)

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
            with (best_dir / "best_of.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        return best_dir

    def _finish_success(self, rows: List[dict], top_k: int):
        report_path = self._write_report(rows)
        best_dir = self._export_best_of(top_k)
        self.running = False
        self.stop_requested = False
        self.status("Terminé", ok=True)
        self.log(f"Rapport écrit : {report_path}")
        self.log(f"Best-of exporté : {best_dir}")
        self.log("Génération terminée avec succès.")

    def _finish_stopped(self, rows: List[dict], top_k: int):
        if rows:
            report_path = self._write_report(rows)
            best_dir = self._export_best_of(min(top_k, len(rows)))
            self.log(f"Rapport partiel écrit : {report_path}")
            self.log(f"Best-of partiel exporté : {best_dir}")
        self.running = False
        self.stop_requested = False
        self.status("Arrêté", ok=False)
        self.log("Génération arrêtée proprement.")

    def _finish_error(self, message: str):
        self.running = False
        self.stop_requested = False
        self.status("Erreur", ok=False)
        self.log(f"Erreur : {message}")

    # ---------------- UI thread updates ----------------

    @mainthread
    def status(self, text: str, ok: bool = True):
        self.status_label.text = text
        self.status_label.color = (0.78, 0.90, 0.78, 1) if ok else (0.95, 0.66, 0.66, 1)

    @mainthread
    def log(self, line: str):
        self.log_view.append(line)

    @mainthread
    def update_progress(self, current: int, total: int):
        self.progress_global.max = total
        self.progress_global.value = current

    @mainthread
    def update_attempt_status(self, image_idx: int, attempt_idx: int, rs: np.ndarray, extra_scores: Dict[str, float]):
        self.progress_attempt_label.text = f"Image {image_idx:03d} | essai {attempt_idx:04d}"
        self.stat_color.text = (
            f"C={rs[camo.IDX_COYOTE]*100:.2f}%  "
            f"O={rs[camo.IDX_OLIVE]*100:.2f}%  "
            f"T={rs[camo.IDX_TERRE]*100:.2f}%  "
            f"G={rs[camo.IDX_GRIS]*100:.2f}%"
        )
        self.stat_score.text = f"Score final = {extra_scores['score_final']:.3f}"
        self.stat_main.text = (
            f"Silhouette={extra_scores['score_silhouette']:.3f} | "
            f"Contour={extra_scores['score_contour']:.3f} | "
            f"Main={extra_scores['score_main']:.3f}"
        )
        self.stat_extra.text = (
            f"Contour break={extra_scores['contour_break_score']:.3f} | "
            f"Outline={extra_scores['outline_band_diversity']:.3f} | "
            f"Small-scale={extra_scores['small_scale_structural_score']:.3f}"
        )

    @mainthread
    def update_preview(self, pil_img: PILImage.Image, silhouette_img: PILImage.Image):
        self.current_preview_img = pil_img
        self.current_silhouette_preview = silhouette_img
        self.preview_img.texture = pil_to_coreimage(pil_img).texture
        self.preview_silhouette.texture = pil_to_coreimage(silhouette_img).texture


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    CamouflageApp().run()