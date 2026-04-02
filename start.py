# -*- coding: utf-8 -*-
"""
start.py
Front-end Kivy réaligné avec le main.py actuel.

Corrections principales :
- supprime les appels à des fonctions absentes dans main.py ;
- remplace les anciens scores "visuels" par les vraies métriques strictes du backend ;
- garde la génération séquentielle stricte avec aperçu live ;
- ajoute une analyse locale des règles de rejet à partir des seuils de main.py ;
- conserve l'export rapport + best_of côté front ;
- intègre une projection mannequin plus robuste et physiquement cohérente
  avec la géométrie du main.py.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import json
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

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageFilter

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
    import main as camo
except Exception as exc:
    raise RuntimeError("Impossible d'importer main.py depuis start.py") from exc

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
DEFAULT_OUTPUT_DIR = Path(getattr(camo, "OUTPUT_DIR", "camouflages_federale_europe_8k"))
DEFAULT_TARGET_COUNT = int(getattr(camo, "N_VARIANTS_REQUIRED", 100))
DEFAULT_TOP_K = 20
REPORT_NAME = "rapport_textures.csv" if hasattr(camo, "validate_with_reasons") else "rapport_camouflages.csv"
BEST_DIR_NAME = "best_of"
MANNEQUIN_DIR_NAME = "mannequin_previews"

RUN_MODE_BLOCKING = "blocking"
RUN_MODE_NON_BLOCKING = "non_blocking"
RUN_MODE_SKIP_TESTS = "skip_tests"

THUMB_SIZE = (240, 150)
GALLERY_COLUMNS = 3
MAX_GALLERY_ITEMS = 24
SCRIPT_DIR = Path(__file__).resolve().parent
SOLDIER_MODEL_BASENAMES = ("soldat_modele_vert.png", "1774949910078.png", "soldat_modele.png")

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


DEFAULT_UI_MOTIF_SCALE = float(getattr(camo, "DEFAULT_MOTIF_SCALE", 0.58))
DEFAULT_BACKEND_MACHINE_INTENSITY = float(getattr(camo, "DEFAULT_MACHINE_INTENSITY", 0.98))


BACKEND_IDX_0 = getattr(camo, "IDX_0", getattr(camo, "IDX_COYOTE", 0))
BACKEND_IDX_1 = getattr(camo, "IDX_1", getattr(camo, "IDX_OLIVE", 1))
BACKEND_IDX_2 = getattr(camo, "IDX_2", getattr(camo, "IDX_TERRE", 2))
BACKEND_IDX_3 = getattr(camo, "IDX_3", getattr(camo, "IDX_GRIS", 3))
BACKEND_CLASS_LABELS = list(getattr(camo, "CLASS_NAMES", ["class_0", "class_1", "class_2", "class_3"]))
PRIMARY_COMPONENT_METRIC = "largest_component_ratio_class_1" if hasattr(camo, "validate_with_reasons") else "largest_olive_component_ratio"


def palette_map() -> Dict[Tuple[int, int, int], int]:
    return {
        tuple(camo.RGB[BACKEND_IDX_0].tolist()): BACKEND_IDX_0,
        tuple(camo.RGB[BACKEND_IDX_1].tolist()): BACKEND_IDX_1,
        tuple(camo.RGB[BACKEND_IDX_2].tolist()): BACKEND_IDX_2,
        tuple(camo.RGB[BACKEND_IDX_3].tolist()): BACKEND_IDX_3,
    }


PALETTE_TO_INDEX = palette_map()


def rgb_image_to_index_canvas(img: PILImage.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    for rgb, idx in PALETTE_TO_INDEX.items():
        mask = np.all(arr == np.array(rgb, dtype=np.uint8), axis=-1)
        out[mask] = idx
    return out


def backend_machine_intensity(percent_value: float) -> float:
    return max(0.10, min(1.00, float(percent_value) / 100.0))


def safe_metric(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def extract_backend_scores(ratios: np.ndarray, metrics: Dict[str, float]) -> Dict[str, float]:
    abs_err = np.abs(ratios - camo.TARGET)
    return {
        "ratio_mae": float(np.mean(abs_err)),
        "ratio_max_abs": float(np.max(abs_err)),
        "primary_component_ratio": safe_metric(metrics, PRIMARY_COMPONENT_METRIC),
        "boundary_density": safe_metric(metrics, "boundary_density"),
        "boundary_density_small": safe_metric(metrics, "boundary_density_small"),
        "boundary_density_tiny": safe_metric(metrics, "boundary_density_tiny"),
        "mirror_similarity": safe_metric(metrics, "mirror_similarity"),
        "edge_contact_ratio": safe_metric(metrics, "edge_contact_ratio"),
        "overscan": safe_metric(metrics, "overscan"),
        "shift_strength": safe_metric(metrics, "shift_strength"),
        "px_per_cm": safe_metric(metrics, "px_per_cm"),
        "bestof_score": safe_metric(metrics, "bestof_score"),
    }


def rejection_rules_for_candidate(candidate: camo.CandidateResult, outcome: Optional[Any] = None) -> List[str]:
    if outcome is None:
        validator = getattr(camo, "validate_with_reasons", None)
        if callable(validator):
            try:
                outcome = validator(candidate)
            except Exception:
                outcome = None

    if outcome is not None and hasattr(outcome, "reasons"):
        reasons = list(getattr(outcome, "reasons", []) or [])
        if reasons:
            return reasons
        if hasattr(outcome, "accepted") and not bool(getattr(outcome, "accepted")):
            return ["rejet_backend_non_detaille"]
        return []

    ratios = candidate.ratios
    metrics = candidate.metrics
    rules: List[str] = []

    abs_err = np.abs(ratios - camo.TARGET)
    for idx, err in enumerate(abs_err):
        if float(err) > float(camo.MAX_ABS_ERROR_PER_COLOR[idx]):
            rules.append(f"ratio_class_{idx}_abs_error")

    if float(np.mean(abs_err)) > float(camo.MAX_MEAN_ABS_ERROR):
        rules.append("mean_abs_error")

    bd = safe_metric(metrics, "boundary_density")
    if bd < float(camo.MIN_BOUNDARY_DENSITY):
        rules.append("boundary_density_trop_faible")
    elif bd > float(camo.MAX_BOUNDARY_DENSITY):
        rules.append("boundary_density_trop_forte")

    bd_small = safe_metric(metrics, "boundary_density_small")
    if bd_small < float(camo.MIN_BOUNDARY_DENSITY_SMALL):
        rules.append("boundary_density_small_trop_faible")
    elif bd_small > float(camo.MAX_BOUNDARY_DENSITY_SMALL):
        rules.append("boundary_density_small_trop_forte")

    bd_tiny = safe_metric(metrics, "boundary_density_tiny")
    if bd_tiny < float(camo.MIN_BOUNDARY_DENSITY_TINY):
        rules.append("boundary_density_tiny_trop_faible")
    elif bd_tiny > float(camo.MAX_BOUNDARY_DENSITY_TINY):
        rules.append("boundary_density_tiny_trop_forte")

    mirror = safe_metric(metrics, "mirror_similarity")
    if mirror > float(camo.MAX_MIRROR_SIMILARITY):
        rules.append("mirror_similarity")

    primary = safe_metric(metrics, PRIMARY_COMPONENT_METRIC)
    threshold_name = "MIN_LARGEST_COMPONENT_RATIO_CLASS_1" if hasattr(camo, "MIN_LARGEST_COMPONENT_RATIO_CLASS_1") else "MIN_LARGEST_OLIVE_COMPONENT_RATIO"
    threshold = float(getattr(camo, threshold_name, 0.08))
    if primary < threshold:
        rules.append(PRIMARY_COMPONENT_METRIC)

    edge = safe_metric(metrics, "edge_contact_ratio")
    if edge > float(camo.MAX_EDGE_CONTACT_RATIO):
        rules.append("edge_contact_ratio")

    return rules


def candidate_rank_key(record: CandidateRecord) -> Tuple[float, float, float, float]:
    ratios = record.ratios
    metrics = record.metrics
    ratio_mae = float(np.mean(np.abs(ratios - camo.TARGET)))
    ratio_max = float(np.max(np.abs(ratios - camo.TARGET)))
    mirror = safe_metric(metrics, "mirror_similarity")
    edge = safe_metric(metrics, "edge_contact_ratio")
    return (ratio_mae, ratio_max, mirror, edge)


async def async_generate_candidate_from_seed(seed: int) -> camo.CandidateResult:
    return await asyncio.to_thread(camo.generate_candidate_from_seed, seed)


async def async_validate_candidate_result(candidate: camo.CandidateResult) -> Any:
    validator = getattr(camo, "validate_with_reasons", None)
    if callable(validator):
        return await asyncio.to_thread(validator, candidate)

    simple_validator = getattr(camo, "validate_candidate_result", None)
    if callable(simple_validator):
        accepted = await asyncio.to_thread(simple_validator, candidate)
        return accepted

    raise RuntimeError("Aucune fonction de validation compatible trouvée dans main.py")


async def async_save_candidate_image(candidate: camo.CandidateResult, path: Path) -> Path:
    return await asyncio.to_thread(camo.save_candidate_image, candidate, path)


async def async_write_report(rows: List[dict], output_dir: Path, filename: str = REPORT_NAME) -> Path:
    return await asyncio.to_thread(camo.write_report, rows, output_dir, filename)


def save_mannequin_projection_sync(projection_img: PILImage.Image, saved_camo_path: Path, output_dir: Path) -> Path:
    mannequin_dir = Path(output_dir) / MANNEQUIN_DIR_NAME
    mannequin_dir.mkdir(parents=True, exist_ok=True)
    out_path = mannequin_dir / f"{Path(saved_camo_path).stem}__mannequin.png"
    projection_img.save(out_path, format="PNG")
    return out_path


async def async_save_mannequin_projection(projection_img: PILImage.Image, saved_camo_path: Path, output_dir: Path) -> Path:
    return await asyncio.to_thread(save_mannequin_projection_sync, projection_img, saved_camo_path, output_dir)


def build_backend_compatible_output_path(
    output_dir: Path,
    target_index: int,
    local_attempt: int,
    global_attempt: int,
    candidate: camo.CandidateResult,
) -> Path:
    for builder_name in ("build_unique_pattern_path", "build_unique_camo_path"):
        builder = getattr(camo, builder_name, None)
        if callable(builder):
            try:
                return Path(builder(
                    output_dir=output_dir,
                    target_index=target_index,
                    seed=int(candidate.seed),
                    local_attempt=local_attempt,
                    global_attempt=global_attempt,
                ))
            except Exception:
                pass

    prefix = "pattern" if hasattr(camo, "validate_with_reasons") else "camouflage"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nano = time.time_ns() % 1_000_000_000
    return output_dir / (
        f"{prefix}_{int(target_index):03d}"
        f"_s{int(candidate.seed)}"
        f"_a{int(local_attempt):04d}"
        f"_g{int(global_attempt):06d}"
        f"_{timestamp}_{nano:09d}.png"
    )


def build_candidate_row_compatible(
    target_index: int,
    local_attempt: int,
    global_attempt: int,
    candidate: camo.CandidateResult,
    outcome: Optional[Any],
    saved_path: Path,
) -> Dict[str, Any]:
    row_builder = getattr(camo, "candidate_row", None)
    if callable(row_builder):
        try:
            return row_builder(
                target_index,
                local_attempt,
                global_attempt,
                candidate,
                outcome,
                image_name=saved_path.name,
                image_path=str(saved_path),
            )
        except TypeError:
            try:
                return row_builder(
                    target_index,
                    local_attempt,
                    global_attempt,
                    candidate,
                    image_name=saved_path.name,
                    image_path=str(saved_path),
                )
            except TypeError:
                row = row_builder(target_index, local_attempt, global_attempt, candidate)
                if isinstance(row, dict):
                    row["image_name"] = saved_path.name
                    row["image_path"] = str(saved_path)
                    if outcome is not None and hasattr(outcome, "bestof_score"):
                        row.setdefault("bestof_score", float(getattr(outcome, "bestof_score", 0.0)))
                    if outcome is not None and hasattr(outcome, "reasons"):
                        row.setdefault("reasons", "|".join(getattr(outcome, "reasons", []) or []))
                    if outcome is not None and hasattr(outcome, "accepted"):
                        row.setdefault("accepted", int(bool(getattr(outcome, "accepted"))))
                    return row
    return {
        "index": target_index,
        "seed": int(candidate.seed),
        "attempts_for_this_image": local_attempt,
        "global_attempt": global_attempt,
        "image_name": saved_path.name,
        "image_path": str(saved_path),
        "bestof_score": float(getattr(outcome, "bestof_score", 0.0)) if outcome is not None else 0.0,
        "accepted": int(bool(getattr(outcome, "accepted", False))) if outcome is not None else 0,
        "reasons": "|".join(getattr(outcome, "reasons", []) or []) if outcome is not None else "",
    }


# ============================================================
# PROJECTION SUR SOLDAT MODÈLE (aperçu uniquement)
# ============================================================

@dataclass
class ProjectionConfig:
    # Détection initiale large du vert.
    hue_min: int = 35
    hue_max: int = 95
    sat_min: int = 20
    val_min: int = 20
    sat_max: int = 255
    val_max: int = 190

    # Protection des zones sombres non à repeindre.
    dark_val_max: int = 70
    dark_sat_max: int = 120

    # Morphologie.
    open_kernel: int = 3
    close_kernel: int = 7
    blur_radius: int = 7

    # Répartition verticale du vêtement.
    hat_ratio: float = 0.13
    jacket_ratio: float = 0.41
    pants_ratio: float = 0.46

    # Approximation physique du mannequin visible.
    uniform_visible_height_cm: float = 160.0

    # Adaptation physique quand la source est un camouflage plein format.
    hat_scale_multiplier: float = 0.92
    jacket_scale_multiplier: float = 1.00
    pants_scale_multiplier: float = 1.08

    # Fallback quand la source est un petit tile.
    tile_hat_width_ratio: float = 0.32
    tile_jacket_width_ratio: float = 0.25
    tile_pants_width_ratio: float = 0.22

    # Bornes globales de scale.
    min_region_scale: float = 0.16
    max_region_scale: float = 1.55

    # Composition.
    shadow_strength: float = 0.95
    detail_strength: float = 0.42
    edge_darkening: float = 0.25
    alpha_gamma: float = 1.0


PROJECTION_CFG = ProjectionConfig()
_PROJECTION_SUBJECT_CACHE: Optional[np.ndarray] = None
_PROJECTION_SUBJECT_PATH: Optional[Path] = None
_PROJECTION_ANALYSIS_CACHE: Optional["ProjectionSubjectAnalysis"] = None
_PROJECTION_ANALYSIS_PATH: Optional[Path] = None


@dataclass
class ProjectionSubjectAnalysis:
    model_path: Path
    subject_bgr: np.ndarray
    uniform_mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    regions: Dict[str, np.ndarray]
    mannequin_px_per_cm: float


def ensure_odd(v: int) -> int:
    return v if int(v) % 2 == 1 else int(v) + 1


def pil_rgb_to_bgr(img: PILImage.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil_rgb(bgr: np.ndarray) -> PILImage.Image:
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


def read_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path}")
    return img


def read_pil_rgb(path: str | Path) -> PILImage.Image:
    with PILImage.open(path) as img:
        return img.convert("RGB")


def resolve_soldier_model_path() -> Path:
    env_value = os.getenv("CAMO_SOLDIER_MODEL", "").strip()
    candidates: List[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())

    for name in SOLDIER_MODEL_BASENAMES:
        candidates.extend([
            SCRIPT_DIR / name,
            Path.cwd() / name,
            SCRIPT_DIR / "images" / name,
            SCRIPT_DIR / "assets" / name,
            Path.cwd() / "images" / name,
            Path.cwd() / "assets" / name,
        ])

    candidates.append(Path("/mnt/data/1774949910078.png"))

    seen = set()
    uniq: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)

    for candidate in uniq:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n - ".join(str(p) for p in uniq)
    raise FileNotFoundError(
        "Image modèle introuvable. Place '1774949910078.png' ou 'soldat_modele_vert.png' à côté de start.py, "
        "ou définis la variable CAMO_SOLDIER_MODEL. Emplacements testés:\n - " + searched
    )


def get_projection_subject_bgr() -> np.ndarray:
    global _PROJECTION_SUBJECT_CACHE, _PROJECTION_SUBJECT_PATH
    model_path = resolve_soldier_model_path()
    if _PROJECTION_SUBJECT_CACHE is None or _PROJECTION_SUBJECT_PATH != model_path:
        _PROJECTION_SUBJECT_PATH = model_path
        _PROJECTION_SUBJECT_CACHE = read_bgr(model_path)
    return _PROJECTION_SUBJECT_CACHE.copy()


def white_background_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 190)).astype(np.uint8) * 255
    white = cv2.GaussianBlur(white, (0, 0), 3)
    return white


def person_mask_from_foreground(bgr: np.ndarray) -> np.ndarray:
    white = white_background_mask(bgr)
    fg = 255 - white
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(
        fg,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return fg
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    out = np.zeros_like(fg)
    out[labels == largest] = 255
    out = cv2.GaussianBlur(out, (0, 0), 2)
    return out


def largest_component_mask(mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        zero = np.zeros_like(mask)
        return zero, (0, 0, 0, 0), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))

    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    area = int(stats[best, cv2.CC_STAT_AREA])

    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out, (x, y, w, h), area


def refine_green_thresholds(subject_bgr: np.ndarray, broad_mask: np.ndarray, cfg: ProjectionConfig) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[broad_mask > 0]
    if px.size == 0:
        lower = np.array([cfg.hue_min, cfg.sat_min, cfg.val_min], dtype=np.uint8)
        upper = np.array([cfg.hue_max, cfg.sat_max, cfg.val_max], dtype=np.uint8)
        return lower, upper

    h_vals = px[:, 0].astype(np.float32)
    s_vals = px[:, 1].astype(np.float32)
    v_vals = px[:, 2].astype(np.float32)

    lower = np.array([
        max(cfg.hue_min, int(np.quantile(h_vals, 0.02)) - 4),
        max(18, int(np.quantile(s_vals, 0.03)) - 10),
        max(18, int(np.quantile(v_vals, 0.03)) - 10),
    ], dtype=np.uint8)

    upper = np.array([
        min(cfg.hue_max, int(np.quantile(h_vals, 0.98)) + 4),
        255,
        min(cfg.val_max, int(np.quantile(v_vals, 0.98)) + 25),
    ], dtype=np.uint8)

    return lower, upper


def green_uniform_mask(subject_bgr: np.ndarray, cfg: ProjectionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2HSV)
    person = person_mask_from_foreground(subject_bgr)

    lower0 = np.array([cfg.hue_min, cfg.sat_min, cfg.val_min], dtype=np.uint8)
    upper0 = np.array([cfg.hue_max, cfg.sat_max, cfg.val_max], dtype=np.uint8)

    broad = cv2.inRange(hsv, lower0, upper0)
    broad = cv2.bitwise_and(broad, person)

    dark = (((hsv[:, :, 2] <= cfg.dark_val_max) & (hsv[:, :, 1] <= cfg.dark_sat_max))).astype(np.uint8) * 255
    broad = cv2.bitwise_and(broad, 255 - dark)

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))

    broad = cv2.morphologyEx(broad, cv2.MORPH_OPEN, open_k)
    broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, close_k)

    broad_main, _bbox, area = largest_component_mask(broad)
    if area <= 0:
        blur_k = ensure_odd(cfg.blur_radius)
        return cv2.GaussianBlur(broad, (blur_k, blur_k), 0)

    lower, upper = refine_green_thresholds(subject_bgr, broad_main, cfg)

    refined = cv2.inRange(hsv, lower, upper)
    refined = cv2.bitwise_and(refined, person)
    refined = cv2.bitwise_and(refined, 255 - dark)

    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_k)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_k)

    refined_main, _bbox2, _area2 = largest_component_mask(refined)

    blur_k = ensure_odd(cfg.blur_radius)
    refined_main = cv2.GaussianBlur(refined_main, (blur_k, blur_k), 0)
    return refined_main


def split_uniform_regions(mask: np.ndarray, cfg: ProjectionConfig) -> Dict[str, np.ndarray]:
    yy, xx = np.where(mask > 32)
    if len(xx) == 0:
        zero = np.zeros_like(mask)
        return {"hat": zero, "jacket": zero, "pants": zero}

    x1, x2 = int(np.min(xx)), int(np.max(xx))
    y1, y2 = int(np.min(yy)), int(np.max(yy))
    total_h = max(1, y2 - y1)

    hat_y2 = y1 + int(total_h * cfg.hat_ratio)
    jacket_y2 = hat_y2 + int(total_h * cfg.jacket_ratio)

    hat = np.zeros_like(mask)
    jacket = np.zeros_like(mask)
    pants = np.zeros_like(mask)

    hat[y1:hat_y2, x1:x2 + 1] = mask[y1:hat_y2, x1:x2 + 1]
    jacket[hat_y2:jacket_y2, x1:x2 + 1] = mask[hat_y2:jacket_y2, x1:x2 + 1]
    pants[jacket_y2:y2 + 1, x1:x2 + 1] = mask[jacket_y2:y2 + 1, x1:x2 + 1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hat = cv2.GaussianBlur(cv2.morphologyEx(hat, cv2.MORPH_OPEN, kernel), (0, 0), 1.0)
    jacket = cv2.GaussianBlur(cv2.morphologyEx(jacket, cv2.MORPH_OPEN, kernel), (0, 0), 1.0)
    pants = cv2.GaussianBlur(cv2.morphologyEx(pants, cv2.MORPH_OPEN, kernel), (0, 0), 1.0)

    return {"hat": hat, "jacket": jacket, "pants": pants}


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    yy, xx = np.where(mask > 32)
    if len(xx) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(np.min(xx)), int(np.max(xx))
    y1, y2 = int(np.min(yy)), int(np.max(yy))
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def build_projection_subject_analysis(cfg: ProjectionConfig = PROJECTION_CFG) -> ProjectionSubjectAnalysis:
    model_path = resolve_soldier_model_path()
    subject_bgr = get_projection_subject_bgr()
    uniform_mask = green_uniform_mask(subject_bgr, cfg)
    bbox = bbox_from_mask(uniform_mask)
    regions = split_uniform_regions(uniform_mask, cfg)

    _, _, _, bbox_h = bbox
    mannequin_px_per_cm = (bbox_h / max(1.0, cfg.uniform_visible_height_cm)) if bbox_h > 0 else 1.0

    return ProjectionSubjectAnalysis(
        model_path=model_path,
        subject_bgr=subject_bgr,
        uniform_mask=uniform_mask,
        bbox=bbox,
        regions=regions,
        mannequin_px_per_cm=float(max(0.01, mannequin_px_per_cm)),
    )


def get_projection_subject_analysis(cfg: ProjectionConfig = PROJECTION_CFG) -> ProjectionSubjectAnalysis:
    global _PROJECTION_ANALYSIS_CACHE, _PROJECTION_ANALYSIS_PATH
    model_path = resolve_soldier_model_path()
    if _PROJECTION_ANALYSIS_CACHE is None or _PROJECTION_ANALYSIS_PATH != model_path:
        _PROJECTION_ANALYSIS_CACHE = build_projection_subject_analysis(cfg)
        _PROJECTION_ANALYSIS_PATH = model_path
    return _PROJECTION_ANALYSIS_CACHE


def is_full_camo_canvas(camo_bgr: np.ndarray) -> bool:
    h, w = camo_bgr.shape[:2]
    expected_w = int(getattr(camo, "WIDTH", 0))
    expected_h = int(getattr(camo, "HEIGHT", 0))

    if expected_w > 0 and expected_h > 0:
        ar_src = w / max(1, h)
        ar_ref = expected_w / max(1, expected_h)
        if w >= min(2048, expected_w // 2) and h >= min(1024, expected_h // 2):
            if abs(ar_src - ar_ref) < 0.20:
                return True

    return w >= 2048 and h >= 1024


def estimate_camo_px_per_cm(camo_bgr: np.ndarray) -> float:
    h, w = camo_bgr.shape[:2]

    physical_w = float(getattr(camo, "PHYSICAL_WIDTH_CM", max(1, w)))
    physical_h = float(getattr(camo, "PHYSICAL_HEIGHT_CM", max(1, h)))

    ppcm_x = w / max(1.0, physical_w)
    ppcm_y = h / max(1.0, physical_h)

    fallback = float(getattr(camo, "PX_PER_CM", max(1.0, min(ppcm_x, ppcm_y))))
    ppcm = min(ppcm_x, ppcm_y)
    return float(ppcm if ppcm > 0 else fallback)


def adaptive_region_scale(
    region_name: str,
    region_mask: np.ndarray,
    camo_bgr: np.ndarray,
    analysis: ProjectionSubjectAnalysis,
    cfg: ProjectionConfig,
) -> float:
    _rx, _ry, rw, rh = bbox_from_mask(region_mask)
    if rw <= 0 or rh <= 0:
        return 1.0

    if is_full_camo_canvas(camo_bgr):
        mannequin_ppcm = float(analysis.mannequin_px_per_cm)
        camo_ppcm = float(max(1e-6, estimate_camo_px_per_cm(camo_bgr)))

        base_scale = mannequin_ppcm / camo_ppcm

        if region_name == "hat":
            scale = base_scale * cfg.hat_scale_multiplier
        elif region_name == "pants":
            scale = base_scale * cfg.pants_scale_multiplier
        else:
            scale = base_scale * cfg.jacket_scale_multiplier

        _ux, _uy, uniform_w, _uh = analysis.bbox
        if uniform_w > 0:
            width_ratio = rw / max(1.0, uniform_w)
            scale *= np.clip(0.85 + width_ratio * 0.75, 0.80, 1.18)

        return float(np.clip(scale, cfg.min_region_scale, cfg.max_region_scale))

    _camo_h, camo_w = camo_bgr.shape[:2]
    if region_name == "hat":
        target_w = rw * cfg.tile_hat_width_ratio
    elif region_name == "pants":
        target_w = rw * cfg.tile_pants_width_ratio
    else:
        target_w = rw * cfg.tile_jacket_width_ratio

    scale = target_w / max(1.0, float(camo_w))
    return float(np.clip(scale, cfg.min_region_scale, cfg.max_region_scale))


def tile_camo(camo_bgr: np.ndarray, shape_hw: Tuple[int, int], scale: float, seed: int) -> np.ndarray:
    h, w = shape_hw
    ch, cw = camo_bgr.shape[:2]

    scale = max(0.05, float(scale))
    nw = max(32, int(round(cw * scale)))
    nh = max(32, int(round(ch * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    camo_resized = cv2.resize(camo_bgr, (nw, nh), interpolation=interp)

    rep_x = int(np.ceil(w / nw)) + 2
    rep_y = int(np.ceil(h / nh)) + 2
    tiled = np.tile(camo_resized, (rep_y, rep_x, 1))

    ox = (seed * 37 + w * 3) % nw
    oy = (seed * 53 + h * 5) % nh
    return tiled[oy:oy + h, ox:ox + w]


def shading_detail_edges(subject_bgr: np.ndarray, alpha_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha = alpha_mask.astype(np.float32) / 255.0

    soft = cv2.GaussianBlur(gray, (0, 0), sigmaX=23, sigmaY=23)
    soft = np.clip(soft, 1e-4, 1.0)
    shading = np.clip(gray / soft, 0.55, 1.50)

    detail = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=4, sigmaY=4)
    detail = np.clip(detail, -0.25, 0.25)

    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 110).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=2, sigmaY=2)
    edges *= alpha
    return shading, detail, edges


def compose_region(base_bgr: np.ndarray, region_mask: np.ndarray, camo_bgr: np.ndarray, scale: float, seed: int, cfg: ProjectionConfig) -> np.ndarray:
    alpha = (region_mask.astype(np.float32) / 255.0) ** max(0.2, cfg.alpha_gamma)
    alpha = alpha[..., None]

    tiled = tile_camo(camo_bgr, base_bgr.shape[:2], scale=scale, seed=seed).astype(np.float32) / 255.0
    shading, detail, edges = shading_detail_edges(base_bgr, region_mask)

    camo_layer = tiled.copy()
    camo_layer *= (1.0 + (shading[..., None] - 1.0) * cfg.shadow_strength)
    camo_layer += detail[..., None] * cfg.detail_strength
    camo_layer *= (1.0 - edges[..., None] * cfg.edge_darkening)
    camo_layer = np.clip(camo_layer, 0.0, 1.0)

    base = base_bgr.astype(np.float32) / 255.0
    out = base * (1.0 - alpha) + camo_layer * alpha
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def apply_camo_to_reference(subject_bgr: np.ndarray, camo_bgr: np.ndarray, cfg: ProjectionConfig = PROJECTION_CFG) -> Tuple[np.ndarray, np.ndarray]:
    analysis = get_projection_subject_analysis(cfg)
    mask = analysis.uniform_mask
    regions = analysis.regions

    hat_scale = adaptive_region_scale("hat", regions["hat"], camo_bgr, analysis, cfg)
    jacket_scale = adaptive_region_scale("jacket", regions["jacket"], camo_bgr, analysis, cfg)
    pants_scale = adaptive_region_scale("pants", regions["pants"], camo_bgr, analysis, cfg)

    out = subject_bgr.copy()
    out = compose_region(out, regions["hat"], camo_bgr, hat_scale, seed=11, cfg=cfg)
    out = compose_region(out, regions["jacket"], camo_bgr, jacket_scale, seed=29, cfg=cfg)
    out = compose_region(out, regions["pants"], camo_bgr, pants_scale, seed=47, cfg=cfg)
    return out, mask


def projection_preview_image(camo_img: PILImage.Image, cfg: ProjectionConfig = PROJECTION_CFG) -> PILImage.Image:
    analysis = get_projection_subject_analysis(cfg)
    subject_bgr = analysis.subject_bgr.copy()
    camo_bgr = pil_rgb_to_bgr(camo_img)
    projected_bgr, _mask = apply_camo_to_reference(subject_bgr, camo_bgr, cfg=cfg)
    return bgr_to_pil_rgb(projected_bgr)


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

    def set_thumbnail_pil(self, pil_img: PILImage.Image):
        try:
            self.thumb.texture = pil_to_coreimage(pil_img).texture
        except Exception:
            pass

    def load_thumbnail(self):
        try:
            img = read_pil_rgb(self.image_path)
            self.set_thumbnail_pil(make_thumbnail(img, THUMB_SIZE))
            self.app_ref.request_gallery_projection(self.image_path, self)
        except Exception:
            pass

    def _open_preview(self, *_):
        try:
            pil_img = read_pil_rgb(self.image_path)
            self.app_ref.update_preview(pil_img, pil_img)
            self.app_ref.request_preview_projection(self.image_path, pil_img)
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
        self.machine_intensity = DEFAULT_BACKEND_MACHINE_INTENSITY * 100.0
        self.motif_scale = DEFAULT_UI_MOTIF_SCALE
        self.process = psutil.Process() if psutil else None
        self.tests_ran = False
        self.tests_ok = False
        self.tests_summary = "Préflight non lancé."
        self.run_mode = RUN_MODE_BLOCKING
        self.diag_total = 0
        self.diag_accepts = 0
        self.diag_rejects = 0
        self.diag_rule_counter: Counter[str] = Counter()
        self.diag_last_rules: List[str] = []
        self._runtime_subscription_active = False
        self._runtime_subscriber_callback = None
        self.gallery_projection_cache: Dict[str, PILImage.Image] = {}
        self.gallery_projection_pending: Dict[str, List["GalleryThumb"]] = {}
        self.preview_projection_cache: Dict[str, PILImage.Image] = {}

        self.status_label: Optional[Label] = None
        self.attempt_text: Optional[Label] = None
        self.count_input: Optional[SoftTextInput] = None
        self.start_btn: Optional[SoftButton] = None
        self.stop_btn: Optional[SoftButton] = None
        self.open_btn: Optional[SoftButton] = None
        self.progress_bar: Optional[GlassProgressBar] = None
        self.progress_text: Optional[Label] = None
        self.motif_scale_slider: Optional[Slider] = None
        self.motif_scale_label: Optional[Label] = None
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

        controls.add_widget(self._label("Scale des motifs"))
        motif_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8))
        self.motif_scale_slider = Slider(min=0.35, max=1.20, value=DEFAULT_UI_MOTIF_SCALE, step=0.01)
        self.motif_scale_label = self._small_label(f"{DEFAULT_UI_MOTIF_SCALE:.2f}", size_hint_x=0.22)
        self.motif_scale_slider.bind(value=self._on_motif_scale_change)
        motif_row.add_widget(self.motif_scale_slider)
        motif_row.add_widget(self.motif_scale_label)
        controls.add_widget(motif_row)

        controls.add_widget(self._label("Monitoring"))
        self.resource_text = self._small_label("CPU -- | RAM -- | Disque -- | Processus -- | scale --")
        controls.add_widget(self.resource_text)

        controls.add_widget(self._label("Validation backend"))
        self.score_text = self._small_label("MAE ratio -- | max abs -- | olive comp. -- | miroir --")
        self.color_text = self._small_label("C -- | O -- | T -- | G --")
        self.extra_text = self._small_label("bd -- | bd/4 -- | bd/8 -- | bord --")
        self.struct_text = self._small_label("overscan -- | shift -- | px/cm --")
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
        previews.add_widget(self._carded_view("Projection sur soldat modèle", self.preview_silhouette))

        live_card = GlassCard(orientation="vertical")
        live_card.add_widget(self._label("Suivi direct de construction"))
        self.live_stage_label = self._small_label("Étape : attente")
        self.live_counts_label = self._small_label("État backend strict")
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
        self._refresh_diag_labels()
        Clock.schedule_interval(self._update_resource_monitor, 1.0)
        Clock.schedule_once(lambda dt: self.reload_gallery(), 0.2)
        return root

    # ---------- helpers UI ----------
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

    def _projection_cache_key(self, image_path: Path) -> str:
        try:
            stat = image_path.stat()
            return f"{image_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
        except Exception:
            return str(image_path.resolve())

    def request_gallery_projection(self, image_path: Path, thumb_widget: "GalleryThumb"):
        key = self._projection_cache_key(image_path)
        cached = self.gallery_projection_cache.get(key)
        if cached is not None:
            thumb_widget.set_thumbnail_pil(cached)
            return
        waiters = self.gallery_projection_pending.setdefault(key, [])
        waiters.append(thumb_widget)
        if len(waiters) > 1:
            return
        fut = self.async_runner.submit(self._async_build_gallery_projection(image_path))
        fut.add_done_callback(lambda f, k=key: self._on_gallery_projection_done(k, f))

    async def _async_build_gallery_projection(self, image_path: Path) -> PILImage.Image:
        pil_img = await asyncio.to_thread(read_pil_rgb, image_path)
        await asyncio.to_thread(get_projection_subject_analysis, PROJECTION_CFG)
        projected = await asyncio.to_thread(projection_preview_image, pil_img)
        return await asyncio.to_thread(make_thumbnail, projected, THUMB_SIZE)

    def _on_gallery_projection_done(self, key: str, fut: Future):
        try:
            thumb_img = fut.result()
        except Exception as exc:
            thumb_img = None
            self.log(f"Projection galerie impossible : {exc}")
        Clock.schedule_once(lambda dt: self._apply_gallery_projection_done(key, thumb_img), 0)

    @mainthread
    def _apply_gallery_projection_done(self, key: str, thumb_img: Optional[PILImage.Image]):
        waiters = self.gallery_projection_pending.pop(key, [])
        if thumb_img is None:
            return
        self.gallery_projection_cache[key] = thumb_img
        for thumb in waiters:
            try:
                thumb.set_thumbnail_pil(thumb_img)
            except Exception:
                pass

    def request_preview_projection(self, image_path: Path, raw_img: PILImage.Image):
        key = self._projection_cache_key(image_path)
        cached = self.preview_projection_cache.get(key)
        if cached is not None:
            self.update_preview(raw_img, cached)
            return
        fut = self.async_runner.submit(self._async_build_preview_projection(image_path))
        fut.add_done_callback(lambda f, k=key, img=raw_img.copy(): self._on_preview_projection_done(k, img, f))

    async def _async_build_preview_projection(self, image_path: Path) -> PILImage.Image:
        pil_img = await asyncio.to_thread(read_pil_rgb, image_path)
        await asyncio.to_thread(get_projection_subject_analysis, PROJECTION_CFG)
        return await asyncio.to_thread(projection_preview_image, pil_img)

    def _on_preview_projection_done(self, key: str, raw_img: PILImage.Image, fut: Future):
        try:
            projected = fut.result()
        except Exception as exc:
            self.log(f"Projection aperçu impossible : {exc}")
            return
        Clock.schedule_once(lambda dt, k=key, r=raw_img, p=projected: self._apply_preview_projection_done(k, r, p), 0)

    @mainthread
    def _apply_preview_projection_done(self, key: str, raw_img: PILImage.Image, projected: PILImage.Image):
        self.preview_projection_cache[key] = projected
        self.update_preview(raw_img, projected)

    # ---------- runtime ----------
    def on_start(self):
        try:
            Window.maximize()
        except Exception:
            pass
        self._subscribe_runtime_feed()
        self._emit_runtime("INFO", "start", "Interface Kivy démarrée")
        try:
            model_path = resolve_soldier_model_path()
            self.log(f"Modèle soldat chargé : {model_path}")
            # préchauffe le cache d'analyse du mannequin
            _ = get_projection_subject_analysis(PROJECTION_CFG)
        except Exception as exc:
            self.log(str(exc))
            self.status("Modèle soldat introuvable", ok=False)

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
    def update_preview(self, pil_img: PILImage.Image, projection_img: PILImage.Image):
        if self.preview_img is not None:
            self.preview_img.texture = pil_to_coreimage(pil_img).texture
        if self.preview_silhouette is not None:
            self.preview_silhouette.texture = pil_to_coreimage(projection_img).texture

    @mainthread
    def update_live_stage(
        self,
        stage: str,
        target_index: Optional[int] = None,
        local_attempt: Optional[int] = None,
        seed: Optional[int] = None,
        metrics_text: Optional[str] = None,
        pil_img: Optional[PILImage.Image] = None,
        preview_path: Optional[str] = None,
    ):
        if self.live_stage_label is not None:
            self.live_stage_label.text = f"Étape : {stage}"
        if self.live_counts_label is not None:
            self.live_counts_label.text = metrics_text or "État backend strict"
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
        target_index = payload.get("target_index")
        local_attempt = payload.get("local_attempt")
        seed = payload.get("seed")
        metrics_text = None
        if "metrics" in payload and isinstance(payload["metrics"], dict):
            m = payload["metrics"]
            metrics_text = (
                f"bd {m.get('boundary_density', '--')} | "
                f"miroir {m.get('mirror_similarity', '--')} | "
                f"bord {m.get('edge_contact_ratio', '--')}"
            )
        if stage or preview_path:
            self.update_live_stage(
                stage=str(stage or "runtime"),
                target_index=int(target_index) if target_index is not None else None,
                local_attempt=int(local_attempt) if local_attempt is not None else None,
                seed=int(seed) if seed is not None else None,
                metrics_text=metrics_text,
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

    def _apply_backend_motif_scale(self) -> float:
        scale = max(0.25, float(self.motif_scale))
        setter = getattr(camo, "set_motif_scale", None)
        if callable(setter):
            setter(scale)
        else:
            camo.set_canvas_geometry(
                width=int(getattr(camo, "WIDTH", 7680)),
                height=int(getattr(camo, "HEIGHT", 4320)),
                physical_width_cm=float(getattr(camo, "PHYSICAL_WIDTH_CM", 240.0)),
                physical_height_cm=float(getattr(camo, "PHYSICAL_HEIGHT_CM", 135.0)),
                motif_scale=scale,
            )
        return scale

    def _on_motif_scale_change(self, _slider, value):
        self.motif_scale = float(value)
        applied = self._apply_backend_motif_scale()
        if self.motif_scale_label is not None:
            self.motif_scale_label.text = f"{applied:.2f}"

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
        paths = sorted(self.current_output_dir.glob("camouflage_*.png"))
        if MAX_GALLERY_ITEMS > 0:
            paths = paths[-MAX_GALLERY_ITEMS:]
        for p in reversed(paths):
            self.gallery_grid.add_widget(GalleryThumb(self, p))

    # ---------- preflight ----------
    @mainthread
    def _update_preflight_label(self, text: str, ok: Optional[bool] = None):
        if self.tests_label is None:
            return
        self.tests_label.text = text
        self.tests_label.color = C["success"] if ok is True else C["danger"] if ok is False else C["text_soft"]

    async def _async_run_preflight(self) -> Tuple[bool, str]:
        try:
            try:
                count = int((self.count_input.text if self.count_input is not None else str(DEFAULT_TARGET_COUNT)).strip())
            except Exception:
                count = DEFAULT_TARGET_COUNT

            self._apply_backend_motif_scale()
            intensity = backend_machine_intensity(self.machine_intensity)
            output_dir = self.current_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(
                camo.validate_generation_request,
                target_count=max(1, count),
                output_dir=output_dir,
                base_seed=int(getattr(camo, "DEFAULT_BASE_SEED", 0)),
                machine_intensity=intensity,
                max_workers=1,
                attempt_batch_size=1,
            )

            if hasattr(camo, "sample_process_resources"):
                snap = await asyncio.to_thread(camo.sample_process_resources, intensity, output_dir)
                summary = (
                    f"Préflight OK | CPU={snap.cpu_count} | RAM dispo={snap.system_available_mb:.0f} Mo | "
                    f"disque libre={snap.disk_free_mb:.0f} Mo"
                )
            else:
                summary = "Préflight OK"
            return True, summary
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
            self.status("Préflight KO", ok=False)

    # ---------- diagnostics ----------
    async def _register_live_diag(self, candidate: camo.CandidateResult, target_index: int, local_attempt: int, outcome: Any):
        self.diag_total += 1
        accepted = bool(getattr(outcome, "accepted", bool(outcome)))
        if accepted:
            self.diag_accepts += 1
            self.diag_last_rules = []
            self.diag_log(f"[img={target_index:03d} essai={local_attempt:04d}] accepté | seed={candidate.seed}")
            self._refresh_diag_labels()
            return
        self.diag_rejects += 1
        rules = rejection_rules_for_candidate(candidate, outcome)
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
            self.tests_ok = True
            self.tests_ran = True
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

        self._apply_backend_motif_scale()
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
        self.log(f"Démarrage : {count} camouflage(s) | sortie={self.current_output_dir} | motif_scale={self.motif_scale:.2f}")
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
        if extra > 0:
            await asyncio.sleep(extra)

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
                    candidate = await async_generate_candidate_from_seed(seed)
                    outcome = await async_validate_candidate_result(candidate)
                    valid = bool(getattr(outcome, "accepted", bool(outcome)))
                    scores = extract_backend_scores(candidate.ratios, candidate.metrics)
                    try:
                        projection_img = await asyncio.to_thread(projection_preview_image, candidate.image)
                    except Exception as exc:
                        projection_img = candidate.image
                        self.log(f"Projection modèle indisponible : {exc}")
                    self.update_preview(candidate.image, projection_img)
                    await self._register_live_diag(candidate, target_index, local_attempt, outcome)

                    metrics_text = (
                        f"bd {scores['boundary_density']:.4f} | "
                        f"miroir {scores['mirror_similarity']:.4f} | "
                        f"bord {scores['edge_contact_ratio']:.4f}"
                    )

                    if not valid:
                        self.update_live_stage("rejeté", target_index, local_attempt, seed, metrics_text=metrics_text, pil_img=projection_img)
                        self._update_attempt_status(
                            target_index,
                            local_attempt,
                            total_attempts,
                            seed,
                            target_count,
                            self.accepted_count,
                            self.diag_rejects,
                            False,
                            candidate.ratios,
                            scores,
                            candidate.metrics,
                        )
                        self.log(
                            f"[img={target_index:03d} essai={local_attempt:04d}] rejeté | "
                            f"MAE={scores['ratio_mae']:.6f} | bd={scores['boundary_density']:.4f}"
                        )
                        await self._adaptive_pause()
                        await asyncio.sleep(0)
                        continue

                    filename = build_backend_compatible_output_path(
                        output_dir=self.current_output_dir,
                        target_index=target_index,
                        local_attempt=local_attempt,
                        global_attempt=total_attempts,
                        candidate=candidate,
                    )
                    self.update_live_stage("export image", target_index, local_attempt, seed, metrics_text=metrics_text, pil_img=projection_img)
                    saved_path = await async_save_candidate_image(candidate, filename)
                    mannequin_saved_path = await async_save_mannequin_projection(
                        projection_img,
                        saved_path,
                        self.current_output_dir,
                    )
                    record = CandidateRecord(
                        index=target_index,
                        seed=candidate.seed,
                        local_attempt=local_attempt,
                        global_attempt=total_attempts,
                        image_path=saved_path,
                        metrics={k: float(v) for k, v in candidate.metrics.items()},
                        ratios=candidate.ratios.copy(),
                    )
                    self.best_records.append(record)
                    self.best_records.sort(key=candidate_rank_key)
                    row = build_candidate_row_compatible(
                        target_index=target_index,
                        local_attempt=local_attempt,
                        global_attempt=total_attempts,
                        candidate=candidate,
                        outcome=outcome,
                        saved_path=saved_path,
                    )
                    row["mannequin_image_name"] = mannequin_saved_path.name
                    row["mannequin_image_path"] = str(mannequin_saved_path)
                    rows.append(row)
                    self.accepted_count = len(rows)
                    self.update_progress(len(rows), target_count)
                    self.update_live_stage("accepté", target_index, local_attempt, seed, metrics_text=metrics_text, pil_img=projection_img)
                    self._update_attempt_status(
                        target_index,
                        local_attempt,
                        total_attempts,
                        seed,
                        target_count,
                        self.accepted_count,
                        self.diag_rejects,
                        True,
                        candidate.ratios,
                        scores,
                        candidate.metrics,
                    )
                    self.log(
                        f"[img={target_index:03d}] accepté -> {saved_path.name} | "
                        f"mannequin -> {mannequin_saved_path.name} | "
                        f"MAE={scores['ratio_mae']:.6f} | composant={scores['primary_component_ratio']:.4f}"
                    )
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
            labels = BACKEND_CLASS_LABELS[:4] if len(BACKEND_CLASS_LABELS) >= 4 else ["class_0", "class_1", "class_2", "class_3"]
            self.color_text.text = (
                f"{labels[0]} {rs[BACKEND_IDX_0]*100:.4f}% | "
                f"{labels[1]} {rs[BACKEND_IDX_1]*100:.4f}% | "
                f"{labels[2]} {rs[BACKEND_IDX_2]*100:.4f}% | "
                f"{labels[3]} {rs[BACKEND_IDX_3]*100:.4f}%"
            )
        if self.score_text is not None:
            self.score_text.text = (
                f"MAE ratio {scores['ratio_mae']:.6f} | "
                f"max abs {scores['ratio_max_abs']:.6f} | "
                f"composant {scores['primary_component_ratio']:.4f} | "
                f"miroir {scores['mirror_similarity']:.4f}"
            )
        if self.extra_text is not None:
            self.extra_text.text = (
                f"bd {scores['boundary_density']:.4f} | "
                f"bd/4 {scores['boundary_density_small']:.4f} | "
                f"bd/8 {scores['boundary_density_tiny']:.4f} | "
                f"bord {scores['edge_contact_ratio']:.4f}"
            )
        if self.struct_text is not None:
            self.struct_text.text = (
                f"overscan {safe_metric(metrics, 'overscan'):.4f} | "
                f"shift {safe_metric(metrics, 'shift_strength'):.4f} | "
                f"px/cm {safe_metric(metrics, 'px_per_cm'):.4f} | "
                f"scale {safe_metric(metrics, 'motif_scale', self.motif_scale):.4f}"
            )

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
            if rec.image_path.exists():
                dst = best_dir / f"best_{rank:03d}_camouflage_{rec.index:03d}.png"
                await asyncio.to_thread(shutil.copy2, rec.image_path, dst)
            rows.append({
                "rank": rank,
                "source_index": rec.index,
                "seed": rec.seed,
                "global_attempt": rec.global_attempt,
                "attempts_for_this_image": rec.local_attempt,
                "ratio_mae": round(float(np.mean(np.abs(rec.ratios - camo.TARGET))), 8),
                "ratio_max_abs": round(float(np.max(np.abs(rec.ratios - camo.TARGET))), 8),
                "primary_component_ratio": round(safe_metric(rec.metrics, PRIMARY_COMPONENT_METRIC), 6),
                "boundary_density": round(safe_metric(rec.metrics, "boundary_density"), 6),
                "boundary_density_small": round(safe_metric(rec.metrics, "boundary_density_small"), 6),
                "boundary_density_tiny": round(safe_metric(rec.metrics, "boundary_density_tiny"), 6),
                "mirror_similarity": round(safe_metric(rec.metrics, "mirror_similarity"), 6),
                "edge_contact_ratio": round(safe_metric(rec.metrics, "edge_contact_ratio"), 6),
                "class_0_pct": round(float(rec.ratios[BACKEND_IDX_0] * 100), 4),
                "class_1_pct": round(float(rec.ratios[BACKEND_IDX_1] * 100), 4),
                "class_2_pct": round(float(rec.ratios[BACKEND_IDX_2] * 100), 4),
                "class_3_pct": round(float(rec.ratios[BACKEND_IDX_3] * 100), 4),
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
        report_path = await async_write_report(rows, self.current_output_dir, filename=REPORT_NAME)
        best_dir = await self._async_export_best_of(min(DEFAULT_TOP_K, len(self.best_records)))
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
        report_path = await async_write_report(rows, self.current_output_dir, filename=REPORT_NAME)
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
            self.resource_text.text = (
                f"CPU {cpu:.0f}% | RAM {ram:.0f}% | Disque {disk:.0f}% | "
                f"Processus {proc_cpu:.0f}% / {proc_mem:.2f} Go | scale {self.motif_scale:.2f}"
            )
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
