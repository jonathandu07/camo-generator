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
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageOps

try:
    import psutil
except Exception:
    psutil = None

logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.INFO)

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

try:
    import camouflage_ml_dl as camo_mldl
except Exception:
    camo_mldl = None

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
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
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
OUTPUT_IMAGE_GLOB = "pattern_*.png" if hasattr(camo, "validate_with_reasons") else "camouflage_*.png"
DEFAULT_MLDL_REPORT_NAME = "rapport_camouflages_ml_dl.csv"

RUN_MODE_BLOCKING = "blocking"
RUN_MODE_NON_BLOCKING = "non_blocking"
RUN_MODE_SKIP_TESTS = "skip_tests"

THUMB_SIZE = (320, 320)
GALLERY_COLUMNS = 3
MAX_GALLERY_ITEMS = 24
SCRIPT_DIR = Path(__file__).resolve().parent
SOLDIER_MODEL_BASENAMES = ("soldat_modele_vert.png", "file_000000001604724685750b386a71249d.png", "soldat_modele.png")

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


@dataclass
class PendingManualReview:
    target_index: int
    local_attempt: int
    global_attempt: int
    candidate: camo.CandidateResult
    outcome: Any
    projection_img: PILImage.Image
    metrics_text: str
    manually_saved: bool = False


@dataclass
class GenerationConfigSnapshot:
    target_count: int
    motif_scale: float
    projection_scale: float
    machine_intensity: float
    output_dir: Path
    max_workers: int
    attempt_batch_size: int
    parallel_attempts: bool


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
    img = pil_img.convert("RGB")
    return ImageOps.fit(img, size, method=PILImage.Resampling.LANCZOS, centering=(0.5, 0.5))


def make_fill_image(**kwargs) -> Image:
    try:
        return Image(fit_mode="fill", **kwargs)
    except TypeError:
        img = Image(**kwargs)
        if hasattr(img, "fit_mode"):
            try:
                img.fit_mode = "fill"
                return img
            except Exception:
                pass
        # Fallback ancien Kivy
        try:
            img.allow_stretch = True
            img.keep_ratio = False
        except Exception:
            pass
        return img


DEFAULT_UI_MOTIF_SCALE = float(max(0.18, min(getattr(camo, "DEFAULT_MOTIF_SCALE", 0.55), 1.20)))
DEFAULT_BACKEND_MACHINE_INTENSITY = float(getattr(camo, "DEFAULT_MACHINE_INTENSITY", 0.98))
DEFAULT_PROJECTION_PREVIEW_SCALE = 0.40


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


def extract_backend_scores(ratios: np.ndarray, metrics: Dict[str, float], outcome: Optional[Any] = None) -> Dict[str, float]:
    abs_err = np.abs(ratios - camo.TARGET)
    bestof_score = safe_metric(metrics, "bestof_score")
    if outcome is not None and hasattr(outcome, "bestof_score"):
        try:
            bestof_score = float(getattr(outcome, "bestof_score", bestof_score))
        except Exception:
            pass

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
        "bestof_score": float(bestof_score),
        "seed_macros_total": safe_metric(metrics, "seed_macros_total"),
        "growth_rounds": safe_metric(metrics, "growth_rounds"),
        "safe_rebalanced_pixels": safe_metric(metrics, "safe_rebalanced_pixels"),
        "orphan_pixels_fixed": safe_metric(metrics, "orphan_pixels_fixed"),
        "repair_round": safe_metric(metrics, "repair_round"),
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


def candidate_rank_key(record: CandidateRecord) -> Tuple[float, float, float, float, float]:
    ratios = record.ratios
    metrics = record.metrics
    ratio_mae = float(np.mean(np.abs(ratios - camo.TARGET)))
    ratio_max = float(np.max(np.abs(ratios - camo.TARGET)))
    mirror = safe_metric(metrics, "mirror_similarity")
    edge = safe_metric(metrics, "edge_contact_ratio")
    bestof = safe_metric(metrics, "bestof_score")
    return (-bestof, ratio_mae, ratio_max, mirror, edge)


async def async_generate_candidate_from_seed(seed: int) -> camo.CandidateResult:
    return await asyncio.to_thread(camo.generate_candidate_from_seed, seed)


async def async_generate_and_validate_from_seed(
    seed: int,
    *,
    tolerance_profile: Optional[Any] = None,
    max_repair_rounds: Optional[int] = None,
    anti_pixel: Optional[bool] = None,
) -> Tuple[camo.CandidateResult, Any]:
    combined = getattr(camo, "generate_and_validate_from_seed", None)
    if callable(combined):
        kwargs: Dict[str, Any] = {}
        if tolerance_profile is not None:
            kwargs["tolerance_profile"] = tolerance_profile
        if max_repair_rounds is not None:
            kwargs["max_repair_rounds"] = int(max_repair_rounds)
        if anti_pixel is not None:
            kwargs["anti_pixel"] = bool(anti_pixel)
        try:
            return await asyncio.to_thread(combined, seed, **kwargs)
        except TypeError:
            return await asyncio.to_thread(combined, seed)
    candidate = await asyncio.to_thread(camo.generate_candidate_from_seed, seed)
    outcome = await async_validate_candidate_result(candidate)
    return candidate, outcome


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


async def async_wait_generation_future(fut: asyncio.Future, attempt_no: int, seed: int) -> Tuple[int, int, camo.CandidateResult, Any]:
    candidate, outcome = await fut
    return attempt_no, seed, candidate, outcome


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
    manual_accept: bool = False,
) -> Dict[str, Any]:
    row_builder = getattr(camo, "candidate_row", None)
    if callable(row_builder):
        try:
            row = row_builder(
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
                row = row_builder(
                    target_index,
                    local_attempt,
                    global_attempt,
                    candidate,
                    image_name=saved_path.name,
                    image_path=str(saved_path),
                )
            except TypeError:
                row = row_builder(target_index, local_attempt, global_attempt, candidate)
        if not isinstance(row, dict):
            row = {}
    else:
        row = {}

    if not row:
        row = {
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

    backend_accepted = int(bool(getattr(outcome, "accepted", False))) if outcome is not None else 0
    manual_accepted = int(bool(manual_accept))
    final_accepted = 1 if manual_accepted else backend_accepted
    row["image_name"] = saved_path.name
    row["image_path"] = str(saved_path)
    row["accepted"] = final_accepted
    row["backend_accepted"] = backend_accepted
    row["manual_accepted"] = manual_accepted
    row["accepted_source"] = "manual" if manual_accepted else "backend"
    row["review_status"] = "accepted_manual" if manual_accepted else "accepted_backend"
    return row


# ============================================================
# PROJECTION SUR SOLDAT MODÈLE (aperçu uniquement)
# ============================================================
# PROJECTION SUR SOLDAT MODÈLE (aperçu uniquement)
# ============================================================

@dataclass
class ProjectionConfig:
    # Détection du fond clair studio.
    bg_sat_max: int = 24
    bg_val_min: int = 185

    # Graine verte robuste : on cherche le textile vert dominant.
    hue_min: int = 28
    hue_max: int = 95
    sat_min: int = 30
    val_min: int = 22
    sat_max: int = 255
    val_max: int = 248

    # Dominance RGB pour rester sur le textile vert, y compris en ombre.
    green_dom_g_over_r: int = 8
    green_dom_g_over_b: int = 5
    green_dom_g_min: int = 38

    # Extension du masque aux ombres du textile déjà connectées à la graine.
    shadow_hue_min: int = 20
    shadow_hue_max: int = 100
    shadow_green_dom_g_over_r: int = 1
    shadow_green_dom_g_over_b: int = -6
    shadow_val_min: int = 18

    # Protection des zones non textiles ou interdites à repeindre.
    dark_val_max: int = 38
    dark_sat_max: int = 80
    tan_hue_min: int = 6
    tan_hue_max: int = 28
    tan_sat_min: int = 18
    tan_val_min: int = 40
    tan_val_max: int = 240
    metal_sat_max: int = 30
    metal_val_min: int = 30
    metal_val_max: int = 210
    neutral_sat_max: int = 20
    neutral_val_min: int = 25
    neutral_val_max: int = 150
    neutral_green_margin: int = 4

    # Morphologie.
    open_kernel: int = 3
    close_kernel: int = 5
    protect_dilate_kernel: int = 5
    blur_radius: int = 5
    min_component_area_px: int = 220
    min_connected_shadow_area_px: int = 60
    min_uniform_pixels: int = 1500

    # Répartition verticale du vêtement.
    hat_ratio: float = 0.13
    jacket_ratio: float = 0.41
    pants_ratio: float = 0.46

    # Approximation physique du mannequin visible.
    uniform_visible_height_cm: float = 160.0

    # Adaptation physique quand la source est un camouflage plein format.
    hat_scale_multiplier: float = 0.48
    jacket_scale_multiplier: float = 0.52
    pants_scale_multiplier: float = 0.56

    # Fallback quand la source est un petit tile.
    tile_hat_width_ratio: float = 0.14
    tile_jacket_width_ratio: float = 0.12
    tile_pants_width_ratio: float = 0.11

    # Bornes globales de scale.
    min_region_scale: float = 0.06
    max_region_scale: float = 0.95

    # Composition.
    shadow_strength: float = 0.62
    detail_strength: float = 0.18
    edge_darkening: float = 0.09
    alpha_gamma: float = 0.92
    alpha_hard_threshold: float = 0.35
    min_alpha_inside_mask: float = 0.90

    # Réparation et vérification stricte des résidus verts.
    residual_rounds: int = 5
    residual_threshold: int = 12
    repair_seed_base: int = 211
    repair_seed_step: int = 17
    repair_scale_decay: float = 0.10

    verify_hue_window: int = 14
    verify_sat_margin: int = 18
    verify_val_min: int = 12
    verify_lab_distance_max: float = 30.0
    verify_lab_green_distance_max: float = 42.0
    verify_rgb_delta_max: float = 28.0
    verify_rgb_green_delta_max: float = 36.0
    max_residual_ratio_quality: float = 0.015
    max_residual_pixels_quality: int = 180
    max_residual_ratio_fast: float = 0.030
    max_residual_pixels_fast: int = 320


PROJECTION_CFG = ProjectionConfig()
_PROJECTION_SUBJECT_CACHE: Optional[np.ndarray] = None
_PROJECTION_SUBJECT_PATH: Optional[Path] = None
_PROJECTION_ANALYSIS_CACHE: Optional["ProjectionSubjectAnalysis"] = None
_PROJECTION_ANALYSIS_PATH: Optional[Path] = None


@dataclass
class ProjectionVerificationReport:
    uniform_pixels: int
    residual_pixels: int
    still_green_pixels: int
    residual_ratio: float
    mean_lab_distance: float
    mean_rgb_delta: float
    valid: bool


@dataclass
class ProjectionSubjectAnalysis:
    model_path: Path
    subject_bgr: np.ndarray
    person_mask: np.ndarray
    protect_mask: np.ndarray
    uniform_mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    regions: Dict[str, np.ndarray]
    mannequin_px_per_cm: float
    green_reference_lab: np.ndarray
    green_reference_hsv: np.ndarray


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


def white_background_mask(bgr: np.ndarray, cfg: ProjectionConfig = PROJECTION_CFG) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] <= cfg.bg_sat_max) & (hsv[:, :, 2] >= cfg.bg_val_min)).astype(np.uint8) * 255
    white = cv2.GaussianBlur(white, (0, 0), 2.0)
    return white


def person_mask_from_foreground(bgr: np.ndarray, cfg: ProjectionConfig = PROJECTION_CFG) -> np.ndarray:
    white = white_background_mask(bgr, cfg)
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
    out = cv2.GaussianBlur(out, (0, 0), 1.8)
    return out


def keep_significant_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    out = np.zeros_like(mask)
    min_area = max(1, int(min_area))
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label_idx] = 255
    return out


def build_protection_mask(subject_bgr: np.ndarray, person_mask: np.ndarray, cfg: ProjectionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2HSV)
    b = subject_bgr[:, :, 0].astype(np.int16)
    g = subject_bgr[:, :, 1].astype(np.int16)
    r = subject_bgr[:, :, 2].astype(np.int16)

    person_bool = person_mask > 0
    dark = (
        (hsv[:, :, 2] <= cfg.dark_val_max)
        & (hsv[:, :, 1] <= cfg.dark_sat_max)
    )

    tan_hsv = (
        (hsv[:, :, 0] >= cfg.tan_hue_min)
        & (hsv[:, :, 0] <= cfg.tan_hue_max)
        & (hsv[:, :, 1] >= cfg.tan_sat_min)
        & (hsv[:, :, 2] >= cfg.tan_val_min)
        & (hsv[:, :, 2] <= cfg.tan_val_max)
    )
    tan_rgb = (r >= g + 2) & (r >= b - 6)
    tan = tan_hsv & tan_rgb

    metal = (
        (hsv[:, :, 1] <= cfg.metal_sat_max)
        & (hsv[:, :, 2] >= cfg.metal_val_min)
        & (hsv[:, :, 2] <= cfg.metal_val_max)
    )

    neutral = (
        (hsv[:, :, 1] <= cfg.neutral_sat_max)
        & (hsv[:, :, 2] >= cfg.neutral_val_min)
        & (hsv[:, :, 2] <= cfg.neutral_val_max)
        & (g < r + cfg.neutral_green_margin)
    )

    protect = (dark | tan | metal | neutral).astype(np.uint8) * 255
    protect = cv2.bitwise_and(protect, person_mask)
    protect = cv2.dilate(
        protect,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.protect_dilate_kernel, cfg.protect_dilate_kernel)),
        iterations=1,
    )
    return protect


def _grow_connected_green_regions(
    grow_mask: np.ndarray,
    seed_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    grow_u8 = (grow_mask.astype(np.uint8) * 255) if grow_mask.dtype != np.uint8 else grow_mask.copy()
    seed_bool = seed_mask > 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grow_u8, connectivity=8)
    out = np.zeros_like(grow_u8)
    min_area = max(1, int(min_area))
    for label_idx in range(1, num_labels):
        comp = labels == label_idx
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if np.any(seed_bool & comp):
            out[comp] = 255
    return out


def _sample_uniform_reference(subject_bgr: np.ndarray, uniform_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = uniform_mask > 32
    if not np.any(mask):
        lab = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        hsv = np.array([60.0, 128.0, 128.0], dtype=np.float32)
        return lab, hsv
    subj_lab = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2LAB)
    subj_hsv = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2HSV)
    ref_lab = np.median(subj_lab[mask], axis=0).astype(np.float32)
    ref_hsv = np.median(subj_hsv[mask], axis=0).astype(np.float32)
    return ref_lab, ref_hsv


def green_uniform_mask(subject_bgr: np.ndarray, cfg: ProjectionConfig) -> np.ndarray:
    hsv = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2HSV)
    person = person_mask_from_foreground(subject_bgr, cfg)

    b = subject_bgr[:, :, 0].astype(np.int16)
    g = subject_bgr[:, :, 1].astype(np.int16)
    r = subject_bgr[:, :, 2].astype(np.int16)

    person_bool = person > 0
    seed_mask = (
        (hsv[:, :, 0] >= cfg.hue_min)
        & (hsv[:, :, 0] <= cfg.hue_max)
        & (hsv[:, :, 1] >= cfg.sat_min)
        & (hsv[:, :, 2] >= cfg.val_min)
        & (hsv[:, :, 2] <= cfg.val_max)
        & (g >= r + cfg.green_dom_g_over_r)
        & (g >= b + cfg.green_dom_g_over_b)
        & (g >= cfg.green_dom_g_min)
        & person_bool
    )

    protect = build_protection_mask(subject_bgr, person, cfg)
    allowed = person_bool & (protect == 0)

    grow_mask = (
        (hsv[:, :, 0] >= cfg.shadow_hue_min)
        & (hsv[:, :, 0] <= cfg.shadow_hue_max)
        & (hsv[:, :, 2] >= cfg.shadow_val_min)
        & (g >= r + cfg.shadow_green_dom_g_over_r)
        & (g >= b + cfg.shadow_green_dom_g_over_b)
        & allowed
    )

    seed_u8 = (seed_mask.astype(np.uint8) * 255)
    grown = _grow_connected_green_regions(
        (grow_mask.astype(np.uint8) * 255),
        seed_u8,
        cfg.min_connected_shadow_area_px,
    )

    if int(np.sum(grown > 0)) < cfg.min_uniform_pixels and int(np.sum(seed_u8 > 0)) > 0:
        grown = seed_u8.copy()

    grown = cv2.morphologyEx(
        grown,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel)),
    )
    grown = cv2.morphologyEx(
        grown,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel)),
    )
    grown = keep_significant_components(grown, max(cfg.min_component_area_px, cfg.min_connected_shadow_area_px))

    blur_k = ensure_odd(cfg.blur_radius)
    grown = cv2.GaussianBlur(grown, (blur_k, blur_k), 0)
    grown = cv2.bitwise_and(grown, 255 - protect)
    grown = cv2.bitwise_and(grown, person)
    return grown


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


def build_projection_analysis_from_bgr(
    subject_bgr: np.ndarray,
    cfg: ProjectionConfig = PROJECTION_CFG,
    model_path: Optional[Path] = None,
) -> ProjectionSubjectAnalysis:
    person_mask = person_mask_from_foreground(subject_bgr, cfg)
    protect_mask = build_protection_mask(subject_bgr, person_mask, cfg)
    uniform_mask = green_uniform_mask(subject_bgr, cfg)
    bbox = bbox_from_mask(uniform_mask)
    regions = split_uniform_regions(uniform_mask, cfg)
    ref_lab, ref_hsv = _sample_uniform_reference(subject_bgr, uniform_mask)
    _, _, _, bbox_h = bbox
    mannequin_px_per_cm = (bbox_h / max(1.0, cfg.uniform_visible_height_cm)) if bbox_h > 0 else 1.0
    return ProjectionSubjectAnalysis(
        model_path=model_path or Path('.'),
        subject_bgr=subject_bgr,
        person_mask=person_mask,
        protect_mask=protect_mask,
        uniform_mask=uniform_mask,
        bbox=bbox,
        regions=regions,
        mannequin_px_per_cm=float(max(0.01, mannequin_px_per_cm)),
        green_reference_lab=ref_lab,
        green_reference_hsv=ref_hsv,
    )


def build_projection_subject_analysis(cfg: ProjectionConfig = PROJECTION_CFG) -> ProjectionSubjectAnalysis:
    model_path = resolve_soldier_model_path()
    subject_bgr = get_projection_subject_bgr()
    return build_projection_analysis_from_bgr(subject_bgr, cfg=cfg, model_path=model_path)


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
    user_scale: float = 1.0,
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
        return float(np.clip(scale * max(0.10, float(user_scale)), cfg.min_region_scale, cfg.max_region_scale))

    _camo_h, camo_w = camo_bgr.shape[:2]
    if region_name == "hat":
        target_w = rw * cfg.tile_hat_width_ratio
    elif region_name == "pants":
        target_w = rw * cfg.tile_pants_width_ratio
    else:
        target_w = rw * cfg.tile_jacket_width_ratio
    scale = target_w / max(1.0, float(camo_w))
    return float(np.clip(scale * max(0.10, float(user_scale)), cfg.min_region_scale, cfg.max_region_scale))


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
    alpha_raw = np.clip(region_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    strong = alpha_raw >= float(cfg.alpha_hard_threshold)
    alpha_raw[strong] = np.maximum(alpha_raw[strong], float(cfg.min_alpha_inside_mask))
    alpha = np.clip(alpha_raw, 0.0, 1.0) ** max(0.2, cfg.alpha_gamma)
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


def _hue_distance_wrap(hue: np.ndarray, ref_hue: float) -> np.ndarray:
    diff = np.abs(hue.astype(np.float32) - float(ref_hue))
    return np.minimum(diff, 180.0 - diff)


def build_residual_uniform_mask(
    subject_bgr: np.ndarray,
    projected_bgr: np.ndarray,
    uniform_mask: np.ndarray,
    protect_mask: np.ndarray,
    cfg: ProjectionConfig,
    reference_lab: Optional[np.ndarray] = None,
    reference_hsv: Optional[np.ndarray] = None,
) -> np.ndarray:
    uniform_bool = uniform_mask > 24
    if not np.any(uniform_bool):
        return np.zeros(subject_bgr.shape[:2], dtype=np.uint8)

    if reference_lab is None or reference_hsv is None:
        reference_lab, reference_hsv = _sample_uniform_reference(subject_bgr, uniform_mask)

    allowed = uniform_bool & ~(protect_mask > 24)
    if not np.any(allowed):
        return np.zeros(subject_bgr.shape[:2], dtype=np.uint8)

    projected_lab = cv2.cvtColor(projected_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    projected_hsv = cv2.cvtColor(projected_bgr, cv2.COLOR_BGR2HSV)

    ref_lab = np.asarray(reference_lab, dtype=np.float32)
    lab_distance = np.linalg.norm(projected_lab - ref_lab[None, None, :], axis=2)
    rgb_delta = np.mean(np.abs(projected_bgr.astype(np.int16) - subject_bgr.astype(np.int16)), axis=2).astype(np.float32)

    hue_distance = _hue_distance_wrap(projected_hsv[:, :, 0], float(reference_hsv[0]))
    still_green = (
        (hue_distance <= float(cfg.verify_hue_window))
        & (projected_hsv[:, :, 1] >= max(12.0, float(reference_hsv[1]) - float(cfg.verify_sat_margin)))
        & (projected_hsv[:, :, 2] >= float(cfg.verify_val_min))
    )

    too_close = (
        (lab_distance <= float(cfg.verify_lab_distance_max))
        & (rgb_delta <= float(cfg.verify_rgb_delta_max))
    )
    green_close = (
        still_green
        & (
            (lab_distance <= float(cfg.verify_lab_green_distance_max))
            | (rgb_delta <= float(cfg.verify_rgb_green_delta_max))
        )
    )

    residual = (allowed & (too_close | green_close)).astype(np.uint8) * 255
    residual = cv2.morphologyEx(
        residual,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    residual = cv2.morphologyEx(
        residual,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    residual = keep_significant_components(residual, 14)
    residual = cv2.GaussianBlur(residual, (0, 0), 1.1)
    residual = cv2.bitwise_and(residual, (allowed.astype(np.uint8) * 255))
    return residual


def evaluate_projection_verification(
    subject_bgr: np.ndarray,
    projected_bgr: np.ndarray,
    uniform_mask: np.ndarray,
    protect_mask: np.ndarray,
    cfg: ProjectionConfig,
    reference_lab: Optional[np.ndarray] = None,
    reference_hsv: Optional[np.ndarray] = None,
    preview_mode: str = 'quality',
) -> ProjectionVerificationReport:
    uniform_bool = uniform_mask > 24
    allowed = uniform_bool & ~(protect_mask > 24)
    uniform_pixels = int(np.sum(allowed))
    if uniform_pixels <= 0:
        return ProjectionVerificationReport(
            uniform_pixels=0,
            residual_pixels=0,
            still_green_pixels=0,
            residual_ratio=1.0,
            mean_lab_distance=0.0,
            mean_rgb_delta=0.0,
            valid=False,
        )

    if reference_lab is None or reference_hsv is None:
        reference_lab, reference_hsv = _sample_uniform_reference(subject_bgr, uniform_mask)

    residual_mask = build_residual_uniform_mask(
        subject_bgr,
        projected_bgr,
        uniform_mask,
        protect_mask,
        cfg,
        reference_lab=reference_lab,
        reference_hsv=reference_hsv,
    )

    projected_lab = cv2.cvtColor(projected_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    projected_hsv = cv2.cvtColor(projected_bgr, cv2.COLOR_BGR2HSV)
    ref_lab = np.asarray(reference_lab, dtype=np.float32)
    lab_distance = np.linalg.norm(projected_lab - ref_lab[None, None, :], axis=2)
    rgb_delta = np.mean(np.abs(projected_bgr.astype(np.int16) - subject_bgr.astype(np.int16)), axis=2).astype(np.float32)
    hue_distance = _hue_distance_wrap(projected_hsv[:, :, 0], float(reference_hsv[0]))
    still_green_mask = (
        (hue_distance <= float(cfg.verify_hue_window))
        & (projected_hsv[:, :, 1] >= max(12.0, float(reference_hsv[1]) - float(cfg.verify_sat_margin)))
        & allowed
    )

    residual_pixels = int(np.sum(residual_mask > cfg.residual_threshold))
    residual_ratio = float(residual_pixels / max(1, uniform_pixels))
    still_green_pixels = int(np.sum(still_green_mask))
    mean_lab_distance = float(np.mean(lab_distance[allowed])) if np.any(allowed) else 0.0
    mean_rgb_delta = float(np.mean(rgb_delta[allowed])) if np.any(allowed) else 0.0

    if preview_mode == 'fast':
        ratio_max = float(cfg.max_residual_ratio_fast)
        pixels_max = int(cfg.max_residual_pixels_fast)
    else:
        ratio_max = float(cfg.max_residual_ratio_quality)
        pixels_max = int(cfg.max_residual_pixels_quality)

    valid = residual_ratio <= ratio_max and residual_pixels <= pixels_max
    return ProjectionVerificationReport(
        uniform_pixels=uniform_pixels,
        residual_pixels=residual_pixels,
        still_green_pixels=still_green_pixels,
        residual_ratio=residual_ratio,
        mean_lab_distance=mean_lab_distance,
        mean_rgb_delta=mean_rgb_delta,
        valid=bool(valid),
    )


def correct_projection_residuals(
    subject_bgr: np.ndarray,
    projected_bgr: np.ndarray,
    uniform_mask: np.ndarray,
    protect_mask: np.ndarray,
    camo_bgr: np.ndarray,
    base_scale: float,
    cfg: ProjectionConfig,
    reference_lab: Optional[np.ndarray] = None,
    reference_hsv: Optional[np.ndarray] = None,
    preview_mode: str = 'quality',
    rounds: Optional[int] = None,
) -> Tuple[np.ndarray, ProjectionVerificationReport]:
    out = projected_bgr.copy()
    total_rounds = cfg.residual_rounds if rounds is None else max(0, int(rounds))
    report = evaluate_projection_verification(
        subject_bgr,
        out,
        uniform_mask,
        protect_mask,
        cfg,
        reference_lab=reference_lab,
        reference_hsv=reference_hsv,
        preview_mode=preview_mode,
    )
    if report.valid:
        return out, report

    for round_idx in range(total_rounds):
        residual = build_residual_uniform_mask(
            subject_bgr,
            out,
            uniform_mask,
            protect_mask,
            cfg,
            reference_lab=reference_lab,
            reference_hsv=reference_hsv,
        )
        residual_px = int(np.sum(residual > cfg.residual_threshold))
        if residual_px <= 0:
            break
        local_scale = float(
            np.clip(
                base_scale * max(0.55, 0.92 - cfg.repair_scale_decay * round_idx),
                cfg.min_region_scale,
                cfg.max_region_scale,
            )
        )
        out = compose_region(
            out,
            residual,
            camo_bgr,
            local_scale,
            seed=int(cfg.repair_seed_base + round_idx * cfg.repair_seed_step),
            cfg=cfg,
        )
        report = evaluate_projection_verification(
            subject_bgr,
            out,
            uniform_mask,
            protect_mask,
            cfg,
            reference_lab=reference_lab,
            reference_hsv=reference_hsv,
            preview_mode=preview_mode,
        )
        if report.valid:
            break
    return out, report


def apply_camo_to_reference(
    subject_bgr: np.ndarray,
    camo_bgr: np.ndarray,
    cfg: ProjectionConfig = PROJECTION_CFG,
    user_scale: float = 1.0,
    analysis: Optional[ProjectionSubjectAnalysis] = None,
    fast_preview: bool = False,
) -> Tuple[np.ndarray, np.ndarray, ProjectionVerificationReport]:
    analysis = analysis or build_projection_analysis_from_bgr(subject_bgr, cfg=cfg)
    full_mask = analysis.uniform_mask.copy()
    if int(np.sum(full_mask > 24)) < int(cfg.min_uniform_pixels):
        raise RuntimeError(
            f'Masque uniforme trop faible pour une projection fiable ({int(np.sum(full_mask > 24))} px).'
        )

    protect_mask = analysis.protect_mask.copy()
    full_mask = cv2.bitwise_and(full_mask, 255 - protect_mask)

    hat_mask = cv2.bitwise_and(analysis.regions.get('hat', np.zeros_like(full_mask)), full_mask)
    jacket_mask = cv2.bitwise_and(analysis.regions.get('jacket', np.zeros_like(full_mask)), full_mask)
    pants_mask = cv2.bitwise_and(analysis.regions.get('pants', np.zeros_like(full_mask)), full_mask)

    out = subject_bgr.copy()
    for region_name, region_mask, region_seed in (
        ('hat', hat_mask, 23),
        ('jacket', jacket_mask, 31),
        ('pants', pants_mask, 47),
    ):
        if int(np.sum(region_mask > 8)) <= 8:
            continue
        scale = adaptive_region_scale(region_name, region_mask, camo_bgr, analysis, cfg, user_scale=user_scale)
        out = compose_region(out, region_mask, camo_bgr, scale, seed=region_seed, cfg=cfg)

    jacket_scale = adaptive_region_scale('jacket', full_mask, camo_bgr, analysis, cfg, user_scale=user_scale)
    out, report = correct_projection_residuals(
        subject_bgr,
        out,
        full_mask,
        protect_mask,
        camo_bgr,
        jacket_scale,
        cfg,
        reference_lab=analysis.green_reference_lab,
        reference_hsv=analysis.green_reference_hsv,
        preview_mode='fast' if fast_preview else 'quality',
        rounds=1 if fast_preview else None,
    )
    return out, full_mask, report


def crop_person_display_16_9(
    projected_bgr: np.ndarray,
    subject_bgr: np.ndarray,
    cfg: ProjectionConfig = PROJECTION_CFG,
    margin_ratio: float = 0.18,
) -> np.ndarray:
    h, w = projected_bgr.shape[:2]
    target_ratio = 16.0 / 9.0
    person = person_mask_from_foreground(subject_bgr, cfg)
    x, y, bw, bh = bbox_from_mask(person)
    if bw <= 0 or bh <= 0:
        return projected_bgr
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    crop_w = bw * (1.0 + 2.0 * margin_ratio)
    crop_h = bh * (1.0 + 2.0 * margin_ratio)
    if crop_w / max(1e-6, crop_h) < target_ratio:
        crop_w = crop_h * target_ratio
    else:
        crop_h = crop_w / target_ratio
    scale = min(w / max(1e-6, crop_w), h / max(1e-6, crop_h), 1.0)
    crop_w *= scale
    crop_h *= scale
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x1 = int(round(cx + crop_w / 2.0))
    y1 = int(round(cy + crop_h / 2.0))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= (x1 - w)
        x1 = w
    if y1 > h:
        y0 -= (y1 - h)
        y1 = h
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return projected_bgr
    cropped = projected_bgr[y0:y1, x0:x1]
    if cropped.size == 0:
        return projected_bgr
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def resize_long_side(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    cur = max(h, w)
    if cur <= max_side:
        return img_bgr
    scale = max_side / float(cur)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def projection_preview_with_report(
    camo_img: PILImage.Image,
    cfg: ProjectionConfig = PROJECTION_CFG,
    user_scale: float = 1.0,
    preview_mode: str = "quality",
) -> Tuple[PILImage.Image, ProjectionVerificationReport]:
    analysis = get_projection_subject_analysis(cfg)
    subject_bgr = analysis.subject_bgr.copy()
    camo_bgr = pil_rgb_to_bgr(camo_img)

    if preview_mode == "fast":
        subject_small = resize_long_side(subject_bgr, 960)
        camo_small = resize_long_side(camo_bgr, 960)
        fast_analysis = build_projection_analysis_from_bgr(subject_small, cfg=cfg, model_path=analysis.model_path)
        projected_bgr, _mask, report = apply_camo_to_reference(
            subject_small,
            camo_small,
            cfg=cfg,
            user_scale=user_scale,
            analysis=fast_analysis,
            fast_preview=True,
        )
        projected_bgr = crop_person_display_16_9(projected_bgr, subject_small, cfg=cfg)
        return bgr_to_pil_rgb(projected_bgr), report

    projected_bgr, _mask, report = apply_camo_to_reference(
        subject_bgr,
        camo_bgr,
        cfg=cfg,
        user_scale=user_scale,
        analysis=analysis,
        fast_preview=False,
    )
    projected_bgr = crop_person_display_16_9(projected_bgr, subject_bgr, cfg=cfg)
    return bgr_to_pil_rgb(projected_bgr), report


def projection_preview_image(
    camo_img: PILImage.Image,
    cfg: ProjectionConfig = PROJECTION_CFG,
    user_scale: float = 1.0,
    preview_mode: str = "quality",
) -> PILImage.Image:
    projected, _report = projection_preview_with_report(
        camo_img,
        cfg=cfg,
        user_scale=user_scale,
        preview_mode=preview_mode,
    )
    return projected

    projected_bgr, _mask, report = apply_camo_to_reference(
        subject_bgr,
        camo_bgr,
        cfg=cfg,
        user_scale=user_scale,
        analysis=analysis,
        fast_preview=False,
    )
    if not report.valid:
        raise RuntimeError(
            f"Projection refusée : vert originel encore visible ({report.residual_pixels} px, {report.residual_ratio:.2%})."
        )
    projected_bgr = crop_person_display_16_9(projected_bgr, subject_bgr, cfg=cfg)
    return bgr_to_pil_rgb(projected_bgr)

    projected_bgr, _mask = apply_camo_to_reference(subject_bgr, camo_bgr, cfg=cfg, user_scale=user_scale, analysis=analysis, fast_preview=False)
    projected_bgr = crop_person_display_16_9(projected_bgr, subject_bgr, cfg=cfg)
    return bgr_to_pil_rgb(projected_bgr)


# ============================================================
# WIDGETS UI
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
        self.thumb = make_fill_image()
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
        self.projection_warmup_future: Optional[Future] = None
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
        self.motif_scale = DEFAULT_UI_MOTIF_SCALE
        self.projection_preview_scale = DEFAULT_PROJECTION_PREVIEW_SCALE
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

        self.dynamic_tolerance_enabled = bool(getattr(camo, "DEFAULT_DYNAMIC_TOLERANCE_ENABLED", True))
        self.rejection_rate_window = int(getattr(camo, "DEFAULT_REJECTION_RATE_WINDOW", 24))
        self.rejection_rate_high = float(getattr(camo, "DEFAULT_REJECTION_RATE_HIGH", 0.90))
        self.rejection_rate_low = float(getattr(camo, "DEFAULT_REJECTION_RATE_LOW", 0.55))
        self.tolerance_min_attempts = int(getattr(camo, "DEFAULT_TOLERANCE_MIN_ATTEMPTS", 24))
        self.tolerance_relax_step = float(getattr(camo, "DEFAULT_TOLERANCE_RELAX_STEP", 0.08))
        self.max_tolerance_relax = float(getattr(camo, "MAX_TOLERANCE_RELAX", 0.40))
        self.max_repair_rounds = int(getattr(camo, "MAX_REPAIR_ROUNDS", 3))
        self.anti_pixel = bool(getattr(camo, "DEFAULT_ENABLE_ANTI_PIXEL", True))
        self.tolerance_relax_level = 0.0
        self.tolerance_profile: Optional[Any] = None
        self.tolerance_initial_profile: Optional[Any] = None
        self.tolerance_last_snapshot: Optional[Dict[str, float]] = None
        self.tolerance_change_count = 0
        self.tolerance_last_change_attempt = 0
        self.tolerance_history: List[Dict[str, Any]] = []
        self.tolerance_runtime: Dict[str, float] = {
            "rejection_rate": 0.0,
            "window_count": 0.0,
            "relax_before": 0.0,
            "relax_after": 0.0,
        }
        self.tolerance_outcomes: List[bool] = []
        self.validation_event_count = 0
        self.last_validation_payload: Optional[Dict[str, Any]] = None

        self._runtime_subscription_active = False
        self._runtime_subscriber_callback = None
        self.gallery_projection_cache: Dict[str, PILImage.Image] = {}
        self.gallery_projection_pending: Dict[str, List["GalleryThumb"]] = {}
        self.preview_projection_cache: Dict[str, PILImage.Image] = {}
        self.gallery_projection_semaphore = asyncio.Semaphore(2)
        self.preview_projection_semaphore = asyncio.Semaphore(1)
        self.requested_target_count = DEFAULT_TARGET_COUNT
        self.current_report_name = REPORT_NAME
        self.current_generation_mode = "classic"
        self.use_mldl = bool(camo_mldl is not None and getattr(camo_mldl, "TORCH_AVAILABLE", False))
        self.mldl_last_stats: Optional[Dict[str, Any]] = None
        self.mldl_runtime_state: Dict[str, Any] = {
            "device": "--",
            "warmup_progress": 0,
            "warmup_total": 0,
            "warmup_remaining": 0,
            "warmup_in_background": False,
            "candidate_pool_size": 0,
            "validate_top_k": 0,
            "dataset_samples": 0,
            "dataset_loaded": False,
            "checkpoint_loaded": False,
            "tolerance_state_loaded": False,
            "resume_used": False,
            "checkpoint_path": "",
            "dataset_path": "",
            "summary_path": "",
            "training_log_path": "",
            "latest_stats": None,
            "latest_error": None,
            "champion": {},
            "surrogate_trained": False,
        }
        self._current_preview_raw_img: Optional[PILImage.Image] = None
        self.pending_manual_review: Optional[PendingManualReview] = None
        self.generated_rows: List[dict] = []
        self._gallery_reload_scheduled = False

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
        self.projection_scale_slider: Optional[Slider] = None
        self.projection_scale_label: Optional[Label] = None
        self.manual_accept_btn: Optional[SoftButton] = None
        self.manual_skip_btn: Optional[SoftButton] = None
        self.manual_review_label: Optional[Label] = None
        self.manual_review_mini_label: Optional[Label] = None
        self.diag_summary_mini_label: Optional[Label] = None
        self.diag_top_rules_mini_label: Optional[Label] = None
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
        root = BoxLayout(orientation="vertical", spacing=dp(12), padding=dp(12))

        # =====================
        # HEADER
        # =====================
        header = GlassCard(orientation="horizontal", size_hint_y=None, height=dp(104), spacing=dp(14))

        hero_left = BoxLayout(orientation="vertical", spacing=dp(2))
        self.attempt_text = self._small_label("Image 000 | essai 0000 | total 000000 | seed --", color=C["text_muted"])
        title = self._label(APP_TITLE, font_size=sp(22), height=dp(30))
        title.bold = True
        subtitle = self._small_label(
            "Pilotage temps réel, validation stricte, galerie, mannequin et revue manuelle des rejets.",
            color=C["text_soft"],
            height=dp(22),
        )
        hero_left.add_widget(self.attempt_text)
        hero_left.add_widget(title)
        hero_left.add_widget(subtitle)

        hero_right = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_x=0.26)
        self.status_label = self._label("Prêt", halign="right", font_size=sp(18), height=dp(28))
        self.status_label.bold = True
        hero_hint = self._small_label("Sortie : textures + mannequins", halign="right", color=C["text_muted"])
        hero_right.add_widget(self.status_label)
        hero_right.add_widget(hero_hint)

        header.add_widget(hero_left)
        header.add_widget(hero_right)
        root.add_widget(header)

        # =====================
        # BODY
        # =====================
        body = BoxLayout(spacing=dp(12))
        left = BoxLayout(orientation="vertical", spacing=dp(12), size_hint_x=0.31)
        right = BoxLayout(orientation="vertical", spacing=dp(12), size_hint_x=0.69)

        # ---------------------
        # LEFT SIDEBAR
        # ---------------------
        control_scroll = ScrollView(do_scroll_x=False, bar_width=dp(8))
        control_content = BoxLayout(orientation="vertical", spacing=dp(12), size_hint_y=None)
        control_content.bind(minimum_height=control_content.setter("height"))

        actions_card = GlassCard(orientation="vertical", size_hint_y=None, spacing=dp(10))
        actions_card.bind(minimum_height=actions_card.setter("height"))
        actions_card.add_widget(self._section_title("Commandes", "Pilotage principal et sortie."))
        self.count_input = SoftTextInput(text=str(DEFAULT_TARGET_COUNT), multiline=False, input_filter="int", size_hint_y=None, height=dp(52))
        actions_card.add_widget(self.count_input)
        btn_row = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(10))
        self.start_btn = self._button("Commencer", "launch", self.start_generation)
        self.stop_btn = self._button("Arrêter", "stop", self.stop_generation)
        btn_row.add_widget(self.start_btn)
        btn_row.add_widget(self.stop_btn)
        actions_card.add_widget(btn_row)
        self.open_btn = self._button("Ouvrir le dossier de sortie", "neutral", lambda *_: open_folder(self.current_output_dir))
        actions_card.add_widget(self.open_btn)
        control_content.add_widget(actions_card)

        runtime_card = GlassCard(orientation="vertical", size_hint_y=None, spacing=dp(10))
        runtime_card.bind(minimum_height=runtime_card.setter("height"))
        runtime_card.add_widget(self._section_title("Exécution", "Préflight, mode de lancement et revue manuelle."))
        self.tests_label = self._small_label(self.tests_summary)
        runtime_card.add_widget(self.tests_label)

        mode_grid = GridLayout(cols=1, size_hint_y=None, spacing=dp(8), height=dp(194))
        self.mode_blocking_btn = self._button("● Tests bloquants", "launch", lambda *_: self._set_run_mode(RUN_MODE_BLOCKING))
        self.mode_non_blocking_btn = self._button("○ Tests non bloquants", "neutral", lambda *_: self._set_run_mode(RUN_MODE_NON_BLOCKING))
        self.mode_skip_tests_btn = self._button("○ Sans tests", "neutral", lambda *_: self._set_run_mode(RUN_MODE_SKIP_TESTS))
        mode_grid.add_widget(self.mode_blocking_btn)
        mode_grid.add_widget(self.mode_non_blocking_btn)
        mode_grid.add_widget(self.mode_skip_tests_btn)
        runtime_card.add_widget(mode_grid)
        self.run_mode_label = self._small_label("Mode actuel : tests bloquants", color=C["text_muted"])
        runtime_card.add_widget(self.run_mode_label)

        runtime_card.add_widget(self._section_title("Dernier rejet mémorisé", "Enregistrement manuel sans stopper la génération."))
        manual_row = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(10))
        self.manual_accept_btn = self._button("Valider ce rejet", "launch", self.manual_accept_current_reject)
        self.manual_skip_btn = self._button("Oublier ce rejet", "neutral", self.manual_skip_current_reject)
        manual_row.add_widget(self.manual_accept_btn)
        manual_row.add_widget(self.manual_skip_btn)
        runtime_card.add_widget(manual_row)
        self.manual_review_label = self._small_label("Aucun rejet en attente.")
        runtime_card.add_widget(self.manual_review_label)
        control_content.add_widget(runtime_card)

        tuning_card = GlassCard(orientation="vertical", size_hint_y=None, spacing=dp(10))
        tuning_card.bind(minimum_height=tuning_card.setter("height"))
        tuning_card.add_widget(self._section_title("Réglages", "Motif et projection mannequin."))

        tuning_card.add_widget(self._small_label("Scale des motifs", color=C["text_muted"]))
        motif_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8))
        self.motif_scale_slider = Slider(min=0.18, max=1.20, value=DEFAULT_UI_MOTIF_SCALE, step=0.01)
        self.motif_scale_label = self._small_label(f"{DEFAULT_UI_MOTIF_SCALE:.2f}", size_hint_x=0.22, halign="right")
        self.motif_scale_slider.bind(value=self._on_motif_scale_change)
        motif_row.add_widget(self.motif_scale_slider)
        motif_row.add_widget(self.motif_scale_label)
        tuning_card.add_widget(motif_row)

        tuning_card.add_widget(self._small_label("Scale projection mannequin", color=C["text_muted"]))
        projection_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8))
        self.projection_scale_slider = Slider(min=0.08, max=0.95, value=self.projection_preview_scale, step=0.01)
        self.projection_scale_label = self._small_label(f"{self.projection_preview_scale:.2f}", size_hint_x=0.22, halign="right")
        self.projection_scale_slider.bind(value=self._on_projection_scale_change)
        projection_row.add_widget(self.projection_scale_slider)
        projection_row.add_widget(self.projection_scale_label)
        tuning_card.add_widget(projection_row)
        control_content.add_widget(tuning_card)

        health_card = GlassCard(orientation="vertical", size_hint_y=None, spacing=dp(8))
        health_card.bind(minimum_height=health_card.setter("height"))
        health_card.add_widget(self._section_title("Santé & qualité", "Résumé compact de l’état backend."))
        self.resource_text = self._small_label("Machine : CPU -- | RAM -- | Disque -- | Processus --", height=dp(40))
        self.score_text = self._small_label("Qualité : MAE -- | max abs -- | composant -- | best-of --", height=dp(30))
        self.color_text = self._small_label("Couleurs : C -- | O -- | T -- | G --", height=dp(30))
        self.extra_text = self._small_label("Contours : bd -- | bd/4 -- | bd/8 -- | bord --", height=dp(30))
        self.struct_text = self._small_label("Tolérance dynamique : --", height=dp(46))
        self.runtime_last_label = self._small_label("Preuve dataset/runtime : --", height=dp(46))
        self.diag_summary_label = self._small_label("Résumé essais : tentatives 0 | acceptés 0 | rejetés 0", height=dp(46))
        self.diag_top_rules_label = self._small_label("Pourquoi ça rejette : --", height=dp(46))
        self.diag_last_fail_label = self._small_label("ML / DL : --", height=dp(46))
        for widget in [
            self.resource_text,
            self.score_text,
            self.color_text,
            self.extra_text,
            self.struct_text,
            self.runtime_last_label,
            self.diag_summary_label,
            self.diag_top_rules_label,
            self.diag_last_fail_label,
        ]:
            health_card.add_widget(widget)
        control_content.add_widget(health_card)

        gallery_card = GlassCard(orientation="vertical", size_hint_y=None, height=dp(360), spacing=dp(10))
        gallery_card.add_widget(self._section_title("Galerie récente", "Clique une vignette pour charger immédiatement l’aperçu."))
        gallery_scroll = ScrollView(do_scroll_x=False, bar_width=dp(8))
        self.gallery_grid = GridLayout(cols=GALLERY_COLUMNS, spacing=dp(10), padding=dp(2), size_hint_y=None)
        self.gallery_grid.bind(minimum_height=self.gallery_grid.setter("height"))
        gallery_scroll.add_widget(self.gallery_grid)
        gallery_card.add_widget(gallery_scroll)
        control_content.add_widget(gallery_card)

        control_scroll.add_widget(control_content)
        left.add_widget(control_scroll)

        # ---------------------
        # RIGHT CONTENT
        # ---------------------
        preview_row = GridLayout(cols=2, spacing=dp(12), size_hint_y=None, height=dp(300))
        self.preview_img = Image()
        self.preview_silhouette = Image()
        preview_row.add_widget(self._carded_view("Camouflage courant", self.preview_img))
        preview_row.add_widget(self._carded_view("Projection sur soldat modèle", self.preview_silhouette))
        right.add_widget(preview_row)

        info_row = GridLayout(cols=3, spacing=dp(12), size_hint_y=None, height=dp(110))

        progress_card = GlassCard(orientation="vertical", spacing=dp(8))
        progress_card.add_widget(self._section_title("Progression", "Suivi du lot courant."))
        self.progress_bar = GlassProgressBar()
        self.progress_text = self._small_label("0 / 0 validé(s) | tentatives 0 | rejetés 0")
        progress_card.add_widget(self.progress_bar)
        progress_card.add_widget(self.progress_text)
        info_row.add_widget(progress_card)

        backend_card = GlassCard(orientation="vertical", spacing=dp(8))
        backend_card.add_widget(self._section_title("Validation stricte", "Synthèse instantanée du backend."))
        self.diag_summary_mini_label = self._small_label("Tentatives 0 | acceptés 0 | rejetés 0 | taux 0.00%")
        self.diag_top_rules_mini_label = self._small_label("Top règles : --")
        backend_card.add_widget(self.diag_summary_mini_label)
        backend_card.add_widget(self.diag_top_rules_mini_label)
        info_row.add_widget(backend_card)

        manual_card = GlassCard(orientation="vertical", spacing=dp(8))
        manual_card.add_widget(self._section_title("Revue manuelle", "Dernier rejet disponible à l’enregistrement."))
        self.manual_review_mini_label = self._small_label("Aucun rejet en attente.")
        manual_card.add_widget(self.manual_review_mini_label)
        info_row.add_widget(manual_card)
        right.add_widget(info_row)

        live_card = GlassCard(orientation="vertical", size_hint_y=None, height=dp(190), spacing=dp(8))
        live_card.add_widget(self._section_title("Suivi direct de construction", "Ce que le pipeline est en train de faire maintenant."))
        self.live_stage_label = self._small_label("Étape : attente")
        self.live_counts_label = self._small_label("État backend strict")
        self.live_meta_label = self._small_label("Image -- | essai -- | seed --")
        live_card.add_widget(self.live_stage_label)
        live_card.add_widget(self.live_counts_label)
        live_card.add_widget(self.live_meta_label)
        self.live_preview_img = Image()
        live_pane = SoftPane(orientation="vertical")
        live_pane.add_widget(self.live_preview_img)
        live_card.add_widget(live_pane)
        right.add_widget(live_card)

        bottom = GridLayout(cols=1, spacing=dp(12), size_hint_y=1)
        log_card = GlassCard(orientation="vertical", spacing=dp(10), size_hint_y=0.54)
        log_card.add_widget(self._section_title("Journal opérationnel", "Vue large pour les événements généraux, exports et états du front."))
        log_actions = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(10))
        log_actions.add_widget(self._button("Ouvrir en grand", "neutral", lambda *_: self._open_text_modal("Journal opérationnel", "log")))
        log_card.add_widget(log_actions)
        self.log_view = LogView()
        log_card.add_widget(self.log_view)
        bottom.add_widget(log_card)

        diag_card = GlassCard(orientation="vertical", spacing=dp(10), size_hint_y=0.46)
        diag_card.add_widget(self._section_title("Diagnostic live", "Vue large des rejets, règles et détails candidat par candidat."))
        diag_actions = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(10))
        diag_actions.add_widget(self._button("Ouvrir en grand", "neutral", lambda *_: self._open_text_modal("Diagnostic live", "diag")))
        diag_card.add_widget(diag_actions)
        self.diag_log_view = LogView()
        diag_card.add_widget(self.diag_log_view)
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

    def _section_title(self, title: str, subtitle: str = "") -> Widget:
        box = BoxLayout(orientation="vertical", spacing=dp(2), size_hint_y=None)
        box.bind(minimum_height=box.setter("height"))
        ttl = self._label(title, font_size=sp(16), height=dp(24))
        ttl.bold = True
        box.add_widget(ttl)
        if subtitle:
            box.add_widget(self._small_label(subtitle, color=C["text_muted"], height=dp(20)))
        return box

    def _button(self, text: str, role: str, callback) -> SoftButton:
        btn = SoftButton(text=text, role=role)
        btn.bind(on_release=callback)
        return btn


    def _collect_log_text(self, source: str) -> str:
        view = self.log_view if source == "log" else self.diag_log_view
        if view is None or getattr(view, "label", None) is None:
            return "Aucun contenu disponible."
        text = str(view.label.text or "").strip()
        return text or "Aucun contenu disponible."

    def _open_text_modal(self, title: str, source: str) -> None:
        content = BoxLayout(orientation="vertical", spacing=dp(12), padding=dp(12))

        heading = self._label(title, font_size=sp(19), height=dp(30))
        heading.bold = True
        content.add_widget(heading)

        viewer = TextInput(
            text=self._collect_log_text(source),
            readonly=True,
            multiline=True,
            font_size=sp(15),
            background_normal="",
            background_active="",
            background_color=C["bg_input"],
            foreground_color=C["text_main"],
            cursor_color=C["text_main"],
            padding=[dp(14), dp(14), dp(14), dp(14)],
        )
        content.add_widget(viewer)

        buttons = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(10))
        refresh_btn = self._button("Actualiser", "neutral", lambda *_: setattr(viewer, "text", self._collect_log_text(source)))
        close_btn = self._button("Fermer", "launch", lambda *_: popup.dismiss())
        buttons.add_widget(refresh_btn)
        buttons.add_widget(close_btn)
        content.add_widget(buttons)

        popup = Popup(
            title="",
            content=content,
            size_hint=(0.96, 0.92),
            separator_height=0,
            auto_dismiss=True,
        )
        popup.open()

    def _set_image_widget_texture(self, image_widget: Image, pil_img: Optional[PILImage.Image]):
        placeholder = getattr(image_widget, "_placeholder_label", None)
        try:
            if pil_img is None:
                image_widget.texture = None
                image_widget.opacity = 0.0
                if placeholder is not None:
                    placeholder.opacity = 1.0
                return
            image_widget.texture = pil_to_coreimage(pil_img).texture
            image_widget.opacity = 1.0
            if placeholder is not None:
                placeholder.opacity = 0.0
        except Exception:
            image_widget.texture = None
            image_widget.opacity = 0.0
            if placeholder is not None:
                placeholder.opacity = 1.0

    def _carded_view(self, title: str, image_widget: Image) -> Widget:
        box = GlassCard(orientation="vertical", spacing=dp(8))
        box.add_widget(self._section_title(title))
        pane = SoftPane(orientation="vertical", size_hint_y=None)

        def _sync_ratio(*_args):
            inner_w = max(dp(120), pane.width - dp(16))
            pane.height = (inner_w * 9.0 / 16.0) + dp(16)

        pane.bind(width=_sync_ratio)
        Clock.schedule_once(lambda _dt: _sync_ratio(), 0)

        stage = FloatLayout()
        try:
            image_widget.fit_mode = "contain"
        except Exception:
            try:
                image_widget.allow_stretch = True
                image_widget.keep_ratio = True
            except Exception:
                pass
        image_widget.size_hint = (1, 1)
        image_widget.pos_hint = {"x": 0, "y": 0}
        image_widget.opacity = 0.0

        placeholder = Label(
            text=f"{title}\nAperçu en attente",
            font_size=sp(16),
            halign="center",
            valign="middle",
            color=C["text_muted"],
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
        )
        placeholder.bind(size=lambda *a: setattr(placeholder, "text_size", placeholder.size))
        image_widget._placeholder_label = placeholder

        stage.add_widget(image_widget)
        stage.add_widget(placeholder)
        pane.add_widget(stage)
        box.add_widget(pane)
        return box

    def schedule_gallery_reload(self, delay: float = 0.12) -> None:
        Clock.unschedule(self._run_scheduled_gallery_reload)
        self._gallery_reload_scheduled = True
        Clock.schedule_once(self._run_scheduled_gallery_reload, delay)

    def _run_scheduled_gallery_reload(self, _dt) -> None:
        self._gallery_reload_scheduled = False
        self.reload_gallery()

    def _projection_cache_key(self, image_path: Path) -> str:
        try:
            stat = image_path.stat()
            return f"{image_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}::projscale={self.projection_preview_scale:.2f}"
        except Exception:
            return f"{image_path.resolve()}::projscale={self.projection_preview_scale:.2f}"

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
        async with self.gallery_projection_semaphore:
            pil_img = await asyncio.to_thread(read_pil_rgb, image_path)
            await asyncio.to_thread(get_projection_subject_analysis, PROJECTION_CFG)
            projected = await asyncio.to_thread(
                projection_preview_image,
                pil_img,
                PROJECTION_CFG,
                self.projection_preview_scale,
                "fast",
            )
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
        async with self.preview_projection_semaphore:
            pil_img = await asyncio.to_thread(read_pil_rgb, image_path)
            await asyncio.to_thread(get_projection_subject_analysis, PROJECTION_CFG)
            return await asyncio.to_thread(
                projection_preview_image,
                pil_img,
                PROJECTION_CFG,
                self.projection_preview_scale,
                "quality",
            )

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
        Clock.schedule_once(lambda _dt: self._deferred_startup_projection_warmup(), 0)

    def _deferred_startup_projection_warmup(self):
        try:
            model_path = resolve_soldier_model_path()
            self.log(f"Modèle soldat chargé : {model_path}")
            if not self.running and not self.preflight_running:
                self.status("Initialisation mannequin…", ok=True)
        except Exception as exc:
            self.log(str(exc))
            self.status("Modèle soldat introuvable", ok=False)
            return

        try:
            fut = self.async_runner.submit(self._async_warmup_projection_subject())
            self.projection_warmup_future = fut
            fut.add_done_callback(self._on_projection_warmup_done)
        except Exception as exc:
            self.log(f"Préchargement mannequin impossible : {exc}")
            if not self.running and not self.preflight_running:
                self.status("Mannequin indisponible", ok=False)

    async def _async_warmup_projection_subject(self) -> ProjectionSubjectAnalysis:
        return await asyncio.to_thread(get_projection_subject_analysis, PROJECTION_CFG)

    def _on_projection_warmup_done(self, fut: Future):
        try:
            analysis = fut.result()
            model_path = getattr(analysis, "model_path", None)
            payload = str(model_path) if model_path is not None else ""
            Clock.schedule_once(lambda _dt, p=payload: self._finish_projection_warmup(True, p), 0)
        except Exception as exc:
            Clock.schedule_once(lambda _dt, e=str(exc): self._finish_projection_warmup(False, e), 0)

    @mainthread
    def _finish_projection_warmup(self, ok: bool, payload: str):
        self.projection_warmup_future = None
        if ok:
            self.log(f"Analyse mannequin prête : {payload}")
            if not self.running and not self.preflight_running:
                self.status("Prêt", ok=True)
            return

        self.log(f"Préchargement mannequin impossible : {payload}")
        if not self.running and not self.preflight_running:
            self.status("Mannequin indisponible", ok=False)

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
            short = line if len(line) <= 180 else line[:177] + "..."
            self.runtime_last_label.text = f"Preuve dataset/runtime : {short}"

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
        self._current_preview_raw_img = pil_img.copy()
        if self.preview_img is not None:
            self._set_image_widget_texture(self.preview_img, pil_img)
        if self.preview_silhouette is not None:
            self._set_image_widget_texture(self.preview_silhouette, projection_img)

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
                self._set_image_widget_texture(self.live_preview_img, pil_img)
            elif preview_path and self.live_preview_img is not None and Path(preview_path).exists():
                img = PILImage.open(preview_path).convert("RGB")
                self._set_image_widget_texture(self.live_preview_img, img)
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

    @mainthread
    def _refresh_controls_state(self):
        if self.start_btn is not None:
            self.start_btn.disabled = self.running or self.stopping or self.preflight_running
        if self.stop_btn is not None:
            self.stop_btn.disabled = not (self.running or self.preflight_running or self.stopping)
        if self.open_btn is not None:
            self.open_btn.disabled = False
        manual_enabled = bool(self.pending_manual_review is not None and not bool(getattr(self.pending_manual_review, "manually_saved", False)))
        if self.manual_accept_btn is not None:
            self.manual_accept_btn.disabled = not manual_enabled
        if self.manual_skip_btn is not None:
            self.manual_skip_btn.disabled = self.pending_manual_review is None
        self._refresh_run_mode_buttons()

    def _apply_backend_motif_scale(self) -> float:
        scale = max(0.18, float(self.motif_scale))
        setter = getattr(camo, "set_motif_scale", None)
        if callable(setter):
            setter(scale)
        else:
            camo.set_canvas_geometry(
                width=int(getattr(camo, "WIDTH", 7680)),
                height=int(getattr(camo, "HEIGHT", 4320)),
                physical_width_cm=float(getattr(camo, "PHYSICAL_WIDTH_CM", 768.0)),
                physical_height_cm=float(getattr(camo, "PHYSICAL_HEIGHT_CM", 432.0)),
                motif_scale=scale,
            )
        return scale

    def _on_motif_scale_change(self, _slider, value):
        self.motif_scale = float(value)
        applied = self._apply_backend_motif_scale()
        if self.motif_scale_label is not None:
            self.motif_scale_label.text = f"{applied:.2f}"

    def _on_projection_scale_change(self, _slider, value):
        self.projection_preview_scale = float(value)
        if self.projection_scale_label is not None:
            self.projection_scale_label.text = f"{self.projection_preview_scale:.2f}"

        self.gallery_projection_cache.clear()
        self.preview_projection_cache.clear()

        if self._current_preview_raw_img is not None:
            raw = self._current_preview_raw_img.copy()
            fut = self.async_runner.submit(
                asyncio.to_thread(
                    projection_preview_image,
                    raw,
                    PROJECTION_CFG,
                    self.projection_preview_scale,
                    "fast",
                )
            )
            fut.add_done_callback(lambda f, img=raw: self._on_live_projection_scale_done(img, f))

        self.schedule_gallery_reload(0.08)

    def _on_live_projection_scale_done(self, raw_img: PILImage.Image, fut: Future):
        try:
            projected = fut.result()
        except Exception as exc:
            self.log(f"Projection live impossible : {exc}")
            return
        Clock.schedule_once(lambda _dt, r=raw_img, p=projected: self.update_preview(r, p), 0)

    @mainthread
    def _arm_manual_review(self, review: PendingManualReview):
        self.pending_manual_review = review
        rules = " | ".join(rejection_rules_for_candidate(review.candidate, review.outcome)[:4]) or "rejet backend"
        text = (
            f"Dernier rejet mémorisé : image {review.target_index:03d} | essai {review.local_attempt:04d} | "
            f"seed {review.candidate.seed} | {rules}"
        )
        if self.manual_review_label is not None:
            self.manual_review_label.text = text
        if self.manual_review_mini_label is not None:
            self.manual_review_mini_label.text = text
        self._refresh_controls_state()

    @mainthread
    def _clear_manual_review(self):
        self.pending_manual_review = None
        if self.manual_review_label is not None:
            self.manual_review_label.text = "Aucun rejet en attente."
        if self.manual_review_mini_label is not None:
            self.manual_review_mini_label.text = "Aucun rejet en attente."
        self._refresh_controls_state()

    def manual_accept_current_reject(self, *_):
        review = self.pending_manual_review
        if review is None or bool(review.manually_saved):
            return
        self.log("Validation manuelle demandée pour le dernier rejet mémorisé.")
        fut = self.async_runner.submit(self._async_manual_accept_review(review))
        fut.add_done_callback(lambda f: self._on_manual_accept_done(f))

    def manual_skip_current_reject(self, *_):
        if self.pending_manual_review is None:
            return
        self.log("Dernier rejet oublié manuellement.")
        self._clear_manual_review()

    async def _async_manual_accept_review(self, review: PendingManualReview) -> Tuple[Path, Path, PendingManualReview]:
        saved_path, mannequin_saved_path = await self._async_save_candidate_bundle(
            self.generated_rows,
            target_index=review.target_index,
            local_attempt=review.local_attempt,
            global_attempt=review.global_attempt,
            candidate=review.candidate,
            outcome=review.outcome,
            projection_img=review.projection_img,
            manual_accept=True,
        )
        review.manually_saved = True
        return saved_path, mannequin_saved_path, review

    def _on_manual_accept_done(self, fut: Future):
        try:
            saved_path, mannequin_saved_path, review = fut.result()
        except Exception as exc:
            Clock.schedule_once(lambda _dt, e=str(exc): self.log(f"Validation manuelle impossible : {e}"), 0)
            return
        Clock.schedule_once(lambda _dt, sp=saved_path, mp=mannequin_saved_path, rv=review: self._finish_manual_accept(sp, mp, rv), 0)

    @mainthread
    def _finish_manual_accept(self, saved_path: Path, mannequin_saved_path: Path, review: PendingManualReview):
        self.accepted_count = len(self.generated_rows)
        self.log(
            f"Rejet validé manuellement -> {saved_path.name} | mannequin -> {(mannequin_saved_path.name if mannequin_saved_path is not None else 'non_enregistré')}"
        )
        text = f"Rejet déjà enregistré : image {review.target_index:03d} | seed {review.candidate.seed}"
        if self.manual_review_label is not None:
            self.manual_review_label.text = text
        if self.manual_review_mini_label is not None:
            self.manual_review_mini_label.text = text
        self.reload_gallery()
        self._refresh_controls_state()

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

    def _reset_dynamic_tolerance_state(self):
        self.tolerance_relax_level = 0.0
        builder = getattr(camo, "build_validation_tolerance_profile", None)
        if callable(builder):
            try:
                self.tolerance_profile = builder(0.0)
            except Exception:
                self.tolerance_profile = None
        else:
            self.tolerance_profile = None
        self.tolerance_initial_profile = self.tolerance_profile
        self.tolerance_last_snapshot = self._profile_snapshot(self.tolerance_profile)
        self.tolerance_change_count = 0
        self.tolerance_last_change_attempt = 0
        self.tolerance_history = []
        self.tolerance_runtime = {
            "rejection_rate": 0.0,
            "window_count": 0.0,
            "relax_before": 0.0,
            "relax_after": 0.0,
        }
        self.tolerance_outcomes = []
        self.validation_event_count = 0
        self.last_validation_payload = None

    def _update_dynamic_tolerance_profile(
        self,
        accepted: Optional[bool] = None,
        global_attempt: Optional[int] = None,
        *extra_args: Any,
        sync_target: Optional[Any] = None,
        **extra_kwargs: Any,
    ):
        adapter = getattr(camo, "adapt_tolerance_relax_level", None)
        builder = getattr(camo, "build_validation_tolerance_profile", None)

        normalized_accepted: Optional[bool] = None
        normalized_global_attempt: Optional[int] = None

        probe_values = [accepted, global_attempt, *extra_args]
        outcome_obj = extra_kwargs.get("outcome")
        if outcome_obj is not None:
            probe_values.append(outcome_obj)

        for value in probe_values:
            if normalized_accepted is None and isinstance(value, bool):
                normalized_accepted = bool(value)
                continue
            if normalized_global_attempt is None and isinstance(value, int) and not isinstance(value, bool):
                normalized_global_attempt = int(value)
                continue
            if normalized_accepted is None and hasattr(value, "accepted"):
                try:
                    normalized_accepted = bool(getattr(value, "accepted"))
                except Exception:
                    pass

        if normalized_accepted is not None:
            self._remember_tolerance_outcome(normalized_accepted)

        if callable(adapter):
            try:
                relax, profile, runtime = adapter(
                    self.tolerance_relax_level,
                    self.tolerance_outcomes,
                    window=self.rejection_rate_window,
                    rejection_rate_high=self.rejection_rate_high,
                    rejection_rate_low=self.rejection_rate_low,
                    min_attempts=self.tolerance_min_attempts,
                    relax_step=self.tolerance_relax_step,
                    enabled=self.dynamic_tolerance_enabled,
                )
                self.tolerance_relax_level = float(relax)
                self.tolerance_profile = profile
                self.tolerance_runtime = {
                    "rejection_rate": float(runtime.get("rejection_rate", 0.0)),
                    "window_count": float(runtime.get("window_count", 0.0)),
                    "relax_before": float(runtime.get("relax_before", 0.0)),
                    "relax_after": float(runtime.get("relax_after", 0.0)),
                }
            except Exception:
                if callable(builder):
                    try:
                        self.tolerance_profile = builder(self.tolerance_relax_level)
                    except Exception:
                        self.tolerance_profile = None
        elif callable(builder):
            try:
                self.tolerance_profile = builder(self.tolerance_relax_level)
            except Exception:
                self.tolerance_profile = None

        if sync_target is not None:
            try:
                setattr(sync_target, "tolerance_profile", self.tolerance_profile)
            except Exception:
                pass

        if normalized_global_attempt is not None:
            self._remember_tolerance_change(normalized_global_attempt)

    def _remember_tolerance_outcome(self, accepted: bool):
        self.tolerance_outcomes.append(bool(accepted))
        keep = max(self.rejection_rate_window * 8, self.tolerance_min_attempts * 4, 64)
        if len(self.tolerance_outcomes) > keep:
            self.tolerance_outcomes = self.tolerance_outcomes[-keep:]

    def _tolerance_debug_text(self) -> str:
        rejection_rate = float(self.tolerance_runtime.get("rejection_rate", 0.0))
        window_count = int(round(float(self.tolerance_runtime.get("window_count", 0.0))))
        bestof_min = None
        if self.tolerance_profile is not None:
            try:
                bestof_min = float(getattr(self.tolerance_profile, "bestof_min_score"))
            except Exception:
                bestof_min = None
        txt = (
            f"tol relax {self.tolerance_relax_level:.2f}/{self.max_tolerance_relax:.2f} | "
            f"rej-fen {rejection_rate:.0%} | fen {window_count}"
        )
        if bestof_min is not None:
            txt += f" | best-min {bestof_min:.4f}"
        return txt

    def _profile_snapshot(self, profile: Optional[Any]) -> Dict[str, float]:
        if profile is None:
            return {}
        def _g(name: str, default: float = 0.0) -> float:
            try:
                return float(getattr(profile, name, default))
            except Exception:
                return float(default)
        try:
            per_color = tuple(float(x) for x in getattr(profile, "max_abs_error_per_color", (0.0, 0.0, 0.0, 0.0)))
        except Exception:
            per_color = (0.0, 0.0, 0.0, 0.0)
        return {
            "relax_level": _g("relax_level"),
            "mean_abs_error": _g("max_mean_abs_error"),
            "bestof_min_score": _g("bestof_min_score"),
            "mirror_max": _g("max_mirror_similarity"),
            "largest_min": _g("min_largest_component_ratio_class_1"),
            "edge_max": _g("max_edge_contact_ratio"),
            "orphan_max": _g("max_orphan_ratio"),
            "micro_max": _g("max_micro_islands_per_mp"),
            "ratio_abs_mean": float(sum(per_color) / max(1, len(per_color))),
        }

    def _remember_tolerance_change(self, global_attempt: int):
        snap = self._profile_snapshot(self.tolerance_profile)
        if not snap:
            return
        if self.tolerance_last_snapshot is None:
            self.tolerance_last_snapshot = dict(snap)
            return
        changed = any(abs(float(snap.get(k, 0.0)) - float(self.tolerance_last_snapshot.get(k, 0.0))) > 1e-12 for k in snap.keys())
        if changed:
            self.tolerance_change_count += 1
            self.tolerance_last_change_attempt = int(global_attempt)
            self.tolerance_history.append({
                "global_attempt": int(global_attempt),
                "snapshot": dict(snap),
                "runtime": dict(self.tolerance_runtime),
            })
            self.tolerance_last_snapshot = dict(snap)

    def _count_file_lines(self, path: Path) -> int:
        try:
            if not path.exists():
                return 0
            with path.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _ml_dl_status_text(self) -> str:
        out = Path(self.current_output_dir)
        checkpoint = out / "surrogate_camouflage.pt"
        if checkpoint.exists():
            return "ML/DL : artefacts détectés, mais non synchronisé finement avec le front"
        return "ML/DL : non synchronisé / indisponible côté front"

    def _tolerance_proof_text(self) -> str:
        if self.last_validation_payload:
            return "Tolérance dynamique : pilotée par le backend (état partiel reçu)"
        return "Tolérance dynamique : non synchronisé côté front"

    def _human_rejection_text(self) -> str:
        rules = [name for name, _count in self.diag_rule_counter.most_common(3)]
        if not rules:
            return "aucun rejet enregistré"
        explanation = " | ".join(rules)
        if "orphan_pixels" in rules or "micro_islands" in rules:
            explanation += " | cause dominante: orphelins / micro-îlots restent stricts, donc la relaxation ne les desserre pas"
        elif "not_bestof" in rules:
            explanation += " | cause dominante: le best-of s'est desserré, mais le candidat ne passe toujours pas les règles strictes"
        return explanation

    @mainthread
    def _refresh_diag_labels(self):
        rate = (self.diag_accepts / self.diag_total) if self.diag_total else 0.0
        summary_text = (
            f"Résumé essais : tentatives {self.diag_total} | acceptés {self.diag_accepts} | "
            f"rejetés {self.diag_rejects} | taux {rate:.2%}"
        )
        if self.diag_summary_label is not None:
            self.diag_summary_label.text = summary_text
        if self.diag_summary_mini_label is not None:
            self.diag_summary_mini_label.text = summary_text

        top = " | ".join(f"{n}:{c}" for n, c in self.diag_rule_counter.most_common(3)) if self.diag_rule_counter else "--"
        why_text = f"Pourquoi ça rejette : {top} | {self._human_rejection_text()}"
        if self.diag_top_rules_label is not None:
            self.diag_top_rules_label.text = why_text
        if self.diag_top_rules_mini_label is not None:
            self.diag_top_rules_mini_label.text = why_text

        ml_text = self._ml_dl_status_text()
        if self.diag_last_fail_label is not None:
            self.diag_last_fail_label.text = f"ML / DL : {ml_text}"

    # ---------- gallery ----------
    @mainthread
    def reload_gallery(self):
        if self.gallery_grid is None:
            return
        self.gallery_grid.clear_widgets()
        if not self.current_output_dir.exists():
            return
        paths = sorted(self.current_output_dir.glob(OUTPUT_IMAGE_GLOB))
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
                count = int(self.requested_target_count)
            except Exception:
                count = DEFAULT_TARGET_COUNT

            self._apply_backend_motif_scale()
            intensity = backend_machine_intensity(self.machine_intensity)
            output_dir = self.current_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            max_workers = int(getattr(camo, "CPU_COUNT", os.cpu_count() or 1))
            attempt_batch_size = max(max_workers, max_workers * 2)
            await asyncio.to_thread(
                camo.validate_generation_request,
                target_count=max(1, count),
                output_dir=output_dir,
                base_seed=int(getattr(camo, "DEFAULT_BASE_SEED", 0)),
                machine_intensity=1.0,
                max_workers=max_workers,
                attempt_batch_size=attempt_batch_size,
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
    def _mldl_available(self) -> bool:
        return bool(camo_mldl is not None and getattr(camo_mldl, "TORCH_AVAILABLE", False))

    def _build_mldl_config(self, target_count: int):
        if camo_mldl is None:
            raise RuntimeError("camouflage_ml_dl.py est indisponible.")
        warmup_samples = max(64, min(192, int(target_count) * 6))
        kwargs = dict(
            target_count=int(target_count),
            warmup_samples=int(warmup_samples),
            candidate_pool_size=8,
            validate_top_k=3,
            max_attempts_per_target=max(120, int(target_count) * 12),
            train_epochs=24,
            batch_size=32,
            learning_rate=1e-3,
            hidden_dim=128,
            device="auto",
            base_seed=int(getattr(camo, "DEFAULT_BASE_SEED", 202604010001)),
            output_dir=str(self.current_output_dir),
            report_name=DEFAULT_MLDL_REPORT_NAME,
            random_seed=int(getattr(camo, "DEFAULT_BASE_SEED", 202604010001) & 0x7FFF_FFFF),
            parallel_train_enabled=True,
            parallel_train_min_interval_s=3.0,
            min_train_size=12,
            retrain_every=8,
        )
        fields = getattr(getattr(camo_mldl, "MLDLConfig", None), "__dataclass_fields__", {}) or {}
        if "pretrain_relax_level" in fields:
            kwargs["pretrain_relax_level"] = 0.18
        if "pretrain_max_orphan_ratio" in fields:
            kwargs["pretrain_max_orphan_ratio"] = 0.0015
        if "pretrain_max_micro_islands_per_mp" in fields:
            kwargs["pretrain_max_micro_islands_per_mp"] = 2.0
        if "warmup_persist_every" in fields:
            kwargs["warmup_persist_every"] = 8
        if "tolerance_state_name" in fields:
            kwargs["tolerance_state_name"] = "adaptive_tolerance_state.json"
        if "bootstrap_first_candidate" in fields:
            kwargs["bootstrap_first_candidate"] = True
        if "bootstrap_image_name" in fields:
            kwargs["bootstrap_image_name"] = "bootstrap_reference.png"
        return camo_mldl.MLDLConfig(**kwargs)

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

        self.requested_target_count = count
        self._apply_backend_motif_scale()
        self.current_output_dir = DEFAULT_OUTPUT_DIR
        self.current_output_dir.mkdir(parents=True, exist_ok=True)
        self.best_records.clear()
        self.generated_rows = []
        self.stop_flag = False
        self.stopping = False
        self.running = True
        self.accepted_count = 0
        self.total_attempts = 0
        self.diag_total = self.diag_accepts = self.diag_rejects = 0
        self.diag_rule_counter = Counter()
        self.diag_last_rules = []
        self._reset_dynamic_tolerance_state()
        self.current_generation_mode = "mldl" if self._mldl_available() and self.use_mldl else "classic"
        self.current_report_name = DEFAULT_MLDL_REPORT_NAME if self.current_generation_mode == "mldl" else REPORT_NAME
        self.update_progress(0, count)
        self._refresh_diag_labels()
        prevent_sleep(True)
        self.status("Génération en cours…", ok=True)
        if self.use_mldl and not self._mldl_available():
            import_err = getattr(camo_mldl, "TORCH_IMPORT_ERROR", None) if camo_mldl is not None else None
            self.log(f"ML/DL indisponible, bascule sur le backend classique. Détail: {import_err}")
        self.log(
            f"Démarrage : {count} camouflage(s) | sortie={self.current_output_dir} | motif_scale={self.motif_scale:.2f} | mode={self.current_generation_mode}"
        )
        self._refresh_controls_state()
        worker_coro = self._async_worker_generate_mldl(count) if self.current_generation_mode == "mldl" else self._async_worker_generate(count)
        fut = self.async_runner.submit(worker_coro)
        self.current_future = fut
        fut.add_done_callback(self._on_generation_done)

    def _on_generation_done(self, fut: Future):
        try:
            fut.result()
        except Exception as exc:
            err = str(exc)
            Clock.schedule_once(lambda dt, e=err: self._finish_error(e), 0)
        finally:
            Clock.schedule_once(lambda dt, f=fut: self._clear_future(f), 0)

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

    def _max_resource_plan(self) -> Tuple[int, int, bool, float]:
        intensity = backend_machine_intensity(self.machine_intensity)
        try:
            sampler = getattr(camo, "sample_process_resources", None)
            sample = sampler(machine_intensity=intensity, output_dir=self.current_output_dir) if callable(sampler) else None
            planner = getattr(camo, "compute_runtime_tuning", None)
            if callable(planner):
                tuning = planner(
                    max_workers=None,
                    attempt_batch_size=None,
                    parallel_attempts=True,
                    machine_intensity=intensity,
                    sample=sample,
                )
                max_workers = max(1, int(getattr(tuning, "max_workers", 1)))
                attempt_batch_size = max(1, int(getattr(tuning, "attempt_batch_size", max_workers)))
                parallel = bool(getattr(tuning, "parallel_attempts", max_workers > 1))
                tuned_intensity = float(getattr(tuning, "machine_intensity", intensity))
                # Front Kivy: on garde toujours 1 coeur pour l'UI si possible.
                if max_workers > 1:
                    max_workers = max(1, min(max_workers, int(getattr(camo, "CPU_COUNT", os.cpu_count() or 1)) - 1 or 1))
                attempt_batch_size = max(1, min(attempt_batch_size, max_workers * 2))
                return max_workers, attempt_batch_size, parallel, tuned_intensity
        except Exception:
            pass

        cpu_count = int(getattr(camo, "CPU_COUNT", os.cpu_count() or 1))
        max_workers = max(1, cpu_count - 1) if cpu_count > 2 else 1
        attempt_batch_size = max(max_workers, max_workers * 2)
        return max_workers, attempt_batch_size, max_workers > 1, intensity

    def _rebuild_best_records_from_rows(self, rows: List[dict]):
        self.best_records.clear()
        for row in rows:
            try:
                image_path = Path(str(row.get("image_path", "")))
                if not image_path.exists():
                    continue
                metrics = {
                    "largest_component_ratio_class_1": float(row.get("largest_component_ratio_class_1", 0.0)),
                    "boundary_density": float(row.get("boundary_density", 0.0)),
                    "boundary_density_small": float(row.get("boundary_density_small", 0.0)),
                    "boundary_density_tiny": float(row.get("boundary_density_tiny", 0.0)),
                    "mirror_similarity": float(row.get("mirror_similarity", 0.0)),
                    "edge_contact_ratio": float(row.get("edge_contact_ratio", 0.0)),
                    "overscan": float(row.get("overscan", 0.0)),
                    "shift_strength": float(row.get("shift_strength", 0.0)),
                    "px_per_cm": float(row.get("px_per_cm", 0.0)),
                    "bestof_score": float(row.get("bestof_score", 0.0)),
                }
                ratios = np.array([
                    float(row.get("class_0_pct", 0.0)) / 100.0,
                    float(row.get("class_1_pct", 0.0)) / 100.0,
                    float(row.get("class_2_pct", 0.0)) / 100.0,
                    float(row.get("class_3_pct", 0.0)) / 100.0,
                ], dtype=np.float32)
                self.best_records.append(CandidateRecord(
                    index=int(row.get("index", 0)),
                    seed=int(row.get("seed", 0)),
                    local_attempt=int(row.get("attempts_for_this_image", 0)),
                    global_attempt=int(row.get("global_attempt", 0)),
                    image_path=image_path,
                    metrics=metrics,
                    ratios=ratios,
                ))
            except Exception:
                continue
        self.best_records.sort(key=candidate_rank_key)

    async def _async_generate_missing_mannequins(self, rows: List[dict]):
        for row in rows:
            try:
                image_path = Path(str(row.get("image_path", "")))
                if not image_path.exists():
                    continue
                mannequin_path = self.current_output_dir / MANNEQUIN_DIR_NAME / f"{image_path.stem}__mannequin.png"
                if mannequin_path.exists():
                    row["mannequin_image_name"] = mannequin_path.name
                    row["mannequin_image_path"] = str(mannequin_path)
                    continue
                pil_img = await asyncio.to_thread(read_pil_rgb, image_path)
                projection_img = await asyncio.to_thread(projection_preview_image, pil_img, PROJECTION_CFG, self.projection_preview_scale)
                saved = await async_save_mannequin_projection(projection_img, image_path, self.current_output_dir)
                row["mannequin_image_name"] = saved.name
                row["mannequin_image_path"] = str(saved)
            except Exception as exc:
                self.log(f"Projection mannequin post-run impossible : {exc}")

    async def _async_worker_generate_mldl(self, target_count: int):
        if not self._mldl_available() or camo_mldl is None:
            await self._async_worker_generate(target_count)
            return

        rows: List[dict] = []
        cfg = self._build_mldl_config(target_count)
        runner = camo_mldl.CamouflageMLDLGenerator(cfg)
        self.current_report_name = str(cfg.report_name)
        self.mldl_last_stats = None

        def _buffer_len() -> int:
            try:
                return int(len(getattr(getattr(runner, "buffer", None), "features", []) or []))
            except Exception:
                return 0

        async def _refresh_mldl_runtime_state() -> None:
            dataset_samples = _buffer_len()
            self.mldl_runtime_state["device"] = str(getattr(runner, "device", "--"))
            self.mldl_runtime_state["warmup_progress"] = int(min(dataset_samples, int(cfg.warmup_samples)))
            self.mldl_runtime_state["warmup_total"] = int(cfg.warmup_samples)
            self.mldl_runtime_state["dataset_samples"] = int(dataset_samples)
            self.mldl_runtime_state["dataset_loaded"] = bool(dataset_samples > 0)
            self.mldl_runtime_state["checkpoint_loaded"] = bool((self.current_output_dir / getattr(cfg, "checkpoint_name", "surrogate_camouflage.pt")).exists())
            if hasattr(runner, "surrogate"):
                self.mldl_runtime_state["surrogate_trained"] = bool(getattr(runner.surrogate, "trained", False))
            self.mldl_runtime_state["warmup_in_background"] = True
            self.mldl_runtime_state["warmup_remaining"] = int(max(0, int(cfg.warmup_samples) - dataset_samples))
            self.mldl_runtime_state["candidate_pool_size"] = int(cfg.candidate_pool_size)
            self.mldl_runtime_state["validate_top_k"] = int(cfg.validate_top_k)
            try:
                self._refresh_diag_labels()
            except Exception:
                pass

        def _persist_buffer_if_needed(sample_count: int) -> None:
            every = max(1, int(getattr(cfg, "warmup_persist_every", 8) or 8))
            if sample_count <= 0 or (sample_count % every != 0):
                return
            try:
                getattr(runner, "buffer").save(self.current_output_dir / getattr(cfg, "dataset_name", "dataset_camouflage_ml_dl.npz"))
            except Exception as exc:
                self.log(f"Persist dataset ML/DL impossible : {exc}")

        existing_samples = _buffer_len()
        await _refresh_mldl_runtime_state()
        self._emit_runtime(
            "INFO",
            "mldl",
            "Mode ML/DL activé",
            target_count=int(target_count),
            output_dir=str(self.current_output_dir),
            warmup_samples=int(cfg.warmup_samples),
            warmup_remaining=int(max(0, int(cfg.warmup_samples) - existing_samples)),
            dataset_samples=int(existing_samples),
            checkpoint_loaded=bool((self.current_output_dir / getattr(cfg, "checkpoint_name", "surrogate_camouflage.pt")).exists()),
            candidate_pool_size=int(cfg.candidate_pool_size),
            validate_top_k=int(cfg.validate_top_k),
            warmup_in_background=True,
        )
        self.log(
            f"Mode ML/DL : device={runner.device} | warmup intégré aux tentatives | "
            f"dataset={existing_samples}/{cfg.warmup_samples} | pool={cfg.candidate_pool_size} | top_k={cfg.validate_top_k}"
        )

        # Si un dataset repris est déjà suffisant, on lance l'entraînement en arrière-plan immédiatement.
        if existing_samples >= int(getattr(cfg, "min_train_size", 32) or 32):
            try:
                runner._schedule_background_train(force=True)
            except Exception:
                pass

        try:
            current_analysis = getattr(runner, "last_analysis", None)
            if getattr(runner, "last_rejected_candidate", None) is None:
                runner.last_rejected_candidate = None
                runner.last_analysis = None

            for target_index in range(1, target_count + 1):
                local_attempt = 1
                accepted_payload = None
                current_analysis = getattr(runner, "last_analysis", current_analysis)

                while local_attempt <= cfg.max_attempts_per_target:
                    if await self._async_should_stop():
                        runner.rows = rows
                        try:
                            runner._flush_background_train()
                        except Exception:
                            pass
                        runner._trainer_pool.shutdown(wait=False, cancel_futures=False)
                        await self._async_finish_stopped(rows)
                        return

                    polled = runner._poll_background_train()
                    if polled and isinstance(polled, dict):
                        stats = dict(polled.get("stats") or {})
                        if stats:
                            self.mldl_last_stats = stats
                            self.mldl_runtime_state["latest_stats"] = dict(stats)
                            self.mldl_runtime_state["checkpoint_loaded"] = True
                            self.mldl_runtime_state["dataset_loaded"] = True
                            self.log(f"DL mis à jour : {json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
                            self._emit_runtime("INFO", "mldl", "Checkpoint DL mis à jour", stats=stats)
                            await _refresh_mldl_runtime_state()

                    warmup_done = _buffer_len() >= int(cfg.warmup_samples)
                    stage_name = "génération ML/DL" if warmup_done else "génération ML/DL + warmup arrière-plan"
                    self.update_live_stage(
                        stage_name,
                        target_index=target_index,
                        local_attempt=local_attempt,
                        seed=None,
                        metrics_text=f"dataset={_buffer_len()}/{cfg.warmup_samples} | rejets et acceptations alimentent le warmup",
                    )

                    action_indexes = runner._select_action_indexes(current_analysis)
                    proposals: List[Any] = []
                    for offset, action_idx in enumerate(action_indexes, start=0):
                        action_name, action = camo_mldl.ACTION_LIBRARY[action_idx]
                        base_seed = camo.build_seed(target_index, local_attempt + offset, cfg.base_seed)
                        seed = camo_mldl.propose_seed(base_seed, action)
                        candidate = await asyncio.to_thread(camo.generate_candidate_from_seed, seed, self.anti_pixel)
                        if runner.surrogate.trained:
                            pred_valid, pred_reward = runner.surrogate.predict(camo_mldl.candidate_to_feature_vector(candidate))
                            pred_valid_f = float(pred_valid[0])
                            pred_reward_f = float(pred_reward[0])
                        else:
                            pred_valid_f = 0.5
                            pred_reward_f = 0.0
                        proposals.append(camo_mldl.Proposal(
                            seed=int(seed),
                            action_idx=int(action_idx),
                            action_name=str(action_name),
                            candidate=candidate,
                            pred_valid=pred_valid_f,
                            pred_reward=pred_reward_f,
                        ))
                    proposals.sort(key=lambda p: (p.pred_valid, p.pred_reward), reverse=True)

                    self.log(
                        f"[ML/DL][img={target_index:03d}] pool={len(proposals)} | top={proposals[0].action_name if proposals else '--'} | pred={proposals[0].pred_valid if proposals else 0.0:.4f} | dataset={_buffer_len()}/{cfg.warmup_samples}"
                    )

                    best_analysis = None
                    best_reward = -1e18
                    accepted_payload = None

                    for rank, proposal in enumerate(proposals[: cfg.validate_top_k], start=1):
                        if await self._async_should_stop():
                            runner.rows = rows
                            try:
                                runner._flush_background_train()
                            except Exception:
                                pass
                            runner._trainer_pool.shutdown(wait=False, cancel_futures=False)
                            await self._async_finish_stopped(rows)
                            return

                        active_profile = getattr(runner, "tolerance_profile", None) or self.tolerance_profile
                        candidate, outcome = await asyncio.to_thread(
                            camo.generate_and_validate_from_seed,
                            proposal.seed,
                            self.max_repair_rounds,
                            active_profile,
                            self.anti_pixel,
                        )
                        runner.total_attempts += 1
                        total_attempts = int(runner.total_attempts)
                        self.total_attempts = total_attempts

                        valid = bool(getattr(outcome, "accepted", False))
                        self._update_dynamic_tolerance_profile(
                            valid,
                            total_attempts,
                            candidate,
                            outcome,
                            sync_target=runner,
                            outcome=outcome,
                        )
                        try:
                            update_tol = getattr(runner, "_update_tolerance_from_candidate", None)
                            if callable(update_tol):
                                update_tol(candidate, outcome, total_attempts)
                        except Exception:
                            pass
                        self.tolerance_profile = getattr(runner, "tolerance_profile", self.tolerance_profile)
                        scores = extract_backend_scores(candidate.ratios, candidate.metrics, outcome)

                        reward = runner.buffer.add(candidate, valid)
                        dataset_samples = _buffer_len()
                        _persist_buffer_if_needed(dataset_samples)

                        if dataset_samples >= int(getattr(cfg, "min_train_size", 32) or 32) and not bool(getattr(runner.surrogate, "trained", False)):
                            runner._schedule_background_train(force=True)
                        else:
                            runner._schedule_background_train(force=False)
                        polled = runner._poll_background_train()
                        if polled and isinstance(polled, dict):
                            stats = dict(polled.get("stats") or {})
                            if stats:
                                self.mldl_last_stats = stats
                                self.mldl_runtime_state["latest_stats"] = dict(stats)
                                self.mldl_runtime_state["checkpoint_loaded"] = True
                                self.mldl_runtime_state["dataset_loaded"] = True

                        await _refresh_mldl_runtime_state()

                        if hasattr(camo, "emit_validation_payload"):
                            try:
                                payload = await asyncio.to_thread(
                                    camo.emit_validation_payload,
                                    output_dir=self.current_output_dir,
                                    target_index=target_index,
                                    local_attempt=local_attempt + rank - 1,
                                    global_attempt=total_attempts,
                                    candidate=candidate,
                                    outcome=outcome,
                                    tolerance_profile=(getattr(runner, "tolerance_profile", None) or self.tolerance_profile),
                                    tolerance_runtime=self.tolerance_runtime,
                                )
                                self.last_validation_payload = payload
                                self.validation_event_count += 1
                            except Exception as exc:
                                self.log(f"Export validation_payload impossible : {exc}")

                        preview_mode = "quality" if valid else "fast"
                        try:
                            projection_img = await asyncio.to_thread(
                                projection_preview_image,
                                candidate.image,
                                PROJECTION_CFG,
                                self.projection_preview_scale,
                                preview_mode,
                            )
                        except Exception as exc:
                            projection_img = candidate.image
                            self.log(f"Projection modèle indisponible : {exc}")

                        self.update_preview(candidate.image, projection_img)
                        await self._register_live_diag(candidate, target_index, local_attempt + rank - 1, outcome)

                        metrics_text = (
                            f"ML/DL {proposal.action_name} | pred {proposal.pred_valid:.3f}/{proposal.pred_reward:.3f} | "
                            f"best {scores['bestof_score']:.4f} | bd {scores['boundary_density']:.4f} | warmup={min(dataset_samples, cfg.warmup_samples)}/{cfg.warmup_samples}"
                        )
                        self._update_attempt_status(
                            target_index,
                            local_attempt + rank - 1,
                            total_attempts,
                            proposal.seed,
                            target_count,
                            self.accepted_count,
                            self.diag_rejects,
                            valid,
                            candidate.ratios,
                            scores,
                            candidate.metrics,
                        )

                        self._emit_runtime(
                            "INFO" if valid else "WARNING",
                            "mldl",
                            "Validation candidate ML/DL",
                            target_index=int(target_index),
                            local_attempt=int(local_attempt + rank - 1),
                            global_attempt=int(total_attempts),
                            seed=int(proposal.seed),
                            action_name=str(proposal.action_name),
                            pred_valid=float(proposal.pred_valid),
                            pred_reward=float(proposal.pred_reward),
                            accepted=bool(valid),
                            dataset_samples=int(dataset_samples),
                            warmup_progress=int(min(dataset_samples, int(cfg.warmup_samples))),
                            warmup_in_background=True,
                        )

                        if valid:
                            context = camo_mldl.build_context_vector(candidate, None)
                            runner.bandit.update(proposal.action_idx, context, reward)
                            accepted_payload = (proposal, candidate, outcome, projection_img, scores)
                            self.update_live_stage("accepté ML/DL", target_index, local_attempt + rank - 1, proposal.seed, metrics_text=metrics_text, pil_img=projection_img)
                            break

                        analysis = await asyncio.to_thread(
                            camo_mldl.analyze_rejection,
                            candidate,
                            target_index,
                            local_attempt + rank - 1,
                        )
                        if reward > best_reward:
                            best_reward = reward
                            best_analysis = analysis
                        context = camo_mldl.build_context_vector(candidate, analysis)
                        runner.bandit.update(proposal.action_idx, context, reward)
                        runner.last_rejected_candidate = candidate
                        runner.last_analysis = analysis

                        review = PendingManualReview(
                            target_index=target_index,
                            local_attempt=local_attempt + rank - 1,
                            global_attempt=total_attempts,
                            candidate=candidate,
                            outcome=outcome,
                            projection_img=projection_img,
                            metrics_text=metrics_text,
                        )
                        self._arm_manual_review(review)
                        self.update_live_stage("rejeté ML/DL", target_index, local_attempt + rank - 1, proposal.seed, metrics_text=metrics_text, pil_img=projection_img)
                        await asyncio.sleep(0)

                    current_analysis = best_analysis

                    if accepted_payload is None:
                        local_attempt += max(1, cfg.candidate_pool_size)
                        await self._adaptive_pause()
                        continue

                    proposal, candidate, outcome, projection_img, scores = accepted_payload
                    try:
                        projection_img = await asyncio.to_thread(
                            projection_preview_image,
                            candidate.image,
                            PROJECTION_CFG,
                            self.projection_preview_scale,
                            "quality",
                        )
                    except Exception:
                        pass

                    saved_path, mannequin_saved_path = await self._async_save_candidate_bundle(
                        rows,
                        target_index=target_index,
                        local_attempt=local_attempt,
                        global_attempt=self.total_attempts,
                        candidate=candidate,
                        outcome=outcome,
                        projection_img=projection_img,
                        manual_accept=False,
                    )
                    rows[-1]["mldl_enabled"] = 1
                    rows[-1]["mldl_action_name"] = str(proposal.action_name)
                    rows[-1]["mldl_action_idx"] = int(proposal.action_idx)
                    rows[-1]["mldl_pred_valid"] = float(proposal.pred_valid)
                    rows[-1]["mldl_pred_reward"] = float(proposal.pred_reward)
                    rows[-1]["mldl_device"] = str(runner.device)
                    rows[-1]["mldl_warmup_progress"] = int(min(_buffer_len(), int(cfg.warmup_samples)))
                    rows[-1]["mldl_dataset_samples"] = int(_buffer_len())
                    if runner._latest_train_stats:
                        rows[-1]["mldl_latest_train_stats"] = json.dumps(runner._latest_train_stats, ensure_ascii=False, sort_keys=True)

                    self.update_live_stage(
                        "accepté ML/DL",
                        target_index,
                        local_attempt,
                        candidate.seed,
                        metrics_text=f"{proposal.action_name} | best {scores['bestof_score']:.4f} | export {saved_path.name}",
                        pil_img=projection_img,
                    )
                    self.log(
                        f"[ML/DL][img={target_index:03d}] accepté -> {saved_path.name} | mannequin -> {mannequin_saved_path.name} | action={proposal.action_name} | pred={proposal.pred_valid:.4f}"
                    )
                    await self._adaptive_pause()
                    break

                if accepted_payload is None and local_attempt > cfg.max_attempts_per_target:
                    raise RuntimeError(
                        f"Impossible d'obtenir un camouflage valide en mode ML/DL pour target_index={target_index} dans la limite de {cfg.max_attempts_per_target} tentatives locales."
                    )

            try:
                runner._flush_background_train()
            except Exception:
                pass
            if runner._latest_train_stats:
                self.mldl_last_stats = dict(runner._latest_train_stats)
            runner.rows = rows
            runner._write_summary()
            await self._async_finish_success(rows)
        except asyncio.CancelledError:
            runner.rows = rows
            try:
                runner._flush_background_train()
            except Exception:
                pass
            await self._async_finish_stopped(rows)
            raise
        except Exception as exc:
            await self._async_finish_error(str(exc))
        finally:
            runner._trainer_pool.shutdown(wait=False, cancel_futures=False)

    async def _async_worker_generate(self, target_count: int):
        rows: List[dict] = []
        total_attempts = 0
        max_workers, attempt_batch_size, parallel_attempts, machine_intensity = self._max_resource_plan()
        self.log(
            f"Plan ressources GUI : workers={max_workers} | batch={attempt_batch_size} | "
            f"parallel={parallel_attempts} | intensity={machine_intensity:.2f}"
        )
        # Choix assumé : ThreadPoolExecutor plutôt que le ProcessPool du backend.
        # Motif : sous Windows/Kivy, l'utilisation d'un ProcessPool depuis le script GUI
        # peut provoquer des sous-processus qui réimportent la couche Kivy et ouvrent
        # des fenêtres noires parasites. Ce plan est moins agressif qu'un vrai pool de
        # processus, mais beaucoup plus fiable pour un front Kivy.
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="camo-ui")
        try:
            for target_index in range(1, target_count + 1):
                local_attempt = 1
                while True:
                    if await self._async_should_stop():
                        await self._async_finish_stopped(rows)
                        return

                    self._update_dynamic_tolerance_profile()
                    batch = camo.build_batch(target_index, local_attempt, attempt_batch_size, camo.DEFAULT_BASE_SEED)
                    if not parallel_attempts:
                        batch = batch[:1]

                    tasks = []
                    for attempt_no, seed in batch:
                        fut = loop.run_in_executor(
                            executor,
                            camo.generate_and_validate_from_seed,
                            seed,
                            self.max_repair_rounds,
                            self.tolerance_profile,
                            self.anti_pixel,
                        )
                        tasks.append(asyncio.create_task(async_wait_generation_future(fut, attempt_no, seed)))

                    ordered_results: List[Tuple[int, camo.CandidateResult, Any, PILImage.Image, Dict[str, float]]] = []
                    for idx, done in enumerate(asyncio.as_completed(tasks), start=1):
                        attempt_no, seed, candidate, outcome = await done
                        total_attempts += 1
                        self.total_attempts = total_attempts
                        valid = bool(getattr(outcome, "accepted", bool(outcome)))
                        self._remember_tolerance_outcome(valid)
                        self._remember_tolerance_change(total_attempts)
                        scores = extract_backend_scores(candidate.ratios, candidate.metrics, outcome)

                        if hasattr(camo, "emit_validation_payload"):
                            try:
                                payload = await asyncio.to_thread(
                                    camo.emit_validation_payload,
                                    output_dir=self.current_output_dir,
                                    target_index=target_index,
                                    local_attempt=attempt_no,
                                    global_attempt=total_attempts,
                                    candidate=candidate,
                                    outcome=outcome,
                                    tolerance_profile=(getattr(runner, "tolerance_profile", None) or self.tolerance_profile),
                                    tolerance_runtime=self.tolerance_runtime,
                                )
                                self.last_validation_payload = payload
                                self.validation_event_count += 1
                            except Exception as exc:
                                self.log(f"Export validation_payload impossible : {exc}")

                        preview_mode = "quality" if valid else "fast"
                        try:
                            projection_img = await asyncio.to_thread(
                                projection_preview_image,
                                candidate.image,
                                PROJECTION_CFG,
                                self.projection_preview_scale,
                                preview_mode,
                            )
                        except Exception as exc:
                            projection_img = candidate.image
                            self.log(f"Projection modèle indisponible : {exc}")

                        ordered_results.append((attempt_no, candidate, outcome, projection_img, scores))
                        self.update_preview(candidate.image, projection_img)
                        await self._register_live_diag(candidate, target_index, attempt_no, outcome)

                        metrics_text = (
                            f"best {scores['bestof_score']:.4f} | bd {scores['boundary_density']:.4f} | "
                            f"miroir {scores['mirror_similarity']:.4f} | {self._tolerance_proof_text()}"
                        )
                        self._update_attempt_status(
                            target_index,
                            attempt_no,
                            total_attempts,
                            seed,
                            target_count,
                            self.accepted_count,
                            self.diag_rejects,
                            valid,
                            candidate.ratios,
                            scores,
                            candidate.metrics,
                        )
                        if not valid:
                            review = PendingManualReview(
                                target_index=target_index,
                                local_attempt=attempt_no,
                                global_attempt=total_attempts,
                                candidate=candidate,
                                outcome=outcome,
                                projection_img=projection_img,
                                metrics_text=metrics_text,
                            )
                            self._arm_manual_review(review)
                            self.update_live_stage("rejeté", target_index, attempt_no, seed, metrics_text=metrics_text, pil_img=projection_img)
                        else:
                            self.update_live_stage("accepté potentiel", target_index, attempt_no, seed, metrics_text=metrics_text, pil_img=projection_img)
                        await asyncio.sleep(0)

                    ordered_results.sort(key=lambda x: x[0])
                    accepted_item = next(((a, c, o, p, s) for a, c, o, p, s in ordered_results if bool(getattr(o, "accepted", False))), None)
                    if accepted_item is None:
                        local_attempt += max(1, len(batch))
                        await self._adaptive_pause()
                        continue

                    accepted_attempt, accepted_candidate, accepted_outcome, projection_img, scores = accepted_item
                    try:
                        projection_img = await asyncio.to_thread(
                            projection_preview_image,
                            accepted_candidate.image,
                            PROJECTION_CFG,
                            self.projection_preview_scale,
                            "quality",
                        )
                    except Exception:
                        pass
                    saved_path, mannequin_saved_path = await self._async_save_candidate_bundle(
                        rows,
                        target_index=target_index,
                        local_attempt=accepted_attempt,
                        global_attempt=total_attempts,
                        candidate=accepted_candidate,
                        outcome=accepted_outcome,
                        projection_img=projection_img,
                        manual_accept=False,
                    )
                    self.update_live_stage(
                        "accepté backend",
                        target_index,
                        accepted_attempt,
                        accepted_candidate.seed,
                        metrics_text=f"best {scores['bestof_score']:.4f} | export {saved_path.name}",
                        pil_img=projection_img,
                    )
                    self.log(
                        f"[img={target_index:03d}] accepté -> {saved_path.name} | mannequin -> {(mannequin_saved_path.name if mannequin_saved_path is not None else 'non_enregistré')} | "
                        f"best={scores['bestof_score']:.4f}"
                    )
                    await self._adaptive_pause()
                    break

            await self._async_finish_success(rows)
        except asyncio.CancelledError:
            await self._async_finish_stopped(rows)
            raise
        except Exception as exc:
            await self._async_finish_error(str(exc))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _async_save_candidate_bundle(
        self,
        rows: List[dict],
        *,
        target_index: int,
        local_attempt: int,
        global_attempt: int,
        candidate: camo.CandidateResult,
        outcome: Any,
        projection_img: PILImage.Image,
        manual_accept: bool = False,
    ) -> Tuple[Path, Optional[Path]]:
        filename = build_backend_compatible_output_path(
            output_dir=self.current_output_dir,
            target_index=target_index,
            local_attempt=local_attempt,
            global_attempt=global_attempt,
            candidate=candidate,
        )
        saved_path = await async_save_candidate_image(candidate, filename)
        mannequin_saved_path: Optional[Path] = None
        try:
            projection_img_to_save, projection_report = await asyncio.to_thread(
                projection_preview_with_report,
                candidate.image,
                PROJECTION_CFG,
                self.projection_preview_scale,
                "quality",
            )
            if bool(getattr(projection_report, "valid", False)):
                mannequin_saved_path = await async_save_mannequin_projection(
                    projection_img_to_save,
                    saved_path,
                    self.current_output_dir,
                )
            else:
                self.log(
                    f"Projection mannequin affichée mais non enregistrée : résidu vert {int(getattr(projection_report, 'residual_pixels', 0))} px | ratio {float(getattr(projection_report, 'residual_ratio', 0.0)):.2%}"
                )
        except Exception as exc:
            self.log(f"Projection mannequin affichée mais non enregistrée : {exc}")
        record = CandidateRecord(
            index=target_index,
            seed=candidate.seed,
            local_attempt=local_attempt,
            global_attempt=global_attempt,
            image_path=saved_path,
            metrics={k: float(v) for k, v in candidate.metrics.items()},
            ratios=candidate.ratios.copy(),
        )
        self.best_records.append(record)
        self.best_records.sort(key=candidate_rank_key)
        row = build_candidate_row_compatible(
            target_index=target_index,
            local_attempt=local_attempt,
            global_attempt=global_attempt,
            candidate=candidate,
            outcome=outcome,
            saved_path=saved_path,
            manual_accept=manual_accept,
        )
        row["mannequin_saved"] = int(mannequin_saved_path is not None)
        row["mannequin_image_name"] = mannequin_saved_path.name if mannequin_saved_path is not None else ""
        row["mannequin_image_path"] = str(mannequin_saved_path) if mannequin_saved_path is not None else ""
        row["manual_accept"] = int(bool(manual_accept))
        row["tolerance_relax_level"] = float(self.tolerance_relax_level)
        row["tolerance_rejection_rate"] = float(self.tolerance_runtime.get("rejection_rate", 0.0))
        row["tolerance_window_count"] = int(round(float(self.tolerance_runtime.get("window_count", 0.0))))
        if self.tolerance_profile is not None:
            try:
                row["tolerance_bestof_min_score"] = float(getattr(self.tolerance_profile, "bestof_min_score"))
            except Exception:
                pass
        rows.append(row)
        self.accepted_count = len(rows)
        self.update_progress(len(rows), int(self.requested_target_count))
        self.schedule_gallery_reload(0.02)
        return saved_path, mannequin_saved_path

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
                f"Couleurs : {labels[0]} {rs[BACKEND_IDX_0]*100:.4f}% | "
                f"{labels[1]} {rs[BACKEND_IDX_1]*100:.4f}% | "
                f"{labels[2]} {rs[BACKEND_IDX_2]*100:.4f}% | "
                f"{labels[3]} {rs[BACKEND_IDX_3]*100:.4f}%"
            )
        if self.score_text is not None:
            self.score_text.text = (
                f"Qualité : MAE {scores['ratio_mae']:.6f} | "
                f"max abs {scores['ratio_max_abs']:.6f} | "
                f"composant {scores['primary_component_ratio']:.4f} | "
                f"best-of {scores['bestof_score']:.4f}"
            )
        if self.extra_text is not None:
            self.extra_text.text = (
                f"Contours : bd {scores['boundary_density']:.4f} | "
                f"bd/4 {scores['boundary_density_small']:.4f} | "
                f"bd/8 {scores['boundary_density_tiny']:.4f} | "
                f"bord {scores['edge_contact_ratio']:.4f}"
            )
        if self.struct_text is not None:
            self.struct_text.text = (
                f"Tolérance dynamique : {self._tolerance_proof_text()}"
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
                dst = best_dir / f"best_{rank:03d}_{("pattern" if hasattr(camo, "validate_with_reasons") else "camouflage")}_{rec.index:03d}.png"
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
                "bestof_score": round(safe_metric(rec.metrics, "bestof_score"), 6),
                "boundary_density": round(safe_metric(rec.metrics, "boundary_density"), 6),
                "boundary_density_small": round(safe_metric(rec.metrics, "boundary_density_small"), 6),
                "boundary_density_tiny": round(safe_metric(rec.metrics, "boundary_density_tiny"), 6),
                "mirror_similarity": round(safe_metric(rec.metrics, "mirror_similarity"), 6),
                "edge_contact_ratio": round(safe_metric(rec.metrics, "edge_contact_ratio"), 6),
                "seed_macros_total": round(safe_metric(rec.metrics, "seed_macros_total"), 3),
                "growth_rounds": round(safe_metric(rec.metrics, "growth_rounds"), 3),
                "safe_rebalanced_pixels": round(safe_metric(rec.metrics, "safe_rebalanced_pixels"), 3),
                "orphan_pixels_fixed": round(safe_metric(rec.metrics, "orphan_pixels_fixed"), 3),
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

    @mainthread
    def _apply_run_finished(self, status_text: str, ok: bool, log_lines: List[str], reload_gallery: bool = True):
        prevent_sleep(False)
        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status(status_text, ok=ok)
        for line in log_lines:
            self.log(line)
        if reload_gallery:
            self.reload_gallery()
        self._refresh_controls_state()

    async def _async_finish_success(self, rows: List[dict]):
        report_path = await async_write_report(rows, self.current_output_dir, filename=self.current_report_name)
        best_dir = await self._async_export_best_of(min(DEFAULT_TOP_K, len(self.best_records)))
        Clock.schedule_once(
            lambda _dt, rp=str(report_path), bd=str(best_dir): self._apply_run_finished(
                "Terminé",
                True,
                [f"Rapport écrit : {rp}", f"Best-of exporté : {bd}"],
                True,
            ),
            0,
        )

    async def _async_finish_stopped(self, rows: List[dict]):
        report_path = await async_write_report(rows, self.current_output_dir, filename=self.current_report_name)
        if rows:
            await self._async_export_best_of(min(DEFAULT_TOP_K, len(rows)))
        Clock.schedule_once(
            lambda _dt, rp=str(report_path): self._apply_run_finished(
                "Arrêté",
                False,
                [f"Rapport partiel : {rp}"],
                True,
            ),
            0,
        )

    async def _async_finish_error(self, message: str):
        Clock.schedule_once(
            lambda _dt, msg=str(message): self._finish_error(msg),
            0,
        )

    @mainthread
    def _finish_error(self, message: str):
        prevent_sleep(False)
        self.running = False
        self.stopping = False
        self.stop_flag = False
        self.status("Erreur", ok=False)
        self.log(f"Erreur : {message}")
        self.diag_log(f"Erreur diagnostic : {message}")
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
                f"Machine : CPU {cpu:.0f}% | RAM {ram:.0f}% | Disque {disk:.0f}% | "
                f"Processus {proc_cpu:.0f}% / {proc_mem:.2f} Go | motif {self.motif_scale:.2f} | mannequin {self.projection_preview_scale:.2f}"
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
