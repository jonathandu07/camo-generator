
# -*- coding: utf-8 -*-
"""
main.py
Camouflage Armée Fédérale Europe — version MACRO-ONLY + supervision continue.

Objectifs :
- uniquement des macro-formes anguleuses ;
- génération séquentielle stricte par image ;
- exploitation forte du CPU sans thrash mémoire ;
- asynchrone musclé avec consommation streaming des résultats ;
- contrôles en amont ;
- télémétrie continue et corrections en direct via log.py.
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

OUTPUT_DIR = Path("camouflages_federale_europe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 1400
HEIGHT = 2000
PX_PER_CM = 4.6

N_VARIANTS_REQUIRED = 100
DEFAULT_BASE_SEED = 202603120000

IDX_COYOTE = 0
IDX_OLIVE = 1
IDX_TERRE = 2
IDX_GRIS = 3

COLOR_NAMES = [
    "coyote_brown",
    "vert_olive",
    "terre_de_france",
    "vert_de_gris",
]

RGB = np.array([
    (0x81, 0x61, 0x3C),
    (0x55, 0x54, 0x3F),
    (0x7C, 0x6D, 0x66),
    (0x57, 0x5D, 0x57),
], dtype=np.uint8)

TARGET = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)

ORIGIN_BACKGROUND = 0
ORIGIN_MACRO = 1
ORIGIN_TRANSITION = 2
ORIGIN_MICRO = 3

MAX_ABS_ERROR_PER_COLOR = np.array([0.045, 0.045, 0.040, 0.040], dtype=float)
MAX_MEAN_ABS_ERROR = 0.026

BASE_ANGLES = [-35, -30, -25, -20, -15, 0, 15, 20, 25, 30, 35]

DENSITY_ZONES = [
    (0.02, 0.26, 0.02, 0.18, 1.80),
    (0.74, 0.98, 0.02, 0.18, 1.80),
    (0.00, 0.22, 0.18, 0.72, 1.60),
    (0.78, 1.00, 0.18, 0.72, 1.60),
    (0.20, 0.42, 0.62, 0.96, 1.55),
    (0.58, 0.80, 0.62, 0.96, 1.55),
    (0.30, 0.70, 0.18, 0.62, 0.60),
]

MACRO_LENGTH_CM = (40, 90)
MACRO_WIDTH_CM = (15, 35)

VISIBLE_MACRO_OLIVE_TARGET = TARGET[IDX_OLIVE]
VISIBLE_MACRO_TERRE_TARGET = TARGET[IDX_TERRE]
VISIBLE_MACRO_GRIS_TARGET = TARGET[IDX_GRIS]

MIN_OLIVE_CONNECTED_COMPONENT_RATIO = 0.17
MIN_OLIVE_MULTIZONE_SHARE = 0.48

MAX_COYOTE_CENTER_EMPTY_RATIO = 0.58
MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL = 0.64

MIN_BOUNDARY_DENSITY = 0.090
MAX_BOUNDARY_DENSITY = 0.220
MIN_BOUNDARY_DENSITY_SMALL = 0.060
MAX_BOUNDARY_DENSITY_SMALL = 0.205

MAX_MIRROR_SIMILARITY = 0.74

MIN_OBLIQUE_SHARE = 0.64
MIN_VERTICAL_SHARE = 0.10
MAX_VERTICAL_SHARE = 0.30
MAX_ANGLE_DOMINANCE_RATIO = 0.32

MAX_CENTER_TORSO_OVERLAP_OLIVE = 0.36
MAX_CENTER_TORSO_OVERLAP_TERRE = 0.24
MAX_CENTER_TORSO_OVERLAP_GRIS = 0.20

MAX_MACRO_PLACEMENT_ATTEMPTS_OLIVE = 900
MAX_MACRO_PLACEMENT_ATTEMPTS_TERRE = 700
MAX_MACRO_PLACEMENT_ATTEMPTS_GRIS = 500

MIN_MACRO_OLIVE_VISIBLE_RATIO = 0.22
MIN_MACRO_TERRE_VISIBLE_RATIO = 0.16
MIN_MACRO_GRIS_VISIBLE_RATIO = 0.12

MIN_TOTAL_MACRO_COUNT = 16
MIN_OLIVE_MACRO_COUNT = 7
MIN_TERRE_MACRO_COUNT = 5
MIN_GRIS_MACRO_COUNT = 4
MIN_GLOBAL_MACRO_MULTIZONE_RATIO = 0.52
MAX_SINGLE_MACRO_MASK_RATIO = 0.095

MIN_PERIPHERY_NON_COYOTE_RATIO = 1.12
MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO = 1.08
MAX_CENTRAL_BROWN_CONTINUITY = 0.42

MIN_OLIVE_MACRO_HEIGHT_SPAN_RATIO = 0.16
MIN_TERRE_MACRO_HEIGHT_SPAN_RATIO = 0.12
MIN_GRIS_MACRO_HEIGHT_SPAN_RATIO = 0.10
MIN_OLIVE_HIGH_DENSITY_OVERLAP = 0.22
MIN_TERRE_HIGH_DENSITY_OVERLAP = 0.16
MIN_GRIS_HIGH_DENSITY_OVERLAP = 0.12
MIN_STRUCTURAL_CORE_OVERLAP = 0.06
MAX_MACRO_EDGE_LOCK_RATIO = 0.82

TARGET_VERTICAL_SHARE = 0.16
MAX_VERTICAL_SOFT_TARGET = 0.24
TARGET_PERIPHERY_REPAIR_STEPS = 96
TARGET_CENTER_REPAIR_STEPS = 64

CPU_COUNT = max(1, os.cpu_count() or 1)
DEFAULT_MAX_WORKERS = max(1, CPU_COUNT)
DEFAULT_ATTEMPT_BATCH_SIZE = max(1, DEFAULT_MAX_WORKERS)
DEFAULT_MACHINE_INTENSITY = 0.94
DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES = 1

VISUAL_MIN_SILHOUETTE_COLOR_DIVERSITY = 0.62
VISUAL_MIN_CONTOUR_BREAK_SCORE = 0.44
VISUAL_MIN_OUTLINE_BAND_DIVERSITY = 0.58
VISUAL_MIN_SMALL_SCALE_STRUCTURAL_SCORE = 0.42
VISUAL_MIN_FINAL_SCORE = 0.60
VISUAL_MIN_MILITARY_SCORE = 0.62


# ============================================================
# STRUCTURES
# ============================================================

@dataclass
class VariantProfile:
    seed: int
    allowed_angles: List[int]
    angle_pool: Tuple[int, ...]
    zone_weight_boosts: Tuple[float, ...]
    macro_width_variation: float
    macro_lateral_jitter: float
    macro_tip_taper: float
    macro_edge_break: float
    olive_macro_target_scale: float = 1.0
    terre_macro_target_scale: float = 1.0
    gris_macro_target_scale: float = 1.0
    center_torso_overlap_scale: float = 1.0
    extra_macro_attempts: int = 0


@dataclass
class MacroRecord:
    color_idx: int
    poly: List[Tuple[float, float]]
    angle_deg: int
    center: Tuple[int, int]
    mask: np.ndarray
    zone_count: int


@dataclass
class CandidateResult:
    seed: int
    profile: VariantProfile
    image: Image.Image
    ratios: np.ndarray
    metrics: Dict[str, float]


@dataclass
class ResourceSnapshot:
    ts: float
    cpu_count: int
    process_cpu_percent: float
    system_cpu_percent: float
    process_rss_mb: float
    system_available_mb: float
    system_total_mb: float
    disk_free_mb: float
    machine_intensity: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ts": float(self.ts),
            "cpu_count": float(self.cpu_count),
            "process_cpu_percent": float(self.process_cpu_percent),
            "system_cpu_percent": float(self.system_cpu_percent),
            "process_rss_mb": float(self.process_rss_mb),
            "system_available_mb": float(self.system_available_mb),
            "system_total_mb": float(self.system_total_mb),
            "disk_free_mb": float(self.disk_free_mb),
            "machine_intensity": float(self.machine_intensity),
        }


@dataclass
class RuntimeTuning:
    max_workers: int
    attempt_batch_size: int
    parallel_attempts: bool
    machine_intensity: float
    reason: str = "initial"

    def normalized(self) -> "RuntimeTuning":
        max_workers = max(1, int(self.max_workers))
        batch = max(1, int(self.attempt_batch_size))
        return RuntimeTuning(
            max_workers=max_workers,
            attempt_batch_size=batch,
            parallel_attempts=bool(self.parallel_attempts and max_workers > 1 and batch > 1),
            machine_intensity=max(0.10, min(1.00, float(self.machine_intensity))),
            reason=str(self.reason),
        )


ProgressCallback = Callable[[int, int, int, int, CandidateResult, bool], None]
AsyncProgressCallback = Callable[[int, int, int, int, CandidateResult, bool], Awaitable[None]]
StopCallback = Callable[[], bool]
AsyncStopCallback = Callable[[], Awaitable[bool]]
SupervisorCallback = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


# ============================================================
# LOG / SUPERVISEUR DYNAMIQUE
# ============================================================

_LOG_MODULE_CACHE: Any = None
_LOG_MODULE_ATTEMPTED = False


def _get_log_module() -> Any:
    global _LOG_MODULE_CACHE, _LOG_MODULE_ATTEMPTED
    if _LOG_MODULE_CACHE is not None:
        return _LOG_MODULE_CACHE
    if _LOG_MODULE_ATTEMPTED:
        return None
    _LOG_MODULE_ATTEMPTED = True

    module = sys.modules.get("log")
    if module is not None:
        _LOG_MODULE_CACHE = module
        return module

    try:
        module = importlib.import_module("log")
    except Exception:
        module = None

    _LOG_MODULE_CACHE = module
    return module


def _runtime_log(level: str, source: str, message: str, **payload: Any) -> None:
    mod = _get_log_module()
    fn = getattr(mod, "log_event", None) if mod is not None else None
    if callable(fn):
        try:
            fn(level, source, message, **payload)
        except Exception:
            pass


def _run_log_preflight(strict: bool, output_dir: Path, module_names: Sequence[str] | None = None) -> None:
    mod = _get_log_module()
    fn = getattr(mod, "run_generation_preflight", None) if mod is not None else None
    if callable(fn):
        result = fn(output_dir=output_dir, strict=strict, module_names=module_names)
        if isinstance(result, dict) and not bool(result.get("ok", True)) and strict:
            raise RuntimeError(str(result.get("message", "Préflight refusé par log.py")))


def _supervisor_feedback(event_type: str, **payload: Any) -> Optional[Dict[str, Any]]:
    mod = _get_log_module()
    fn = getattr(mod, "feedback_runtime_event", None) if mod is not None else None
    if callable(fn):
        try:
            out = fn(event_type=event_type, **payload)
            if isinstance(out, dict):
                return out
        except Exception:
            return None
    return None


def _merge_supervisor_tuning(
    tuning: RuntimeTuning,
    advice: Optional[Dict[str, Any]],
    *,
    fallback_machine_intensity: Optional[float] = None,
) -> RuntimeTuning:
    if not advice:
        return tuning

    max_workers = int(advice.get("max_workers", tuning.max_workers))
    attempt_batch_size = int(advice.get("attempt_batch_size", tuning.attempt_batch_size))
    parallel_attempts = bool(advice.get("parallel_attempts", tuning.parallel_attempts))
    machine_intensity = float(advice.get(
        "machine_intensity",
        tuning.machine_intensity if fallback_machine_intensity is None else fallback_machine_intensity,
    ))
    reason = str(advice.get("reason", tuning.reason))
    return RuntimeTuning(
        max_workers=max_workers,
        attempt_batch_size=attempt_batch_size,
        parallel_attempts=parallel_attempts,
        machine_intensity=machine_intensity,
        reason=reason,
    ).normalized()


# ============================================================
# OUTILS SYSTÈME
# ============================================================

_PROCESS_POOL: Optional[ProcessPoolExecutor] = None
_PROCESS_POOL_WORKERS: Optional[int] = None


def _worker_initializer() -> None:
    if str(os.environ.get("CAMO_LIMIT_NUMERIC_THREADS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def shutdown_process_pool() -> None:
    global _PROCESS_POOL, _PROCESS_POOL_WORKERS
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
        _PROCESS_POOL = None
        _PROCESS_POOL_WORKERS = None


def get_process_pool(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    global _PROCESS_POOL, _PROCESS_POOL_WORKERS
    wanted = max(1, int(max_workers or DEFAULT_MAX_WORKERS))

    if _PROCESS_POOL is not None and _PROCESS_POOL_WORKERS != wanted:
        shutdown_process_pool()

    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(
            max_workers=wanted,
            initializer=_worker_initializer,
        )
        _PROCESS_POOL_WORKERS = wanted

    return _PROCESS_POOL


# ============================================================
# PRÉFLIGHT / RESSOURCES
# ============================================================

def _clip_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sample_process_resources(machine_intensity: float = DEFAULT_MACHINE_INTENSITY, output_dir: Path = OUTPUT_DIR) -> ResourceSnapshot:
    machine_intensity = _clip_float(float(machine_intensity), 0.10, 1.00)
    disk = shutil.disk_usage(str(Path(output_dir).resolve()))
    process_cpu = 0.0
    system_cpu = 0.0
    process_rss = 0.0
    available_mb = 0.0
    total_mb = 0.0

    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            process_cpu = float(proc.cpu_percent(interval=0.0))
            system_cpu = float(psutil.cpu_percent(interval=0.0))
            process_rss = float(proc.memory_info().rss / (1024 * 1024))
            mem = psutil.virtual_memory()
            available_mb = float(mem.available / (1024 * 1024))
            total_mb = float(mem.total / (1024 * 1024))
        except Exception:
            pass

    return ResourceSnapshot(
        ts=time.time(),
        cpu_count=CPU_COUNT,
        process_cpu_percent=process_cpu,
        system_cpu_percent=system_cpu,
        process_rss_mb=process_rss,
        system_available_mb=available_mb,
        system_total_mb=total_mb,
        disk_free_mb=float(disk.free / (1024 * 1024)),
        machine_intensity=machine_intensity,
    )


def compute_runtime_tuning(
    *,
    max_workers: Optional[int] = None,
    attempt_batch_size: Optional[int] = None,
    parallel_attempts: bool = True,
    machine_intensity: float = DEFAULT_MACHINE_INTENSITY,
    sample: Optional[ResourceSnapshot] = None,
) -> RuntimeTuning:
    intensity = _clip_float(float(machine_intensity), 0.10, 1.00)
    sample = sample or sample_process_resources(machine_intensity=intensity)
    baseline_workers = max(1, int(round(CPU_COUNT * intensity)))

    if sample.system_available_mb > 0.0:
        if sample.system_available_mb < 1024:
            baseline_workers = min(baseline_workers, 1)
        elif sample.system_available_mb < 2048:
            baseline_workers = min(baseline_workers, max(1, CPU_COUNT // 4))
        elif sample.system_available_mb < 4096:
            baseline_workers = min(baseline_workers, max(1, CPU_COUNT // 2))

    chosen_workers = max_workers if max_workers is not None else baseline_workers
    chosen_workers = max(1, min(CPU_COUNT, int(chosen_workers)))

    chosen_batch = attempt_batch_size if attempt_batch_size is not None else chosen_workers
    chosen_batch = max(1, int(chosen_batch))
    chosen_batch = min(max(chosen_workers, chosen_batch), max(1, CPU_COUNT * 2))

    return RuntimeTuning(
        max_workers=chosen_workers,
        attempt_batch_size=chosen_batch,
        parallel_attempts=bool(parallel_attempts),
        machine_intensity=intensity,
        reason="resource_plan",
    ).normalized()


def validate_generation_request(
    *,
    target_count: int,
    output_dir: Path,
    base_seed: int,
    machine_intensity: float,
    max_workers: Optional[int],
    attempt_batch_size: Optional[int],
) -> None:
    if int(target_count) <= 0:
        raise ValueError("target_count doit être > 0")
    if WIDTH <= 0 or HEIGHT <= 0:
        raise ValueError("WIDTH et HEIGHT doivent être > 0")
    if PX_PER_CM <= 0:
        raise ValueError("PX_PER_CM doit être > 0")
    if int(base_seed) < 0:
        raise ValueError("base_seed doit être >= 0")

    intensity = _clip_float(float(machine_intensity), 0.10, 1.00)
    if not (0.10 <= intensity <= 1.00):
        raise ValueError("machine_intensity doit être compris entre 0.10 et 1.00")

    if max_workers is not None and int(max_workers) <= 0:
        raise ValueError("max_workers doit être > 0")
    if attempt_batch_size is not None and int(attempt_batch_size) <= 0:
        raise ValueError("attempt_batch_size doit être > 0")

    ensure_output_dir(output_dir)
    test_file = Path(output_dir) / ".write_probe.tmp"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink()

    snapshot = sample_process_resources(machine_intensity=intensity, output_dir=output_dir)
    if snapshot.disk_free_mb < 256:
        raise RuntimeError("Espace disque insuffisant pour générer les sorties")
    _runtime_log("INFO", "main_preflight", "Préflight local validé", snapshot=snapshot.to_dict())


# ============================================================
# PROFILS
# ============================================================

def build_seed(target_index: int, local_attempt: int, base_seed: int = DEFAULT_BASE_SEED) -> int:
    return int(base_seed + target_index * 100000 + local_attempt)


def make_profile(seed: int) -> VariantProfile:
    rng = random.Random(seed)
    angles = BASE_ANGLES[:]
    rng.shuffle(angles)
    allowed = sorted(set([0] + angles[:rng.randint(8, len(BASE_ANGLES))]))
    angle_pool: List[int] = list(allowed)
    obliques = [a for a in allowed if a != 0]
    if obliques:
        angle_pool.extend(obliques * rng.randint(2, 4))
    if 0 in allowed:
        angle_pool.extend([0] * rng.randint(1, 3))

    zone_weight_boosts = []
    asym_side = rng.choice([0, 1])
    for idx, _zone in enumerate(DENSITY_ZONES):
        w = 1.0
        if idx in (0, 2, 4) and asym_side == 0:
            w += rng.uniform(0.12, 0.34)
        if idx in (1, 3, 5) and asym_side == 1:
            w += rng.uniform(0.12, 0.34)
        if idx == 6:
            w *= rng.uniform(0.60, 0.90)
        zone_weight_boosts.append(_clip_float(w, 0.45, 1.80))

    return VariantProfile(
        seed=seed,
        allowed_angles=allowed,
        angle_pool=tuple(int(a) for a in angle_pool),
        zone_weight_boosts=tuple(zone_weight_boosts),
        macro_width_variation=rng.uniform(0.20, 0.32),
        macro_lateral_jitter=rng.uniform(0.12, 0.22),
        macro_tip_taper=rng.uniform(0.34, 0.46),
        macro_edge_break=rng.uniform(0.10, 0.17),
        olive_macro_target_scale=rng.uniform(0.96, 1.06),
        terre_macro_target_scale=rng.uniform(0.95, 1.05),
        gris_macro_target_scale=rng.uniform(0.95, 1.05),
        center_torso_overlap_scale=rng.uniform(0.88, 1.00),
        extra_macro_attempts=rng.randint(0, 120),
    )


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def cm_to_px(cm: float) -> int:
    return max(1, int(round(cm * PX_PER_CM)))


def compute_ratios(canvas: np.ndarray) -> np.ndarray:
    counts = np.bincount(canvas.ravel(), minlength=4).astype(float)
    return counts / canvas.size


def render_canvas(canvas: np.ndarray) -> Image.Image:
    return Image.fromarray(RGB[canvas], "RGB")


def rotate(x: float, y: float, deg: float) -> Tuple[float, float]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def choose_biased_center(rng: random.Random, zone_weight_boosts: Optional[Sequence[float]] = None) -> Tuple[int, int]:
    weights: List[float] = []
    for i, z in enumerate(DENSITY_ZONES):
        boost = 1.0
        if zone_weight_boosts is not None and i < len(zone_weight_boosts):
            boost = _clip_float(_safe_float(zone_weight_boosts[i], 1.0), 0.35, 2.40)
        weights.append(z[4] * boost)

    z = rng.choices(DENSITY_ZONES, weights=weights, k=1)[0]
    x = int(rng.uniform(z[0], z[1]) * WIDTH)
    y = int(rng.uniform(z[2], z[3]) * HEIGHT)
    x = min(max(x, 60), WIDTH - 60)
    y = min(max(y, 60), HEIGHT - 60)
    return x, y


def polygon_mask(poly: Sequence[Tuple[float, float]]) -> np.ndarray:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x = max(0, int(math.floor(min(xs))) - 2)
    max_x = min(WIDTH - 1, int(math.ceil(max(xs))) + 2)
    min_y = max(0, int(math.floor(min(ys))) - 2)
    max_y = min(HEIGHT - 1, int(math.ceil(max(ys))) + 2)

    if min_x > max_x or min_y > max_y:
        return np.zeros((HEIGHT, WIDTH), dtype=bool)

    local_w = max_x - min_x + 1
    local_h = max_y - min_y + 1
    local_poly = [(x - min_x, y - min_y) for x, y in poly]

    img = Image.new("L", (local_w, local_h), 0)
    ImageDraw.Draw(img).polygon(local_poly, fill=255)
    local_mask = np.array(img, dtype=np.uint8) > 0

    full = np.zeros((HEIGHT, WIDTH), dtype=bool)
    full[min_y:max_y + 1, min_x:max_x + 1] = local_mask
    return full


def compute_boundary_mask(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape
    diff = np.zeros((h, w), dtype=bool)
    diff[1:, :] |= (canvas[1:, :] != canvas[:-1, :])
    diff[:-1, :] |= (canvas[:-1, :] != canvas[1:, :])
    diff[:, 1:] |= (canvas[:, 1:] != canvas[:, :-1])
    diff[:, :-1] |= (canvas[:, :-1] != canvas[:, 1:])
    return diff


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
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


def downsample_nearest(canvas: np.ndarray, factor: int) -> np.ndarray:
    return canvas[::factor, ::factor]


def boundary_density(canvas: np.ndarray) -> float:
    return float(np.mean(compute_boundary_mask(canvas)))


def mirror_similarity_score(canvas: np.ndarray) -> float:
    mid = canvas.shape[1] // 2
    left = canvas[:, :mid]
    right = canvas[:, canvas.shape[1] - mid:]
    right_flipped = np.fliplr(right)
    h = min(left.shape[0], right_flipped.shape[0])
    w = min(left.shape[1], right_flipped.shape[1])
    return float(np.mean(left[:h, :w] == right_flipped[:h, :w]))


# ============================================================
# ZONES ANATOMIQUES
# ============================================================

def rect_mask(x1: float, x2: float, y1: float, y2: float) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    xa = int(WIDTH * x1)
    xb = int(WIDTH * x2)
    ya = int(HEIGHT * y1)
    yb = int(HEIGHT * y2)
    mask[ya:yb, xa:xb] = True
    return mask


def anatomy_zone_masks() -> Dict[str, np.ndarray]:
    zones = {}
    zones["left_shoulder"] = rect_mask(0.02, 0.26, 0.02, 0.18)
    zones["right_shoulder"] = rect_mask(0.74, 0.98, 0.02, 0.18)
    zones["left_flank"] = rect_mask(0.00, 0.22, 0.18, 0.72)
    zones["right_flank"] = rect_mask(0.78, 1.00, 0.18, 0.72)
    zones["left_thigh"] = rect_mask(0.20, 0.42, 0.62, 0.96)
    zones["right_thigh"] = rect_mask(0.58, 0.80, 0.62, 0.96)
    zones["center_torso"] = rect_mask(0.30, 0.70, 0.18, 0.62)
    return zones


ANATOMY_ZONES = anatomy_zone_masks()
ANATOMY_ZONE_VALUES = tuple(ANATOMY_ZONES.values())
HIGH_DENSITY_ZONE_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_flank",
    "right_flank",
    "left_thigh",
    "right_thigh",
)


def combine_zone_masks(names: Sequence[str]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for name in names:
        mask |= ANATOMY_ZONES[name]
    return mask


HIGH_DENSITY_ZONE_MASK = combine_zone_masks(HIGH_DENSITY_ZONE_NAMES)
CENTER_TORSO_MASK = ANATOMY_ZONES["center_torso"]
CORE_CORRIDOR_MASK = rect_mask(0.28, 0.72, 0.15, 0.84)


def macro_zone_count(mask: np.ndarray) -> int:
    count = 0
    for zone_mask in ANATOMY_ZONE_VALUES:
        overlap = int((mask & zone_mask).sum())
        if overlap >= 600:
            count += 1
    return count


def zone_overlap_ratio(mask: np.ndarray, zone_mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area <= 0:
        return 0.0
    return float(np.mean(zone_mask[mask]))


def center_empty_ratio(canvas: np.ndarray) -> float:
    zone = ANATOMY_ZONES["center_torso"]
    return float(np.mean(canvas[zone] == IDX_COYOTE))


# ============================================================
# ANALYSE MORPHOLOGIQUE
# ============================================================

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


def orientation_score(macro_records: Sequence[MacroRecord]) -> Dict[str, float]:
    if not macro_records:
        return {"oblique_share": 0.0, "vertical_share": 0.0, "dominance_ratio": 1.0}
    angles = np.array([m.angle_deg for m in macro_records], dtype=int)
    abs_angles = np.abs(angles)
    oblique_share = float(np.mean(abs_angles >= 15))
    vertical_share = float(np.mean(abs_angles == 0))
    _, counts = np.unique(angles, return_counts=True)
    dominance_ratio = float(counts.max() / counts.sum())
    return {
        "oblique_share": oblique_share,
        "vertical_share": vertical_share,
        "dominance_ratio": dominance_ratio,
    }


def macro_angle_histogram(macros: Sequence[MacroRecord]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for item in macros:
        counts[item.angle_deg] = counts.get(item.angle_deg, 0) + 1
    return counts


def pick_macro_angle(
    macros: Sequence[MacroRecord],
    profile: VariantProfile,
    rng: random.Random,
    force_vertical_floor: bool = True,
) -> int:
    allowed = list(profile.allowed_angles) or BASE_ANGLES[:]
    counts = macro_angle_histogram(macros)
    total = len(macros)
    vertical_count = counts.get(0, 0)
    vertical_share = (vertical_count / total) if total else 0.0

    obliques = [a for a in allowed if a != 0] or [a for a in BASE_ANGLES if a != 0]

    if force_vertical_floor and 0 in allowed and total >= 4 and vertical_share < TARGET_VERTICAL_SHARE:
        return 0

    if 0 in allowed and total >= 5 and vertical_share >= MAX_VERTICAL_SOFT_TARGET:
        allowed = [a for a in allowed if a != 0] or obliques[:]

    dom_angle = None
    dom_count = -1
    for angle, cnt in counts.items():
        if cnt > dom_count:
            dom_angle = angle
            dom_count = cnt

    if dom_angle is not None and total >= 6 and dom_count / total > (MAX_ANGLE_DOMINANCE_RATIO - 0.02):
        filtered = [a for a in allowed if a != dom_angle]
        if filtered:
            allowed = filtered

    if total >= 5:
        local_counts = {a: counts.get(a, 0) for a in allowed}
        min_count = min(local_counts.values())
        least_used = [a for a, cnt in local_counts.items() if cnt == min_count]
        if least_used:
            allowed = least_used

    return int(rng.choice(allowed or BASE_ANGLES))


# ============================================================
# FORMES
# ============================================================

def jagged_spine_poly(
    rng: random.Random,
    cx: float,
    cy: float,
    length_px: float,
    width_px: float,
    angle_from_vertical_deg: float,
    segments: int,
    width_variation: float,
    lateral_jitter: float,
    tip_taper: float,
    edge_break: float,
) -> List[Tuple[float, float]]:
    half_len = length_px / 2.0
    half_w = width_px / 2.0
    ys = np.linspace(-half_len, half_len, segments)

    left, right = [], []
    for y in ys:
        t = abs(y) / half_len
        taper = max(0.35, 1.0 - tip_taper * t)
        local_half_w = half_w * rng.uniform(1.0 - width_variation, 1.0 + width_variation) * taper
        axis_dx = half_w * lateral_jitter * rng.uniform(-1.0, 1.0)
        ejl = local_half_w * edge_break * rng.uniform(-1.0, 1.0)
        ejr = local_half_w * edge_break * rng.uniform(-1.0, 1.0)

        left.append((axis_dx - local_half_w + ejl, y))
        right.append((axis_dx + local_half_w + ejr, y))

    poly = left + right[::-1]
    rot = [rotate(x, y, angle_from_vertical_deg) for x, y in poly]
    return [(cx + x, cy + y) for x, y in rot]


def apply_mask(canvas: np.ndarray, origin_map: np.ndarray, mask: np.ndarray, color_idx: int) -> None:
    if not np.any(mask):
        return
    canvas[mask] = color_idx
    origin_map[mask] = ORIGIN_MACRO


# ============================================================
# CONTRÔLES STRUCTURELS MACRO
# ============================================================

def macro_candidate_diagnostics(mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "zone_count": 0.0,
            "height_span_ratio": 0.0,
            "width_span_ratio": 0.0,
            "high_density_overlap": 0.0,
            "center_overlap": 0.0,
            "core_overlap": 0.0,
            "edge_lock_ratio": 1.0,
        }

    x_span = float((xs.max() - xs.min() + 1) / WIDTH)
    y_span = float((ys.max() - ys.min() + 1) / HEIGHT)
    left_edge = np.mean(xs <= int(WIDTH * 0.14))
    right_edge = np.mean(xs >= int(WIDTH * 0.86))
    return {
        "zone_count": float(macro_zone_count(mask)),
        "height_span_ratio": y_span,
        "width_span_ratio": x_span,
        "high_density_overlap": float(np.mean(HIGH_DENSITY_ZONE_MASK[mask])),
        "center_overlap": float(np.mean(CENTER_TORSO_MASK[mask])),
        "core_overlap": float(np.mean(CORE_CORRIDOR_MASK[mask])),
        "edge_lock_ratio": float(max(left_edge, right_edge)),
    }


def angle_distance_deg(a: int, b: int) -> int:
    return abs(a - b)


def local_parallel_conflict(
    macros: Sequence[MacroRecord],
    center: Tuple[int, int],
    angle_deg: int,
    dist_threshold_px: int = 260,
    angle_threshold_deg: int = 8,
) -> bool:
    cx, cy = center
    nearby_same = 0
    for m in macros:
        mx, my = m.center
        d = math.hypot(cx - mx, cy - my)
        if d > dist_threshold_px:
            continue
        if angle_distance_deg(angle_deg, m.angle_deg) <= angle_threshold_deg:
            nearby_same += 1
    return nearby_same >= 2


def macro_candidate_is_valid(
    mask: np.ndarray,
    color_idx: int,
    angle_deg: int,
    canvas: np.ndarray,
    macros: Sequence[MacroRecord],
    require_cross_core: bool = False,
) -> bool:
    if mask.sum() == 0:
        return False

    diag = macro_candidate_diagnostics(mask)
    if diag["edge_lock_ratio"] > MAX_MACRO_EDGE_LOCK_RATIO:
        return False

    ys, xs = np.where(mask)
    center = (int(np.mean(xs)), int(np.mean(ys)))

    if float(mask.sum()) / float(canvas.size) > MAX_SINGLE_MACRO_MASK_RATIO:
        return False

    if color_idx == IDX_OLIVE:
        if diag["zone_count"] < 2.0:
            return False
        if diag["height_span_ratio"] < MIN_OLIVE_MACRO_HEIGHT_SPAN_RATIO:
            return False
        if diag["high_density_overlap"] < MIN_OLIVE_HIGH_DENSITY_OVERLAP:
            return False
        if float(np.mean(canvas[mask] == IDX_OLIVE)) > 0.46:
            return False
    elif color_idx == IDX_TERRE:
        if diag["zone_count"] < 1.0:
            return False
        if diag["height_span_ratio"] < MIN_TERRE_MACRO_HEIGHT_SPAN_RATIO:
            return False
        if diag["high_density_overlap"] < MIN_TERRE_HIGH_DENSITY_OVERLAP:
            return False
    else:
        if diag["zone_count"] < 1.0:
            return False
        if diag["height_span_ratio"] < MIN_GRIS_MACRO_HEIGHT_SPAN_RATIO:
            return False
        if diag["high_density_overlap"] < MIN_GRIS_HIGH_DENSITY_OVERLAP:
            return False

    if require_cross_core and diag["core_overlap"] < MIN_STRUCTURAL_CORE_OVERLAP:
        return False

    center_count = sum(1 for m in macros if zone_overlap_ratio(m.mask, CENTER_TORSO_MASK) > 0.14)
    if diag["center_overlap"] > 0.48 and center_count >= 2:
        return False

    if local_parallel_conflict(macros, center, angle_deg, dist_threshold_px=240, angle_threshold_deg=7):
        return False

    return True


def zone_center(name: str) -> Tuple[int, int]:
    ys, xs = np.where(ANATOMY_ZONES[name])
    return int(np.mean(xs)), int(np.mean(ys))


def angle_from_points_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return float(math.degrees(math.atan2(dx, dy)))


def nearest_allowed_angle(angle: float, allowed: Sequence[int]) -> int:
    allowed = list(allowed) or BASE_ANGLES
    return int(min(allowed, key=lambda a: abs(a - angle)))


def macro_color_count(macros: Sequence[MacroRecord], color_idx: int) -> int:
    return sum(1 for m in macros if m.color_idx == color_idx)


def macro_counts(macros: Sequence[MacroRecord]) -> Dict[int, int]:
    return {
        IDX_OLIVE: macro_color_count(macros, IDX_OLIVE),
        IDX_TERRE: macro_color_count(macros, IDX_TERRE),
        IDX_GRIS: macro_color_count(macros, IDX_GRIS),
    }


def macro_visible_pixels(canvas: np.ndarray, origin_map: np.ndarray, color_idx: int) -> int:
    return int(np.sum((canvas == color_idx) & (origin_map == ORIGIN_MACRO)))


def macro_system_metrics(macros: Sequence[MacroRecord], canvas: np.ndarray, origin_map: np.ndarray) -> Dict[str, float]:
    counts = macro_counts(macros)
    multizone_ratio = float(np.mean([m.zone_count >= 2 for m in macros])) if macros else 0.0
    largest_mask_ratio = max((float(m.mask.sum()) / float(canvas.size) for m in macros), default=0.0)
    return {
        "macro_total_count": float(len(macros)),
        "macro_olive_count": float(counts[IDX_OLIVE]),
        "macro_terre_count": float(counts[IDX_TERRE]),
        "macro_gris_count": float(counts[IDX_GRIS]),
        "macro_multizone_ratio": multizone_ratio,
        "macro_visible_total_ratio": float(np.mean(origin_map == ORIGIN_MACRO)),
        "largest_macro_mask_ratio": largest_mask_ratio,
    }


def absolute_origin_color_ratios(canvas: np.ndarray, origin_map: np.ndarray) -> Dict[str, float]:
    total = float(canvas.size)
    return {
        "macro_olive_visible_ratio": float(np.sum((canvas == IDX_OLIVE) & (origin_map == ORIGIN_MACRO)) / total),
        "macro_terre_visible_ratio": float(np.sum((canvas == IDX_TERRE) & (origin_map == ORIGIN_MACRO)) / total),
        "macro_gris_visible_ratio": float(np.sum((canvas == IDX_GRIS) & (origin_map == ORIGIN_MACRO)) / total),
    }


def spatial_discipline_metrics(canvas: np.ndarray) -> Dict[str, float]:
    boundary = compute_boundary_mask(canvas)
    non_coyote = canvas != IDX_COYOTE
    periphery_boundary_density = float(np.mean(boundary[HIGH_DENSITY_ZONE_MASK]))
    center_boundary_density = float(np.mean(boundary[CENTER_TORSO_MASK]))
    periphery_boundary_ratio = periphery_boundary_density / max(center_boundary_density, 1e-6)

    periphery_non_coyote = float(np.mean(non_coyote[HIGH_DENSITY_ZONE_MASK]))
    center_non_coyote = float(np.mean(non_coyote[CENTER_TORSO_MASK]))
    periphery_non_coyote_ratio = periphery_non_coyote / max(center_non_coyote, 1e-6)

    return {
        "periphery_boundary_density": periphery_boundary_density,
        "center_boundary_density": center_boundary_density,
        "periphery_boundary_density_ratio": float(periphery_boundary_ratio),
        "periphery_non_coyote_density": periphery_non_coyote,
        "center_non_coyote_density": center_non_coyote,
        "periphery_non_coyote_ratio": float(periphery_non_coyote_ratio),
    }


def central_brown_continuity(canvas: np.ndarray) -> float:
    x1 = int(canvas.shape[1] * 0.38)
    x2 = int(canvas.shape[1] * 0.62)
    y1 = int(canvas.shape[0] * 0.10)
    y2 = int(canvas.shape[0] * 0.94)
    band = (canvas[y1:y2, x1:x2] == IDX_COYOTE)
    if band.size == 0:
        return 0.0
    return float(np.max([largest_component_ratio(band), float(np.mean(np.any(band, axis=1)))]))


# ============================================================
# GÉNÉRATION DES MACROS
# ============================================================

def _macro_size_config(color_idx: int, long_mode: bool) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int]]:
    if color_idx == IDX_OLIVE:
        if long_mode:
            return (56, 90), (17, 32), (7, 10)
        return (44, 74), (15, 26), (7, 9)
    if color_idx == IDX_TERRE:
        if long_mode:
            return (48, 76), (15, 26), (6, 9)
        return (40, 66), (15, 22), (6, 8)
    if long_mode:
        return (44, 62), (15, 22), (6, 8)
    return (40, 54), (15, 18), (5, 7)


def try_place_validated_macro(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: List[MacroRecord],
    color_idx: int,
    profile: VariantProfile,
    rng: random.Random,
    *,
    long_mode: bool = False,
    require_cross_core: bool = False,
) -> bool:
    length_range, width_range, segment_range = _macro_size_config(color_idx, long_mode)
    cx, cy = choose_biased_center(rng, profile.zone_weight_boosts)
    angle = pick_macro_angle(macros, profile, rng, force_vertical_floor=(color_idx == IDX_OLIVE))

    poly = jagged_spine_poly(
        rng=rng,
        cx=cx,
        cy=cy,
        length_px=cm_to_px(rng.uniform(*length_range)),
        width_px=cm_to_px(rng.uniform(*width_range)),
        angle_from_vertical_deg=angle,
        segments=rng.randint(*segment_range),
        width_variation=profile.macro_width_variation,
        lateral_jitter=profile.macro_lateral_jitter,
        tip_taper=profile.macro_tip_taper,
        edge_break=profile.macro_edge_break,
    )
    mask = polygon_mask(poly)
    if mask.sum() == 0:
        return False

    if not macro_candidate_is_valid(mask, color_idx, angle, canvas, macros, require_cross_core=require_cross_core):
        return False

    center_overlap = zone_overlap_ratio(mask, CENTER_TORSO_MASK)
    if color_idx == IDX_OLIVE:
        max_center = MAX_CENTER_TORSO_OVERLAP_OLIVE * profile.center_torso_overlap_scale
        if angle == 0:
            max_center = min(0.42, max_center + 0.05)
        if center_overlap > max_center:
            return False
    elif color_idx == IDX_TERRE:
        if center_overlap > MAX_CENTER_TORSO_OVERLAP_TERRE * profile.center_torso_overlap_scale:
            return False
        cur = canvas[mask]
        if float(np.mean(np.isin(cur, [IDX_COYOTE, IDX_OLIVE, IDX_TERRE]))) < 0.52:
            return False
    else:
        if center_overlap > MAX_CENTER_TORSO_OVERLAP_GRIS:
            return False
        cur = canvas[mask]
        if float(np.mean(np.isin(cur, [IDX_OLIVE, IDX_TERRE, IDX_GRIS]))) < 0.40:
            return False

    apply_mask(canvas, origin_map, mask, color_idx)
    macros.append(MacroRecord(color_idx, poly, angle, (cx, cy), mask, macro_zone_count(mask)))
    return True


def add_forced_structural_macros(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: List[MacroRecord],
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    main_side = rng.choice(["left", "right"])
    other_side = "right" if main_side == "left" else "left"

    plan = [
        (IDX_OLIVE, (f"{main_side}_shoulder", f"{other_side}_flank"), True),
        (IDX_OLIVE, (f"{main_side}_flank", f"{other_side}_thigh"), False),
        (IDX_TERRE, (f"{other_side}_shoulder", f"{main_side}_flank"), True),
        (IDX_GRIS, (f"{other_side}_flank", f"{main_side}_thigh"), False),
    ]

    for color_idx, (z1, z2), require_cross_core in plan:
        p1 = zone_center(z1)
        p2 = zone_center(z2)
        cx = (p1[0] + p2[0]) / 2.0 + rng.uniform(-22.0, 22.0)
        cy = (p1[1] + p2[1]) / 2.0 + rng.uniform(-32.0, 32.0)
        base_angle = angle_from_points_vertical(p1, p2)
        angle = nearest_allowed_angle(base_angle + rng.uniform(-6.0, 6.0), profile.allowed_angles)

        length_range, width_range, segment_range = _macro_size_config(color_idx, True)
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=max(cm_to_px(rng.uniform(*length_range)), int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 1.05)),
            width_px=cm_to_px(rng.uniform(*width_range)),
            angle_from_vertical_deg=angle,
            segments=rng.randint(*segment_range),
            width_variation=max(0.18, profile.macro_width_variation - 0.02),
            lateral_jitter=max(0.12, profile.macro_lateral_jitter - 0.01),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        mask = polygon_mask(poly)
        if not macro_candidate_is_valid(mask, color_idx, angle, canvas, macros, require_cross_core=require_cross_core):
            continue

        center_overlap = zone_overlap_ratio(mask, CENTER_TORSO_MASK)
        if color_idx == IDX_OLIVE and center_overlap > MAX_CENTER_TORSO_OVERLAP_OLIVE:
            continue
        if color_idx == IDX_TERRE and center_overlap > MAX_CENTER_TORSO_OVERLAP_TERRE:
            continue
        if color_idx == IDX_GRIS and center_overlap > MAX_CENTER_TORSO_OVERLAP_GRIS:
            continue

        apply_mask(canvas, origin_map, mask, color_idx)
        macros.append(MacroRecord(color_idx, poly, angle, (int(cx), int(cy)), mask, macro_zone_count(mask)))


def add_macros(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: List[MacroRecord],
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    target_pixels = {
        IDX_OLIVE: int(VISIBLE_MACRO_OLIVE_TARGET * canvas.size * profile.olive_macro_target_scale),
        IDX_TERRE: int(VISIBLE_MACRO_TERRE_TARGET * canvas.size * profile.terre_macro_target_scale),
        IDX_GRIS: int(VISIBLE_MACRO_GRIS_TARGET * canvas.size * profile.gris_macro_target_scale),
    }

    min_counts = {
        IDX_OLIVE: MIN_OLIVE_MACRO_COUNT,
        IDX_TERRE: MIN_TERRE_MACRO_COUNT,
        IDX_GRIS: MIN_GRIS_MACRO_COUNT,
    }

    max_attempts = {
        IDX_OLIVE: MAX_MACRO_PLACEMENT_ATTEMPTS_OLIVE + profile.extra_macro_attempts,
        IDX_TERRE: MAX_MACRO_PLACEMENT_ATTEMPTS_TERRE + profile.extra_macro_attempts,
        IDX_GRIS: MAX_MACRO_PLACEMENT_ATTEMPTS_GRIS + profile.extra_macro_attempts,
    }

    for color_idx in (IDX_OLIVE, IDX_TERRE, IDX_GRIS):
        attempts = 0
        while True:
            current_pixels = macro_visible_pixels(canvas, origin_map, color_idx)
            current_count = macro_color_count(macros, color_idx)
            enough_pixels = current_pixels >= target_pixels[color_idx]
            enough_count = current_count >= min_counts[color_idx]
            if enough_pixels and enough_count:
                break

            attempts += 1
            if attempts > max_attempts[color_idx]:
                break

            long_mode = bool(current_count < max(2, min_counts[color_idx] // 2))
            require_cross_core = bool(color_idx in (IDX_OLIVE, IDX_TERRE) and current_count < max(2, min_counts[color_idx] // 2))

            try_place_validated_macro(
                canvas,
                origin_map,
                macros,
                color_idx,
                profile,
                rng,
                long_mode=long_mode,
                require_cross_core=require_cross_core,
            )


def enforce_macro_population(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: List[MacroRecord],
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    repair_attempts = 0
    while repair_attempts < 360:
        repair_attempts += 1
        stats = macro_system_metrics(macros, canvas, origin_map)

        if (
            stats["macro_total_count"] >= MIN_TOTAL_MACRO_COUNT
            and stats["macro_olive_count"] >= MIN_OLIVE_MACRO_COUNT
            and stats["macro_terre_count"] >= MIN_TERRE_MACRO_COUNT
            and stats["macro_gris_count"] >= MIN_GRIS_MACRO_COUNT
            and stats["macro_multizone_ratio"] >= MIN_GLOBAL_MACRO_MULTIZONE_RATIO
        ):
            break

        priority: List[int] = []
        if stats["macro_olive_count"] < MIN_OLIVE_MACRO_COUNT:
            priority.extend([IDX_OLIVE, IDX_OLIVE])
        if stats["macro_terre_count"] < MIN_TERRE_MACRO_COUNT:
            priority.extend([IDX_TERRE, IDX_TERRE])
        if stats["macro_gris_count"] < MIN_GRIS_MACRO_COUNT:
            priority.append(IDX_GRIS)
        if stats["macro_multizone_ratio"] < MIN_GLOBAL_MACRO_MULTIZONE_RATIO:
            priority.extend([IDX_OLIVE, IDX_TERRE])
        if stats["macro_total_count"] < MIN_TOTAL_MACRO_COUNT:
            priority.extend([IDX_OLIVE, IDX_TERRE, IDX_GRIS])

        color_idx = priority[(repair_attempts - 1) % len(priority)] if priority else IDX_OLIVE
        require_cross_core = bool(stats["macro_multizone_ratio"] < MIN_GLOBAL_MACRO_MULTIZONE_RATIO and color_idx in (IDX_OLIVE, IDX_TERRE))

        try_place_validated_macro(
            canvas,
            origin_map,
            macros,
            color_idx,
            profile,
            rng,
            long_mode=True,
            require_cross_core=require_cross_core,
        )


def repair_center_and_periphery(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: List[MacroRecord],
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    for _ in range(TARGET_PERIPHERY_REPAIR_STEPS):
        spatial = spatial_discipline_metrics(canvas)
        center_empty = center_empty_ratio(canvas)
        macro_state = absolute_origin_color_ratios(canvas, origin_map)

        need_periphery = (
            spatial["periphery_boundary_density_ratio"] < MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO
            or spatial["periphery_non_coyote_ratio"] < MIN_PERIPHERY_NON_COYOTE_RATIO
        )
        need_center_fill = center_empty > MAX_COYOTE_CENTER_EMPTY_RATIO
        need_olive = macro_state["macro_olive_visible_ratio"] < MIN_MACRO_OLIVE_VISIBLE_RATIO
        need_terre = macro_state["macro_terre_visible_ratio"] < MIN_MACRO_TERRE_VISIBLE_RATIO
        need_gris = macro_state["macro_gris_visible_ratio"] < MIN_MACRO_GRIS_VISIBLE_RATIO

        if not (need_periphery or need_center_fill or need_olive or need_terre or need_gris):
            break

        if need_olive:
            color = IDX_OLIVE
        elif need_terre:
            color = IDX_TERRE
        elif need_gris:
            color = IDX_GRIS
        else:
            color = rng.choices([IDX_OLIVE, IDX_TERRE, IDX_GRIS], weights=[0.48, 0.32, 0.20], k=1)[0]

        require_cross_core = bool(need_center_fill and color in (IDX_OLIVE, IDX_TERRE))
        try_place_validated_macro(
            canvas,
            origin_map,
            macros,
            color,
            profile,
            rng,
            long_mode=True,
            require_cross_core=require_cross_core,
        )


def enforce_macro_angle_discipline(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: List[MacroRecord],
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    repair_attempts = 0
    while repair_attempts < TARGET_CENTER_REPAIR_STEPS:
        repair_attempts += 1
        orient = orientation_score(macros)
        if (
            orient["oblique_share"] >= MIN_OBLIQUE_SHARE
            and MIN_VERTICAL_SHARE <= orient["vertical_share"] <= MAX_VERTICAL_SHARE
            and orient["dominance_ratio"] <= MAX_ANGLE_DOMINANCE_RATIO
        ):
            break

        if orient["vertical_share"] < MIN_VERTICAL_SHARE and 0 in profile.allowed_angles:
            angle = 0
        else:
            angle = pick_macro_angle(macros, profile, rng, force_vertical_floor=False)

        color = IDX_OLIVE if rng.random() < 0.70 else IDX_TERRE
        length_range, width_range, segment_range = _macro_size_config(color, False)
        cx, cy = choose_biased_center(rng, profile.zone_weight_boosts)
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(max(40, length_range[0]), max(48, length_range[1]))),
            width_px=cm_to_px(rng.uniform(max(15, width_range[0]), max(18, width_range[1]))),
            angle_from_vertical_deg=angle,
            segments=rng.randint(*segment_range),
            width_variation=max(0.18, profile.macro_width_variation - 0.02),
            lateral_jitter=max(0.12, profile.macro_lateral_jitter - 0.01),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue
        if not macro_candidate_is_valid(mask, color, angle, canvas, macros, require_cross_core=False):
            continue

        center_overlap = zone_overlap_ratio(mask, CENTER_TORSO_MASK)
        if color == IDX_OLIVE and center_overlap > MAX_CENTER_TORSO_OVERLAP_OLIVE:
            continue
        if color == IDX_TERRE and center_overlap > MAX_CENTER_TORSO_OVERLAP_TERRE:
            continue

        apply_mask(canvas, origin_map, mask, color)
        ys, xs = np.where(mask)
        macros.append(MacroRecord(color, poly, angle, (int(np.mean(xs)), int(np.mean(ys))), mask, macro_zone_count(mask)))


# ============================================================
# MÉTRIQUES MULTI-ÉCHELLE
# ============================================================

def center_empty_ratio_upscaled_proxy(canvas_small: np.ndarray) -> float:
    h, w = canvas_small.shape
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.18)
    y2 = int(h * 0.62)
    zone = canvas_small[y1:y2, x1:x2]
    return float(np.mean(zone == IDX_COYOTE))


def multiscale_metrics(canvas: np.ndarray) -> Dict[str, float]:
    small = downsample_nearest(canvas, 4)
    tiny = downsample_nearest(canvas, 8)
    return {
        "boundary_density_small": boundary_density(small),
        "boundary_density_tiny": boundary_density(tiny),
        "center_empty_ratio_small": center_empty_ratio_upscaled_proxy(small),
        "largest_olive_component_ratio_small": largest_component_ratio(small == IDX_OLIVE),
    }


# ============================================================
# VALIDATION VISUELLE
# ============================================================

def build_silhouette_mask(width: int, height: int) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
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
    band = dilate_mask(bound, radius=5) & sil
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
    olive_ratio = largest_component_ratio(small == IDX_OLIVE)
    bd = float(np.mean(compute_boundary_mask(small)))
    s1 = clamp01((olive_ratio - 0.08) / 0.18)
    s2 = 1.0 - min(1.0, abs(bd - 0.11) / 0.12)
    return clamp01(0.58 * s1 + 0.42 * s2)


def ratio_score(rs: np.ndarray) -> float:
    err = np.abs(rs - TARGET)
    mae = float(np.mean(err))
    return clamp01(1.0 - mae / 0.05)


def main_metrics_score(metrics: Dict[str, float]) -> float:
    parts = [
        clamp01((metrics["largest_olive_component_ratio"] - 0.12) / 0.18),
        clamp01(1.0 - metrics["center_empty_ratio"] / 0.66),
        clamp01(1.0 - metrics.get("mirror_similarity", 1.0) / 0.90),
        clamp01(1.0 - metrics.get("central_brown_continuity", 1.0) / 0.55),
        clamp01((metrics["olive_multizone_share"] - 0.30) / 0.35),
        clamp01(1.0 - abs(metrics["boundary_density"] - 0.13) / 0.11),
        clamp01((metrics.get("macro_total_count", 0.0) - 12.0) / 8.0),
        clamp01((metrics.get("macro_multizone_ratio", 0.0) - 0.35) / 0.30),
        clamp01((metrics.get("macro_olive_visible_ratio", 0.0) - 0.18) / 0.12),
        clamp01((metrics.get("macro_terre_visible_ratio", 0.0) - 0.12) / 0.12),
        clamp01((metrics.get("macro_gris_visible_ratio", 0.0) - 0.08) / 0.12),
    ]
    return float(np.mean(parts))


def evaluate_visual_metrics(index_canvas: np.ndarray, rs: np.ndarray, metrics: Dict[str, float]) -> Dict[str, float]:
    sil_div = silhouette_color_diversity_score(index_canvas)
    contour_score, outline_band_div = contour_break_score(index_canvas)
    small_scale_score = small_scale_structural_score(index_canvas)

    s_ratio = ratio_score(rs)
    s_main = main_metrics_score(metrics)
    s_sil = sil_div
    s_contour = clamp01(0.65 * contour_score + 0.35 * outline_band_div)

    final_score = (
        0.28 * s_ratio
        + 0.30 * s_sil
        + 0.24 * s_contour
        + 0.18 * s_main
    )

    visual_valid = (
        sil_div >= VISUAL_MIN_SILHOUETTE_COLOR_DIVERSITY
        and contour_score >= VISUAL_MIN_CONTOUR_BREAK_SCORE
        and outline_band_div >= VISUAL_MIN_OUTLINE_BAND_DIVERSITY
        and small_scale_score >= VISUAL_MIN_SMALL_SCALE_STRUCTURAL_SCORE
        and final_score >= VISUAL_MIN_FINAL_SCORE
    )

    return {
        "visual_score_final": float(final_score),
        "visual_score_ratio": float(s_ratio),
        "visual_score_silhouette": float(s_sil),
        "visual_score_contour": float(s_contour),
        "visual_score_main": float(s_main),
        "visual_silhouette_color_diversity": float(sil_div),
        "visual_contour_break_score": float(contour_score),
        "visual_outline_band_diversity": float(outline_band_div),
        "visual_small_scale_structural_score": float(small_scale_score),
        "visual_validation_passed": 1.0 if visual_valid else 0.0,
    }


def military_visual_discipline_score(metrics: Dict[str, float]) -> Dict[str, float]:
    parts = [
        clamp01((metrics.get("visual_score_final", 0.0) - 0.45) / 0.30),
        clamp01((metrics.get("periphery_boundary_density_ratio", 0.0) - 0.95) / 0.40),
        clamp01((metrics.get("periphery_non_coyote_ratio", 0.0) - 0.95) / 0.40),
        clamp01((metrics.get("macro_olive_visible_ratio", 0.0) - 0.14) / 0.12),
        clamp01((metrics.get("macro_terre_visible_ratio", 0.0) - 0.10) / 0.10),
        clamp01((metrics.get("macro_gris_visible_ratio", 0.0) - 0.07) / 0.10),
        clamp01((metrics.get("oblique_share", 0.0) - 0.55) / 0.25),
        clamp01(1.0 - metrics.get("mirror_similarity", 1.0) / 0.85),
    ]
    score = float(np.mean(parts))
    return {
        "visual_military_score": score,
        "visual_military_passed": 1.0 if score >= VISUAL_MIN_MILITARY_SCORE else 0.0,
    }


# ============================================================
# GÉNÉRATION D'UNE VARIANTE
# ============================================================

def generate_one_variant(profile: VariantProfile) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
    rng = random.Random(profile.seed)
    canvas = np.full((HEIGHT, WIDTH), IDX_COYOTE, dtype=np.uint8)
    origin_map = np.full((HEIGHT, WIDTH), ORIGIN_BACKGROUND, dtype=np.uint8)

    macros: List[MacroRecord] = []
    add_forced_structural_macros(canvas, origin_map, macros, profile, rng)
    add_macros(canvas, origin_map, macros, profile, rng)
    enforce_macro_population(canvas, origin_map, macros, profile, rng)
    enforce_macro_angle_discipline(canvas, origin_map, macros, profile, rng)
    enforce_macro_population(canvas, origin_map, macros, profile, rng)
    repair_center_and_periphery(canvas, origin_map, macros, profile, rng)
    enforce_macro_population(canvas, origin_map, macros, profile, rng)

    rs = compute_ratios(canvas)
    orient = orientation_score(macros)
    multi = multiscale_metrics(canvas)

    olive_macros = [m for m in macros if m.color_idx == IDX_OLIVE]
    olive_multizone_share = float(np.mean([m.zone_count >= 2 for m in olive_macros])) if olive_macros else 0.0

    metrics = {
        "largest_olive_component_ratio": largest_component_ratio(canvas == IDX_OLIVE),
        "center_empty_ratio": center_empty_ratio(canvas),
        "boundary_density": boundary_density(canvas),
        "mirror_similarity": mirror_similarity_score(canvas),
        "central_brown_continuity": central_brown_continuity(canvas),
        "oblique_share": orient["oblique_share"],
        "vertical_share": orient["vertical_share"],
        "angle_dominance_ratio": orient["dominance_ratio"],
        "olive_multizone_share": olive_multizone_share,
        **absolute_origin_color_ratios(canvas, origin_map),
        **spatial_discipline_metrics(canvas),
        **macro_system_metrics(macros, canvas, origin_map),
        **multi,
    }
    metrics.update(evaluate_visual_metrics(canvas, rs, metrics))
    metrics.update(military_visual_discipline_score(metrics))

    return render_canvas(canvas), rs, metrics


def generate_candidate_from_seed(seed: int) -> CandidateResult:
    profile = make_profile(seed)
    image, ratios, metrics = generate_one_variant(profile)
    return CandidateResult(seed=seed, profile=profile, image=image, ratios=ratios, metrics=metrics)


async def async_generate_candidate_from_seed(seed: int) -> CandidateResult:
    return await asyncio.to_thread(generate_candidate_from_seed, seed)


# ============================================================
# VALIDATION
# ============================================================

def variant_is_valid(rs: np.ndarray, metrics: Dict[str, float]) -> bool:
    abs_err = np.abs(rs - TARGET)
    if np.any(abs_err > MAX_ABS_ERROR_PER_COLOR):
        return False
    if float(np.mean(abs_err)) > MAX_MEAN_ABS_ERROR:
        return False

    if rs[IDX_COYOTE] < 0.27 or rs[IDX_COYOTE] > 0.37:
        return False
    if rs[IDX_OLIVE] < 0.24 or rs[IDX_OLIVE] > 0.33:
        return False
    if rs[IDX_TERRE] < 0.19 or rs[IDX_TERRE] > 0.26:
        return False
    if rs[IDX_GRIS] < 0.14 or rs[IDX_GRIS] > 0.21:
        return False

    if metrics["largest_olive_component_ratio"] < MIN_OLIVE_CONNECTED_COMPONENT_RATIO:
        return False
    if metrics["largest_olive_component_ratio_small"] < 0.12:
        return False
    if metrics["olive_multizone_share"] < MIN_OLIVE_MULTIZONE_SHARE:
        return False

    if metrics["center_empty_ratio"] > MAX_COYOTE_CENTER_EMPTY_RATIO:
        return False
    if metrics["center_empty_ratio_small"] > MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL:
        return False

    if metrics["boundary_density"] < MIN_BOUNDARY_DENSITY or metrics["boundary_density"] > MAX_BOUNDARY_DENSITY:
        return False
    if metrics["boundary_density_small"] < MIN_BOUNDARY_DENSITY_SMALL or metrics["boundary_density_small"] > MAX_BOUNDARY_DENSITY_SMALL:
        return False

    if metrics["mirror_similarity"] > MAX_MIRROR_SIMILARITY:
        return False
    if metrics["central_brown_continuity"] > MAX_CENTRAL_BROWN_CONTINUITY:
        return False

    if metrics["oblique_share"] < MIN_OBLIQUE_SHARE:
        return False
    if metrics["vertical_share"] < MIN_VERTICAL_SHARE or metrics["vertical_share"] > MAX_VERTICAL_SHARE:
        return False
    if metrics["angle_dominance_ratio"] > MAX_ANGLE_DOMINANCE_RATIO:
        return False

    if metrics["macro_olive_visible_ratio"] < MIN_MACRO_OLIVE_VISIBLE_RATIO:
        return False
    if metrics["macro_terre_visible_ratio"] < MIN_MACRO_TERRE_VISIBLE_RATIO:
        return False
    if metrics["macro_gris_visible_ratio"] < MIN_MACRO_GRIS_VISIBLE_RATIO:
        return False

    if metrics["macro_total_count"] < MIN_TOTAL_MACRO_COUNT:
        return False
    if metrics["macro_olive_count"] < MIN_OLIVE_MACRO_COUNT:
        return False
    if metrics["macro_terre_count"] < MIN_TERRE_MACRO_COUNT:
        return False
    if metrics["macro_gris_count"] < MIN_GRIS_MACRO_COUNT:
        return False
    if metrics["macro_multizone_ratio"] < MIN_GLOBAL_MACRO_MULTIZONE_RATIO:
        return False
    if metrics["largest_macro_mask_ratio"] > MAX_SINGLE_MACRO_MASK_RATIO:
        return False

    if metrics["periphery_boundary_density_ratio"] < MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO:
        return False
    if metrics["periphery_non_coyote_ratio"] < MIN_PERIPHERY_NON_COYOTE_RATIO:
        return False

    if metrics["visual_silhouette_color_diversity"] < VISUAL_MIN_SILHOUETTE_COLOR_DIVERSITY:
        return False
    if metrics["visual_contour_break_score"] < VISUAL_MIN_CONTOUR_BREAK_SCORE:
        return False
    if metrics["visual_outline_band_diversity"] < VISUAL_MIN_OUTLINE_BAND_DIVERSITY:
        return False
    if metrics["visual_small_scale_structural_score"] < VISUAL_MIN_SMALL_SCALE_STRUCTURAL_SCORE:
        return False
    if metrics["visual_score_final"] < VISUAL_MIN_FINAL_SCORE:
        return False
    if metrics["visual_military_score"] < VISUAL_MIN_MILITARY_SCORE:
        return False

    return True


def validate_candidate_result(candidate: CandidateResult) -> bool:
    return variant_is_valid(candidate.ratios, candidate.metrics)


async def async_validate_candidate_result(candidate: CandidateResult) -> bool:
    return await asyncio.to_thread(validate_candidate_result, candidate)


# ============================================================
# EXPORT / RAPPORT
# ============================================================

def save_candidate_image(candidate: CandidateResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate.image.save(path)
    return path


async def async_save_candidate_image(candidate: CandidateResult, path: Path) -> Path:
    return await asyncio.to_thread(save_candidate_image, candidate, path)


def candidate_row(target_index: int, local_attempt: int, global_attempt: int, candidate: CandidateResult) -> Dict[str, object]:
    rs = candidate.ratios
    metrics = candidate.metrics
    return {
        "index": target_index,
        "seed": candidate.seed,
        "attempts_for_this_image": local_attempt,
        "global_attempt": global_attempt,
        "coyote_brown_pct": round(float(rs[IDX_COYOTE] * 100), 2),
        "vert_olive_pct": round(float(rs[IDX_OLIVE] * 100), 2),
        "terre_de_france_pct": round(float(rs[IDX_TERRE] * 100), 2),
        "vert_de_gris_pct": round(float(rs[IDX_GRIS] * 100), 2),
        "largest_olive_component_ratio": round(metrics["largest_olive_component_ratio"], 5),
        "largest_olive_component_ratio_small": round(metrics["largest_olive_component_ratio_small"], 5),
        "olive_multizone_share": round(metrics["olive_multizone_share"], 5),
        "center_empty_ratio": round(metrics["center_empty_ratio"], 5),
        "center_empty_ratio_small": round(metrics["center_empty_ratio_small"], 5),
        "boundary_density": round(metrics["boundary_density"], 5),
        "boundary_density_small": round(metrics["boundary_density_small"], 5),
        "boundary_density_tiny": round(metrics["boundary_density_tiny"], 5),
        "mirror_similarity": round(metrics["mirror_similarity"], 5),
        "central_brown_continuity": round(metrics["central_brown_continuity"], 5),
        "oblique_share": round(metrics["oblique_share"], 5),
        "vertical_share": round(metrics["vertical_share"], 5),
        "angle_dominance_ratio": round(metrics["angle_dominance_ratio"], 5),
        "macro_olive_visible_ratio": round(metrics["macro_olive_visible_ratio"], 5),
        "macro_terre_visible_ratio": round(metrics["macro_terre_visible_ratio"], 5),
        "macro_gris_visible_ratio": round(metrics["macro_gris_visible_ratio"], 5),
        "macro_total_count": int(metrics["macro_total_count"]),
        "macro_olive_count": int(metrics["macro_olive_count"]),
        "macro_terre_count": int(metrics["macro_terre_count"]),
        "macro_gris_count": int(metrics["macro_gris_count"]),
        "macro_multizone_ratio": round(metrics["macro_multizone_ratio"], 5),
        "largest_macro_mask_ratio": round(metrics["largest_macro_mask_ratio"], 5),
        "periphery_boundary_density_ratio": round(metrics["periphery_boundary_density_ratio"], 5),
        "periphery_non_coyote_ratio": round(metrics["periphery_non_coyote_ratio"], 5),
        "visual_score_final": round(metrics["visual_score_final"], 5),
        "visual_silhouette_color_diversity": round(metrics["visual_silhouette_color_diversity"], 5),
        "visual_contour_break_score": round(metrics["visual_contour_break_score"], 5),
        "visual_outline_band_diversity": round(metrics["visual_outline_band_diversity"], 5),
        "visual_small_scale_structural_score": round(metrics["visual_small_scale_structural_score"], 5),
        "visual_military_score": round(metrics["visual_military_score"], 5),
        "angles": " ".join(map(str, candidate.profile.allowed_angles)),
    }


def write_report(rows: List[Dict[str, object]], output_dir: Path, filename: str = "rapport_camouflages.csv") -> Path:
    output_dir = ensure_output_dir(output_dir)
    csv_path = output_dir / filename

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


async def async_write_report(rows: List[Dict[str, object]], output_dir: Path, filename: str = "rapport_camouflages.csv") -> Path:
    return await asyncio.to_thread(write_report, rows, output_dir, filename)


# ============================================================
# AIDES PARALLÉLISATION
# ============================================================

def generate_and_validate_from_seed(seed: int) -> Tuple[CandidateResult, bool]:
    candidate = generate_candidate_from_seed(seed)
    accepted = validate_candidate_result(candidate)
    return candidate, accepted


def _batch_attempt_seeds(target_index: int, start_attempt: int, batch_size: int, base_seed: int) -> List[Tuple[int, int]]:
    return [
        (local_attempt, build_seed(target_index, local_attempt, base_seed=base_seed))
        for local_attempt in range(start_attempt, start_attempt + batch_size)
    ]


# ============================================================
# GÉNÉRATION SYNCHRONE / STREAMING
# ============================================================

def generate_all(
    target_count: int = N_VARIANTS_REQUIRED,
    output_dir: Path = OUTPUT_DIR,
    base_seed: int = DEFAULT_BASE_SEED,
    progress_callback: Optional[ProgressCallback] = None,
    stop_requested: Optional[StopCallback] = None,
    max_workers: Optional[int] = None,
    attempt_batch_size: Optional[int] = None,
    parallel_attempts: bool = True,
    machine_intensity: float = DEFAULT_MACHINE_INTENSITY,
    strict_preflight: bool = True,
    preflight_modules: Sequence[str] | None = ("test_main", "test_start"),
    resource_sample_every_batches: int = DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES,
    enable_live_supervisor: bool = True,
) -> List[Dict[str, object]]:
    output_dir = ensure_output_dir(output_dir)
    validate_generation_request(
        target_count=target_count,
        output_dir=output_dir,
        base_seed=base_seed,
        machine_intensity=machine_intensity,
        max_workers=max_workers,
        attempt_batch_size=attempt_batch_size,
    )
    _run_log_preflight(strict=strict_preflight, output_dir=output_dir, module_names=preflight_modules)

    resource_sample_every_batches = max(1, int(resource_sample_every_batches))
    tuning = compute_runtime_tuning(
        max_workers=max_workers,
        attempt_batch_size=attempt_batch_size,
        parallel_attempts=parallel_attempts,
        machine_intensity=machine_intensity,
        sample=sample_process_resources(machine_intensity=machine_intensity, output_dir=output_dir),
    )
    _runtime_log("INFO", "main_generate_all", "Plan de charge initial", tuning=tuning.__dict__)

    if enable_live_supervisor:
        advice = _supervisor_feedback("generation_started", tuning=tuning.__dict__, output_dir=str(output_dir), target_count=target_count)
        tuning = _merge_supervisor_tuning(tuning, advice, fallback_machine_intensity=machine_intensity)

    rows: List[Dict[str, object]] = []
    total_attempts = 0
    batch_counter = 0

    for target_index in range(1, target_count + 1):
        local_attempt = 1

        while True:
            if stop_requested is not None and stop_requested():
                write_report(rows, output_dir)
                return rows

            batch_counter += 1
            if batch_counter % resource_sample_every_batches == 0:
                snapshot = sample_process_resources(machine_intensity=tuning.machine_intensity, output_dir=output_dir)
                if enable_live_supervisor:
                    advice = _supervisor_feedback(
                        "resource_snapshot",
                        target_index=target_index,
                        local_attempt=local_attempt,
                        snapshot=snapshot.to_dict(),
                        tuning=tuning.__dict__,
                    )
                    tuning = _merge_supervisor_tuning(tuning, advice)
                else:
                    snapshot = snapshot  # explicit

            use_parallel = bool(tuning.parallel_attempts and tuning.max_workers > 1 and tuning.attempt_batch_size > 1)

            if use_parallel:
                pool = get_process_pool(tuning.max_workers)
                batch = _batch_attempt_seeds(target_index, local_attempt, tuning.attempt_batch_size, base_seed)
                submitted: Dict[Any, Tuple[int, int, float]] = {}
                for attempt_no, seed in batch:
                    fut = pool.submit(generate_and_validate_from_seed, seed)
                    submitted[fut] = (attempt_no, seed, time.time())

                batch_results: List[Tuple[int, CandidateResult, bool]] = []
                for fut in as_completed(list(submitted.keys())):
                    attempt_no, seed, started_at = submitted[fut]
                    candidate, accepted = fut.result()
                    duration_s = time.time() - started_at
                    total_attempts += 1
                    batch_results.append((attempt_no, candidate, accepted))

                    if progress_callback is not None:
                        progress_callback(target_index, attempt_no, total_attempts, target_count, candidate, accepted)

                    if enable_live_supervisor:
                        advice = _supervisor_feedback(
                            "attempt_finished",
                            target_index=target_index,
                            local_attempt=attempt_no,
                            seed=seed,
                            accepted=accepted,
                            duration_s=duration_s,
                            total_attempts=total_attempts,
                            tuning=tuning.__dict__,
                            ratios={COLOR_NAMES[i]: float(candidate.ratios[i]) for i in range(4)},
                            metrics={k: float(v) for k, v in candidate.metrics.items()},
                        )
                        tuning = _merge_supervisor_tuning(tuning, advice)

                accepted_result = next(((a, c) for a, c, ok in sorted(batch_results, key=lambda x: x[0]) if ok), None)
                if accepted_result is None:
                    local_attempt += tuning.attempt_batch_size
                    if enable_live_supervisor:
                        advice = _supervisor_feedback(
                            "batch_finished",
                            target_index=target_index,
                            accepted=False,
                            tuning=tuning.__dict__,
                            next_local_attempt=local_attempt,
                        )
                        tuning = _merge_supervisor_tuning(tuning, advice)
                    continue

                accepted_attempt, accepted_candidate = accepted_result

            else:
                seed = build_seed(target_index, local_attempt, base_seed=base_seed)
                started_at = time.time()
                accepted_candidate = generate_candidate_from_seed(seed)
                accepted = validate_candidate_result(accepted_candidate)
                duration_s = time.time() - started_at
                total_attempts += 1

                if progress_callback is not None:
                    progress_callback(target_index, local_attempt, total_attempts, target_count, accepted_candidate, accepted)

                if enable_live_supervisor:
                    advice = _supervisor_feedback(
                        "attempt_finished",
                        target_index=target_index,
                        local_attempt=local_attempt,
                        seed=seed,
                        accepted=accepted,
                        duration_s=duration_s,
                        total_attempts=total_attempts,
                        tuning=tuning.__dict__,
                        ratios={COLOR_NAMES[i]: float(accepted_candidate.ratios[i]) for i in range(4)},
                        metrics={k: float(v) for k, v in accepted_candidate.metrics.items()},
                    )
                    tuning = _merge_supervisor_tuning(tuning, advice)

                if not accepted:
                    local_attempt += 1
                    continue
                accepted_attempt = local_attempt

            filename = output_dir / f"camouflage_{target_index:03d}.png"
            save_candidate_image(accepted_candidate, filename)
            rows.append(candidate_row(target_index, accepted_attempt, total_attempts, accepted_candidate))

            if enable_live_supervisor:
                advice = _supervisor_feedback(
                    "candidate_accepted",
                    target_index=target_index,
                    accepted_attempt=accepted_attempt,
                    total_attempts=total_attempts,
                    tuning=tuning.__dict__,
                    output_file=str(filename),
                )
                tuning = _merge_supervisor_tuning(tuning, advice)

            break

    write_report(rows, output_dir)

    if enable_live_supervisor:
        _supervisor_feedback(
            "generation_finished",
            total_rows=len(rows),
            total_attempts=total_attempts,
            tuning=tuning.__dict__,
            output_dir=str(output_dir),
        )

    return rows


# ============================================================
# GÉNÉRATION ASYNCHRONE / STREAMING
# ============================================================


async def _wrap_async_attempt(
    fut: asyncio.Future,
    attempt_no: int,
    seed: int,
    started_at: float,
) -> Tuple[int, int, float, CandidateResult, bool]:
    candidate, accepted = await fut
    return attempt_no, seed, started_at, candidate, accepted


async def async_generate_all(
    target_count: int = N_VARIANTS_REQUIRED,
    output_dir: Path = OUTPUT_DIR,
    base_seed: int = DEFAULT_BASE_SEED,
    progress_callback: Optional[AsyncProgressCallback] = None,
    stop_requested: Optional[AsyncStopCallback] = None,
    max_workers: Optional[int] = None,
    attempt_batch_size: Optional[int] = None,
    parallel_attempts: bool = True,
    machine_intensity: float = DEFAULT_MACHINE_INTENSITY,
    strict_preflight: bool = True,
    preflight_modules: Sequence[str] | None = ("test_main", "test_start"),
    resource_sample_every_batches: int = DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES,
    enable_live_supervisor: bool = True,
) -> List[Dict[str, object]]:
    output_dir = ensure_output_dir(output_dir)
    validate_generation_request(
        target_count=target_count,
        output_dir=output_dir,
        base_seed=base_seed,
        machine_intensity=machine_intensity,
        max_workers=max_workers,
        attempt_batch_size=attempt_batch_size,
    )
    _run_log_preflight(strict=strict_preflight, output_dir=output_dir, module_names=preflight_modules)

    resource_sample_every_batches = max(1, int(resource_sample_every_batches))
    tuning = compute_runtime_tuning(
        max_workers=max_workers,
        attempt_batch_size=attempt_batch_size,
        parallel_attempts=parallel_attempts,
        machine_intensity=machine_intensity,
        sample=sample_process_resources(machine_intensity=machine_intensity, output_dir=output_dir),
    )
    _runtime_log("INFO", "main_async_generate_all", "Plan de charge initial", tuning=tuning.__dict__)

    if enable_live_supervisor:
        advice = _supervisor_feedback("generation_started", tuning=tuning.__dict__, output_dir=str(output_dir), target_count=target_count)
        tuning = _merge_supervisor_tuning(tuning, advice, fallback_machine_intensity=machine_intensity)

    rows: List[Dict[str, object]] = []
    total_attempts = 0
    batch_counter = 0
    loop = asyncio.get_running_loop()

    for target_index in range(1, target_count + 1):
        local_attempt = 1

        while True:
            if stop_requested is not None and await stop_requested():
                await async_write_report(rows, output_dir)
                return rows

            batch_counter += 1
            if batch_counter % resource_sample_every_batches == 0:
                snapshot = sample_process_resources(machine_intensity=tuning.machine_intensity, output_dir=output_dir)
                if enable_live_supervisor:
                    advice = _supervisor_feedback(
                        "resource_snapshot",
                        target_index=target_index,
                        local_attempt=local_attempt,
                        snapshot=snapshot.to_dict(),
                        tuning=tuning.__dict__,
                    )
                    tuning = _merge_supervisor_tuning(tuning, advice)

            use_parallel = bool(tuning.parallel_attempts and tuning.max_workers > 1 and tuning.attempt_batch_size > 1)

            if use_parallel:
                pool = get_process_pool(tuning.max_workers)
                batch = _batch_attempt_seeds(target_index, local_attempt, tuning.attempt_batch_size, base_seed)

                wrapped_tasks: List[asyncio.Task] = []
                for attempt_no, seed in batch:
                    fut = loop.run_in_executor(pool, generate_and_validate_from_seed, seed)
                    wrapped_tasks.append(
                        asyncio.create_task(_wrap_async_attempt(fut, attempt_no, seed, time.time()))
                    )

                batch_results: List[Tuple[int, CandidateResult, bool]] = []
                for done in asyncio.as_completed(wrapped_tasks):
                    attempt_no, seed, started_at, candidate, accepted = await done
                    duration_s = time.time() - started_at
                    total_attempts += 1
                    batch_results.append((attempt_no, candidate, accepted))

                    if progress_callback is not None:
                        await progress_callback(target_index, attempt_no, total_attempts, target_count, candidate, accepted)

                    if enable_live_supervisor:
                        advice = _supervisor_feedback(
                            "attempt_finished",
                            target_index=target_index,
                            local_attempt=attempt_no,
                            seed=seed,
                            accepted=accepted,
                            duration_s=duration_s,
                            total_attempts=total_attempts,
                            tuning=tuning.__dict__,
                            ratios={COLOR_NAMES[i]: float(candidate.ratios[i]) for i in range(4)},
                            metrics={k: float(v) for k, v in candidate.metrics.items()},
                        )
                        tuning = _merge_supervisor_tuning(tuning, advice)

                accepted_result = next(((a, c) for a, c, ok in sorted(batch_results, key=lambda x: x[0]) if ok), None)
                if accepted_result is None:
                    local_attempt += tuning.attempt_batch_size
                    if enable_live_supervisor:
                        advice = _supervisor_feedback(
                            "batch_finished",
                            target_index=target_index,
                            accepted=False,
                            tuning=tuning.__dict__,
                            next_local_attempt=local_attempt,
                        )
                        tuning = _merge_supervisor_tuning(tuning, advice)
                    continue

                accepted_attempt, accepted_candidate = accepted_result

            else:
                seed = build_seed(target_index, local_attempt, base_seed=base_seed)
                started_at = time.time()
                accepted_candidate = await async_generate_candidate_from_seed(seed)
                accepted = await async_validate_candidate_result(accepted_candidate)
                duration_s = time.time() - started_at
                total_attempts += 1

                if progress_callback is not None:
                    await progress_callback(target_index, local_attempt, total_attempts, target_count, accepted_candidate, accepted)

                if enable_live_supervisor:
                    advice = _supervisor_feedback(
                        "attempt_finished",
                        target_index=target_index,
                        local_attempt=local_attempt,
                        seed=seed,
                        accepted=accepted,
                        duration_s=duration_s,
                        total_attempts=total_attempts,
                        tuning=tuning.__dict__,
                        ratios={COLOR_NAMES[i]: float(accepted_candidate.ratios[i]) for i in range(4)},
                        metrics={k: float(v) for k, v in accepted_candidate.metrics.items()},
                    )
                    tuning = _merge_supervisor_tuning(tuning, advice)

                if not accepted:
                    local_attempt += 1
                    continue

                accepted_attempt = local_attempt

            filename = output_dir / f"camouflage_{target_index:03d}.png"
            await async_save_candidate_image(accepted_candidate, filename)
            rows.append(candidate_row(target_index, accepted_attempt, total_attempts, accepted_candidate))

            if enable_live_supervisor:
                advice = _supervisor_feedback(
                    "candidate_accepted",
                    target_index=target_index,
                    accepted_attempt=accepted_attempt,
                    total_attempts=total_attempts,
                    tuning=tuning.__dict__,
                    output_file=str(filename),
                )
                tuning = _merge_supervisor_tuning(tuning, advice)

            break

    await async_write_report(rows, output_dir)

    if enable_live_supervisor:
        _supervisor_feedback(
            "generation_finished",
            total_rows=len(rows),
            total_attempts=total_attempts,
            tuning=tuning.__dict__,
            output_dir=str(output_dir),
        )

    return rows


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        rows = generate_all(
            target_count=N_VARIANTS_REQUIRED,
            output_dir=OUTPUT_DIR,
            base_seed=DEFAULT_BASE_SEED,
            max_workers=None,
            attempt_batch_size=None,
            parallel_attempts=True,
            machine_intensity=DEFAULT_MACHINE_INTENSITY,
            strict_preflight=False,
            enable_live_supervisor=True,
        )
        csv_path = OUTPUT_DIR / "rapport_camouflages.csv"
        print("\nTerminé.")
        print(f"Images validées : {len(rows)}/{N_VARIANTS_REQUIRED}")
        print(f"Dossier : {OUTPUT_DIR.resolve()}")
        print(f"CSV : {csv_path.resolve()}")
        print(f"Workers max dynamiques : {DEFAULT_MAX_WORKERS}")
    finally:
        shutdown_process_pool()
