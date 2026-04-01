# -*- coding: utf-8 -*-
"""
main.py
Camouflage Armée Fédérale Europe — générateur 8K horizontal, asynchrone,
strict, orienté production, avec suivi temps réel, rejet/acceptation en direct,
et exploitation forte de la machine sous contrainte mémoire.

Corrections principales :
- motifs globalement plus petits ;
- proportions des couleurs verrouillées par une passe finale stricte ;
- compteurs live corrigés ;
- quelques garde-fous de robustesse supplémentaires.

Principes :
- 8K horizontal par défaut : 7680 x 4320
- échelle physique par défaut : 768 cm x 432 cm (1 px = 1 mm à 1 m)
- hiérarchie multi-échelle en centimètres pour lecture longue / moyenne / courte distance
- génération overscan + crop central pour éviter les artefacts de bord
- génération mémoire-maîtrisée : une couche à la fois, crop avant empilement
- pipeline asynchrone + ProcessPoolExecutor
- compteurs live : validés, rejetés, tentatives, en cours
- validation stricte mais adaptée au moteur organique
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

OUTPUT_DIR = Path("camouflages_federale_europe_8k")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WIDTH = 7680
DEFAULT_HEIGHT = 4320
WIDTH = DEFAULT_WIDTH
HEIGHT = DEFAULT_HEIGHT

DEFAULT_PHYSICAL_WIDTH_CM = 768.0
DEFAULT_PHYSICAL_HEIGHT_CM = 432.0
PHYSICAL_WIDTH_CM = DEFAULT_PHYSICAL_WIDTH_CM
PHYSICAL_HEIGHT_CM = DEFAULT_PHYSICAL_HEIGHT_CM
PX_PER_CM = min(WIDTH / PHYSICAL_WIDTH_CM, HEIGHT / PHYSICAL_HEIGHT_CM)

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

CPU_COUNT = max(1, os.cpu_count() or 1)
DEFAULT_MAX_WORKERS = max(1, CPU_COUNT)
DEFAULT_ATTEMPT_BATCH_SIZE = max(1, DEFAULT_MAX_WORKERS)
DEFAULT_MACHINE_INTENSITY = 0.98
DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES = 1
DEFAULT_OVERSCAN = 1.10

# Réduit pour obtenir des motifs plus fins tout en conservant une lecture multi-distance.
MIN_MOTIF_SCALE = 0.10
MAX_MOTIF_SCALE = 1.20
DEFAULT_MOTIF_SCALE = 0.58
MOTIF_SCALE = DEFAULT_MOTIF_SCALE
MIN_PATCH_PX = 2.0

# Validation stricte adaptée au moteur organique 8K.
MAX_ABS_ERROR_PER_COLOR = np.array([0.0015, 0.0015, 0.0015, 0.0015], dtype=float)
MAX_MEAN_ABS_ERROR = 0.0010
MIN_BOUNDARY_DENSITY = 0.010
MAX_BOUNDARY_DENSITY = 0.125
MIN_BOUNDARY_DENSITY_SMALL = 0.015
MAX_BOUNDARY_DENSITY_SMALL = 0.165
MIN_BOUNDARY_DENSITY_TINY = 0.020
MAX_BOUNDARY_DENSITY_TINY = 0.210
MAX_MIRROR_SIMILARITY = 0.88
MIN_LARGEST_OLIVE_COMPONENT_RATIO = 0.08
MAX_EDGE_CONTACT_RATIO = 0.72


# ============================================================
# STRUCTURES
# ============================================================

@dataclass
class VariantProfile:
    seed: int
    overscan: float
    shift_strength: float
    palette_bias: Tuple[float, float, float, float]


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


@dataclass
class LiveCounters:
    target_count: int
    accepted: int = 0                  # images effectivement retenues et sauvegardées
    passed_validation: int = 0         # tentatives passées en validation
    rejected: int = 0
    attempts: int = 0
    in_flight: int = 0
    start_ts: float = field(default_factory=time.time)

    def line(self, current_target: int, workers: int) -> str:
        elapsed = max(0.001, time.time() - self.start_ts)
        rate = self.accepted / elapsed
        return (
            f"\r[{time.strftime('%H:%M:%S')}] "
            f"fait={self.accepted}/{self.target_count} "
            f"valides={self.passed_validation} "
            f"rejets={self.rejected} "
            f"tentatives={self.attempts} "
            f"en_cours={self.in_flight} "
            f"cible={current_target}/{self.target_count} "
            f"workers={workers} "
            f"cadence={rate:.3f}/s"
        )


ProgressCallback = Callable[[int, int, int, int, CandidateResult, bool], None]
AsyncProgressCallback = Callable[[int, int, int, int, CandidateResult, bool], Awaitable[None]]
StopCallback = Callable[[], bool]
AsyncStopCallback = Callable[[], Awaitable[bool]]


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
    if str(os.environ.get("CAMO_LIMIT_NUMERIC_THREADS", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_canvas_geometry(
    width: int,
    height: int,
    physical_width_cm: float = DEFAULT_PHYSICAL_WIDTH_CM,
    physical_height_cm: float = DEFAULT_PHYSICAL_HEIGHT_CM,
    motif_scale: float = DEFAULT_MOTIF_SCALE,
) -> None:
    global WIDTH, HEIGHT, PHYSICAL_WIDTH_CM, PHYSICAL_HEIGHT_CM, PX_PER_CM, MOTIF_SCALE
    width = int(width)
    height = int(height)
    physical_width_cm = float(physical_width_cm)
    physical_height_cm = float(physical_height_cm)
    motif_scale = float(motif_scale)
    if width <= 0 or height <= 0:
        raise ValueError("width et height doivent être > 0")
    if physical_width_cm <= 0 or physical_height_cm <= 0:
        raise ValueError("physical_width_cm et physical_height_cm doivent être > 0")
    if motif_scale <= 0:
        raise ValueError("motif_scale doit être > 0")
    WIDTH = width
    HEIGHT = height
    PHYSICAL_WIDTH_CM = physical_width_cm
    PHYSICAL_HEIGHT_CM = physical_height_cm
    px_per_cm_x = WIDTH / PHYSICAL_WIDTH_CM
    px_per_cm_y = HEIGHT / PHYSICAL_HEIGHT_CM
    PX_PER_CM = min(px_per_cm_x, px_per_cm_y)
    MOTIF_SCALE = max(MIN_MOTIF_SCALE, min(MAX_MOTIF_SCALE, motif_scale))


def set_motif_scale(motif_scale: float) -> float:
    global MOTIF_SCALE
    motif_scale = float(motif_scale)
    if motif_scale <= 0:
        raise ValueError("motif_scale doit être > 0")
    MOTIF_SCALE = max(MIN_MOTIF_SCALE, min(MAX_MOTIF_SCALE, motif_scale))
    return MOTIF_SCALE


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
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=wanted, initializer=_worker_initializer)
        _PROCESS_POOL_WORKERS = wanted
    return _PROCESS_POOL


# ============================================================
# PRÉFLIGHT / RESSOURCES
# ============================================================

def _clip_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def sample_process_resources(
    machine_intensity: float = DEFAULT_MACHINE_INTENSITY,
    output_dir: Path = OUTPUT_DIR,
) -> ResourceSnapshot:
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
        if sample.system_available_mb < 2048:
            baseline_workers = min(baseline_workers, 1)
        elif sample.system_available_mb < 4096:
            baseline_workers = min(baseline_workers, max(1, CPU_COUNT // 4))
        elif sample.system_available_mb < 8192:
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
    probe = Path(output_dir) / ".write_probe.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    snapshot = sample_process_resources(machine_intensity=intensity, output_dir=output_dir)
    if snapshot.disk_free_mb < 1024:
        raise RuntimeError("Espace disque insuffisant pour lancer un lot 8K")
    _runtime_log("INFO", "main_preflight", "Préflight local validé", snapshot=snapshot.to_dict())


# ============================================================
# PROFILS
# ============================================================

def build_seed(target_index: int, local_attempt: int, base_seed: int = DEFAULT_BASE_SEED) -> int:
    return int(base_seed + target_index * 100000 + local_attempt)


def make_profile(seed: int) -> VariantProfile:
    rng = random.Random(seed)
    overscan = _clip_float(DEFAULT_OVERSCAN + rng.uniform(-0.02, 0.03), 1.08, 1.16)

    # Shift un peu réduit pour éviter que les masses fines ne fusionnent trop.
    shift_strength = _clip_float(rng.uniform(0.68, 0.98), 0.50, 1.10)

    # Léger biais de palette, volontairement modéré car les proportions sont verrouillées ensuite.
    palette_bias = (
        rng.uniform(-0.006, 0.006),
        rng.uniform(-0.006, 0.006),
        rng.uniform(-0.006, 0.006),
        rng.uniform(-0.006, 0.006),
    )
    return VariantProfile(seed=seed, overscan=overscan, shift_strength=shift_strength, palette_bias=palette_bias)


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def compute_ratios(canvas: np.ndarray) -> np.ndarray:
    counts = np.bincount(canvas.ravel(), minlength=4).astype(np.float64)
    return counts / canvas.size


def render_canvas(canvas: np.ndarray) -> Image.Image:
    return Image.fromarray(RGB[canvas], "RGB")


def boundary_mask(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape
    diff = np.zeros((h, w), dtype=bool)
    diff[1:, :] |= (canvas[1:, :] != canvas[:-1, :])
    diff[:-1, :] |= (canvas[:-1, :] != canvas[1:, :])
    diff[:, 1:] |= (canvas[:, 1:] != canvas[:, :-1])
    diff[:, :-1] |= (canvas[:, :-1] != canvas[:, 1:])
    return diff


def boundary_density(canvas: np.ndarray) -> float:
    return float(np.mean(boundary_mask(canvas)))


def mirror_similarity_score(canvas: np.ndarray) -> float:
    mid = canvas.shape[1] // 2
    left = canvas[:, :mid]
    right = canvas[:, canvas.shape[1] - mid:]
    right_flipped = np.fliplr(right)
    h = min(left.shape[0], right_flipped.shape[0])
    w = min(left.shape[1], right_flipped.shape[1])
    return float(np.mean(left[:h, :w] == right_flipped[:h, :w]))


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
            best = max(best, size)
    return best / total


def edge_contact_ratio(canvas: np.ndarray) -> float:
    top = canvas[0, :]
    bottom = canvas[-1, :]
    left = canvas[:, 0]
    right = canvas[:, -1]
    all_edges = np.concatenate([top, bottom, left, right])
    hist = np.bincount(all_edges, minlength=4).astype(np.float64)
    hist /= max(1.0, hist.sum())
    return float(hist.max())


def downsample_nearest(canvas: np.ndarray, factor: int) -> np.ndarray:
    return canvas[::factor, ::factor]


def center_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
        y0 = (h - target_h) // 2
        x0 = (w - target_w) // 2
        return arr[y0:y0 + target_h, x0:x0 + target_w]
    if arr.ndim == 3:
        _, h, w = arr.shape
        y0 = (h - target_h) // 2
        x0 = (w - target_w) // 2
        return arr[:, y0:y0 + target_h, x0:x0 + target_w]
    raise ValueError("center_crop attend un tableau 2D ou 3D")


def shift_reflect(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = arr.shape
    pad_y = abs(dy) + 2
    pad_x = abs(dx) + 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[y0:y0 + h, x0:x0 + w]


def cells_for_patch_size(patch_cm_x: float, patch_cm_y: float, width_px: int, height_px: int) -> Tuple[int, int]:
    # On abaisse le plancher pour permettre des motifs réellement plus fins en 8K.
    patch_px_x = max(MIN_PATCH_PX, float(patch_cm_x) * PX_PER_CM)
    patch_px_y = max(MIN_PATCH_PX, float(patch_cm_y) * PX_PER_CM)
    cells_x = max(3, int(round(width_px / patch_px_x)))
    cells_y = max(3, int(round(height_px / patch_px_y)))
    return cells_x, cells_y


def scaled_patch_size(patch_cm_x: float, patch_cm_y: float, motif_scale: float) -> Tuple[float, float]:
    motif_scale = max(MIN_MOTIF_SCALE, min(MAX_MOTIF_SCALE, float(motif_scale)))
    return float(patch_cm_x) * motif_scale, float(patch_cm_y) * motif_scale


# ============================================================
# GÉNÉRATEUR ORGANIQUE 8K HUMAN-SCALE
# ============================================================

def random_blob_layer(
    width: int,
    height: int,
    rng: np.random.Generator,
    cells_x: int,
    cells_y: int,
    angle_deg: float,
    pad_ratio: float = 0.16,
) -> np.ndarray:
    cells_x = max(3, int(cells_x))
    cells_y = max(3, int(cells_y))

    small = rng.integers(0, 256, size=(cells_y, cells_x), dtype=np.uint8)
    small_img = Image.fromarray(small, mode="L")

    pad = int(max(width, height) * pad_ratio)
    work_w = width + 2 * pad
    work_h = height + 2 * pad

    big = small_img.resize((work_w, work_h), resample=Image.Resampling.BICUBIC)
    rot = big.rotate(angle_deg, resample=Image.Resampling.BICUBIC)
    crop = rot.crop((pad, pad, pad + width, pad + height))

    arr = np.asarray(crop, dtype=np.uint8).astype(np.float16) / np.float16(255.0)
    return arr


def build_field(
    width: int,
    height: int,
    rng: np.random.Generator,
    plan: List[Tuple[int, int, float, float]],
    shift_strength: float,
) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float16)
    total_weight = np.float16(0.0)

    # Réduction légère du décalage maximum pour éviter des masses trop larges.
    max_shift_x = max(1, int((width // 24) * shift_strength))
    max_shift_y = max(1, int((height // 24) * shift_strength))

    for cells_x, cells_y, angle, weight in plan:
        layer = random_blob_layer(width, height, rng, cells_x, cells_y, angle)
        shift_x = int(rng.integers(-max_shift_x, max_shift_x + 1))
        shift_y = int(rng.integers(-max_shift_y, max_shift_y + 1))
        layer = shift_reflect(layer, shift_y, shift_x)
        field += layer * np.float16(weight)
        total_weight += np.float16(weight)
        del layer

    field = field / max(total_weight, np.float16(1e-3))
    fmin = np.float16(field.min())
    fmax = np.float16(field.max())
    denom = np.float16(max(float(fmax - fmin), 1e-3))
    field = (field - fmin) / denom
    return field.astype(np.float16, copy=False)


def build_all_fields(
    work_width: int,
    work_height: int,
    profile: VariantProfile,
    crop_height: int,
    crop_width: int,
    motif_scale: float = MOTIF_SCALE,
) -> np.ndarray:
    rng = np.random.default_rng(profile.seed)

    # Motifs globalement plus petits, mais toujours multi-distance.
    s1x, s1y = scaled_patch_size(54.0, 39.0, motif_scale)   # macro réduite
    s2x, s2y = scaled_patch_size(30.0, 22.0, motif_scale)   # moyenne
    s3x, s3y = scaled_patch_size(12.0, 8.8, motif_scale)    # intermédiaire
    s4x, s4y = scaled_patch_size(4.4, 3.2, motif_scale)     # micro
    s5x, s5y = scaled_patch_size(2.0, 1.5, motif_scale)     # mini
    s6x, s6y = scaled_patch_size(1.0, 0.75, motif_scale)    # très fin

    c1x, c1y = cells_for_patch_size(s1x, s1y, work_width, work_height)
    c2x, c2y = cells_for_patch_size(s2x, s2y, work_width, work_height)
    c3x, c3y = cells_for_patch_size(s3x, s3y, work_width, work_height)
    c4x, c4y = cells_for_patch_size(s4x, s4y, work_width, work_height)
    c5x, c5y = cells_for_patch_size(s5x, s5y, work_width, work_height)
    c6x, c6y = cells_for_patch_size(s6x, s6y, work_width, work_height)

    plans = {
        IDX_COYOTE: [
            (c1x, c1y, -18.0, 1.00),
            (c2x, c2y,  22.0, 0.64),
            (c3x, c3y,  -8.0, 0.32),
            (c4x, c4y,   0.0, 0.14),
            (c5x, c5y, -11.0, 0.07),
            (c6x, c6y,  15.0, 0.03),
        ],
        IDX_OLIVE: [
            (c1x, c1y,  16.0, 1.00),
            (c2x, c2y, -26.0, 0.64),
            (c3x, c3y,   7.0, 0.31),
            (c4x, c4y,  12.0, 0.15),
            (c5x, c5y,  24.0, 0.07),
            (c6x, c6y, -18.0, 0.03),
        ],
        IDX_TERRE: [
            (c1x, c1y, -10.0, 0.78),
            (c2x, c2y, -12.0, 0.92),
            (c3x, c3y,  26.0, 0.50),
            (c4x, c4y,  -4.0, 0.19),
            (c5x, c5y,   9.0, 0.08),
            (c6x, c6y, -21.0, 0.03),
        ],
        IDX_GRIS: [
            (c1x, c1y,  18.0, 0.74),
            (c2x, c2y,  20.0, 0.90),
            (c3x, c3y, -24.0, 0.52),
            (c4x, c4y,   0.0, 0.22),
            (c5x, c5y, -15.0, 0.09),
            (c6x, c6y,  11.0, 0.03),
        ],
    }

    fields: List[np.ndarray] = []
    for idx in range(4):
        local_seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(local_seed)
        field_i = build_field(work_width, work_height, local_rng, plans[idx], profile.shift_strength)
        field_i = center_crop(field_i, crop_height, crop_width)
        field_i = field_i + np.float16(profile.palette_bias[idx])
        fields.append(field_i.astype(np.float16, copy=False))
        del field_i

    return np.stack(fields, axis=0)


def sequential_assign(fields: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    _, height, width = fields.shape
    labels = np.zeros((height, width), dtype=np.uint8)
    remaining = np.ones((height, width), dtype=bool)

    for c in (IDX_OLIVE, IDX_TERRE, IDX_GRIS):
        count = int(target_counts[c])
        remaining_flat = np.flatnonzero(remaining.ravel())
        values = fields[c].ravel()[remaining_flat]
        best_local = np.argpartition(values, -count)[-count:]
        selected_flat = remaining_flat[best_local]
        labels.ravel()[selected_flat] = c
        remaining.ravel()[selected_flat] = False

    return labels


def exactify_proportions(labels: np.ndarray, fields: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    n_classes, height, width = fields.shape
    labels = labels.copy()

    for _ in range(24):
        counts = np.bincount(labels.ravel(), minlength=n_classes)
        delta = target_counts - counts
        if np.all(delta == 0):
            break

        flat_labels = labels.ravel()
        flat_fields = fields.reshape(n_classes, -1)
        flat_boundary = boundary_mask(labels).ravel()
        changed = False

        under = np.where(delta > 0)[0]
        over = np.where(delta < 0)[0]
        if len(under) == 0 or len(over) == 0:
            break

        for u in under:
            need = int(delta[u])
            if need <= 0:
                continue
            candidate_mask = flat_boundary & np.isin(flat_labels, over)
            idx = np.where(candidate_mask)[0]
            if idx.size == 0:
                continue
            current = flat_labels[idx]
            gain = flat_fields[u, idx].astype(np.float32) - flat_fields[current, idx].astype(np.float32)
            order = np.argsort(gain)[::-1]
            selected = idx[order[:need]]
            if selected.size == 0:
                continue
            flat_labels[selected] = u
            changed = True

        labels = flat_labels.reshape(height, width)
        if not changed:
            break

    return labels.astype(np.uint8)


def force_exact_target_counts(labels: np.ndarray, fields: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    """
    Passe finale stricte :
    force exactement les comptes cibles par couleur.
    """
    labels = labels.copy()
    flat_labels = labels.ravel()
    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)
    shape = labels.shape

    for _ in range(16):
        counts = np.bincount(flat_labels, minlength=fields.shape[0]).astype(int)
        delta = target_counts.astype(int) - counts
        if np.all(delta == 0):
            break

        boundary = boundary_mask(flat_labels.reshape(shape)).ravel()
        under = [int(c) for c in np.where(delta > 0)[0]]
        over = [int(c) for c in np.where(delta < 0)[0]]
        moved = False

        for dst in under:
            need = int(delta[dst])
            if need <= 0:
                continue

            for src in over:
                excess = -int(delta[src])
                if excess <= 0:
                    continue

                take = min(need, excess)

                idx_boundary = np.where((flat_labels == src) & boundary)[0]
                idx_pool = idx_boundary

                if idx_pool.size < take:
                    idx_inner = np.where((flat_labels == src) & (~boundary))[0]
                    if idx_inner.size:
                        idx_pool = np.concatenate([idx_boundary, idx_inner], axis=0)

                if idx_pool.size == 0:
                    continue

                gain = flat_fields[dst, idx_pool] - flat_fields[src, idx_pool]
                order = np.argsort(gain)[::-1]
                picked = idx_pool[order[:take]]

                if picked.size == 0:
                    continue

                flat_labels[picked] = dst
                delta[dst] -= picked.size
                delta[src] += picked.size
                need -= picked.size
                moved = True

                if need <= 0:
                    break

        if not moved:
            break

    return flat_labels.reshape(shape).astype(np.uint8)


def generate_one_variant(profile: VariantProfile) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
    width = int(WIDTH)
    height = int(HEIGHT)
    physical_width_cm = float(PHYSICAL_WIDTH_CM)
    physical_height_cm = float(PHYSICAL_HEIGHT_CM)
    px_per_cm = float(PX_PER_CM)
    motif_scale = float(MOTIF_SCALE)

    work_width = max(width + 64, int(round(width * profile.overscan)))
    work_height = max(height + 64, int(round(height * profile.overscan)))

    fields = build_all_fields(
        work_width,
        work_height,
        profile,
        crop_height=height,
        crop_width=width,
        motif_scale=motif_scale,
    )

    target_counts = np.rint(TARGET * (width * height)).astype(int)
    target_counts[-1] = (width * height) - int(target_counts[:-1].sum())

    canvas = sequential_assign(fields, target_counts)
    canvas = exactify_proportions(canvas, fields, target_counts)
    canvas = force_exact_target_counts(canvas, fields, target_counts)

    rs = compute_ratios(canvas)
    small = downsample_nearest(canvas, 4)
    tiny = downsample_nearest(canvas, 8)

    metrics = {
        "largest_olive_component_ratio": largest_component_ratio(canvas == IDX_OLIVE),
        "boundary_density": boundary_density(canvas),
        "boundary_density_small": boundary_density(small),
        "boundary_density_tiny": boundary_density(tiny),
        "mirror_similarity": mirror_similarity_score(canvas),
        "edge_contact_ratio": edge_contact_ratio(canvas),
        "overscan": float(profile.overscan),
        "shift_strength": float(profile.shift_strength),
        "width": float(width),
        "height": float(height),
        "physical_width_cm": float(physical_width_cm),
        "physical_height_cm": float(physical_height_cm),
        "px_per_cm": float(px_per_cm),
        "motif_scale": float(motif_scale),
    }
    return render_canvas(canvas), rs, metrics


def generate_candidate_from_seed(seed: int) -> CandidateResult:
    profile = make_profile(seed)
    image, ratios, metrics = generate_one_variant(profile)
    return CandidateResult(seed=seed, profile=profile, image=image, ratios=ratios, metrics=metrics)


def generate_and_validate_from_seed(seed: int) -> Tuple[CandidateResult, bool]:
    candidate = generate_candidate_from_seed(seed)
    accepted = validate_candidate_result(candidate)
    return candidate, accepted


# ============================================================
# VALIDATION STRICTE
# ============================================================

def variant_is_valid(rs: np.ndarray, metrics: Dict[str, float]) -> bool:
    abs_err = np.abs(rs - TARGET)
    if np.any(abs_err > MAX_ABS_ERROR_PER_COLOR):
        return False
    if float(np.mean(abs_err)) > MAX_MEAN_ABS_ERROR:
        return False

    if not (MIN_BOUNDARY_DENSITY <= float(metrics["boundary_density"]) <= MAX_BOUNDARY_DENSITY):
        return False
    if not (MIN_BOUNDARY_DENSITY_SMALL <= float(metrics["boundary_density_small"]) <= MAX_BOUNDARY_DENSITY_SMALL):
        return False
    if not (MIN_BOUNDARY_DENSITY_TINY <= float(metrics["boundary_density_tiny"]) <= MAX_BOUNDARY_DENSITY_TINY):
        return False

    if float(metrics["mirror_similarity"]) > MAX_MIRROR_SIMILARITY:
        return False
    if float(metrics["largest_olive_component_ratio"]) < MIN_LARGEST_OLIVE_COMPONENT_RATIO:
        return False
    if float(metrics["edge_contact_ratio"]) > MAX_EDGE_CONTACT_RATIO:
        return False
    return True


def validate_candidate_result(candidate: CandidateResult) -> bool:
    return variant_is_valid(candidate.ratios, candidate.metrics)


# ============================================================
# EXPORT / RAPPORT
# ============================================================

def build_unique_camo_name(
    target_index: int,
    seed: int,
    local_attempt: int,
    global_attempt: Optional[int] = None,
    *,
    prefix: str = "camouflage",
    ext: str = "png",
) -> str:
    ext = ext.lstrip(".") or "png"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nano = time.time_ns() % 1_000_000_000
    global_part = f"_g{int(global_attempt):06d}" if global_attempt is not None else ""
    return (
        f"{prefix}_{int(target_index):03d}"
        f"_s{int(seed)}"
        f"_a{int(local_attempt):04d}"
        f"{global_part}_{timestamp}_{nano:09d}.{ext}"
    )


def build_unique_camo_path(
    output_dir: Path,
    target_index: int,
    seed: int,
    local_attempt: int,
    global_attempt: Optional[int] = None,
    *,
    prefix: str = "camouflage",
    ext: str = "png",
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    return output_dir / build_unique_camo_name(
        target_index=target_index,
        seed=seed,
        local_attempt=local_attempt,
        global_attempt=global_attempt,
        prefix=prefix,
        ext=ext,
    )


def _dedupe_output_path(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}__dup{counter:03d}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def save_candidate_image(candidate: CandidateResult, path: Path) -> Path:
    path = _dedupe_output_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate.image.save(path)
    return path


def candidate_row(
    target_index: int,
    local_attempt: int,
    global_attempt: int,
    candidate: CandidateResult,
    image_name: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Dict[str, object]:
    rs = candidate.ratios
    metrics = candidate.metrics
    return {
        "index": target_index,
        "seed": candidate.seed,
        "attempts_for_this_image": local_attempt,
        "global_attempt": global_attempt,
        "coyote_brown_pct": round(float(rs[IDX_COYOTE] * 100), 4),
        "vert_olive_pct": round(float(rs[IDX_OLIVE] * 100), 4),
        "terre_de_france_pct": round(float(rs[IDX_TERRE] * 100), 4),
        "vert_de_gris_pct": round(float(rs[IDX_GRIS] * 100), 4),
        "largest_olive_component_ratio": round(float(metrics["largest_olive_component_ratio"]), 6),
        "boundary_density": round(float(metrics["boundary_density"]), 6),
        "boundary_density_small": round(float(metrics["boundary_density_small"]), 6),
        "boundary_density_tiny": round(float(metrics["boundary_density_tiny"]), 6),
        "mirror_similarity": round(float(metrics["mirror_similarity"]), 6),
        "edge_contact_ratio": round(float(metrics["edge_contact_ratio"]), 6),
        "overscan": round(float(metrics["overscan"]), 6),
        "shift_strength": round(float(metrics["shift_strength"]), 6),
        "width": int(metrics["width"]),
        "height": int(metrics["height"]),
        "physical_width_cm": round(float(metrics["physical_width_cm"]), 3),
        "physical_height_cm": round(float(metrics["physical_height_cm"]), 3),
        "px_per_cm": round(float(metrics["px_per_cm"]), 6),
        "motif_scale": round(float(metrics["motif_scale"]), 6),
        "image_name": image_name or "",
        "image_path": image_path or "",
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


# ============================================================
# GÉNÉRATION ASYNCHRONE / LIVE
# ============================================================

async def _await_attempt(
    fut: asyncio.Future,
    attempt_no: int,
    seed: int,
) -> Tuple[int, int, CandidateResult, bool]:
    candidate, accepted = await fut
    return attempt_no, seed, candidate, accepted


def build_batch(target_index: int, start_attempt: int, batch_size: int, base_seed: int) -> List[Tuple[int, int]]:
    return [(local_attempt, build_seed(target_index, local_attempt, base_seed)) for local_attempt in range(start_attempt, start_attempt + batch_size)]


def console_progress(counters: LiveCounters, current_target: int, workers: int) -> None:
    sys.stdout.write(counters.line(current_target=current_target, workers=workers))
    sys.stdout.flush()


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
    live_console: bool = True,
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
        advice = _supervisor_feedback(
            "generation_started",
            tuning=tuning.__dict__,
            output_dir=str(output_dir),
            target_count=target_count,
        )
        tuning = _merge_supervisor_tuning(tuning, advice, fallback_machine_intensity=machine_intensity)

    counters = LiveCounters(target_count=target_count)
    rows: List[Dict[str, object]] = []
    batch_counter = 0
    loop = asyncio.get_running_loop()

    for target_index in range(1, target_count + 1):
        local_attempt = 1

        while True:
            if stop_requested is not None and await stop_requested():
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

            batch = build_batch(target_index, local_attempt, tuning.attempt_batch_size, base_seed)
            use_parallel = bool(tuning.parallel_attempts and tuning.max_workers > 1 and len(batch) > 1)

            tasks: List[asyncio.Task] = []
            if use_parallel:
                pool = get_process_pool(tuning.max_workers)
                for attempt_no, seed in batch:
                    fut = loop.run_in_executor(pool, generate_and_validate_from_seed, seed)
                    tasks.append(asyncio.create_task(_await_attempt(fut, attempt_no, seed)))
            else:
                attempt_no, seed = batch[0]
                fut = loop.run_in_executor(None, generate_and_validate_from_seed, seed)
                tasks.append(asyncio.create_task(_await_attempt(fut, attempt_no, seed)))

            counters.in_flight = len(tasks)
            if live_console:
                console_progress(counters, current_target=target_index, workers=tuning.max_workers)

            ordered_results: List[Tuple[int, CandidateResult, bool]] = []

            for idx, done in enumerate(asyncio.as_completed(tasks), start=1):
                attempt_no, _seed, candidate, accepted = await done
                counters.attempts += 1
                counters.in_flight = max(0, len(tasks) - idx)

                if accepted:
                    counters.passed_validation += 1
                else:
                    counters.rejected += 1

                ordered_results.append((attempt_no, candidate, accepted))

                if progress_callback is not None:
                    await progress_callback(target_index, attempt_no, counters.attempts, target_count, candidate, accepted)

                if enable_live_supervisor:
                    advice = _supervisor_feedback(
                        "attempt_finished",
                        target_index=target_index,
                        local_attempt=attempt_no,
                        accepted=accepted,
                        total_attempts=counters.attempts,
                        tuning=tuning.__dict__,
                        ratios={COLOR_NAMES[i]: float(candidate.ratios[i]) for i in range(4)},
                        metrics={k: float(v) for k, v in candidate.metrics.items()},
                    )
                    tuning = _merge_supervisor_tuning(tuning, advice)

                if live_console:
                    console_progress(counters, current_target=target_index, workers=tuning.max_workers)

            ordered_results.sort(key=lambda x: x[0])
            accepted_item = next(((a, c) for a, c, ok in ordered_results if ok), None)

            if accepted_item is None:
                local_attempt += max(1, tuning.attempt_batch_size)
                continue

            accepted_attempt, accepted_candidate = accepted_item
            filename = build_unique_camo_path(
                output_dir=output_dir,
                target_index=target_index,
                seed=accepted_candidate.seed,
                local_attempt=accepted_attempt,
                global_attempt=counters.attempts,
            )
            saved_path = save_candidate_image(accepted_candidate, filename)
            rows.append(candidate_row(
                target_index,
                accepted_attempt,
                counters.attempts,
                accepted_candidate,
                image_name=saved_path.name,
                image_path=str(saved_path),
            ))
            counters.accepted += 1

            if live_console:
                console_progress(counters, current_target=target_index, workers=tuning.max_workers)
            break

    write_report(rows, output_dir)
    if live_console:
        sys.stdout.write("\n")
        sys.stdout.flush()

    summary = {
        "mode": "same_type_organic_8k_human_scale_async",
        "target_count": target_count,
        "accepted": counters.accepted,
        "passed_validation": counters.passed_validation,
        "rejected": counters.rejected,
        "attempts": counters.attempts,
        "output_dir": str(Path(output_dir).resolve()),
        "report": str((Path(output_dir) / "rapport_camouflages.csv").resolve()),
        "width": int(WIDTH),
        "height": int(HEIGHT),
        "physical_width_cm": float(PHYSICAL_WIDTH_CM),
        "physical_height_cm": float(PHYSICAL_HEIGHT_CM),
        "px_per_cm": float(PX_PER_CM),
        "motif_scale": float(MOTIF_SCALE),
    }
    (Path(output_dir) / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


# ============================================================
# INTÉGRATION ML / DL GUIDÉE
# ============================================================

def _build_mldl_config_from_args(args: argparse.Namespace) -> Any:
    mldl = importlib.import_module("camouflage_ml_dl_guided")
    if hasattr(mldl, "build_config_from_main_args"):
        return mldl.build_config_from_main_args(args)
    return mldl.MLDLConfig(
        target_count=int(args.target_count),
        warmup_samples=int(args.mldl_warmup_samples),
        candidate_pool_size=int(args.mldl_candidate_pool_size),
        validate_top_k=int(args.mldl_validate_top_k),
        max_attempts_per_target=int(args.mldl_max_attempts_per_target),
        train_epochs=int(args.mldl_train_epochs),
        batch_size=int(args.mldl_batch_size),
        learning_rate=float(args.mldl_learning_rate),
        hidden_dim=int(args.mldl_hidden_dim),
        device=str(args.mldl_device),
        base_seed=int(args.base_seed),
        output_dir=str(args.output_dir),
        alpha_ucb=float(args.mldl_alpha_ucb),
        min_train_size=int(args.mldl_min_train_size),
        retrain_every=int(args.mldl_retrain_every),
        random_seed=int(args.random_seed),
    )


def run_guided_generation_from_main(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    mldl = importlib.import_module("camouflage_ml_dl_guided")
    cfg = _build_mldl_config_from_args(args)
    if hasattr(mldl, "run_guided_generation"):
        return mldl.run_guided_generation(cfg)

    runner = mldl.CamouflageMLDLGenerator(cfg)
    rows = runner.generate()
    return rows, {
        "total_attempts": runner.total_attempts,
        "output_dir": str(Path(cfg.output_dir).resolve()),
        "report": str((Path(cfg.output_dir) / cfg.report_name).resolve()),
    }


# ============================================================
# CLI
# ============================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Générateur organique 8K horizontal, strict, async, avec suivi live")
    parser.add_argument("--target-count", type=int, default=N_VARIANTS_REQUIRED)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--physical-width-cm", type=float, default=DEFAULT_PHYSICAL_WIDTH_CM)
    parser.add_argument("--physical-height-cm", type=float, default=DEFAULT_PHYSICAL_HEIGHT_CM)
    parser.add_argument("--motif-scale", type=float, default=DEFAULT_MOTIF_SCALE)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--attempt-batch-size", type=int, default=None)
    parser.add_argument("--machine-intensity", type=float, default=DEFAULT_MACHINE_INTENSITY)
    parser.add_argument("--disable-parallel-attempts", action="store_true")
    parser.add_argument("--strict-preflight", action="store_true")
    parser.add_argument("--disable-live-supervisor", action="store_true")
    parser.add_argument("--no-live-console", action="store_true")

    # Mode guidé ML / DL
    parser.add_argument("--guided-ml-dl", action="store_true", help="Active la recherche guidée par ML/DL en ligne")
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--mldl-warmup-samples", type=int, default=128)
    parser.add_argument("--mldl-candidate-pool-size", type=int, default=8)
    parser.add_argument("--mldl-validate-top-k", type=int, default=3)
    parser.add_argument("--mldl-max-attempts-per-target", type=int, default=120)
    parser.add_argument("--mldl-train-epochs", type=int, default=24)
    parser.add_argument("--mldl-batch-size", type=int, default=32)
    parser.add_argument("--mldl-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mldl-hidden-dim", type=int, default=128)
    parser.add_argument("--mldl-device", type=str, default="auto")
    parser.add_argument("--mldl-alpha-ucb", type=float, default=1.25)
    parser.add_argument("--mldl-min-train-size", type=int, default=32)
    parser.add_argument("--mldl-retrain-every", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    set_canvas_geometry(
        width=args.width,
        height=args.height,
        physical_width_cm=args.physical_width_cm,
        physical_height_cm=args.physical_height_cm,
        motif_scale=args.motif_scale,
    )

    try:
        if args.guided_ml_dl:
            rows, guided_summary = run_guided_generation_from_main(args)
            csv_path = Path(args.output_dir) / "rapport_camouflages_ml_dl.csv"
            print("Terminé.")
            print("Mode : same_type_organic_8k_human_scale_guided_ml_dl")
            print(f"Résolution : {WIDTH}x{HEIGHT}")
            print(f"Format physique : {PHYSICAL_WIDTH_CM} cm x {PHYSICAL_HEIGHT_CM} cm")
            print(f"Densité : {PX_PER_CM:.3f} px/cm")
            print(f"Motif scale : {MOTIF_SCALE:.3f}")
            print(f"Images validées : {len(rows)}/{args.target_count}")
            print(f"Tentatives totales : {int(guided_summary.get('total_attempts', 0))}")
            print(f"Dossier : {Path(args.output_dir).resolve()}")
            print(f"CSV : {csv_path.resolve()}")
        else:
            rows = asyncio.run(
                async_generate_all(
                    target_count=args.target_count,
                    output_dir=Path(args.output_dir),
                    base_seed=args.base_seed,
                    max_workers=args.max_workers,
                    attempt_batch_size=args.attempt_batch_size,
                    parallel_attempts=not args.disable_parallel_attempts,
                    machine_intensity=args.machine_intensity,
                    strict_preflight=bool(args.strict_preflight),
                    enable_live_supervisor=not args.disable_live_supervisor,
                    live_console=not args.no_live_console,
                )
            )
            csv_path = Path(args.output_dir) / "rapport_camouflages.csv"
            print("Terminé.")
            print("Mode : same_type_organic_8k_human_scale_async")
            print(f"Résolution : {WIDTH}x{HEIGHT}")
            print(f"Format physique : {PHYSICAL_WIDTH_CM} cm x {PHYSICAL_HEIGHT_CM} cm")
            print(f"Densité : {PX_PER_CM:.3f} px/cm")
            print(f"Motif scale : {MOTIF_SCALE:.3f}")
            print(f"Images validées : {len(rows)}/{args.target_count}")
            print(f"Dossier : {Path(args.output_dir).resolve()}")
            print(f"CSV : {csv_path.resolve()}")
            print(f"Workers max dynamiques : {DEFAULT_MAX_WORKERS}")
    finally:
        shutdown_process_pool()


if __name__ == "__main__":
    main()