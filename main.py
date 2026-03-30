# -*- coding: utf-8 -*-
"""
main.py
Camouflage Armée Fédérale Europe — générateur 8K horizontal, organique et
hiérarchisé pour impression grand format à l'échelle humaine.

Important :
- cette version est pensée pour une sortie 8K horizontale ;
- les tailles des motifs sont pilotées en centimètres via une échelle physique ;
- la hiérarchie multi-échelle vise une lecture utile à longue, moyenne et courte
  distance ;
- la génération reste sans artefacts de bord grâce à overscan + crop central.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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

OUTPUT_DIR = Path("camouflages_federale_europe_same_type")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WIDTH = 7680
DEFAULT_HEIGHT = 4320
WIDTH = DEFAULT_WIDTH
HEIGHT = DEFAULT_HEIGHT
DEFAULT_PHYSICAL_WIDTH_CM = 240.0
DEFAULT_PHYSICAL_HEIGHT_CM = 135.0
PHYSICAL_WIDTH_CM = DEFAULT_PHYSICAL_WIDTH_CM
PHYSICAL_HEIGHT_CM = DEFAULT_PHYSICAL_HEIGHT_CM
PX_PER_CM = WIDTH / PHYSICAL_WIDTH_CM
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
DEFAULT_MACHINE_INTENSITY = 0.94
DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES = 1

# Overscan + crop central pour éviter les artefacts de bord.
DEFAULT_OVERSCAN = 1.35

# Validation : volontairement alignée sur le générateur organique.
MAX_ABS_ERROR_PER_COLOR = np.array([0.0015, 0.0015, 0.0015, 0.0015], dtype=float)
MAX_MEAN_ABS_ERROR = 0.0010
MIN_BOUNDARY_DENSITY = 0.015
MAX_BOUNDARY_DENSITY = 0.120
MAX_MIRROR_SIMILARITY = 0.86


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


def set_canvas_geometry(
    width: int,
    height: int,
    physical_width_cm: float = DEFAULT_PHYSICAL_WIDTH_CM,
    physical_height_cm: float = DEFAULT_PHYSICAL_HEIGHT_CM,
) -> None:
    global WIDTH, HEIGHT, PHYSICAL_WIDTH_CM, PHYSICAL_HEIGHT_CM, PX_PER_CM
    width = int(width)
    height = int(height)
    physical_width_cm = float(physical_width_cm)
    physical_height_cm = float(physical_height_cm)
    if width <= 0 or height <= 0:
        raise ValueError("width et height doivent être > 0")
    if physical_width_cm <= 0 or physical_height_cm <= 0:
        raise ValueError("physical_width_cm et physical_height_cm doivent être > 0")
    WIDTH = width
    HEIGHT = height
    PHYSICAL_WIDTH_CM = physical_width_cm
    PHYSICAL_HEIGHT_CM = physical_height_cm
    px_per_cm_x = WIDTH / PHYSICAL_WIDTH_CM
    px_per_cm_y = HEIGHT / PHYSICAL_HEIGHT_CM
    PX_PER_CM = min(px_per_cm_x, px_per_cm_y)


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
    overscan = _clip_float(DEFAULT_OVERSCAN + rng.uniform(-0.06, 0.08), 1.22, 1.55)
    shift_strength = _clip_float(rng.uniform(0.72, 1.18), 0.50, 1.40)
    palette_bias = (
        rng.uniform(-0.010, 0.010),
        rng.uniform(-0.010, 0.010),
        rng.uniform(-0.010, 0.010),
        rng.uniform(-0.010, 0.010),
    )
    return VariantProfile(
        seed=seed,
        overscan=overscan,
        shift_strength=shift_strength,
        palette_bias=palette_bias,
    )


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def compute_ratios(canvas: np.ndarray) -> np.ndarray:
    counts = np.bincount(canvas.ravel(), minlength=4).astype(float)
    return counts / canvas.size


def render_canvas(canvas: np.ndarray) -> Image.Image:
    return Image.fromarray(RGB[canvas], "RGB")


def boundary_density(canvas: np.ndarray) -> float:
    h, w = canvas.shape
    diff = np.zeros((h, w), dtype=bool)
    diff[1:, :] |= (canvas[1:, :] != canvas[:-1, :])
    diff[:-1, :] |= (canvas[:-1, :] != canvas[1:, :])
    diff[:, 1:] |= (canvas[:, 1:] != canvas[:, :-1])
    diff[:, :-1] |= (canvas[:, :-1] != canvas[:, 1:])
    return float(np.mean(diff))


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
    hist = np.bincount(all_edges, minlength=4).astype(float)
    hist /= max(1.0, hist.sum())
    return float(hist.max())


# ============================================================
# GÉNÉRATEUR ORGANIQUE — MÊME TYPE VISUEL
# ============================================================

def shift_reflect(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = arr.shape
    pad_y = abs(dy) + 2
    pad_x = abs(dx) + 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[y0:y0 + h, x0:x0 + w]


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


def cells_for_patch_size(patch_cm_x: float, patch_cm_y: float) -> Tuple[int, int]:
    patch_px_x = max(8.0, float(patch_cm_x) * PX_PER_CM)
    patch_px_y = max(8.0, float(patch_cm_y) * PX_PER_CM)
    cells_x = max(3, int(round(WIDTH / patch_px_x)))
    cells_y = max(3, int(round(HEIGHT / patch_px_y)))
    return cells_x, cells_y


def random_blob_layer(
    width: int,
    height: int,
    rng: np.random.Generator,
    cells_x: int,
    cells_y: int,
    angle_deg: float,
    pad_ratio: float = 0.30,
) -> np.ndarray:
    cells_x = max(3, int(cells_x))
    cells_y = max(3, int(cells_y))

    small = rng.random((cells_y, cells_x), dtype=np.float32)
    small_img = Image.fromarray((small * 255).astype(np.uint8), mode="L")

    pad = int(max(width, height) * pad_ratio)
    work_w = width + 2 * pad
    work_h = height + 2 * pad

    big = small_img.resize((work_w, work_h), resample=Image.Resampling.BICUBIC)
    rot = big.rotate(angle_deg, resample=Image.Resampling.BICUBIC)
    crop = rot.crop((pad, pad, pad + width, pad + height))

    arr = np.asarray(crop, dtype=np.float32) / 255.0
    return arr


def build_field(
    width: int,
    height: int,
    rng: np.random.Generator,
    plan: List[Tuple[int, int, float, float]],
    shift_strength: float,
) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float32)
    total_weight = 0.0

    max_shift_x = max(1, int((width // 18) * shift_strength))
    max_shift_y = max(1, int((height // 18) * shift_strength))

    for cells_x, cells_y, angle, weight in plan:
        layer = random_blob_layer(width, height, rng, cells_x, cells_y, angle)
        shift_x = int(rng.integers(-max_shift_x, max_shift_x + 1))
        shift_y = int(rng.integers(-max_shift_y, max_shift_y + 1))
        layer = shift_reflect(layer, shift_y, shift_x)
        field += layer * weight
        total_weight += weight

    field /= max(total_weight, 1e-6)
    field = (field - field.min()) / max(field.max() - field.min(), 1e-6)
    return field.astype(np.float32)


def build_all_fields(width: int, height: int, profile: VariantProfile) -> np.ndarray:
    rng = np.random.default_rng(profile.seed)

    # Plans en centimètres, pensés pour une lecture multi-distance sur silhouette humaine.
    # - macro : 40–90 cm
    # - intermédiaire : 10–30 cm
    # - micro : 2–8 cm
    c1x, c1y = cells_for_patch_size(82, 64)
    c2x, c2y = cells_for_patch_size(54, 40)
    c3x, c3y = cells_for_patch_size(22, 16)
    c4x, c4y = cells_for_patch_size(7, 5)

    plans = {
        IDX_COYOTE: [
            (c1x, c1y, -18.0, 1.00),
            (c2x, c2y, 22.0, 0.46),
            (c3x, c3y, -8.0, 0.18),
            (c4x, c4y, 0.0, 0.05),
        ],
        IDX_OLIVE: [
            (c1x, c1y, 16.0, 1.00),
            (c2x, c2y, -26.0, 0.48),
            (c3x, c3y, 7.0, 0.18),
            (c4x, c4y, 12.0, 0.05),
        ],
        IDX_TERRE: [
            (c2x, c2y, -12.0, 0.94),
            (c3x, c3y, 26.0, 0.42),
            (c4x, c4y, -4.0, 0.10),
        ],
        IDX_GRIS: [
            (c2x, c2y, 20.0, 0.88),
            (c3x, c3y, -24.0, 0.42),
            (c4x, c4y, 0.0, 0.16),
        ],
    }

    fields = []
    for idx in range(4):
        local_seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(local_seed)
        field = build_field(width, height, local_rng, plans[idx], profile.shift_strength)
        field = field + float(profile.palette_bias[idx])
        fields.append(field.astype(np.float32))

    return np.stack(fields, axis=0)


def boundary_mask(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape
    diff = np.zeros((h, w), dtype=bool)
    diff[1:, :] |= (canvas[1:, :] != canvas[:-1, :])
    diff[:-1, :] |= (canvas[:-1, :] != canvas[1:, :])
    diff[:, 1:] |= (canvas[:, 1:] != canvas[:, :-1])
    diff[:, :-1] |= (canvas[:, :-1] != canvas[:, 1:])
    return diff


def exactify_proportions(labels: np.ndarray, fields: np.ndarray, target_counts: np.ndarray) -> np.ndarray:
    n_classes, height, width = fields.shape
    labels = labels.copy()

    for _ in range(30):
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
            gain = flat_fields[u, idx] - flat_fields[current, idx]
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


def generate_one_variant(profile: VariantProfile) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
    work_width = max(WIDTH + 64, int(round(WIDTH * profile.overscan)))
    work_height = max(HEIGHT + 64, int(round(HEIGHT * profile.overscan)))

    fields = build_all_fields(work_width, work_height, profile)
    fields = center_crop(fields, HEIGHT, WIDTH)

    target_counts = np.rint(TARGET * (WIDTH * HEIGHT)).astype(int)
    target_counts[-1] = (WIDTH * HEIGHT) - int(target_counts[:-1].sum())

    canvas = sequential_assign(fields, target_counts)
    canvas = exactify_proportions(canvas, fields, target_counts)

    rs = compute_ratios(canvas)
    metrics = {
        "largest_olive_component_ratio": largest_component_ratio(canvas == IDX_OLIVE),
        "boundary_density": boundary_density(canvas),
        "mirror_similarity": mirror_similarity_score(canvas),
        "edge_contact_ratio": edge_contact_ratio(canvas),
        "overscan": float(profile.overscan),
        "shift_strength": float(profile.shift_strength),
    }
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

    bd = float(metrics.get("boundary_density", 0.0))
    if bd < MIN_BOUNDARY_DENSITY or bd > MAX_BOUNDARY_DENSITY:
        return False

    if float(metrics.get("mirror_similarity", 1.0)) > MAX_MIRROR_SIMILARITY:
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
        "coyote_brown_pct": round(float(rs[IDX_COYOTE] * 100), 4),
        "vert_olive_pct": round(float(rs[IDX_OLIVE] * 100), 4),
        "terre_de_france_pct": round(float(rs[IDX_TERRE] * 100), 4),
        "vert_de_gris_pct": round(float(rs[IDX_GRIS] * 100), 4),
        "largest_olive_component_ratio": round(float(metrics["largest_olive_component_ratio"]), 6),
        "boundary_density": round(float(metrics["boundary_density"]), 6),
        "mirror_similarity": round(float(metrics["mirror_similarity"]), 6),
        "edge_contact_ratio": round(float(metrics["edge_contact_ratio"]), 6),
        "overscan": round(float(metrics["overscan"]), 6),
        "shift_strength": round(float(metrics["shift_strength"]), 6),
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

    summary = {
        "mode": "same_type_organic_8k_human_scale",
        "target_count": target_count,
        "total_rows": len(rows),
        "total_attempts": total_attempts,
        "output_dir": str(Path(output_dir).resolve()),
        "report": str((Path(output_dir) / "rapport_camouflages.csv").resolve()),
        "width": int(WIDTH),
        "height": int(HEIGHT),
        "physical_width_cm": float(PHYSICAL_WIDTH_CM),
        "physical_height_cm": float(PHYSICAL_HEIGHT_CM),
        "px_per_cm": float(PX_PER_CM),
    }
    (Path(output_dir) / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

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
                    wrapped_tasks.append(asyncio.create_task(_wrap_async_attempt(fut, attempt_no, seed, time.time())))

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
                    continue

                accepted_attempt, accepted_candidate = accepted_result

            else:
                seed = build_seed(target_index, local_attempt, base_seed=base_seed)
                accepted_candidate = await async_generate_candidate_from_seed(seed)
                accepted = await async_validate_candidate_result(accepted_candidate)
                total_attempts += 1

                if progress_callback is not None:
                    await progress_callback(target_index, local_attempt, total_attempts, target_count, accepted_candidate, accepted)

                if not accepted:
                    local_attempt += 1
                    continue
                accepted_attempt = local_attempt

            filename = output_dir / f"camouflage_{target_index:03d}.png"
            await async_save_candidate_image(accepted_candidate, filename)
            rows.append(candidate_row(target_index, accepted_attempt, total_attempts, accepted_candidate))
            break

    await async_write_report(rows, output_dir)
    return rows


# ============================================================
# CLI
# ============================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Générateur organique 8K horizontal, dimensionné pour impression à l'échelle humaine")
    parser.add_argument("--target-count", type=int, default=N_VARIANTS_REQUIRED)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--physical-width-cm", type=float, default=DEFAULT_PHYSICAL_WIDTH_CM)
    parser.add_argument("--physical-height-cm", type=float, default=DEFAULT_PHYSICAL_HEIGHT_CM)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--attempt-batch-size", type=int, default=None)
    parser.add_argument("--machine-intensity", type=float, default=DEFAULT_MACHINE_INTENSITY)
    parser.add_argument("--disable-parallel-attempts", action="store_true")
    parser.add_argument("--strict-preflight", action="store_true")
    parser.add_argument("--disable-live-supervisor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()

    random.seed(12345)
    np.random.seed(12345)

    set_canvas_geometry(
        width=args.width,
        height=args.height,
        physical_width_cm=args.physical_width_cm,
        physical_height_cm=args.physical_height_cm,
    )

    try:
        rows = generate_all(
            target_count=args.target_count,
            output_dir=Path(args.output_dir),
            base_seed=args.base_seed,
            max_workers=args.max_workers,
            attempt_batch_size=args.attempt_batch_size,
            parallel_attempts=not args.disable_parallel_attempts,
            machine_intensity=args.machine_intensity,
            strict_preflight=bool(args.strict_preflight),
            enable_live_supervisor=not args.disable_live_supervisor,
        )
        csv_path = Path(args.output_dir) / "rapport_camouflages.csv"
        print("\nTerminé.")
        print("Mode : same_type_organic_8k_human_scale")
        print(f"Résolution : {WIDTH}x{HEIGHT}")
        print(f"Format physique : {PHYSICAL_WIDTH_CM} cm x {PHYSICAL_HEIGHT_CM} cm")
        print(f"Densité : {PX_PER_CM:.3f} px/cm")
        print(f"Images validées : {len(rows)}/{args.target_count}")
        print(f"Dossier : {Path(args.output_dir).resolve()}")
        print(f"CSV : {csv_path.resolve()}")
        print(f"Workers max dynamiques : {DEFAULT_MAX_WORKERS}")
    finally:
        shutdown_process_pool()


if __name__ == "__main__":
    main()
