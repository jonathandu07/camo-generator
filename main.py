# -*- coding: utf-8 -*-
"""
main.py
Générateur générique de textures multi-classes 8K horizontal, asynchrone,
strict, orienté production, avec suivi temps réel, rejet/acceptation en direct,
best-of obligatoire et export exhaustif des motifs de rejet vers logs / ML / DL.

Version corrigée :
- évite la constitution de pixels orphelins dès le début ;
- sème des macros aléatoires par classe ;
- fait croître les classes uniquement depuis leur frontière ;
- garde un rééquilibrage des proportions sécurisé ;
- conserve les outils de réparation, logs, best-of et orchestration async.

Usage :
    python main.py
    python main.py --target-count 20 --width 3840 --height 2160
"""

from __future__ import annotations

import argparse
import asyncio
import csv
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

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

OUTPUT_DIR = Path("textures_bestof_8k")
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
DEFAULT_BASE_SEED = 202604010001

IDX_0 = 0
IDX_1 = 1
IDX_2 = 2
IDX_3 = 3
IDX_COYOTE = IDX_0
IDX_OLIVE = IDX_1
IDX_TERRE = IDX_2
IDX_GRIS = IDX_3
N_CLASSES = 4

CLASS_NAMES = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
]

RGB = np.array(
    [
        (0x81, 0x61, 0x3C),
        (0x55, 0x54, 0x3F),
        (0x7C, 0x6D, 0x66),
        (0x57, 0x5D, 0x57),
    ],
    dtype=np.uint8,
)

TARGET = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)

CPU_COUNT = max(1, os.cpu_count() or 1)
DEFAULT_MAX_WORKERS = max(1, CPU_COUNT)
DEFAULT_ATTEMPT_BATCH_SIZE = max(1, DEFAULT_MAX_WORKERS)
DEFAULT_MACHINE_INTENSITY = 0.90
DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES = 1
DEFAULT_OVERSCAN = 1.10

MIN_MOTIF_SCALE = 0.08
MAX_MOTIF_SCALE = 1.20
DEFAULT_MOTIF_SCALE = 0.55
MOTIF_SCALE = DEFAULT_MOTIF_SCALE
MIN_PATCH_PX = 2.0

# Validation stricte.
MAX_ABS_ERROR_PER_COLOR = np.array([0.0015, 0.0015, 0.0015, 0.0015], dtype=float)
MAX_MEAN_ABS_ERROR = 0.0010
MIN_BOUNDARY_DENSITY = 0.010
MAX_BOUNDARY_DENSITY = 0.125
MIN_BOUNDARY_DENSITY_SMALL = 0.015
MAX_BOUNDARY_DENSITY_SMALL = 0.165
MIN_BOUNDARY_DENSITY_TINY = 0.020
MAX_BOUNDARY_DENSITY_TINY = 0.210
MAX_MIRROR_SIMILARITY = 0.88
MIN_LARGEST_COMPONENT_RATIO_CLASS_1 = 0.08
MIN_LARGEST_OLIVE_COMPONENT_RATIO = MIN_LARGEST_COMPONENT_RATIO_CLASS_1
MAX_EDGE_CONTACT_RATIO = 0.72

# Nettoyage / fragmentation.
STRAY_CLEANUP_PASSES = 4
ORPHAN_MAX_SAME_NEIGHBORS = 1
ORPHAN_MIN_WINNER_NEIGHBORS = 4
MIN_COMPONENT_PIXELS = (12, 12, 12, 12)
MAX_ORPHAN_RATIO = 0.0
MAX_MICRO_ISLANDS_PER_MP = 0.0

# Best-of obligatoire.
BESTOF_REQUIRED = True
BESTOF_MIN_SCORE = 0.945

# Exports logs / ML / DL.
EVENTS_JSONL = "validation_events.jsonl"
ACCEPTS_JSONL = "ml_accepts.jsonl"
REJECTIONS_JSONL = "ml_rejections.jsonl"
FULL_DATASET_JSONL = "ml_dataset_all_attempts.jsonl"

MAX_REPAIR_ROUNDS = 3

# Nouveau pipeline topologique.
UNASSIGNED_LABEL = np.uint8(255)

SEED_CANDIDATE_SAMPLES = 96
SEED_RETRY_LIMIT = 32

SEEDS_PER_MP = (0.26, 0.15, 0.22, 0.20)
MIN_SEEDS_PER_CLASS = (5, 3, 4, 4)
MAX_SEEDS_PER_CLASS = (18, 10, 14, 12)

SEED_RADIUS_CM_BASE = (0.60, 0.72, 0.56, 0.52)
PRIMARY_SEED_RADIUS_MULTIPLIER = (1.45, 2.20, 1.55, 1.50)

GROWTH_MIN_BATCH = 2048
GROWTH_MAX_BATCH = 262144

SAFE_TRANSFER_MIN_SOURCE_SAME = 2
SAFE_TRANSFER_MIN_DEST_NEIGHBORS = 1
SAFE_REBALANCE_ROUNDS = 24


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
    label_map: np.ndarray
    ratios: np.ndarray
    metrics: Dict[str, float]


@dataclass
class ValidationOutcome:
    accepted: bool
    passed_strict: bool
    bestof_ok: bool
    bestof_score: float
    reasons: List[str] = field(default_factory=list)
    fragmentation: Dict[str, Any] = field(default_factory=dict)
    subscores: Dict[str, float] = field(default_factory=dict)
    repair_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "passed_strict": bool(self.passed_strict),
            "bestof_ok": bool(self.bestof_ok),
            "bestof_score": float(self.bestof_score),
            "reasons": list(self.reasons),
            "fragmentation": self.fragmentation,
            "subscores": self.subscores,
            "repair_trace": list(self.repair_trace),
        }


@dataclass
class RepairPlan:
    seed: int
    overscan: float
    shift_strength: float
    palette_bias: Tuple[float, float, float, float]
    motif_scale: float
    extra_cleanup_passes: int = 0
    merge_micro: bool = False
    rebalance_edges: bool = False
    anti_mirror: bool = False


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
    accepted: int = 0
    passed_validation: int = 0
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


AsyncProgressCallback = Callable[
    [int, int, int, int, CandidateResult, ValidationOutcome],
    Awaitable[None],
]
AsyncStopCallback = Callable[[], Awaitable[bool]]


# ============================================================
# OUTILS SYSTÈME
# ============================================================

_PROCESS_POOL: Optional[ProcessPoolExecutor] = None
_PROCESS_POOL_WORKERS: Optional[int] = None


def _worker_initializer() -> None:
    if str(os.environ.get("TEXTURE_LIMIT_NUMERIC_THREADS", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _clip_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


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
    PX_PER_CM = min(WIDTH / PHYSICAL_WIDTH_CM, HEIGHT / PHYSICAL_HEIGHT_CM)
    MOTIF_SCALE = max(MIN_MOTIF_SCALE, min(MAX_MOTIF_SCALE, motif_scale))


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
        if sample.system_available_mb < 3072:
            baseline_workers = 1
        elif sample.system_available_mb < 6144:
            baseline_workers = min(baseline_workers, max(1, CPU_COUNT // 4))
        elif sample.system_available_mb < 12288:
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
        raise RuntimeError("Espace disque insuffisant pour lancer un lot")
    (Path(output_dir) / "preflight_snapshot.json").write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# PROFILS
# ============================================================

def build_seed(target_index: int, local_attempt: int, base_seed: int = DEFAULT_BASE_SEED) -> int:
    return int(base_seed + target_index * 100000 + local_attempt)


def make_profile(seed: int) -> VariantProfile:
    rng = random.Random(seed)
    overscan = _clip_float(DEFAULT_OVERSCAN + rng.uniform(-0.02, 0.03), 1.08, 1.16)
    shift_strength = _clip_float(rng.uniform(0.66, 0.98), 0.50, 1.10)
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

def compute_ratios(label_map: np.ndarray) -> np.ndarray:
    counts = np.bincount(label_map.ravel(), minlength=N_CLASSES).astype(np.float64)
    return counts / label_map.size


def render_label_map(label_map: np.ndarray) -> Image.Image:
    return Image.fromarray(RGB[label_map], "RGB")


def boundary_mask(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    diff = np.zeros((h, w), dtype=bool)
    diff[1:, :] |= (label_map[1:, :] != label_map[:-1, :])
    diff[:-1, :] |= (label_map[:-1, :] != label_map[1:, :])
    diff[:, 1:] |= (label_map[:, 1:] != label_map[:, :-1])
    diff[:, :-1] |= (label_map[:, :-1] != label_map[:, 1:])
    return diff


def boundary_density(label_map: np.ndarray) -> float:
    return float(np.mean(boundary_mask(label_map)))


def mirror_similarity_score(label_map: np.ndarray) -> float:
    mid = label_map.shape[1] // 2
    left = label_map[:, :mid]
    right = label_map[:, label_map.shape[1] - mid:]
    right_flipped = np.fliplr(right)
    h = min(left.shape[0], right_flipped.shape[0])
    w = min(left.shape[1], right_flipped.shape[1])
    return float(np.mean(left[:h, :w] == right_flipped[:h, :w]))


def largest_component_ratio(mask: np.ndarray) -> float:
    total = int(mask.sum())
    if total == 0:
        return 0.0

    if cv2 is not None:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return float(np.max(areas) / total)

    h, w = mask.shape
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
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            best = max(best, size)
    return float(best / total)


def edge_contact_ratio(label_map: np.ndarray) -> float:
    top = label_map[0, :]
    bottom = label_map[-1, :]
    left = label_map[:, 0]
    right = label_map[:, -1]
    all_edges = np.concatenate([top, bottom, left, right])
    hist = np.bincount(all_edges, minlength=N_CLASSES).astype(np.float64)
    hist /= max(1.0, hist.sum())
    return float(hist.max())


def downsample_nearest(label_map: np.ndarray, factor: int) -> np.ndarray:
    return label_map[::factor, ::factor]


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
    patch_px_x = max(MIN_PATCH_PX, float(patch_cm_x) * PX_PER_CM)
    patch_px_y = max(MIN_PATCH_PX, float(patch_cm_y) * PX_PER_CM)
    cells_x = max(3, int(round(width_px / patch_px_x)))
    cells_y = max(3, int(round(height_px / patch_px_y)))
    return cells_x, cells_y


def scaled_patch_size(patch_cm_x: float, patch_cm_y: float, motif_scale: float) -> Tuple[float, float]:
    motif_scale = max(MIN_MOTIF_SCALE, min(MAX_MOTIF_SCALE, float(motif_scale)))
    return float(patch_cm_x) * motif_scale, float(patch_cm_y) * motif_scale


def _neighbor_count_from_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), mode="constant")
    return (
        padded[0:-2, 0:-2] +
        padded[0:-2, 1:-1] +
        padded[0:-2, 2:] +
        padded[1:-1, 0:-2] +
        padded[1:-1, 2:] +
        padded[2:, 0:-2] +
        padded[2:, 1:-1] +
        padded[2:, 2:]
    ).astype(np.uint8, copy=False)


def _disk_offsets(radius_px: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius_px = max(1, int(radius_px))
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    mask = (yy * yy + xx * xx) <= (radius_px * radius_px)
    dy = yy[mask].astype(np.int32, copy=False)
    dx = xx[mask].astype(np.int32, copy=False)
    dist2 = (dy.astype(np.int64) * dy.astype(np.int64) + dx.astype(np.int64) * dx.astype(np.int64))
    return dy, dx, dist2


def estimate_seed_radius_px(class_idx: int, motif_scale: float) -> int:
    base_cm = float(SEED_RADIUS_CM_BASE[class_idx] if class_idx < len(SEED_RADIUS_CM_BASE) else SEED_RADIUS_CM_BASE[-1])
    ms = _clip_float(float(motif_scale), MIN_MOTIF_SCALE, MAX_MOTIF_SCALE)
    cm = max(0.28, base_cm * (0.65 + 0.35 * ms))
    radius_px = int(round(cm * PX_PER_CM))
    return max(2, radius_px)


def estimate_seed_count(class_idx: int, target_count: int, total_pixels: int, motif_scale: float) -> int:
    megapixels = max(1e-9, total_pixels / 1_000_000.0)
    per_mp = float(SEEDS_PER_MP[class_idx] if class_idx < len(SEEDS_PER_MP) else SEEDS_PER_MP[-1])
    ms = _clip_float(float(motif_scale), MIN_MOTIF_SCALE, MAX_MOTIF_SCALE)
    scale_factor = 1.25 - 0.35 * ms
    raw = int(round(per_mp * megapixels * scale_factor))
    min_n = int(MIN_SEEDS_PER_CLASS[class_idx] if class_idx < len(MIN_SEEDS_PER_CLASS) else MIN_SEEDS_PER_CLASS[-1])
    max_n = int(MAX_SEEDS_PER_CLASS[class_idx] if class_idx < len(MAX_SEEDS_PER_CLASS) else MAX_SEEDS_PER_CLASS[-1])
    return max(min_n, min(max_n, raw))


def pick_seed_center(
    labels: np.ndarray,
    field_for_class: np.ndarray,
    rng: np.random.Generator,
    samples: int = SEED_CANDIDATE_SAMPLES,
) -> Optional[Tuple[int, int]]:
    unassigned = np.flatnonzero(labels.ravel() == UNASSIGNED_LABEL)
    if unassigned.size == 0:
        return None

    take = min(int(samples), int(unassigned.size))
    choice = rng.choice(unassigned, size=take, replace=False)
    scores = field_for_class.ravel()[choice].astype(np.float32, copy=False)
    best_idx = int(choice[int(np.argmax(scores))])
    h, w = labels.shape
    y = best_idx // w
    x = best_idx % w
    return int(y), int(x)


def stamp_seed_macro(
    labels: np.ndarray,
    class_idx: int,
    center_y: int,
    center_x: int,
    radius_px: int,
    remaining_need: int,
    field_for_class: np.ndarray,
) -> int:
    if remaining_need <= 0:
        return 0

    h, w = labels.shape
    dy, dx, dist2 = _disk_offsets(radius_px)

    yy = center_y + dy
    xx = center_x + dx

    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    if not np.any(valid):
        return 0

    yy = yy[valid]
    xx = xx[valid]
    dist2 = dist2[valid]

    flat = yy.astype(np.int64) * int(w) + xx.astype(np.int64)
    flat_labels = labels.ravel()

    avail_mask = flat_labels[flat] == UNASSIGNED_LABEL
    if not np.any(avail_mask):
        return 0

    flat = flat[avail_mask]
    yy = yy[avail_mask]
    xx = xx[avail_mask]
    dist2 = dist2[avail_mask]

    if flat.size == 0:
        return 0

    if flat.size > remaining_need:
        local_scores = field_for_class[yy, xx].astype(np.float32, copy=False)
        order = np.lexsort((-local_scores, dist2))
        flat = flat[order[:remaining_need]]

    flat_labels[flat] = np.uint8(class_idx)
    return int(flat.size)


def frontier_mask_for_class(labels: np.ndarray, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    class_mask = (labels == class_idx)
    support = _neighbor_count_from_mask(class_mask)
    frontier = (labels == UNASSIGNED_LABEL) & (support > 0)
    return frontier, support


def grow_from_frontier(
    labels: np.ndarray,
    class_idx: int,
    need: int,
    field_for_class: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if need <= 0:
        return 0

    frontier, support = frontier_mask_for_class(labels, class_idx)
    frontier_idx = np.flatnonzero(frontier.ravel())
    if frontier_idx.size == 0:
        return 0

    frontier_size = int(frontier_idx.size)
    dynamic_batch = max(
        GROWTH_MIN_BATCH,
        min(
            GROWTH_MAX_BATCH,
            frontier_size,
            int(max(need * 0.12, frontier_size * 0.28)),
        ),
    )
    take = min(int(need), int(dynamic_batch))
    if take <= 0:
        return 0

    flat_scores = field_for_class.ravel()[frontier_idx].astype(np.float32, copy=False)
    flat_support = support.ravel()[frontier_idx].astype(np.float32, copy=False)
    jitter = rng.random(frontier_idx.size, dtype=np.float32) * np.float32(1e-4)
    composite = flat_scores + (flat_support * np.float32(0.035)) + jitter

    if frontier_idx.size > take:
        local = np.argpartition(composite, -take)[-take:]
        chosen = frontier_idx[local]
    else:
        chosen = frontier_idx

    labels.ravel()[chosen] = np.uint8(class_idx)
    return int(chosen.size)


def macro_seeded_assign(
    fields: np.ndarray,
    target_counts: np.ndarray,
    seed: int,
    motif_scale: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _, height, width = fields.shape
    total_pixels = int(height * width)
    labels = np.full((height, width), UNASSIGNED_LABEL, dtype=np.uint8)
    rng = np.random.default_rng(int(seed) ^ 0xA5A5A5A5)

    counts = np.zeros(N_CLASSES, dtype=np.int64)
    seed_counts_used = [0] * N_CLASSES
    seed_radius_used = [estimate_seed_radius_px(c, motif_scale) for c in range(N_CLASSES)]

    # Une macro primaire par classe pour éviter l'éparpillement.
    for c in range(N_CLASSES):
        remaining_need = int(target_counts[c] - counts[c])
        if remaining_need <= 0:
            continue

        center = pick_seed_center(labels, fields[c], rng)
        if center is not None:
            primary_radius = max(
                seed_radius_used[c],
                int(round(seed_radius_used[c] * float(
                    PRIMARY_SEED_RADIUS_MULTIPLIER[c]
                    if c < len(PRIMARY_SEED_RADIUS_MULTIPLIER)
                    else PRIMARY_SEED_RADIUS_MULTIPLIER[-1]
                )))
            )
            placed = stamp_seed_macro(
                labels=labels,
                class_idx=c,
                center_y=center[0],
                center_x=center[1],
                radius_px=primary_radius,
                remaining_need=remaining_need,
                field_for_class=fields[c],
            )
            if placed > 0:
                seed_counts_used[c] += 1
                counts[c] += int(placed)

    # Semis complémentaires.
    for c in range(N_CLASSES):
        wanted = int(target_counts[c])
        radius_px = int(seed_radius_used[c])
        n_seeds = estimate_seed_count(c, wanted, total_pixels, motif_scale)

        for _ in range(max(0, n_seeds - seed_counts_used[c])):
            remaining_need = int(target_counts[c] - counts[c])
            if remaining_need <= 0:
                break

            placed = 0
            for _retry in range(SEED_RETRY_LIMIT):
                center = pick_seed_center(labels, fields[c], rng)
                if center is None:
                    break

                placed = stamp_seed_macro(
                    labels=labels,
                    class_idx=c,
                    center_y=center[0],
                    center_x=center[1],
                    radius_px=radius_px,
                    remaining_need=remaining_need,
                    field_for_class=fields[c],
                )
                if placed > 0:
                    seed_counts_used[c] += 1
                    counts[c] += int(placed)
                    break

    # Croissance depuis la frontière uniquement.
    growth_rounds = 0
    while True:
        remaining = target_counts.astype(np.int64) - counts
        if np.all(remaining <= 0):
            break

        growth_rounds += 1
        progressed = False

        active_classes = [int(c) for c in np.argsort(-remaining) if remaining[c] > 0]
        for c in active_classes:
            need = int(remaining[c])
            if need <= 0:
                continue

            added = grow_from_frontier(
                labels=labels,
                class_idx=c,
                need=need,
                field_for_class=fields[c],
                rng=rng,
            )
            if added > 0:
                counts[c] += int(added)
                progressed = True
                continue

            # Si la classe n'a plus de frontière, on replante une macro.
            radius_px = int(seed_radius_used[c])
            for _retry in range(SEED_RETRY_LIMIT):
                center = pick_seed_center(labels, fields[c], rng)
                if center is None:
                    break

                added = stamp_seed_macro(
                    labels=labels,
                    class_idx=c,
                    center_y=center[0],
                    center_x=center[1],
                    radius_px=radius_px,
                    remaining_need=need,
                    field_for_class=fields[c],
                )
                if added > 0:
                    seed_counts_used[c] += 1
                    counts[c] += int(added)
                    progressed = True
                    break

        if not progressed:
            break

    # Fallback strict : normalement rare.
    remaining = target_counts.astype(np.int64) - counts
    if np.any(remaining > 0):
        unassigned = np.flatnonzero(labels.ravel() == UNASSIGNED_LABEL)
        for c in np.argsort(-remaining):
            need = int(remaining[c])
            if need <= 0 or unassigned.size == 0:
                continue

            scores = fields[c].ravel()[unassigned].astype(np.float32, copy=False)
            take = min(need, int(unassigned.size))
            if take <= 0:
                continue

            if unassigned.size > take:
                chosen_local = np.argpartition(scores, -take)[-take:]
                chosen = unassigned[chosen_local]
                labels.ravel()[chosen] = np.uint8(c)
                counts[c] += int(chosen.size)

                keep = np.ones(unassigned.size, dtype=bool)
                keep[chosen_local] = False
                unassigned = unassigned[keep]
            else:
                chosen = unassigned
                labels.ravel()[chosen] = np.uint8(c)
                counts[c] += int(chosen.size)
                unassigned = unassigned[:0]

    info = {
        "seed_counts_used": [int(x) for x in seed_counts_used],
        "seed_radius_px": [int(x) for x in seed_radius_used],
        "growth_rounds": int(growth_rounds),
        "assigned_pixels": int(np.sum(labels != UNASSIGNED_LABEL)),
    }
    return labels, info


# ============================================================
# GÉNÉRATEUR DE CHAMPS
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

    max_shift_x = max(1, int((width // 24) * shift_strength))
    max_shift_y = max(1, int((height // 24) * shift_strength))

    for cells_x, cells_y, angle, weight in plan:
        layer = random_blob_layer(width, height, rng, cells_x, cells_y, angle)
        shift_x = int(rng.integers(-max_shift_x, max_shift_x + 1))
        shift_y = int(rng.integers(-max_shift_y, max_shift_y + 1))
        layer = shift_reflect(layer, shift_y, shift_x)
        field += layer * np.float16(weight)
        total_weight += np.float16(weight)

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

    s1x, s1y = scaled_patch_size(54.0, 39.0, motif_scale)
    s2x, s2y = scaled_patch_size(30.0, 22.0, motif_scale)
    s3x, s3y = scaled_patch_size(12.0, 8.8, motif_scale)
    s4x, s4y = scaled_patch_size(4.4, 3.2, motif_scale)
    s5x, s5y = scaled_patch_size(2.0, 1.5, motif_scale)
    s6x, s6y = scaled_patch_size(1.0, 0.75, motif_scale)

    c1x, c1y = cells_for_patch_size(s1x, s1y, work_width, work_height)
    c2x, c2y = cells_for_patch_size(s2x, s2y, work_width, work_height)
    c3x, c3y = cells_for_patch_size(s3x, s3y, work_width, work_height)
    c4x, c4y = cells_for_patch_size(s4x, s4y, work_width, work_height)
    c5x, c5y = cells_for_patch_size(s5x, s5y, work_width, work_height)
    c6x, c6y = cells_for_patch_size(s6x, s6y, work_width, work_height)

    plans = {
        IDX_0: [
            (c1x, c1y, -18.0, 1.00),
            (c2x, c2y,  22.0, 0.64),
            (c3x, c3y,  -8.0, 0.32),
            (c4x, c4y,   0.0, 0.14),
            (c5x, c5y, -11.0, 0.07),
            (c6x, c6y,  15.0, 0.03),
        ],
        IDX_1: [
            (c1x, c1y,  16.0, 1.00),
            (c2x, c2y, -26.0, 0.64),
            (c3x, c3y,   7.0, 0.31),
            (c4x, c4y,  12.0, 0.15),
            (c5x, c5y,  24.0, 0.07),
            (c6x, c6y, -18.0, 0.03),
        ],
        IDX_2: [
            (c1x, c1y, -10.0, 0.78),
            (c2x, c2y, -12.0, 0.92),
            (c3x, c3y,  26.0, 0.50),
            (c4x, c4y,  -4.0, 0.19),
            (c5x, c5y,   9.0, 0.08),
            (c6x, c6y, -21.0, 0.03),
        ],
        IDX_3: [
            (c1x, c1y,  18.0, 0.74),
            (c2x, c2y,  20.0, 0.90),
            (c3x, c3y, -24.0, 0.52),
            (c4x, c4y,   0.0, 0.22),
            (c5x, c5y, -15.0, 0.09),
            (c6x, c6y,  11.0, 0.03),
        ],
    }

    fields: List[np.ndarray] = []
    for idx in range(N_CLASSES):
        local_seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(local_seed)
        field_i = build_field(work_width, work_height, local_rng, plans[idx], profile.shift_strength)
        field_i = center_crop(field_i, crop_height, crop_width)
        field_i = field_i + np.float16(profile.palette_bias[idx])
        fields.append(field_i.astype(np.float16, copy=False))

    return np.stack(fields, axis=0)


# ============================================================
# NETTOYAGE DES PIXELS ORPHELINS
# ============================================================

def _neighbors8(labels: np.ndarray) -> List[np.ndarray]:
    padded = np.pad(labels, ((1, 1), (1, 1)), mode="edge")
    return [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:  ],
        padded[1:-1, 0:-2],
        padded[1:-1, 2:  ],
        padded[2:  , 0:-2],
        padded[2:  , 1:-1],
        padded[2:  , 2:  ],
    ]


def same_neighbor_count(label_map: np.ndarray) -> np.ndarray:
    neigh = _neighbors8(label_map)
    out = np.zeros_like(label_map, dtype=np.uint8)
    for n in neigh:
        out += (n == label_map).astype(np.uint8)
    return out


def dominant_neighbor_class(label_map: np.ndarray, class_count: int) -> Tuple[np.ndarray, np.ndarray]:
    neigh = _neighbors8(label_map)
    counts = np.zeros((class_count, label_map.shape[0], label_map.shape[1]), dtype=np.uint8)
    for cls in range(class_count):
        acc = np.zeros_like(label_map, dtype=np.uint8)
        for n in neigh:
            acc += (n == cls).astype(np.uint8)
        counts[cls] = acc
    winner = np.argmax(counts, axis=0).astype(label_map.dtype, copy=False)
    winner_count = np.max(counts, axis=0).astype(np.uint8, copy=False)
    return winner, winner_count


def cleanup_orphan_pixels(
    label_map: np.ndarray,
    *,
    class_count: int,
    passes: int = STRAY_CLEANUP_PASSES,
    orphan_max_same_neighbors: int = ORPHAN_MAX_SAME_NEIGHBORS,
    orphan_min_winner_neighbors: int = ORPHAN_MIN_WINNER_NEIGHBORS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = np.asarray(label_map).copy()
    total_changed = 0
    changed_per_pass: List[int] = []

    for _ in range(max(1, int(passes))):
        same = same_neighbor_count(labels)
        winner, winner_count = dominant_neighbor_class(labels, class_count=class_count)

        orphan_mask = (
            (same <= int(orphan_max_same_neighbors)) &
            (winner != labels) &
            (winner_count >= int(orphan_min_winner_neighbors))
        )

        changed = int(orphan_mask.sum())
        changed_per_pass.append(changed)
        if changed == 0:
            break

        labels[orphan_mask] = winner[orphan_mask]
        total_changed += changed

    info = {
        "orphan_pixels_fixed": int(total_changed),
        "orphan_cleanup_passes": int(len(changed_per_pass)),
        "orphan_changed_per_pass": changed_per_pass,
    }
    return labels, info


# ============================================================
# FRAGMENTATION / MICRO-ÎLOTS
# ============================================================

def _connected_component_areas_cv2(mask: np.ndarray) -> List[int]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return []
    return stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()


def _connected_component_areas_python(mask: np.ndarray) -> List[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    areas: List[int] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue

            stack = [(y, x)]
            seen[y, x] = True
            area = 0

            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))

            areas.append(area)

    return areas


def connected_component_areas(mask: np.ndarray) -> List[int]:
    if cv2 is not None:
        return _connected_component_areas_cv2(mask)
    return _connected_component_areas_python(mask)


def fragmentation_report(
    label_map: np.ndarray,
    *,
    class_count: int,
    min_component_pixels: Sequence[int] = MIN_COMPONENT_PIXELS,
) -> Dict[str, Any]:
    pixel_count = int(label_map.size)
    megapixels = max(1e-9, pixel_count / 1_000_000.0)

    same = same_neighbor_count(label_map)
    orphan_pixels = int(np.sum(same == 0))
    weak_pixels = int(np.sum(same <= 1))

    by_class: Dict[str, Any] = {}
    micro_total = 0
    all_components = 0

    for cls in range(class_count):
        mask = (label_map == cls)
        areas = connected_component_areas(mask)
        all_components += len(areas)
        min_px = int(min_component_pixels[cls] if cls < len(min_component_pixels) else min_component_pixels[-1])
        micro = [a for a in areas if a < min_px]
        micro_total += len(micro)

        by_class[str(cls)] = {
            "components": int(len(areas)),
            "largest_component_px": int(max(areas) if areas else 0),
            "smallest_component_px": int(min(areas) if areas else 0),
            "micro_components": int(len(micro)),
            "micro_threshold_px": int(min_px),
            "component_area_mean": float(np.mean(areas) if areas else 0.0),
        }

    return {
        "pixel_count": pixel_count,
        "megapixels": float(megapixels),
        "orphan_pixels": int(orphan_pixels),
        "orphan_ratio": float(orphan_pixels / max(1, pixel_count)),
        "weak_pixels": int(weak_pixels),
        "weak_ratio": float(weak_pixels / max(1, pixel_count)),
        "micro_components_total": int(micro_total),
        "micro_components_per_mp": float(micro_total / megapixels),
        "component_count_total": int(all_components),
        "by_class": by_class,
    }


# ============================================================
# BEST-OF / VALIDATION
# ============================================================

def compute_bestof_score(
    *,
    ratios: np.ndarray,
    target: np.ndarray,
    metrics: Dict[str, float],
    fragmentation: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    abs_err = np.abs(np.asarray(ratios, dtype=float) - np.asarray(target, dtype=float))
    mean_abs = float(np.mean(abs_err))

    ratio_score = float(np.clip(1.0 - (mean_abs / max(1e-9, float(MAX_MEAN_ABS_ERROR))), 0.0, 1.0))
    per_class_score = float(np.clip(1.0 - float(np.mean(abs_err / np.maximum(MAX_ABS_ERROR_PER_COLOR, 1e-9))), 0.0, 1.0))

    orphan_ratio = float(fragmentation.get("orphan_ratio", 1.0))
    weak_ratio = float(fragmentation.get("weak_ratio", 1.0))
    micro_per_mp = float(fragmentation.get("micro_components_per_mp", 999.0))

    fragmentation_score = 1.0
    fragmentation_score -= min(1.0, orphan_ratio * 1000.0)
    fragmentation_score -= min(0.5, weak_ratio * 50.0)
    fragmentation_score -= min(1.0, micro_per_mp / 10.0)
    fragmentation_score = float(np.clip(fragmentation_score, 0.0, 1.0))

    symmetry_score = float(np.clip(1.0 - float(metrics.get("mirror_similarity", 1.0)), 0.0, 1.0))
    edge_score = float(np.clip(1.0 - float(metrics.get("edge_contact_ratio", 1.0)), 0.0, 1.0))

    subscores = {
        "ratio_score": ratio_score,
        "per_class_score": per_class_score,
        "fragmentation_score": fragmentation_score,
        "symmetry_score": symmetry_score,
        "edge_score": edge_score,
    }

    score = (
        0.30 * ratio_score +
        0.25 * per_class_score +
        0.30 * fragmentation_score +
        0.10 * symmetry_score +
        0.05 * edge_score
    )
    return float(np.clip(score, 0.0, 1.0)), subscores


def validate_with_reasons(candidate: CandidateResult) -> ValidationOutcome:
    reasons: List[str] = []
    ratios = candidate.ratios
    metrics = candidate.metrics
    label_map = candidate.label_map

    abs_err = np.abs(ratios - TARGET)
    mean_abs = float(np.mean(abs_err))

    for i, err in enumerate(abs_err):
        if float(err) > float(MAX_ABS_ERROR_PER_COLOR[i]):
            reasons.append(f"ratio_class_{i}_abs_error")

    if mean_abs > float(MAX_MEAN_ABS_ERROR):
        reasons.append("mean_abs_error")

    if not (MIN_BOUNDARY_DENSITY <= float(metrics["boundary_density"]) <= MAX_BOUNDARY_DENSITY):
        reasons.append("boundary_density")
    if not (MIN_BOUNDARY_DENSITY_SMALL <= float(metrics["boundary_density_small"]) <= MAX_BOUNDARY_DENSITY_SMALL):
        reasons.append("boundary_density_small")
    if not (MIN_BOUNDARY_DENSITY_TINY <= float(metrics["boundary_density_tiny"]) <= MAX_BOUNDARY_DENSITY_TINY):
        reasons.append("boundary_density_tiny")

    if float(metrics["mirror_similarity"]) > MAX_MIRROR_SIMILARITY:
        reasons.append("mirror_similarity")
    if float(metrics["largest_component_ratio_class_1"]) < MIN_LARGEST_COMPONENT_RATIO_CLASS_1:
        reasons.append("largest_component_ratio_class_1")
    if float(metrics["edge_contact_ratio"]) > MAX_EDGE_CONTACT_RATIO:
        reasons.append("edge_contact_ratio")

    frag = fragmentation_report(
        label_map,
        class_count=N_CLASSES,
        min_component_pixels=MIN_COMPONENT_PIXELS,
    )

    if float(frag["orphan_ratio"]) > MAX_ORPHAN_RATIO:
        reasons.append("orphan_pixels")
    if float(frag["micro_components_per_mp"]) > MAX_MICRO_ISLANDS_PER_MP:
        reasons.append("micro_islands")

    strict_ok = len(reasons) == 0

    bestof_score, subscores = compute_bestof_score(
        ratios=ratios,
        target=TARGET,
        metrics=metrics,
        fragmentation=frag,
    )
    bestof_ok = bool(bestof_score >= BESTOF_MIN_SCORE and strict_ok)

    if BESTOF_REQUIRED and not bestof_ok:
        reasons.append("not_bestof")

    accepted = bool(strict_ok and (bestof_ok if BESTOF_REQUIRED else True))

    return ValidationOutcome(
        accepted=accepted,
        passed_strict=bool(strict_ok),
        bestof_ok=bool(bestof_ok),
        bestof_score=float(bestof_score),
        reasons=reasons,
        fragmentation=frag,
        subscores=subscores,
    )


def validate_candidate_result(candidate: CandidateResult) -> bool:
    return bool(validate_with_reasons(candidate).accepted)


def _stable_reason_salt(reasons: Sequence[str]) -> int:
    return int(sum(sum(ord(ch) for ch in str(r)) for r in reasons))


def validation_rank(outcome: ValidationOutcome) -> Tuple[int, int, float, int]:
    return (
        1 if outcome.accepted else 0,
        1 if outcome.passed_strict else 0,
        float(outcome.bestof_score),
        -len(outcome.reasons),
    )


def _clip_palette_bias(bias: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(list(bias), dtype=np.float32)
    arr = np.clip(arr, -0.025, 0.025)
    return tuple(float(x) for x in arr.tolist())


def derive_repair_plan(
    candidate: CandidateResult,
    outcome: ValidationOutcome,
    repair_round: int,
) -> RepairPlan:
    reasons = set(str(r) for r in outcome.reasons)
    metrics = dict(candidate.metrics)

    overscan = float(candidate.profile.overscan)
    shift_strength = float(candidate.profile.shift_strength)
    motif_scale = float(metrics.get("motif_scale", MOTIF_SCALE))
    bias = np.asarray(candidate.profile.palette_bias, dtype=np.float32).copy()

    extra_cleanup_passes = 0
    merge_micro = False
    rebalance_edges = False
    anti_mirror = False

    bd = float(metrics.get("boundary_density", 0.0))
    bd_small = float(metrics.get("boundary_density_small", 0.0))
    bd_tiny = float(metrics.get("boundary_density_tiny", 0.0))

    if "boundary_density" in reasons:
        if bd < MIN_BOUNDARY_DENSITY:
            motif_scale *= 0.88
            shift_strength += 0.08
        elif bd > MAX_BOUNDARY_DENSITY:
            motif_scale *= 1.12
            shift_strength -= 0.08

    if "boundary_density_small" in reasons or "boundary_density_tiny" in reasons:
        if bd_small < MIN_BOUNDARY_DENSITY_SMALL or bd_tiny < MIN_BOUNDARY_DENSITY_TINY:
            motif_scale *= 0.84
            shift_strength += 0.10
        if bd_small > MAX_BOUNDARY_DENSITY_SMALL or bd_tiny > MAX_BOUNDARY_DENSITY_TINY:
            motif_scale *= 1.16
            shift_strength -= 0.10

    if "mirror_similarity" in reasons:
        anti_mirror = True
        overscan = _clip_float(overscan + 0.02, 1.08, 1.20)

    if "edge_contact_ratio" in reasons:
        rebalance_edges = True
        overscan = _clip_float(overscan + 0.01, 1.08, 1.20)

    if "largest_component_ratio_class_1" in reasons:
        bias[IDX_1] += np.float32(0.0045)
        shift_strength = _clip_float(shift_strength - 0.06, 0.50, 1.10)
        motif_scale *= 1.08

    if "orphan_pixels" in reasons or "micro_islands" in reasons:
        extra_cleanup_passes += 2
        merge_micro = True
        motif_scale *= 1.05

    if any(r.startswith("ratio_class_") for r in reasons) or "mean_abs_error" in reasons:
        bias += ((TARGET - candidate.ratios) * 0.25).astype(np.float32)

    reason_salt = _stable_reason_salt(sorted(reasons))
    new_seed = int((candidate.seed ^ (reason_salt + repair_round * 104729)) & 0x7FFFFFFF)

    return RepairPlan(
        seed=new_seed,
        overscan=_clip_float(overscan, 1.08, 1.20),
        shift_strength=_clip_float(shift_strength, 0.50, 1.10),
        palette_bias=_clip_palette_bias(bias),
        motif_scale=_clip_float(motif_scale, MIN_MOTIF_SCALE, MAX_MOTIF_SCALE),
        extra_cleanup_passes=int(extra_cleanup_passes),
        merge_micro=bool(merge_micro),
        rebalance_edges=bool(rebalance_edges),
        anti_mirror=bool(anti_mirror),
    )


def majority_smoothing_pass(
    label_map: np.ndarray,
    *,
    class_count: int,
    same_threshold: int = 2,
    winner_threshold: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    same = same_neighbor_count(label_map)
    winner, winner_count = dominant_neighbor_class(label_map, class_count=class_count)

    mask = (
        (same <= int(same_threshold))
        & (winner != label_map)
        & (winner_count >= int(winner_threshold))
    )

    out = label_map.copy()
    changed = int(mask.sum())
    if changed:
        out[mask] = winner[mask]

    return out, {
        "majority_smoothed_pixels": int(changed),
        "same_threshold": int(same_threshold),
        "winner_threshold": int(winner_threshold),
    }


def rebalance_edge_pixels(
    label_map: np.ndarray,
    fields: np.ndarray,
    *,
    edge_margin: int = 10,
    fraction: float = 0.12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = label_map.copy()
    m = max(1, int(edge_margin))

    border = np.zeros_like(labels, dtype=bool)
    border[:m, :] = True
    border[-m:, :] = True
    border[:, :m] = True
    border[:, -m:] = True

    edge_values = labels[border]
    hist = np.bincount(edge_values.ravel(), minlength=N_CLASSES)
    dominant_edge_class = int(np.argmax(hist))

    mask = border & (labels == dominant_edge_class)
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return labels, {
            "edge_pixels_rebalanced": 0,
            "dominant_edge_class": dominant_edge_class,
        }

    take = max(1, int(idx.size * float(fraction)))
    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)
    flat_labels = labels.ravel()

    scores = flat_fields[:, idx].copy()
    current = flat_labels[idx]
    scores[current, np.arange(idx.size)] = -1e9
    alt = np.argmax(scores, axis=0).astype(np.uint8, copy=False)
    gains = scores[alt, np.arange(idx.size)] - flat_fields[current, idx]
    order = np.argsort(gains)[::-1]
    chosen = idx[order[:take]]
    chosen_alt = alt[order[:take]]

    flat_labels[chosen] = chosen_alt
    return flat_labels.reshape(labels.shape), {
        "edge_pixels_rebalanced": int(chosen.size),
        "dominant_edge_class": dominant_edge_class,
    }


def break_mirror_pattern(
    label_map: np.ndarray,
    fields: np.ndarray,
    seed: int,
    *,
    fraction: float = 0.015,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = label_map.copy()
    _, w = labels.shape

    half_mask = np.zeros_like(labels, dtype=bool)
    half_mask[:, w // 2:] = True

    candidate_mask = boundary_mask(labels) & half_mask
    idx = np.flatnonzero(candidate_mask.ravel())
    if idx.size == 0:
        return labels, {"anti_mirror_pixels": 0}

    rng = np.random.default_rng(int(seed) ^ 0x9E3779B1)
    take = max(1, int(idx.size * float(fraction)))
    chosen = rng.choice(idx, size=min(take, idx.size), replace=False)

    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)
    flat_labels = labels.ravel()
    current = flat_labels[chosen]

    scores = flat_fields[:, chosen].copy()
    scores[current, np.arange(chosen.size)] = -1e9
    alt = np.argmax(scores, axis=0).astype(np.uint8, copy=False)

    flat_labels[chosen] = alt
    return flat_labels.reshape(labels.shape), {
        "anti_mirror_pixels": int(chosen.size),
    }


def safe_target_rebalance(
    labels: np.ndarray,
    fields: np.ndarray,
    target_counts: np.ndarray,
    max_rounds: int = SAFE_REBALANCE_ROUNDS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = labels.copy()
    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)
    total_moved = 0

    for _ in range(max(1, int(max_rounds))):
        flat_labels = labels.ravel()
        counts = np.bincount(flat_labels, minlength=N_CLASSES).astype(np.int64)
        delta = target_counts.astype(np.int64) - counts
        if np.all(delta == 0):
            break

        boundary = boundary_mask(labels).ravel()
        same = same_neighbor_count(labels).ravel()
        winner, winner_count = dominant_neighbor_class(labels, class_count=N_CLASSES)
        winner = winner.ravel()
        winner_count = winner_count.ravel()

        moved = False
        under = [int(c) for c in np.where(delta > 0)[0]]
        over = [int(c) for c in np.where(delta < 0)[0]]

        for dst in under:
            need = int(delta[dst])
            if need <= 0:
                continue

            for src in over:
                excess = int(-delta[src])
                if excess <= 0:
                    continue

                take = min(need, excess)
                candidate_idx = np.where(
                    (flat_labels == src) &
                    boundary &
                    (winner == dst) &
                    (winner_count >= SAFE_TRANSFER_MIN_DEST_NEIGHBORS) &
                    (same >= SAFE_TRANSFER_MIN_SOURCE_SAME)
                )[0]

                if candidate_idx.size == 0:
                    continue

                gains = flat_fields[dst, candidate_idx] - flat_fields[src, candidate_idx]
                if candidate_idx.size > take:
                    best_local = np.argpartition(gains, -take)[-take:]
                    picked = candidate_idx[best_local]
                else:
                    picked = candidate_idx

                if picked.size == 0:
                    continue

                flat_labels[picked] = np.uint8(dst)
                total_moved += int(picked.size)
                need -= int(picked.size)
                delta[dst] -= int(picked.size)
                delta[src] += int(picked.size)
                moved = True

                if need <= 0:
                    break

        if not moved:
            break

        labels = flat_labels.reshape(labels.shape)

    return labels.astype(np.uint8, copy=False), {
        "safe_rebalanced_pixels": int(total_moved),
    }


# ============================================================
# LOGS / DATASET ML-DL
# ============================================================

def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def emit_validation_payload(
    *,
    output_dir: Path,
    target_index: int,
    local_attempt: int,
    global_attempt: int,
    candidate: CandidateResult,
    outcome: ValidationOutcome,
) -> Dict[str, Any]:
    payload = {
        "ts": time.time(),
        "target_index": int(target_index),
        "local_attempt": int(local_attempt),
        "global_attempt": int(global_attempt),
        "seed": int(candidate.seed),
        "accepted": bool(outcome.accepted),
        "passed_strict": bool(outcome.passed_strict),
        "bestof_ok": bool(outcome.bestof_ok),
        "bestof_score": float(outcome.bestof_score),
        "reasons": list(outcome.reasons),
        "repair_trace": list(outcome.repair_trace),
        "ratios": [float(x) for x in np.asarray(candidate.ratios, dtype=float)],
        "metrics": {str(k): float(v) for k, v in candidate.metrics.items()},
        "fragmentation": outcome.fragmentation,
        "subscores": outcome.subscores,
        "profile": {
            "seed": int(candidate.profile.seed),
            "overscan": float(candidate.profile.overscan),
            "shift_strength": float(candidate.profile.shift_strength),
            "palette_bias": [float(x) for x in candidate.profile.palette_bias],
        },
    }

    out = Path(output_dir)
    _append_jsonl(out / EVENTS_JSONL, payload)
    _append_jsonl(out / FULL_DATASET_JSONL, payload)
    if outcome.accepted:
        _append_jsonl(out / ACCEPTS_JSONL, payload)
    else:
        _append_jsonl(out / REJECTIONS_JSONL, payload)
    return payload


# ============================================================
# GÉNÉRATION D'UNE VARIANTE
# ============================================================

def generate_one_variant(
    profile: VariantProfile,
    *,
    motif_scale_override: Optional[float] = None,
    extra_cleanup_passes: int = 0,
    merge_micro: bool = False,
    rebalance_edges: bool = False,
    anti_mirror: bool = False,
) -> CandidateResult:
    width = int(WIDTH)
    height = int(HEIGHT)
    physical_width_cm = float(PHYSICAL_WIDTH_CM)
    physical_height_cm = float(PHYSICAL_HEIGHT_CM)
    px_per_cm = float(PX_PER_CM)
    motif_scale = float(motif_scale_override if motif_scale_override is not None else MOTIF_SCALE)

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

    target_counts = np.rint(TARGET * (width * height)).astype(np.int64)
    target_counts[-1] = (width * height) - int(target_counts[:-1].sum())

    label_map, seed_growth_info = macro_seeded_assign(
        fields=fields,
        target_counts=target_counts,
        seed=profile.seed,
        motif_scale=motif_scale,
    )

    label_map, rebalance_info_0 = safe_target_rebalance(
        label_map,
        fields,
        target_counts,
        max_rounds=SAFE_REBALANCE_ROUNDS,
    )

    label_map, orphan_info = cleanup_orphan_pixels(
        label_map,
        class_count=N_CLASSES,
        passes=max(1, 1 + int(extra_cleanup_passes)),
        orphan_max_same_neighbors=ORPHAN_MAX_SAME_NEIGHBORS,
        orphan_min_winner_neighbors=ORPHAN_MIN_WINNER_NEIGHBORS,
    )

    label_map, rebalance_info_1 = safe_target_rebalance(
        label_map,
        fields,
        target_counts,
        max_rounds=max(4, SAFE_REBALANCE_ROUNDS // 2),
    )

    smoothing_info = {"majority_smoothed_pixels": 0}
    if merge_micro:
        label_map, info_a = majority_smoothing_pass(
            label_map,
            class_count=N_CLASSES,
            same_threshold=2,
            winner_threshold=5,
        )
        label_map, info_b = majority_smoothing_pass(
            label_map,
            class_count=N_CLASSES,
            same_threshold=3,
            winner_threshold=5,
        )
        smoothing_info = {
            "majority_smoothed_pixels": int(info_a["majority_smoothed_pixels"] + info_b["majority_smoothed_pixels"])
        }

    edge_info = {"edge_pixels_rebalanced": 0, "dominant_edge_class": -1}
    if rebalance_edges:
        label_map, edge_info = rebalance_edge_pixels(label_map, fields)

    mirror_info = {"anti_mirror_pixels": 0}
    if anti_mirror:
        label_map, mirror_info = break_mirror_pattern(label_map, fields, profile.seed)

    label_map, orphan_info_2 = cleanup_orphan_pixels(
        label_map,
        class_count=N_CLASSES,
        passes=1,
        orphan_max_same_neighbors=ORPHAN_MAX_SAME_NEIGHBORS,
        orphan_min_winner_neighbors=ORPHAN_MIN_WINNER_NEIGHBORS,
    )

    label_map, rebalance_info_2 = safe_target_rebalance(
        label_map,
        fields,
        target_counts,
        max_rounds=max(4, SAFE_REBALANCE_ROUNDS // 2),
    )

    residual_unassigned = np.flatnonzero(label_map.ravel() == UNASSIGNED_LABEL)
    if residual_unassigned.size:
        flat_fields = fields.reshape(N_CLASSES, -1).astype(np.float32, copy=False)
        best_cls = np.argmax(flat_fields[:, residual_unassigned], axis=0).astype(np.uint8, copy=False)
        label_map.ravel()[residual_unassigned] = best_cls
        label_map, rebalance_info_3 = safe_target_rebalance(
            label_map,
            fields,
            target_counts,
            max_rounds=max(4, SAFE_REBALANCE_ROUNDS // 2),
        )
    else:
        rebalance_info_3 = {"safe_rebalanced_pixels": 0}

    ratios = compute_ratios(label_map)
    small = downsample_nearest(label_map, 4)
    tiny = downsample_nearest(label_map, 8)

    largest_ratio = largest_component_ratio(label_map == IDX_1)

    metrics = {
        "largest_component_ratio_class_1": largest_ratio,
        "largest_olive_component_ratio": largest_ratio,
        "boundary_density": boundary_density(label_map),
        "boundary_density_small": boundary_density(small),
        "boundary_density_tiny": boundary_density(tiny),
        "mirror_similarity": mirror_similarity_score(label_map),
        "edge_contact_ratio": edge_contact_ratio(label_map),
        "overscan": float(profile.overscan),
        "shift_strength": float(profile.shift_strength),
        "width": float(width),
        "height": float(height),
        "physical_width_cm": float(physical_width_cm),
        "physical_height_cm": float(physical_height_cm),
        "px_per_cm": float(px_per_cm),
        "motif_scale": float(motif_scale),
        "orphan_pixels_fixed": float(orphan_info["orphan_pixels_fixed"] + orphan_info_2["orphan_pixels_fixed"]),
        "orphan_cleanup_passes": float(orphan_info["orphan_cleanup_passes"] + orphan_info_2["orphan_cleanup_passes"]),
        "majority_smoothed_pixels": float(smoothing_info["majority_smoothed_pixels"]),
        "edge_pixels_rebalanced": float(edge_info["edge_pixels_rebalanced"]),
        "anti_mirror_pixels": float(mirror_info["anti_mirror_pixels"]),
        "repair_merge_micro": float(1 if merge_micro else 0),
        "repair_rebalance_edges": float(1 if rebalance_edges else 0),
        "repair_anti_mirror": float(1 if anti_mirror else 0),
        "repair_extra_cleanup_passes": float(extra_cleanup_passes),

        "seed_macros_total": float(sum(seed_growth_info["seed_counts_used"])),
        "seed_macros_class_0": float(seed_growth_info["seed_counts_used"][0]),
        "seed_macros_class_1": float(seed_growth_info["seed_counts_used"][1]),
        "seed_macros_class_2": float(seed_growth_info["seed_counts_used"][2]),
        "seed_macros_class_3": float(seed_growth_info["seed_counts_used"][3]),
        "seed_radius_px_class_0": float(seed_growth_info["seed_radius_px"][0]),
        "seed_radius_px_class_1": float(seed_growth_info["seed_radius_px"][1]),
        "seed_radius_px_class_2": float(seed_growth_info["seed_radius_px"][2]),
        "seed_radius_px_class_3": float(seed_growth_info["seed_radius_px"][3]),
        "growth_rounds": float(seed_growth_info["growth_rounds"]),
        "safe_rebalanced_pixels": float(
            rebalance_info_0["safe_rebalanced_pixels"] +
            rebalance_info_1["safe_rebalanced_pixels"] +
            rebalance_info_2["safe_rebalanced_pixels"] +
            rebalance_info_3["safe_rebalanced_pixels"]
        ),
    }

    return CandidateResult(
        seed=profile.seed,
        profile=profile,
        image=render_label_map(label_map),
        label_map=label_map,
        ratios=ratios,
        metrics=metrics,
    )


def generate_candidate_from_seed(seed: int) -> CandidateResult:
    profile = make_profile(seed)
    return generate_one_variant(profile)


def generate_and_validate_from_seed(
    seed: int,
    max_repair_rounds: int = MAX_REPAIR_ROUNDS,
) -> Tuple[CandidateResult, ValidationOutcome]:
    candidate = generate_candidate_from_seed(seed)
    outcome = validate_with_reasons(candidate)

    trace: List[Dict[str, Any]] = [{
        "round": 0,
        "seed": int(candidate.seed),
        "accepted": bool(outcome.accepted),
        "bestof_score": float(outcome.bestof_score),
        "reasons": list(outcome.reasons),
        "metrics": {k: float(v) for k, v in candidate.metrics.items()},
    }]

    best_candidate = candidate
    best_outcome = outcome

    for repair_round in range(1, max_repair_rounds + 1):
        if best_outcome.accepted:
            break

        plan = derive_repair_plan(best_candidate, best_outcome, repair_round)

        repaired_profile = VariantProfile(
            seed=int(plan.seed),
            overscan=float(plan.overscan),
            shift_strength=float(plan.shift_strength),
            palette_bias=tuple(plan.palette_bias),
        )

        repaired_candidate = generate_one_variant(
            repaired_profile,
            motif_scale_override=plan.motif_scale,
            extra_cleanup_passes=plan.extra_cleanup_passes,
            merge_micro=plan.merge_micro,
            rebalance_edges=plan.rebalance_edges,
            anti_mirror=plan.anti_mirror,
        )
        repaired_candidate.metrics["repair_round"] = float(repair_round)

        repaired_outcome = validate_with_reasons(repaired_candidate)
        trace.append({
            "round": int(repair_round),
            "seed": int(repaired_candidate.seed),
            "accepted": bool(repaired_outcome.accepted),
            "bestof_score": float(repaired_outcome.bestof_score),
            "reasons": list(repaired_outcome.reasons),
            "metrics": {k: float(v) for k, v in repaired_candidate.metrics.items()},
            "applied_plan": {
                "motif_scale": float(plan.motif_scale),
                "overscan": float(plan.overscan),
                "shift_strength": float(plan.shift_strength),
                "palette_bias": [float(x) for x in plan.palette_bias],
                "extra_cleanup_passes": int(plan.extra_cleanup_passes),
                "merge_micro": bool(plan.merge_micro),
                "rebalance_edges": bool(plan.rebalance_edges),
                "anti_mirror": bool(plan.anti_mirror),
            },
        })

        if validation_rank(repaired_outcome) > validation_rank(best_outcome):
            best_candidate = repaired_candidate
            best_outcome = repaired_outcome

    best_outcome.repair_trace = trace
    return best_candidate, best_outcome


# ============================================================
# EXPORT / RAPPORT
# ============================================================

def build_unique_pattern_name(
    target_index: int,
    seed: int,
    local_attempt: int,
    global_attempt: Optional[int] = None,
    *,
    prefix: str = "pattern",
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


def build_unique_pattern_path(
    output_dir: Path,
    target_index: int,
    seed: int,
    local_attempt: int,
    global_attempt: Optional[int] = None,
    *,
    prefix: str = "pattern",
    ext: str = "png",
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    return output_dir / build_unique_pattern_name(
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
    outcome: Optional[ValidationOutcome] = None,
    image_name: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Dict[str, object]:
    if outcome is None:
        outcome = validate_with_reasons(candidate)

    rs = candidate.ratios
    metrics = candidate.metrics
    return {
        "index": target_index,
        "seed": candidate.seed,
        "attempts_for_this_image": local_attempt,
        "global_attempt": global_attempt,
        "class_0_pct": round(float(rs[IDX_0] * 100), 4),
        "class_1_pct": round(float(rs[IDX_1] * 100), 4),
        "class_2_pct": round(float(rs[IDX_2] * 100), 4),
        "class_3_pct": round(float(rs[IDX_3] * 100), 4),
        "largest_component_ratio_class_1": round(float(metrics["largest_component_ratio_class_1"]), 6),
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
        "seed_macros_total": round(float(metrics.get("seed_macros_total", 0.0)), 3),
        "growth_rounds": round(float(metrics.get("growth_rounds", 0.0)), 3),
        "safe_rebalanced_pixels": round(float(metrics.get("safe_rebalanced_pixels", 0.0)), 3),
        "bestof_score": round(float(outcome.bestof_score), 6),
        "accepted": int(outcome.accepted),
        "reasons": "|".join(outcome.reasons),
        "image_name": image_name or "",
        "image_path": image_path or "",
    }


def write_report(rows: List[Dict[str, object]], output_dir: Path, filename: str = "rapport_textures.csv") -> Path:
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
# ASYNC / LIVE
# ============================================================

async def _await_attempt(
    fut: asyncio.Future,
    attempt_no: int,
    seed: int,
) -> Tuple[int, int, CandidateResult, ValidationOutcome]:
    candidate, outcome = await fut
    return attempt_no, seed, candidate, outcome


def build_batch(target_index: int, start_attempt: int, batch_size: int, base_seed: int) -> List[Tuple[int, int]]:
    return [
        (local_attempt, build_seed(target_index, local_attempt, base_seed))
        for local_attempt in range(start_attempt, start_attempt + batch_size)
    ]


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
    resource_sample_every_batches: int = DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES,
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

    resource_sample_every_batches = max(1, int(resource_sample_every_batches))
    tuning = compute_runtime_tuning(
        max_workers=max_workers,
        attempt_batch_size=attempt_batch_size,
        parallel_attempts=parallel_attempts,
        machine_intensity=machine_intensity,
        sample=sample_process_resources(machine_intensity=machine_intensity, output_dir=output_dir),
    )

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
                if snapshot.system_available_mb > 0 and snapshot.system_available_mb < 4096:
                    tuning = RuntimeTuning(
                        max_workers=1,
                        attempt_batch_size=1,
                        parallel_attempts=False,
                        machine_intensity=min(tuning.machine_intensity, 0.50),
                        reason="memory_pressure",
                    ).normalized()

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

            ordered_results: List[Tuple[int, CandidateResult, ValidationOutcome]] = []

            for idx, done in enumerate(asyncio.as_completed(tasks), start=1):
                attempt_no, _seed, candidate, outcome = await done
                counters.attempts += 1
                counters.in_flight = max(0, len(tasks) - idx)

                emit_validation_payload(
                    output_dir=output_dir,
                    target_index=target_index,
                    local_attempt=attempt_no,
                    global_attempt=counters.attempts,
                    candidate=candidate,
                    outcome=outcome,
                )

                if outcome.accepted:
                    counters.passed_validation += 1
                else:
                    counters.rejected += 1

                ordered_results.append((attempt_no, candidate, outcome))

                if progress_callback is not None:
                    await progress_callback(target_index, attempt_no, counters.attempts, target_count, candidate, outcome)

                if live_console:
                    console_progress(counters, current_target=target_index, workers=tuning.max_workers)

            ordered_results.sort(key=lambda x: x[0])
            accepted_item = next(((a, c, o) for a, c, o in ordered_results if o.accepted), None)

            if accepted_item is None:
                local_attempt += max(1, tuning.attempt_batch_size)
                continue

            accepted_attempt, accepted_candidate, accepted_outcome = accepted_item
            filename = build_unique_pattern_path(
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
                accepted_outcome,
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
        "mode": "generic_multiclass_texture_async_bestof",
        "target_count": target_count,
        "accepted": counters.accepted,
        "passed_validation": counters.passed_validation,
        "rejected": counters.rejected,
        "attempts": counters.attempts,
        "output_dir": str(Path(output_dir).resolve()),
        "report": str((Path(output_dir) / "rapport_textures.csv").resolve()),
        "width": int(WIDTH),
        "height": int(HEIGHT),
        "physical_width_cm": float(PHYSICAL_WIDTH_CM),
        "physical_height_cm": float(PHYSICAL_HEIGHT_CM),
        "px_per_cm": float(PX_PER_CM),
        "motif_scale": float(MOTIF_SCALE),
        "bestof_required": bool(BESTOF_REQUIRED),
        "bestof_min_score": float(BESTOF_MIN_SCORE),
    }
    (Path(output_dir) / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


# ============================================================
# CLI
# ============================================================

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Générateur générique de textures multi-classes 8K, strict, async, best-of obligatoire"
    )
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
    parser.add_argument("--no-live-console", action="store_true")
    parser.add_argument("--random-seed", type=int, default=12345)
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
        rows = asyncio.run(
            async_generate_all(
                target_count=args.target_count,
                output_dir=Path(args.output_dir),
                base_seed=args.base_seed,
                max_workers=args.max_workers,
                attempt_batch_size=args.attempt_batch_size,
                parallel_attempts=not args.disable_parallel_attempts,
                machine_intensity=args.machine_intensity,
                live_console=not args.no_live_console,
            )
        )
        csv_path = Path(args.output_dir) / "rapport_textures.csv"
        print("Terminé.")
        print("Mode : generic_multiclass_texture_async_bestof")
        print(f"Résolution : {WIDTH}x{HEIGHT}")
        print(f"Format physique : {PHYSICAL_WIDTH_CM} cm x {PHYSICAL_HEIGHT_CM} cm")
        print(f"Densité : {PX_PER_CM:.3f} px/cm")
        print(f"Motif scale : {MOTIF_SCALE:.3f}")
        print(f"Images acceptées : {len(rows)}/{args.target_count}")
        print(f"Dossier : {Path(args.output_dir).resolve()}")
        print(f"CSV : {csv_path.resolve()}")
    finally:
        shutdown_process_pool()


if __name__ == "__main__":
    main()