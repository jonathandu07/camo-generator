
# -*- coding: utf-8 -*-
"""
main.py
Générateur générique de textures multi-classes 8K horizontal, asynchrone,
strict, orienté production, avec suivi temps réel, rejet/acceptation en direct,
best-of obligatoire et export exhaustif des motifs de rejet vers logs / ML / DL.

Version corrigée :
- prévention des pixels orphelins dès la génération des macros via un macro-guide ;
- ajustements de proportions renforcés par cohérence spatiale ;
- nettoyage final réaligné pour éviter la recréation d’orphelins en fin de pipeline ;
- métriques de fragmentation plus cohérentes avec la définition opérationnelle d’orphelin.

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
from PIL import Image, ImageFilter

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
        (0x81, 0x61, 0x3C),  # Coyote Brown
        (0x55, 0x54, 0x3F),  # Vert Olive
        (0x7C, 0x6D, 0x66),  # Terre de France
        (0x57, 0x5D, 0x57),  # Vert-de-gris
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

# Prévention en amont.
MACRO_GUIDE_MIN_SIDE = 96
MACRO_GUIDE_MAX_SIDE = 192
MACRO_GUIDE_BONUS = 0.085
MACRO_GUIDE_PENALTY = 0.030
MIN_DST_NEIGHBORS_FOR_FLIP = 3
COHERENCE_GAIN_WEIGHT = 0.035

# Best-of informatif : score calculé et exporté, mais non bloquant.
BESTOF_REQUIRED = False
BESTOF_MIN_SCORE = 0.945

# Exports logs / ML / DL.
EVENTS_JSONL = "validation_events.jsonl"
ACCEPTS_JSONL = "ml_accepts.jsonl"
REJECTIONS_JSONL = "ml_rejections.jsonl"
FULL_DATASET_JSONL = "ml_dataset_all_attempts.jsonl"

MAX_REPAIR_ROUNDS = 3

# Tolérance dynamique bornée.
DEFAULT_DYNAMIC_TOLERANCE_ENABLED = True
DEFAULT_REJECTION_RATE_WINDOW = 24
DEFAULT_REJECTION_RATE_HIGH = 0.90
DEFAULT_REJECTION_RATE_LOW = 0.55
DEFAULT_TOLERANCE_MIN_ATTEMPTS = 24
DEFAULT_TOLERANCE_RELAX_STEP = 0.08
MAX_TOLERANCE_RELAX = 0.40

# Anti-pixellisation.
DEFAULT_ENABLE_ANTI_PIXEL = True
ANTI_PIXEL_MODE_FILTER_SIZE = 3
ANTI_PIXEL_PASSES = 1


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
class ValidationToleranceProfile:
    relax_level: float
    max_abs_error_per_color: Tuple[float, float, float, float]
    max_mean_abs_error: float
    min_boundary_density: float
    max_boundary_density: float
    min_boundary_density_small: float
    max_boundary_density_small: float
    min_boundary_density_tiny: float
    max_boundary_density_tiny: float
    max_mirror_similarity: float
    min_largest_component_ratio_class_1: float
    max_edge_contact_ratio: float
    bestof_min_score: float
    max_orphan_ratio: float
    max_micro_islands_per_mp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relax_level": float(self.relax_level),
            "max_abs_error_per_color": [float(x) for x in self.max_abs_error_per_color],
            "max_mean_abs_error": float(self.max_mean_abs_error),
            "min_boundary_density": float(self.min_boundary_density),
            "max_boundary_density": float(self.max_boundary_density),
            "min_boundary_density_small": float(self.min_boundary_density_small),
            "max_boundary_density_small": float(self.max_boundary_density_small),
            "min_boundary_density_tiny": float(self.min_boundary_density_tiny),
            "max_boundary_density_tiny": float(self.max_boundary_density_tiny),
            "max_mirror_similarity": float(self.max_mirror_similarity),
            "min_largest_component_ratio_class_1": float(self.min_largest_component_ratio_class_1),
            "max_edge_contact_ratio": float(self.max_edge_contact_ratio),
            "bestof_min_score": float(self.bestof_min_score),
            "max_orphan_ratio": float(self.max_orphan_ratio),
            "max_micro_islands_per_mp": float(self.max_micro_islands_per_mp),
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


def build_validation_tolerance_profile(relax_level: float = 0.0) -> ValidationToleranceProfile:
    relax = _clip_float(float(relax_level), 0.0, MAX_TOLERANCE_RELAX)

    per_color = np.asarray(MAX_ABS_ERROR_PER_COLOR, dtype=np.float32) * np.float32(1.0 + 0.75 * relax)
    mean_abs = float(MAX_MEAN_ABS_ERROR * (1.0 + 1.00 * relax))

    min_bd = float(max(0.0, MIN_BOUNDARY_DENSITY * (1.0 - 0.35 * relax)))
    max_bd = float(min(1.0, MAX_BOUNDARY_DENSITY * (1.0 + 0.20 * relax)))

    min_bd_small = float(max(0.0, MIN_BOUNDARY_DENSITY_SMALL * (1.0 - 0.35 * relax)))
    max_bd_small = float(min(1.0, MAX_BOUNDARY_DENSITY_SMALL * (1.0 + 0.20 * relax)))

    min_bd_tiny = float(max(0.0, MIN_BOUNDARY_DENSITY_TINY * (1.0 - 0.35 * relax)))
    max_bd_tiny = float(min(1.0, MAX_BOUNDARY_DENSITY_TINY * (1.0 + 0.20 * relax)))

    max_mirror = float(min(0.98, MAX_MIRROR_SIMILARITY + 0.08 * relax))
    min_largest = float(max(0.04, MIN_LARGEST_COMPONENT_RATIO_CLASS_1 * (1.0 - 0.30 * relax)))
    max_edge = float(min(0.90, MAX_EDGE_CONTACT_RATIO + 0.10 * relax))
    bestof_min = float(max(0.90, BESTOF_MIN_SCORE - 0.03 * (relax / max(1e-9, MAX_TOLERANCE_RELAX))))

    relax_ratio = float(relax / max(1e-9, MAX_TOLERANCE_RELAX))
    orphan_cap = float(MAX_ORPHAN_RATIO)
    micro_cap = float(MAX_MICRO_ISLANDS_PER_MP)
    if relax_ratio > 0.0:
        orphan_cap = float(_clip_float(
            max(float(MAX_ORPHAN_RATIO), float(PRECHAMPION_MAX_ORPHAN_RATIO) * relax_ratio),
            float(MAX_ORPHAN_RATIO),
            float(PRECHAMPION_MAX_ORPHAN_RATIO),
        ))
        micro_cap = float(_clip_float(
            max(float(MAX_MICRO_ISLANDS_PER_MP), float(PRECHAMPION_MAX_MICRO_ISLANDS_PER_MP) * relax_ratio),
            float(MAX_MICRO_ISLANDS_PER_MP),
            float(PRECHAMPION_MAX_MICRO_ISLANDS_PER_MP),
        ))

    return ValidationToleranceProfile(
        relax_level=relax,
        max_abs_error_per_color=tuple(float(x) for x in per_color.tolist()),
        max_mean_abs_error=mean_abs,
        min_boundary_density=min_bd,
        max_boundary_density=max_bd,
        min_boundary_density_small=min_bd_small,
        max_boundary_density_small=max_bd_small,
        min_boundary_density_tiny=min_bd_tiny,
        max_boundary_density_tiny=max_bd_tiny,
        max_mirror_similarity=max_mirror,
        min_largest_component_ratio_class_1=min_largest,
        max_edge_contact_ratio=max_edge,
        bestof_min_score=bestof_min,
        max_orphan_ratio=orphan_cap,
        max_micro_islands_per_mp=micro_cap,
    )


def compute_recent_rejection_rate(outcomes: Sequence[bool], window: int) -> float:
    window = max(1, int(window))
    if not outcomes:
        return 0.0
    recent = list(outcomes[-window:])
    accept_rate = float(sum(1 for x in recent if bool(x)) / max(1, len(recent)))
    return float(1.0 - accept_rate)


def adapt_tolerance_relax_level(
    current_relax: float,
    recent_outcomes: Sequence[bool],
    *,
    window: int = DEFAULT_REJECTION_RATE_WINDOW,
    rejection_rate_high: float = DEFAULT_REJECTION_RATE_HIGH,
    rejection_rate_low: float = DEFAULT_REJECTION_RATE_LOW,
    min_attempts: int = DEFAULT_TOLERANCE_MIN_ATTEMPTS,
    relax_step: float = DEFAULT_TOLERANCE_RELAX_STEP,
    enabled: bool = DEFAULT_DYNAMIC_TOLERANCE_ENABLED,
) -> Tuple[float, ValidationToleranceProfile, Dict[str, float]]:
    if not enabled:
        profile = build_validation_tolerance_profile(0.0)
        return 0.0, profile, {
            "rejection_rate": 0.0,
            "window_count": 0.0,
            "relax_before": 0.0,
            "relax_after": 0.0,
        }

    window = max(1, int(window))
    min_attempts = max(1, int(min_attempts))
    relax_step = max(0.0, float(relax_step))
    current_relax = _clip_float(float(current_relax), 0.0, MAX_TOLERANCE_RELAX)

    recent_count = min(len(recent_outcomes), window)
    rejection_rate = compute_recent_rejection_rate(recent_outcomes, window)
    new_relax = current_relax

    if len(recent_outcomes) >= min_attempts:
        if rejection_rate >= float(rejection_rate_high):
            new_relax = _clip_float(current_relax + relax_step, 0.0, MAX_TOLERANCE_RELAX)
        elif rejection_rate <= float(rejection_rate_low):
            new_relax = _clip_float(current_relax - (relax_step * 0.5), 0.0, MAX_TOLERANCE_RELAX)

    profile = build_validation_tolerance_profile(new_relax)
    info = {
        "rejection_rate": float(rejection_rate),
        "window_count": float(recent_count),
        "relax_before": float(current_relax),
        "relax_after": float(new_relax),
    }
    return new_relax, profile, info


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


def resize_nearest_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    y_idx = np.linspace(0, arr.shape[0] - 1, out_h).astype(np.int32)
    x_idx = np.linspace(0, arr.shape[1] - 1, out_w).astype(np.int32)
    return arr[np.ix_(y_idx, x_idx)]


def resize_linear_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    arr_u8 = np.clip(np.asarray(arr, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode="L")
    out = img.resize((int(out_w), int(out_h)), resample=Image.Resampling.BICUBIC)
    return np.asarray(out, dtype=np.float32) / 255.0


# ============================================================
# GÉNÉRATEUR
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
            (c2x, c2y, 22.0, 0.64),
            (c3x, c3y, -8.0, 0.32),
            (c4x, c4y, 0.0, 0.14),
            (c5x, c5y, -11.0, 0.07),
            (c6x, c6y, 15.0, 0.03),
        ],
        IDX_1: [
            (c1x, c1y, 16.0, 1.00),
            (c2x, c2y, -26.0, 0.64),
            (c3x, c3y, 7.0, 0.31),
            (c4x, c4y, 12.0, 0.15),
            (c5x, c5y, 24.0, 0.07),
            (c6x, c6y, -18.0, 0.03),
        ],
        IDX_2: [
            (c1x, c1y, -10.0, 0.78),
            (c2x, c2y, -12.0, 0.92),
            (c3x, c3y, 26.0, 0.50),
            (c4x, c4y, -4.0, 0.19),
            (c5x, c5y, 9.0, 0.08),
            (c6x, c6y, -21.0, 0.03),
        ],
        IDX_3: [
            (c1x, c1y, 18.0, 0.74),
            (c2x, c2y, 20.0, 0.90),
            (c3x, c3y, -24.0, 0.52),
            (c4x, c4y, 0.0, 0.22),
            (c5x, c5y, -15.0, 0.09),
            (c6x, c6y, 11.0, 0.03),
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


def _neighbors8(labels: np.ndarray) -> List[np.ndarray]:
    padded = np.pad(labels, ((1, 1), (1, 1)), mode="edge")
    return [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 0:-2],
        padded[1:-1, 2:],
        padded[2:, 0:-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]


def same_neighbor_count(label_map: np.ndarray) -> np.ndarray:
    neigh = _neighbors8(label_map)
    out = np.zeros_like(label_map, dtype=np.uint8)
    for n in neigh:
        out += (n == label_map).astype(np.uint8)
    return out


def neighbor_count_for_class(label_map: np.ndarray, cls: int) -> np.ndarray:
    neigh = _neighbors8(label_map)
    out = np.zeros_like(label_map, dtype=np.uint8)
    for n in neigh:
        out += (n == cls).astype(np.uint8)
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


def sequential_assign(
    fields: np.ndarray,
    target_counts: np.ndarray,
    macro_guide: Optional[np.ndarray] = None,
    macro_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    _, height, width = fields.shape
    labels = np.zeros((height, width), dtype=np.uint8)
    remaining = np.ones((height, width), dtype=bool)

    guide_flat = macro_guide.ravel() if macro_guide is not None else None
    prior_flat = (
        macro_prior.reshape(macro_prior.shape[0], -1).astype(np.float32, copy=False)
        if macro_prior is not None else None
    )

    for c in (IDX_1, IDX_2, IDX_3):
        count = int(target_counts[c])
        if count <= 0:
            continue

        remaining_flat = np.flatnonzero(remaining.ravel())
        score = fields[c].ravel()[remaining_flat].astype(np.float32, copy=False)

        if guide_flat is not None:
            g = guide_flat[remaining_flat]
            score = score + np.where(
                g == c,
                np.float32(MACRO_GUIDE_BONUS * 0.35),
                np.float32(-MACRO_GUIDE_PENALTY * 0.25),
            )
        if prior_flat is not None:
            score = score + np.float32(MACRO_GUIDE_BONUS) * prior_flat[c, remaining_flat].astype(np.float32)

        best_local = np.argpartition(score, -count)[-count:]
        selected_flat = remaining_flat[best_local]
        labels.ravel()[selected_flat] = c
        remaining.ravel()[selected_flat] = False

    return labels


def exactify_proportions(
    labels: np.ndarray,
    fields: np.ndarray,
    target_counts: np.ndarray,
    macro_guide: Optional[np.ndarray] = None,
    macro_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_classes, height, width = fields.shape
    labels = labels.copy()

    flat_fields = fields.reshape(n_classes, -1).astype(np.float32, copy=False)
    guide_flat = macro_guide.ravel() if macro_guide is not None else None
    prior_flat = (
        macro_prior.reshape(macro_prior.shape[0], -1).astype(np.float32, copy=False)
        if macro_prior is not None else None
    )

    for _ in range(24):
        counts = np.bincount(labels.ravel(), minlength=n_classes)
        delta = target_counts - counts
        if np.all(delta == 0):
            break

        flat_labels = labels.ravel()
        changed = False

        under = np.where(delta > 0)[0]
        over = np.where(delta < 0)[0]
        if len(under) == 0 or len(over) == 0:
            break

        for u in under:
            need = int(delta[u])
            if need <= 0:
                continue

            current_map = flat_labels.reshape(height, width)
            flat_boundary = boundary_mask(current_map).ravel()
            same = same_neighbor_count(current_map).ravel().astype(np.float32)
            dst_neighbors = neighbor_count_for_class(current_map, int(u)).ravel().astype(np.float32)

            candidate_mask = (
                flat_boundary
                & np.isin(flat_labels, over)
                & (dst_neighbors >= MIN_DST_NEIGHBORS_FOR_FLIP)
            )
            idx = np.where(candidate_mask)[0]
            if idx.size == 0:
                continue

            current = flat_labels[idx]
            gain = flat_fields[u, idx] - flat_fields[current, idx]
            gain += COHERENCE_GAIN_WEIGHT * dst_neighbors[idx]
            gain -= COHERENCE_GAIN_WEIGHT * np.maximum(0.0, same[idx] - 4.0)

            if guide_flat is not None:
                gain += np.where(guide_flat[idx] == u, MACRO_GUIDE_BONUS * 0.35, 0.0).astype(np.float32)
                gain -= np.where(guide_flat[idx] == current, MACRO_GUIDE_PENALTY * 0.25, 0.0).astype(np.float32)
            if prior_flat is not None:
                gain += MACRO_GUIDE_BONUS * prior_flat[u, idx]
                gain -= MACRO_GUIDE_PENALTY * prior_flat[current, idx]

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


def force_exact_target_counts_relaxed(
    labels: np.ndarray,
    fields: np.ndarray,
    target_counts: np.ndarray,
    macro_guide: Optional[np.ndarray] = None,
    macro_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    labels = labels.copy()
    flat_labels = labels.ravel()
    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)
    shape = labels.shape
    guide_flat = macro_guide.ravel() if macro_guide is not None else None
    prior_flat = (
        macro_prior.reshape(macro_prior.shape[0], -1).astype(np.float32, copy=False)
        if macro_prior is not None else None
    )

    for _ in range(16):
        counts = np.bincount(flat_labels, minlength=fields.shape[0]).astype(int)
        delta = target_counts.astype(int) - counts
        if np.all(delta == 0):
            break

        current_map = flat_labels.reshape(shape)
        boundary = boundary_mask(current_map).ravel()

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
                if guide_flat is not None:
                    gain += np.where(guide_flat[idx_pool] == dst, MACRO_GUIDE_BONUS * 0.20, 0.0).astype(np.float32)
                    gain -= np.where(guide_flat[idx_pool] == src, MACRO_GUIDE_PENALTY * 0.15, 0.0).astype(np.float32)
                if prior_flat is not None:
                    gain += (MACRO_GUIDE_BONUS * 0.5) * prior_flat[dst, idx_pool]
                    gain -= (MACRO_GUIDE_PENALTY * 0.5) * prior_flat[src, idx_pool]

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


def force_exact_target_counts(
    labels: np.ndarray,
    fields: np.ndarray,
    target_counts: np.ndarray,
    macro_guide: Optional[np.ndarray] = None,
    macro_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    labels = labels.copy()
    flat_labels = labels.ravel()
    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)
    shape = labels.shape
    guide_flat = macro_guide.ravel() if macro_guide is not None else None
    prior_flat = (
        macro_prior.reshape(macro_prior.shape[0], -1).astype(np.float32, copy=False)
        if macro_prior is not None else None
    )

    for _ in range(16):
        counts = np.bincount(flat_labels, minlength=fields.shape[0]).astype(int)
        delta = target_counts.astype(int) - counts
        if np.all(delta == 0):
            break

        current_map = flat_labels.reshape(shape)
        boundary = boundary_mask(current_map).ravel()
        same = same_neighbor_count(current_map).ravel().astype(np.float32)

        under = [int(c) for c in np.where(delta > 0)[0]]
        over = [int(c) for c in np.where(delta < 0)[0]]
        moved = False

        for dst in under:
            need = int(delta[dst])
            if need <= 0:
                continue

            dst_neighbors_full = neighbor_count_for_class(current_map, dst).ravel().astype(np.float32)

            for src in over:
                excess = -int(delta[src])
                if excess <= 0:
                    continue

                take = min(need, excess)

                idx_boundary = np.where(
                    (flat_labels == src)
                    & boundary
                    & (dst_neighbors_full >= MIN_DST_NEIGHBORS_FOR_FLIP)
                )[0]
                idx_pool = idx_boundary

                if idx_pool.size < take:
                    idx_inner = np.where(
                        (flat_labels == src)
                        & (~boundary)
                        & (dst_neighbors_full >= MIN_DST_NEIGHBORS_FOR_FLIP)
                    )[0]
                    if idx_inner.size:
                        idx_pool = np.concatenate([idx_boundary, idx_inner], axis=0)

                if idx_pool.size == 0:
                    continue

                gain = flat_fields[dst, idx_pool] - flat_fields[src, idx_pool]
                gain += COHERENCE_GAIN_WEIGHT * dst_neighbors_full[idx_pool]
                gain -= COHERENCE_GAIN_WEIGHT * np.maximum(0.0, same[idx_pool] - 4.0)

                if guide_flat is not None:
                    gain += np.where(guide_flat[idx_pool] == dst, MACRO_GUIDE_BONUS * 0.35, 0.0).astype(np.float32)
                    gain -= np.where(guide_flat[idx_pool] == src, MACRO_GUIDE_PENALTY * 0.25, 0.0).astype(np.float32)
                if prior_flat is not None:
                    gain += MACRO_GUIDE_BONUS * prior_flat[dst, idx_pool]
                    gain -= MACRO_GUIDE_PENALTY * prior_flat[src, idx_pool]

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

    labels = flat_labels.reshape(shape).astype(np.uint8)

    counts = np.bincount(labels.ravel(), minlength=fields.shape[0]).astype(int)
    if not np.all(counts == target_counts.astype(int)):
        labels = force_exact_target_counts_relaxed(labels, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)

    return labels.astype(np.uint8)


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
            (same <= int(orphan_max_same_neighbors))
            & (winner != labels)
            & (winner_count >= int(orphan_min_winner_neighbors))
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
    orphan_pixels = int(np.sum(same <= ORPHAN_MAX_SAME_NEIGHBORS))
    weak_pixels = int(np.sum(same <= max(1, ORPHAN_MAX_SAME_NEIGHBORS + 1)))

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


def mode_filter_smoothing_pass(
    label_map: np.ndarray,
    *,
    filter_size: int = ANTI_PIXEL_MODE_FILTER_SIZE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = Image.fromarray(label_map.astype(np.uint8), mode="L")
    out = np.asarray(img.filter(ImageFilter.ModeFilter(size=max(3, int(filter_size)))), dtype=np.uint8)
    changed = int(np.sum(out != label_map))
    return out, {
        "mode_filtered_pixels": int(changed),
        "mode_filter_size": int(max(3, int(filter_size))),
    }


def build_macro_prior(fields: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, height, width = fields.shape

    guide_w = int(np.clip(round(width / 48), MACRO_GUIDE_MIN_SIDE, MACRO_GUIDE_MAX_SIDE))
    guide_h = int(np.clip(round(height / 48), MACRO_GUIDE_MIN_SIDE, MACRO_GUIDE_MAX_SIDE))

    coarse_fields = np.stack(
        [resize_nearest_2d(fields[c], guide_h, guide_w).astype(np.float32, copy=False) for c in range(N_CLASSES)],
        axis=0,
    )

    target_counts_small = np.rint(TARGET * (guide_h * guide_w)).astype(int)
    target_counts_small[-1] = (guide_h * guide_w) - int(target_counts_small[:-1].sum())

    guide = sequential_assign(coarse_fields, target_counts_small, macro_guide=None, macro_prior=None)
    guide = exactify_proportions(guide, coarse_fields, target_counts_small, macro_guide=None, macro_prior=None)
    guide = force_exact_target_counts(guide, coarse_fields, target_counts_small, macro_guide=None, macro_prior=None)

    guide, _ = cleanup_orphan_pixels(
        guide,
        class_count=N_CLASSES,
        passes=3,
        orphan_max_same_neighbors=2,
        orphan_min_winner_neighbors=5,
    )
    guide, _ = majority_smoothing_pass(
        guide,
        class_count=N_CLASSES,
        same_threshold=3,
        winner_threshold=5,
    )
    guide, _ = cleanup_orphan_pixels(
        guide,
        class_count=N_CLASSES,
        passes=2,
        orphan_max_same_neighbors=2,
        orphan_min_winner_neighbors=5,
    )

    soft_prior = np.stack(
        [resize_linear_2d((guide == cls).astype(np.float32), height, width) for cls in range(N_CLASSES)],
        axis=0,
    )
    denom = np.maximum(np.sum(soft_prior, axis=0, keepdims=True), 1e-6)
    soft_prior = soft_prior / denom
    hard_guide = np.argmax(soft_prior, axis=0).astype(np.uint8, copy=False)
    return hard_guide, soft_prior


# ============================================================
# BEST-OF / VALIDATION
# ============================================================


def compute_bestof_score(
    *,
    ratios: np.ndarray,
    target: np.ndarray,
    metrics: Dict[str, float],
    fragmentation: Dict[str, Any],
    tolerance_profile: Optional[ValidationToleranceProfile] = None,
) -> Tuple[float, Dict[str, float]]:
    profile = tolerance_profile or build_validation_tolerance_profile(0.0)
    abs_err = np.abs(np.asarray(ratios, dtype=float) - np.asarray(target, dtype=float))
    mean_abs = float(np.mean(abs_err))

    ratio_score = float(np.clip(1.0 - (mean_abs / max(1e-9, float(profile.max_mean_abs_error))), 0.0, 1.0))
    per_class_score = float(
        np.clip(
            1.0 - float(np.mean(abs_err / np.maximum(np.asarray(profile.max_abs_error_per_color, dtype=float), 1e-9))),
            0.0,
            1.0,
        )
    )

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
        0.30 * ratio_score
        + 0.25 * per_class_score
        + 0.30 * fragmentation_score
        + 0.10 * symmetry_score
        + 0.05 * edge_score
    )
    return float(np.clip(score, 0.0, 1.0)), subscores


def validate_with_reasons(
    candidate: CandidateResult,
    tolerance_profile: Optional[ValidationToleranceProfile] = None,
) -> ValidationOutcome:
    profile = tolerance_profile or build_validation_tolerance_profile(0.0)
    reasons: List[str] = []
    ratios = candidate.ratios
    metrics = candidate.metrics
    label_map = candidate.label_map

    abs_err = np.abs(ratios - TARGET)
    mean_abs = float(np.mean(abs_err))

    max_abs_per_color = np.asarray(profile.max_abs_error_per_color, dtype=float)

    for i, err in enumerate(abs_err):
        if float(err) > float(max_abs_per_color[i]):
            reasons.append(f"ratio_class_{i}_abs_error")

    if mean_abs > float(profile.max_mean_abs_error):
        reasons.append("mean_abs_error")

    if not (profile.min_boundary_density <= float(metrics["boundary_density"]) <= profile.max_boundary_density):
        reasons.append("boundary_density")
    if not (profile.min_boundary_density_small <= float(metrics["boundary_density_small"]) <= profile.max_boundary_density_small):
        reasons.append("boundary_density_small")
    if not (profile.min_boundary_density_tiny <= float(metrics["boundary_density_tiny"]) <= profile.max_boundary_density_tiny):
        reasons.append("boundary_density_tiny")

    if float(metrics["mirror_similarity"]) > profile.max_mirror_similarity:
        reasons.append("mirror_similarity")
    if float(metrics["largest_component_ratio_class_1"]) < profile.min_largest_component_ratio_class_1:
        reasons.append("largest_component_ratio_class_1")
    if float(metrics["edge_contact_ratio"]) > profile.max_edge_contact_ratio:
        reasons.append("edge_contact_ratio")

    frag = fragmentation_report(
        label_map,
        class_count=N_CLASSES,
        min_component_pixels=MIN_COMPONENT_PIXELS,
    )

    if float(frag["orphan_ratio"]) > float(profile.max_orphan_ratio):
        reasons.append("orphan_pixels")
    if float(frag["micro_components_per_mp"]) > float(profile.max_micro_islands_per_mp):
        reasons.append("micro_islands")

    strict_ok = len(reasons) == 0

    bestof_score, subscores = compute_bestof_score(
        ratios=ratios,
        target=TARGET,
        metrics=metrics,
        fragmentation=frag,
        tolerance_profile=profile,
    )
    bestof_ok = bool(bestof_score >= float(profile.bestof_min_score) and strict_ok)

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


def validate_candidate_result(
    candidate: CandidateResult,
    tolerance_profile: Optional[ValidationToleranceProfile] = None,
) -> bool:
    return bool(validate_with_reasons(candidate, tolerance_profile=tolerance_profile).accepted)


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
        shift_strength = _clip_float(shift_strength - 0.05, 0.50, 1.10)
        motif_scale *= 1.05

    if "orphan_pixels" in reasons or "micro_islands" in reasons:
        extra_cleanup_passes += 2
        merge_micro = True

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

    current_map = labels.copy()
    flat_labels = labels.ravel()
    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)

    dst_neighbors_full = {
        cls: neighbor_count_for_class(current_map, cls).ravel().astype(np.float32) for cls in range(N_CLASSES)
    }

    mask = border & (labels == dominant_edge_class)
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return labels, {
            "edge_pixels_rebalanced": 0,
            "dominant_edge_class": dominant_edge_class,
        }

    # Ne pas choisir des pixels qui créeraient un flip isolé.
    same = same_neighbor_count(current_map).ravel().astype(np.float32)
    safe_idx = idx[same[idx] <= 6]
    if safe_idx.size == 0:
        safe_idx = idx

    take = max(1, int(safe_idx.size * float(fraction)))
    scores = flat_fields[:, safe_idx].copy()
    current = flat_labels[safe_idx]
    scores[current, np.arange(safe_idx.size)] = -1e9

    for cls in range(N_CLASSES):
        scores[cls, :] += COHERENCE_GAIN_WEIGHT * dst_neighbors_full[cls][safe_idx]

    alt = np.argmax(scores, axis=0).astype(np.uint8, copy=False)
    alt_neighbors = np.array([dst_neighbors_full[int(a)][i] for i, a in zip(safe_idx, alt)], dtype=np.float32)
    gains = scores[alt, np.arange(safe_idx.size)] - flat_fields[current, safe_idx]
    gains[alt_neighbors < MIN_DST_NEIGHBORS_FOR_FLIP] = -1e9

    order = np.argsort(gains)[::-1]
    chosen_local = safe_idx[order[:take]]
    chosen_alt = alt[order[:take]]
    valid = gains[order[:take]] > -1e8

    chosen = chosen_local[valid]
    chosen_alt = chosen_alt[valid]

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

    current_map = labels.copy()
    flat_fields = fields.reshape(fields.shape[0], -1).astype(np.float32, copy=False)
    flat_labels = labels.ravel()
    current = flat_labels[chosen]

    scores = flat_fields[:, chosen].copy()
    scores[current, np.arange(chosen.size)] = -1e9

    for cls in range(N_CLASSES):
        scores[cls, :] += COHERENCE_GAIN_WEIGHT * neighbor_count_for_class(current_map, cls).ravel()[chosen]

    alt = np.argmax(scores, axis=0).astype(np.uint8, copy=False)
    alt_neighbors = np.array(
        [neighbor_count_for_class(current_map, int(a)).ravel()[i] for i, a in zip(chosen, alt)],
        dtype=np.float32,
    )

    valid = alt_neighbors >= MIN_DST_NEIGHBORS_FOR_FLIP
    chosen = chosen[valid]
    alt = alt[valid]

    flat_labels[chosen] = alt
    return flat_labels.reshape(labels.shape), {
        "anti_mirror_pixels": int(chosen.size),
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
    tolerance_profile: Optional[ValidationToleranceProfile] = None,
    tolerance_runtime: Optional[Dict[str, float]] = None,
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
        "tolerance_profile": (tolerance_profile.to_dict() if tolerance_profile is not None else None),
        "tolerance_runtime": ({str(k): float(v) for k, v in tolerance_runtime.items()} if tolerance_runtime is not None else None),
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
    anti_pixel: bool = DEFAULT_ENABLE_ANTI_PIXEL,
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

    macro_guide, macro_prior = build_macro_prior(fields)

    target_counts = np.rint(TARGET * (width * height)).astype(int)
    target_counts[-1] = (width * height) - int(target_counts[:-1].sum())

    label_map = sequential_assign(fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)
    label_map = exactify_proportions(label_map, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)
    label_map = force_exact_target_counts(label_map, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)

    label_map, orphan_info = cleanup_orphan_pixels(
        label_map,
        class_count=N_CLASSES,
        passes=STRAY_CLEANUP_PASSES + int(extra_cleanup_passes),
        orphan_max_same_neighbors=ORPHAN_MAX_SAME_NEIGHBORS,
        orphan_min_winner_neighbors=ORPHAN_MIN_WINNER_NEIGHBORS,
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
            "majority_smoothed_pixels": int(
                info_a["majority_smoothed_pixels"] + info_b["majority_smoothed_pixels"]
            )
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

    label_map = exactify_proportions(label_map, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)
    label_map = force_exact_target_counts(label_map, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)

    # Sécurité finale : éviter qu'une exactification terminale ne recrée des orphelins.
    label_map, orphan_info_3 = cleanup_orphan_pixels(
        label_map,
        class_count=N_CLASSES,
        passes=1,
        orphan_max_same_neighbors=ORPHAN_MAX_SAME_NEIGHBORS,
        orphan_min_winner_neighbors=ORPHAN_MIN_WINNER_NEIGHBORS,
    )
    label_map = force_exact_target_counts(label_map, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)

    anti_pixel_info = {"mode_filtered_pixels": 0, "mode_filter_size": 0}
    if anti_pixel:
        for _ in range(max(1, int(ANTI_PIXEL_PASSES))):
            label_map, anti_pixel_info = mode_filter_smoothing_pass(
                label_map,
                filter_size=ANTI_PIXEL_MODE_FILTER_SIZE,
            )
            label_map = exactify_proportions(label_map, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)
            label_map = force_exact_target_counts(label_map, fields, target_counts, macro_guide=macro_guide, macro_prior=macro_prior)
            label_map, _ = cleanup_orphan_pixels(
                label_map,
                class_count=N_CLASSES,
                passes=1,
                orphan_max_same_neighbors=ORPHAN_MAX_SAME_NEIGHBORS,
                orphan_min_winner_neighbors=ORPHAN_MIN_WINNER_NEIGHBORS,
            )

    frag_metrics = fragmentation_report(
        label_map,
        class_count=N_CLASSES,
        min_component_pixels=MIN_COMPONENT_PIXELS,
    )

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
        "orphan_ratio": float(frag_metrics["orphan_ratio"]),
        "weak_ratio": float(frag_metrics["weak_ratio"]),
        "micro_components_per_mp": float(frag_metrics["micro_components_per_mp"]),
        "mode_filtered_pixels": float(anti_pixel_info["mode_filtered_pixels"]),
        "macro_prior_agreement": float(np.mean(label_map == macro_guide)),
        "orphan_pixels_fixed": float(
            orphan_info["orphan_pixels_fixed"]
            + orphan_info_2["orphan_pixels_fixed"]
            + orphan_info_3["orphan_pixels_fixed"]
        ),
        "orphan_cleanup_passes": float(
            orphan_info["orphan_cleanup_passes"]
            + orphan_info_2["orphan_cleanup_passes"]
            + orphan_info_3["orphan_cleanup_passes"]
        ),
        "majority_smoothed_pixels": float(smoothing_info["majority_smoothed_pixels"]),
        "edge_pixels_rebalanced": float(edge_info["edge_pixels_rebalanced"]),
        "anti_mirror_pixels": float(mirror_info["anti_mirror_pixels"]),
        "repair_merge_micro": float(1 if merge_micro else 0),
        "repair_rebalance_edges": float(1 if rebalance_edges else 0),
        "repair_anti_mirror": float(1 if anti_mirror else 0),
        "repair_extra_cleanup_passes": float(extra_cleanup_passes),
        "macro_guide_agreement": float(np.mean(label_map == macro_guide)),
        "anti_pixel_enabled": float(1 if anti_pixel else 0),
    }

    return CandidateResult(
        seed=profile.seed,
        profile=profile,
        image=render_label_map(label_map),
        label_map=label_map,
        ratios=ratios,
        metrics=metrics,
    )


def generate_candidate_from_seed(seed: int, anti_pixel: bool = DEFAULT_ENABLE_ANTI_PIXEL) -> CandidateResult:
    profile = make_profile(seed)
    return generate_one_variant(profile, anti_pixel=anti_pixel)


def generate_and_validate_from_seed(
    seed: int,
    max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    tolerance_profile: Optional[ValidationToleranceProfile] = None,
    anti_pixel: bool = DEFAULT_ENABLE_ANTI_PIXEL,
) -> Tuple[CandidateResult, ValidationOutcome]:
    candidate = generate_candidate_from_seed(seed, anti_pixel=anti_pixel)
    outcome = validate_with_reasons(candidate, tolerance_profile=tolerance_profile)

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
            anti_pixel=anti_pixel,
        )
        repaired_candidate.metrics["repair_round"] = float(repair_round)

        repaired_outcome = validate_with_reasons(repaired_candidate, tolerance_profile=tolerance_profile)
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
    tolerance_profile: Optional[ValidationToleranceProfile] = None,
) -> Dict[str, object]:
    if outcome is None:
        outcome = validate_with_reasons(candidate, tolerance_profile=tolerance_profile)

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
        "bestof_score": round(float(outcome.bestof_score), 6),
        "accepted": int(outcome.accepted),
        "reasons": "|".join(outcome.reasons),
        "tolerance_relax_level": round(float((tolerance_profile.relax_level if tolerance_profile is not None else 0.0)), 6),
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
    dynamic_tolerance_enabled: bool = DEFAULT_DYNAMIC_TOLERANCE_ENABLED,
    rejection_rate_window: int = DEFAULT_REJECTION_RATE_WINDOW,
    rejection_rate_high: float = DEFAULT_REJECTION_RATE_HIGH,
    rejection_rate_low: float = DEFAULT_REJECTION_RATE_LOW,
    tolerance_min_attempts: int = DEFAULT_TOLERANCE_MIN_ATTEMPTS,
    tolerance_relax_step: float = DEFAULT_TOLERANCE_RELAX_STEP,
    anti_pixel: bool = DEFAULT_ENABLE_ANTI_PIXEL,
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

    tolerance_outcomes: List[bool] = []
    tolerance_relax_level = 0.0
    tolerance_profile = build_validation_tolerance_profile(tolerance_relax_level)
    tolerance_runtime = {
        "rejection_rate": 0.0,
        "window_count": 0.0,
        "relax_before": 0.0,
        "relax_after": 0.0,
    }

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

            tolerance_relax_level, tolerance_profile, tolerance_runtime = adapt_tolerance_relax_level(
                tolerance_relax_level,
                tolerance_outcomes,
                window=rejection_rate_window,
                rejection_rate_high=rejection_rate_high,
                rejection_rate_low=rejection_rate_low,
                min_attempts=tolerance_min_attempts,
                relax_step=tolerance_relax_step,
                enabled=dynamic_tolerance_enabled,
            )

            batch = build_batch(target_index, local_attempt, tuning.attempt_batch_size, base_seed)
            use_parallel = bool(tuning.parallel_attempts and tuning.max_workers > 1 and len(batch) > 1)

            tasks: List[asyncio.Task] = []
            if use_parallel:
                pool = get_process_pool(tuning.max_workers)
                for attempt_no, seed in batch:
                    fut = loop.run_in_executor(pool, generate_and_validate_from_seed, seed, MAX_REPAIR_ROUNDS, tolerance_profile, anti_pixel)
                    tasks.append(asyncio.create_task(_await_attempt(fut, attempt_no, seed)))
            else:
                attempt_no, seed = batch[0]
                fut = loop.run_in_executor(None, generate_and_validate_from_seed, seed, MAX_REPAIR_ROUNDS, tolerance_profile, anti_pixel)
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
                    tolerance_profile=tolerance_profile,
                    tolerance_runtime=tolerance_runtime,
                )

                if outcome.accepted:
                    counters.passed_validation += 1
                    tolerance_outcomes.append(True)
                else:
                    counters.rejected += 1
                    tolerance_outcomes.append(False)

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
                tolerance_profile=tolerance_profile,
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
        "dynamic_tolerance_enabled": bool(dynamic_tolerance_enabled),
        "tolerance_relax_level_final": float(tolerance_relax_level),
        "tolerance_profile_final": tolerance_profile.to_dict(),
        "tolerance_runtime_final": {str(k): float(v) for k, v in tolerance_runtime.items()},
        "anti_pixel_enabled": bool(DEFAULT_ENABLE_ANTI_PIXEL),
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
    parser.add_argument("--disable-dynamic-tolerance", action="store_true")
    parser.add_argument("--disable-anti-pixel", action="store_true")
    parser.add_argument("--rejection-rate-window", type=int, default=DEFAULT_REJECTION_RATE_WINDOW)
    parser.add_argument("--rejection-rate-high", type=float, default=DEFAULT_REJECTION_RATE_HIGH)
    parser.add_argument("--rejection-rate-low", type=float, default=DEFAULT_REJECTION_RATE_LOW)
    parser.add_argument("--tolerance-min-attempts", type=int, default=DEFAULT_TOLERANCE_MIN_ATTEMPTS)
    parser.add_argument("--tolerance-relax-step", type=float, default=DEFAULT_TOLERANCE_RELAX_STEP)
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
                dynamic_tolerance_enabled=not args.disable_dynamic_tolerance,
                rejection_rate_window=args.rejection_rate_window,
                rejection_rate_high=args.rejection_rate_high,
                rejection_rate_low=args.rejection_rate_low,
                tolerance_min_attempts=args.tolerance_min_attempts,
                tolerance_relax_step=args.tolerance_relax_step,
                anti_pixel=not args.disable_anti_pixel,
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
