# -*- coding: utf-8 -*-
"""
main.py
Camouflage Armée Fédérale Europe
Version module-friendly + async-friendly + accélération contrôlée.

Points clés :
- API synchrone conservée
- API asynchrone ajoutée
- génération séquentielle stricte par image conservée
- aucun passage à l'image suivante tant que la précédente n'est pas validée
- accélération par lots de tentatives parallèles pour une même image
- critères de validation inchangés

Corrections / améliorations :
- meilleure exploitation CPU via ProcessPoolExecutor plus agressif ;
- pool recréé automatiquement si le nombre de workers change ;
- limitation des threads BLAS/OpenMP dans les workers pour éviter la sur-saturation ;
- arrêt propre du pool en fin d'exécution ;
- comportement strictement séquentiel par image conservé ;
- aucune limite artificielle sur le nombre de tentatives par image.
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


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

# Couleurs
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
    (0x81, 0x61, 0x3C),  # Coyote Brown
    (0x55, 0x54, 0x3F),  # Vert Olive
    (0x7C, 0x6D, 0x66),  # Terre de France
    (0x57, 0x5D, 0x57),  # Vert-de-gris
], dtype=np.uint8)

TARGET = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)

# Origine visible du pixel final
ORIGIN_BACKGROUND = 0
ORIGIN_MACRO = 1
ORIGIN_TRANSITION = 2
ORIGIN_MICRO = 3

# Tolérances de validation finales
MAX_ABS_ERROR_PER_COLOR = np.array([0.045, 0.045, 0.040, 0.040], dtype=float)
MAX_MEAN_ABS_ERROR = 0.026

# Angles autorisés : vertical + oblique 15° à 35°
BASE_ANGLES = [-35, -30, -25, -20, -15, 0, 15, 20, 25, 30, 35]

# Zones de densité asymétrique
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

TRANSITION_LENGTH_CM = (10, 30)
TRANSITION_WIDTH_CM = (5, 15)

MICRO_SIZE_CM = (2, 8)

VISIBLE_MACRO_OLIVE_TARGET = 0.205
VISIBLE_MACRO_TERRE_TARGET = 0.050
VISIBLE_MACRO_GRIS_TARGET = 0.000

VISIBLE_TOTAL_OLIVE_TARGET = TARGET[IDX_OLIVE]
VISIBLE_TOTAL_TERRE_TARGET = TARGET[IDX_TERRE]
VISIBLE_TOTAL_GRIS_TARGET = TARGET[IDX_GRIS]

MIN_OLIVE_CONNECTED_COMPONENT_RATIO = 0.17
MIN_OLIVE_MULTIZONE_SHARE = 0.42
MAX_COYOTE_CENTER_EMPTY_RATIO = 0.50
MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL = 0.56

MIN_BOUNDARY_DENSITY = 0.095
MAX_BOUNDARY_DENSITY = 0.250
MIN_BOUNDARY_DENSITY_SMALL = 0.060
MAX_BOUNDARY_DENSITY_SMALL = 0.220

MAX_MIRROR_SIMILARITY = 0.76

MIN_VISIBLE_OLIVE_MACRO_SHARE = 0.60
MIN_VISIBLE_TERRE_TRANS_SHARE = 0.50
MIN_VISIBLE_GRIS_MICRO_SHARE = 0.82
MAX_VISIBLE_GRIS_MACRO_SHARE = 0.06

MIN_OBLIQUE_SHARE = 0.64
MIN_VERTICAL_SHARE = 0.10
MAX_VERTICAL_SHARE = 0.30
MAX_ANGLE_DOMINANCE_RATIO = 0.32

BOUNDARY_NUDGE_PASSES = 10
BOUNDARY_NUDGE_SAMPLE_RATIO = 0.0035

MIN_TRANSITION_TOUCH_PIXELS = 22
MIN_MICRO_BOUNDARY_COVERAGE = 0.24
MAX_LOCAL_MASS_RATIO_TRANSITION = 0.72
MAX_CENTER_TORSO_OVERLAP_MACRO = 0.34
MAX_CENTER_TORSO_OVERLAP_TERRAIN_MACRO = 0.22
MAX_MACRO_PLACEMENT_ATTEMPTS_OLIVE = 420
MAX_MACRO_PLACEMENT_ATTEMPTS_TERRE = 240
MAX_MACRO_PLACEMENT_ATTEMPTS_GRIS = 40
MAX_TRANSITION_PLACEMENTS = 900
MAX_MICRO_CLUSTER_PLACEMENTS = 1200

# Accélération contrôlée
CPU_COUNT = max(1, os.cpu_count() or 1)
DEFAULT_MAX_WORKERS = max(1, min(12, CPU_COUNT - 2 if CPU_COUNT > 4 else CPU_COUNT))
DEFAULT_ATTEMPT_BATCH_SIZE = max(1, min(8, DEFAULT_MAX_WORKERS))

# Validation perceptive / visuelle
VISUAL_MIN_SILHOUETTE_COLOR_DIVERSITY = 0.62
VISUAL_MIN_CONTOUR_BREAK_SCORE = 0.44
VISUAL_MIN_OUTLINE_BAND_DIVERSITY = 0.58
VISUAL_MIN_SMALL_SCALE_STRUCTURAL_SCORE = 0.42
VISUAL_MIN_FINAL_SCORE = 0.60
VISUAL_MIN_MILITARY_SCORE = 0.62
MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO = 1.10
MIN_PERIPHERY_NON_COYOTE_RATIO = 1.05
MIN_MACRO_OLIVE_VISIBLE_RATIO = 0.16
MAX_MACRO_TERRE_VISIBLE_RATIO = 0.09
MAX_MACRO_GRIS_VISIBLE_RATIO = 0.015


# Discipline de génération pilotée
TARGET_VERTICAL_SHARE = 0.16
MAX_VERTICAL_SOFT_TARGET = 0.24
TARGET_PERIPHERY_REPAIR_STEPS = 80
TARGET_CENTER_REPAIR_STEPS = 56
SEMANTIC_ALLOWED_COLORS = {
    ORIGIN_BACKGROUND: (IDX_COYOTE, IDX_OLIVE),
    ORIGIN_MACRO: (IDX_OLIVE, IDX_TERRE),
    ORIGIN_TRANSITION: (IDX_COYOTE, IDX_OLIVE, IDX_TERRE),
    ORIGIN_MICRO: (IDX_TERRE, IDX_GRIS),
}


# ============================================================
# STRUCTURES
# ============================================================

@dataclass
class VariantProfile:
    seed: int
    allowed_angles: List[int]
    micro_cluster_min: int
    micro_cluster_max: int
    macro_width_variation: float
    macro_lateral_jitter: float
    macro_tip_taper: float
    macro_edge_break: float
    micro_width_variation: float
    micro_lateral_jitter: float
    micro_tip_taper: float
    micro_edge_break: float
    angle_pool: Tuple[int, ...] = field(default_factory=tuple)
    zone_weight_boosts: Tuple[float, ...] = field(default_factory=lambda: tuple(1.0 for _ in DENSITY_ZONES))
    olive_macro_target_scale: float = 1.0
    terre_macro_target_scale: float = 1.0
    gris_macro_target_scale: float = 1.0
    transition_terre_bias: float = 0.0
    transition_olive_bias: float = 0.0
    transition_coyote_bias: float = 0.0
    micro_gris_bias: float = 0.0
    micro_terre_bias: float = 0.0
    center_torso_overlap_scale: float = 1.0
    extra_boundary_nudge_passes: int = 0
    boundary_sample_ratio_scale: float = 1.0


@dataclass
class AdaptiveGenerationState:
    consecutive_rejections: int = 0
    total_rejections: int = 0
    fail_counter: Dict[str, int] = field(default_factory=dict)
    olive_pressure: float = 0.0
    terre_pressure: float = 0.0
    gris_pressure: float = 0.0
    coyote_deficit_pressure: float = 0.0
    coyote_excess_pressure: float = 0.0
    boundary_low_pressure: float = 0.0
    boundary_high_pressure: float = 0.0
    center_relief_pressure: float = 0.0
    asymmetry_pressure: float = 0.0
    oblique_pressure: float = 0.0
    vertical_pressure: float = 0.0
    vertical_excess_pressure: float = 0.0
    angle_diversity_pressure: float = 0.0
    gris_macro_cap_pressure: float = 0.0
    last_rules: List[str] = field(default_factory=list)

    def _bump(self, name: str, amount: float, cap: float = 6.0) -> None:
        setattr(self, name, min(cap, float(getattr(self, name)) + max(0.0, float(amount))))

    def register_success(self) -> None:
        self.consecutive_rejections = 0
        self.last_rules = []
        decay_fields = (
            "olive_pressure",
            "terre_pressure",
            "gris_pressure",
            "coyote_deficit_pressure",
            "coyote_excess_pressure",
            "boundary_low_pressure",
            "boundary_high_pressure",
            "center_relief_pressure",
            "asymmetry_pressure",
            "oblique_pressure",
            "vertical_pressure",
            "vertical_excess_pressure",
            "angle_diversity_pressure",
            "gris_macro_cap_pressure",
        )
        for name in decay_fields:
            setattr(self, name, float(getattr(self, name)) * 0.52)

    def register_failures(self, failures: Sequence[Dict[str, Any]]) -> None:
        self.consecutive_rejections += 1
        self.total_rejections += 1
        self.last_rules = []

        for item in failures:
            rule = str(item.get("rule", "")).strip()
            if not rule:
                continue

            self.last_rules.append(rule)
            self.fail_counter[rule] = int(self.fail_counter.get(rule, 0)) + 1

            delta = _safe_failure_float(item.get("delta"), default=0.05)
            min_value = item.get("min_value")
            max_value = item.get("max_value")
            actual = _safe_failure_float(item.get("actual"), default=0.0)
            target = _safe_failure_float(item.get("target"), default=0.0)
            intensity = min(2.4, max(0.30, delta * 18.0 + 0.20))

            if rule in {"ratio_olive", "abs_err_olive", "vert_olive_macro_share", "largest_olive_component_ratio", "largest_olive_component_ratio_small", "olive_multizone_share", "macro_olive_visible_ratio"}:
                self._bump("olive_pressure", intensity)

            if rule in {"ratio_terre", "abs_err_terre", "terre_de_france_transition_share"}:
                self._bump("terre_pressure", intensity)

            if rule in {"ratio_gris", "abs_err_gris", "vert_de_gris_micro_share"}:
                self._bump("gris_pressure", intensity)

            if rule in {"vert_de_gris_macro_share", "macro_gris_visible_ratio"}:
                self._bump("gris_macro_cap_pressure", intensity)

            if rule in {"ratio_coyote", "abs_err_coyote"}:
                if (max_value is not None and actual > _safe_failure_float(max_value, default=0.0)) or actual > target:
                    self._bump("coyote_excess_pressure", intensity)
                    self._bump("olive_pressure", intensity * 0.42)
                    self._bump("terre_pressure", intensity * 0.30)
                    self._bump("gris_pressure", intensity * 0.20)
                else:
                    self._bump("coyote_deficit_pressure", intensity)

            if rule == "mean_abs_error":
                self._bump("angle_diversity_pressure", intensity * 0.60)
                self._bump("boundary_low_pressure", intensity * 0.30)

            if rule in {"center_empty_ratio", "center_empty_ratio_small"}:
                self._bump("center_relief_pressure", intensity)

            if rule in {"boundary_density", "boundary_density_small"}:
                if min_value is not None and actual < _safe_failure_float(min_value, default=0.0):
                    self._bump("boundary_low_pressure", intensity)
                else:
                    self._bump("boundary_high_pressure", intensity)

            if rule == "mirror_similarity":
                self._bump("asymmetry_pressure", intensity)
                self._bump("angle_diversity_pressure", intensity * 0.50)

            if rule == "oblique_share":
                self._bump("oblique_pressure", intensity)

            if rule == "vertical_share":
                if min_value is not None and actual < _safe_failure_float(min_value, default=0.0):
                    self._bump("vertical_pressure", intensity)
                else:
                    self._bump("vertical_excess_pressure", intensity)

            if rule == "angle_dominance_ratio":
                self._bump("angle_diversity_pressure", intensity)

    def to_hint(self) -> Dict[str, Any]:
        zone_weights = [1.0 for _ in DENSITY_ZONES]
        edge_gain = min(1.10, 0.14 * self.center_relief_pressure + 0.10 * self.asymmetry_pressure)
        for idx in (0, 1, 2, 3, 4, 5):
            zone_weights[idx] += edge_gain
        zone_weights[6] = max(0.35, zone_weights[6] - (0.28 * self.center_relief_pressure + 0.10 * self.asymmetry_pressure))

        if self.asymmetry_pressure > 0.0:
            skew = min(0.45, 0.08 * self.asymmetry_pressure)
            zone_weights[0] += skew
            zone_weights[2] += skew * 0.85
            zone_weights[4] += skew * 0.65
            zone_weights[1] = max(0.55, zone_weights[1] - skew * 0.30)
            zone_weights[3] = max(0.55, zone_weights[3] - skew * 0.24)

        return {
            "prefer_oblique": _clip_float(self.oblique_pressure + 0.40 * self.angle_diversity_pressure + 0.18 * self.asymmetry_pressure, 0.0, 3.5),
            "prefer_vertical": _clip_float(self.vertical_pressure, 0.0, 3.0),
            "avoid_vertical": _clip_float(self.vertical_excess_pressure + 0.30 * self.angle_diversity_pressure, 0.0, 3.0),
            "diversify_angles": _clip_float(self.angle_diversity_pressure + 0.35 * self.asymmetry_pressure, 0.0, 3.5),
            "olive_macro_target_scale": _clip_float(1.0 + 0.05 * self.olive_pressure + 0.02 * self.boundary_low_pressure, 0.90, 1.30),
            "terre_macro_target_scale": _clip_float(1.0 + 0.04 * self.terre_pressure, 0.88, 1.18),
            "gris_macro_target_scale": _clip_float(1.0 - 0.22 * self.gris_macro_cap_pressure, 0.10, 1.00),
            "transition_terre_bias": _clip_float(0.025 * self.terre_pressure + 0.010 * self.boundary_low_pressure, -0.08, 0.18),
            "transition_olive_bias": _clip_float(0.022 * self.olive_pressure, -0.08, 0.14),
            "transition_coyote_bias": _clip_float(0.030 * self.coyote_deficit_pressure - 0.022 * self.coyote_excess_pressure, -0.10, 0.10),
            "micro_gris_bias": _clip_float(0.028 * self.gris_pressure + 0.015 * self.boundary_low_pressure - 0.030 * self.gris_macro_cap_pressure, -0.05, 0.18),
            "micro_terre_bias": _clip_float(0.020 * self.terre_pressure, -0.04, 0.10),
            "micro_cluster_bonus": int(round(_clip_float(0.40 * self.gris_pressure + 0.35 * self.boundary_low_pressure, 0.0, 2.0))),
            "center_torso_overlap_scale": _clip_float(1.0 - 0.10 * self.center_relief_pressure, 0.55, 1.00),
            "extra_boundary_nudge_passes": int(round(_clip_float(0.60 * self.boundary_low_pressure + 0.20 * self.olive_pressure + 0.15 * self.terre_pressure + 0.15 * self.gris_pressure, 0.0, 10.0))),
            "boundary_sample_ratio_scale": _clip_float(1.0 + 0.08 * self.boundary_low_pressure - 0.05 * self.boundary_high_pressure, 0.75, 1.80),
            "zone_weight_boosts": tuple(_clip_float(v, 0.35, 2.40) for v in zone_weights),
            "consecutive_rejections": int(self.consecutive_rejections),
            "last_rules": list(self.last_rules[:8]),
        }


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


# ============================================================
# OUTILS SYSTÈME
# ============================================================

_PROCESS_POOL: Optional[ProcessPoolExecutor] = None
_PROCESS_POOL_WORKERS: Optional[int] = None


def _worker_initializer() -> None:
    # Évite qu'un worker CPU lance lui-même plusieurs threads BLAS/OpenMP.
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
# PROFIL
# ============================================================

def build_seed(target_index: int, local_attempt: int, base_seed: int = DEFAULT_BASE_SEED) -> int:
    return int(base_seed + target_index * 100000 + local_attempt)


def make_profile(seed: int, adaptive_hint: Optional[Dict[str, Any]] = None) -> VariantProfile:
    rng = random.Random(seed)
    adaptive_hint = dict(adaptive_hint or {})

    angles = BASE_ANGLES[:]
    rng.shuffle(angles)
    allowed = sorted(set([0] + angles[:rng.randint(8, len(BASE_ANGLES))]))

    prefer_oblique = _safe_failure_float(adaptive_hint.get("prefer_oblique"), 0.0)
    prefer_vertical = _safe_failure_float(adaptive_hint.get("prefer_vertical"), 0.0)
    avoid_vertical = _safe_failure_float(adaptive_hint.get("avoid_vertical"), 0.0)
    diversify_angles = _safe_failure_float(adaptive_hint.get("diversify_angles"), 0.0)

    angle_pool: List[int] = list(allowed)
    obliques = [a for a in allowed if a != 0]

    if prefer_oblique > 0.0 and obliques:
        repeats = 1 + int(round(min(5.0, prefer_oblique)))
        angle_pool.extend(obliques * repeats)

    if prefer_vertical > 0.0:
        angle_pool.extend([0] * (1 + int(round(min(4.0, prefer_vertical)))))

    if avoid_vertical > 0.0 and 0 in angle_pool and len(obliques) >= 2:
        removal = min(angle_pool.count(0), max(1, int(round(avoid_vertical))))
        for _ in range(removal):
            try:
                angle_pool.remove(0)
            except ValueError:
                break
        if 0 in allowed and avoid_vertical >= 2.2 and len(allowed) > 6:
            allowed = [a for a in allowed if a != 0]

    if diversify_angles > 0.0 and obliques:
        diversified = sorted(obliques, key=lambda a: (abs(a), a))
        angle_pool.extend(diversified[: min(len(diversified), 6)])
        angle_pool.extend(diversified[-min(len(diversified), 4):])

    angle_pool = angle_pool or list(allowed) or BASE_ANGLES[:]

    zone_weight_boosts_raw = adaptive_hint.get("zone_weight_boosts")
    if isinstance(zone_weight_boosts_raw, (list, tuple)) and zone_weight_boosts_raw:
        zone_weight_boosts = tuple(
            _clip_float(_safe_failure_float(zone_weight_boosts_raw[i] if i < len(zone_weight_boosts_raw) else 1.0, 1.0), 0.35, 2.40)
            for i in range(len(DENSITY_ZONES))
        )
    else:
        zone_weight_boosts = tuple(1.0 for _ in DENSITY_ZONES)

    micro_cluster_bonus = int(round(_clip_float(_safe_failure_float(adaptive_hint.get("micro_cluster_bonus"), 0.0), 0.0, 2.0)))

    return VariantProfile(
        seed=seed,
        allowed_angles=allowed,
        micro_cluster_min=2,
        micro_cluster_max=rng.randint(4, 5) + micro_cluster_bonus,
        macro_width_variation=_clip_float(
            rng.uniform(0.22, 0.30) + 0.008 * _safe_failure_float(adaptive_hint.get("olive_macro_target_scale"), 1.0) - 0.008,
            0.20,
            0.34,
        ),
        macro_lateral_jitter=_clip_float(
            rng.uniform(0.14, 0.21) + 0.010 * prefer_oblique + 0.008 * diversify_angles - 0.010 * prefer_vertical,
            0.12,
            0.26,
        ),
        macro_tip_taper=_clip_float(
            rng.uniform(0.34, 0.43) + 0.012 * prefer_vertical - 0.010 * avoid_vertical,
            0.30,
            0.47,
        ),
        macro_edge_break=_clip_float(
            rng.uniform(0.10, 0.15) + 0.006 * diversify_angles + 0.005 * prefer_oblique,
            0.10,
            0.18,
        ),
        micro_width_variation=_clip_float(
            rng.uniform(0.18, 0.25) + 0.012 * _safe_failure_float(adaptive_hint.get("micro_gris_bias"), 0.0),
            0.16,
            0.30,
        ),
        micro_lateral_jitter=_clip_float(
            rng.uniform(0.12, 0.18) + 0.010 * prefer_oblique + 0.010 * _safe_failure_float(adaptive_hint.get("micro_gris_bias"), 0.0),
            0.10,
            0.22,
        ),
        micro_tip_taper=_clip_float(
            rng.uniform(0.42, 0.52) + 0.010 * prefer_vertical - 0.010 * avoid_vertical,
            0.38,
            0.58,
        ),
        micro_edge_break=_clip_float(
            rng.uniform(0.12, 0.18) + 0.006 * diversify_angles + 0.006 * _safe_failure_float(adaptive_hint.get("micro_gris_bias"), 0.0),
            0.10,
            0.22,
        ),
        angle_pool=tuple(int(a) for a in angle_pool),
        zone_weight_boosts=zone_weight_boosts,
        olive_macro_target_scale=_clip_float(_safe_failure_float(adaptive_hint.get("olive_macro_target_scale"), 1.0), 0.90, 1.30),
        terre_macro_target_scale=_clip_float(_safe_failure_float(adaptive_hint.get("terre_macro_target_scale"), 1.0), 0.88, 1.18),
        gris_macro_target_scale=_clip_float(_safe_failure_float(adaptive_hint.get("gris_macro_target_scale"), 1.0), 0.10, 1.00),
        transition_terre_bias=_clip_float(_safe_failure_float(adaptive_hint.get("transition_terre_bias"), 0.0), -0.08, 0.18),
        transition_olive_bias=_clip_float(_safe_failure_float(adaptive_hint.get("transition_olive_bias"), 0.0), -0.08, 0.14),
        transition_coyote_bias=_clip_float(_safe_failure_float(adaptive_hint.get("transition_coyote_bias"), 0.0), -0.10, 0.10),
        micro_gris_bias=_clip_float(_safe_failure_float(adaptive_hint.get("micro_gris_bias"), 0.0), -0.05, 0.18),
        micro_terre_bias=_clip_float(_safe_failure_float(adaptive_hint.get("micro_terre_bias"), 0.0), -0.04, 0.10),
        center_torso_overlap_scale=_clip_float(_safe_failure_float(adaptive_hint.get("center_torso_overlap_scale"), 1.0), 0.55, 1.00),
        extra_boundary_nudge_passes=int(round(_clip_float(_safe_failure_float(adaptive_hint.get("extra_boundary_nudge_passes"), 0.0), 0.0, 10.0))),
        boundary_sample_ratio_scale=_clip_float(_safe_failure_float(adaptive_hint.get("boundary_sample_ratio_scale"), 1.0), 0.75, 1.80),
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
            boost = _clip_float(_safe_failure_float(zone_weight_boosts[i], 1.0), 0.35, 2.40)
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


def local_color_variety(canvas: np.ndarray, x: int, y: int, radius: int = 2) -> int:
    h, w = canvas.shape
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    return len(np.unique(canvas[y1:y2, x1:x2]))


def downsample_nearest(canvas: np.ndarray, factor: int) -> np.ndarray:
    return canvas[::factor, ::factor]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clip_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_failure_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


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


def infer_origin_from_neighbors(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    x: int,
    y: int,
    chosen_color: int,
    fallback_origin: int,
) -> int:
    h, w = canvas.shape
    y1, y2 = max(0, y - 2), min(h, y + 3)
    x1, x2 = max(0, x - 2), min(w, x + 3)

    neigh_colors = canvas[y1:y2, x1:x2]
    neigh_origins = origin_map[y1:y2, x1:x2]

    same = neigh_colors == chosen_color
    if not np.any(same):
        return fallback_origin

    vals = neigh_origins[same]
    if vals.size == 0:
        return fallback_origin

    counts = np.bincount(vals.astype(int), minlength=4)
    return int(np.argmax(counts))


# ============================================================
# ZONES ANATOMIQUES
# ============================================================

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


def rect_mask(x1: float, x2: float, y1: float, y2: float) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    xa = int(WIDTH * x1)
    xb = int(WIDTH * x2)
    ya = int(HEIGHT * y1)
    yb = int(HEIGHT * y2)
    mask[ya:yb, xa:xb] = True
    return mask


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
        row = mask[y]
        row_visited = visited[y]

        for x in range(w):
            if not row[x] or row_visited[x]:
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
    if not allowed:
        allowed = BASE_ANGLES[:]

    counts = macro_angle_histogram(macros)
    total = len(macros)
    vertical_count = counts.get(0, 0)
    vertical_share = (vertical_count / total) if total else 0.0

    obliques = [a for a in allowed if a != 0]
    if not obliques:
        obliques = [a for a in BASE_ANGLES if a != 0]

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


def preferred_boundary_coordinates(boundary: np.ndarray, preferred_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if preferred_mask is not None:
        ys, xs = np.where(boundary & preferred_mask)
        if len(xs) > 0:
            return ys, xs
    ys, xs = np.where(boundary)
    return ys, xs


def choose_semantic_color_for_origin(deficits: np.ndarray, origin_code: int, current_color: int) -> Optional[int]:
    allowed = SEMANTIC_ALLOWED_COLORS.get(int(origin_code), (IDX_COYOTE, IDX_OLIVE, IDX_TERRE, IDX_GRIS))
    candidates = [c for c in allowed if c != current_color and deficits[c] > 0.0]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (float(deficits[c]), c), reverse=True)
    return int(candidates[0])


def semantic_color_allowed(origin_code: int, color_idx: int) -> bool:
    return int(color_idx) in SEMANTIC_ALLOWED_COLORS.get(int(origin_code), ())
def visible_origin_shares(canvas: np.ndarray, origin_map: np.ndarray) -> Dict[str, float]:
    out = {}

    for color_idx, name in enumerate(COLOR_NAMES):
        mask = canvas == color_idx
        n = int(mask.sum())

        if n == 0:
            out[f"{name}_macro_share"] = 0.0
            out[f"{name}_transition_share"] = 0.0
            out[f"{name}_micro_share"] = 0.0
            continue

        origins = origin_map[mask]
        out[f"{name}_macro_share"] = float(np.mean(origins == ORIGIN_MACRO))
        out[f"{name}_transition_share"] = float(np.mean(origins == ORIGIN_TRANSITION))
        out[f"{name}_micro_share"] = float(np.mean(origins == ORIGIN_MICRO))

    return out


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
    # Le repère local est construit avec l'axe de longueur sur Y : 0° doit donc rester vertical.
    rot = [rotate(x, y, angle_from_vertical_deg) for x, y in poly]
    return [(cx + x, cy + y) for x, y in rot]


def attached_transition(
    rng: random.Random,
    parent: Sequence[Tuple[float, float]],
    length_px: float,
    width_px: float,
) -> List[Tuple[float, float]]:
    i = rng.randint(0, len(parent) - 1)
    p1 = parent[i]
    p2 = parent[(i + 1) % len(parent)]

    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2

    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    n = max(1.0, math.hypot(vx, vy))
    tx, ty = vx / n, vy / n
    nx, ny = -ty, tx

    if rng.random() < 0.5:
        nx, ny = -nx, -ny

    base = width_px * rng.uniform(0.45, 0.70)
    tip = length_px

    a = (mx - tx * base, my - ty * base)
    b = (mx + tx * base, my + ty * base)
    c = (mx + nx * tip, my + ny * tip)
    d1 = (
        mx - tx * (base * 0.95) + nx * (tip * 0.42),
        my - ty * (base * 0.95) + ny * (tip * 0.42),
    )
    d2 = (
        mx + tx * (base * 0.80) + nx * (tip * 0.74),
        my + ty * (base * 0.80) + ny * (tip * 0.74),
    )
    return [a, d1, c, d2, b]


# ============================================================
# CONTRÔLES STRUCTURELS
# ============================================================

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


def transition_is_attached(
    parent_mask: np.ndarray,
    transition_mask: np.ndarray,
    min_touch_pixels: int = MIN_TRANSITION_TOUCH_PIXELS,
) -> bool:
    contact = dilate_mask(parent_mask, radius=1) & transition_mask
    return int(contact.sum()) >= min_touch_pixels


def micro_is_on_boundary(
    boundary: np.ndarray,
    micro_mask: np.ndarray,
    min_boundary_coverage: float = MIN_MICRO_BOUNDARY_COVERAGE,
) -> bool:
    area = int(micro_mask.sum())
    if area == 0:
        return False
    cov = float(np.mean(boundary[micro_mask]))
    return cov >= min_boundary_coverage


def creates_new_mass(
    canvas: np.ndarray,
    new_mask: np.ndarray,
    color_idx: int,
    local_radius: int = 45,
    max_local_area_ratio: float = MAX_LOCAL_MASS_RATIO_TRANSITION,
) -> bool:
    h, w = canvas.shape
    ys, xs = np.where(new_mask)
    if len(xs) == 0:
        return False

    x1 = max(0, int(xs.min()) - local_radius)
    x2 = min(w, int(xs.max()) + local_radius + 1)
    y1 = max(0, int(ys.min()) - local_radius)
    y2 = min(h, int(ys.max()) + local_radius + 1)

    local_existing = canvas[y1:y2, x1:x2]
    local_new = new_mask[y1:y2, x1:x2]
    local_ratio_after = float(np.mean((local_existing == color_idx) | local_new))
    return local_ratio_after > max_local_area_ratio


# ============================================================
# COUCHES
# ============================================================

def apply_mask(canvas: np.ndarray, origin_map: np.ndarray, mask: np.ndarray, color_idx: int, origin_code: int) -> None:
    canvas[mask] = color_idx
    origin_map[mask] = origin_code



def add_macros(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    profile: VariantProfile,
    rng: random.Random,
) -> List[MacroRecord]:
    macros: List[MacroRecord] = []

    target_macro_olive_pixels = int(VISIBLE_MACRO_OLIVE_TARGET * canvas.size * profile.olive_macro_target_scale)
    target_macro_terre_pixels = int(VISIBLE_MACRO_TERRE_TARGET * canvas.size * profile.terre_macro_target_scale)
    target_macro_gris_pixels = int(VISIBLE_MACRO_GRIS_TARGET * canvas.size * profile.gris_macro_target_scale)

    olive_attempts = 0
    while int(np.sum((canvas == IDX_OLIVE) & (origin_map == ORIGIN_MACRO))) < target_macro_olive_pixels:
        olive_attempts += 1
        if olive_attempts > MAX_MACRO_PLACEMENT_ATTEMPTS_OLIVE:
            break

        cx, cy = choose_biased_center(rng, profile.zone_weight_boosts)
        angle = pick_macro_angle(macros, profile, rng, force_vertical_floor=True)

        if local_parallel_conflict(macros, (cx, cy), angle):
            continue

        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(*MACRO_LENGTH_CM)),
            width_px=cm_to_px(rng.uniform(*MACRO_WIDTH_CM)),
            angle_from_vertical_deg=angle,
            segments=rng.randint(7, 10),
            width_variation=profile.macro_width_variation,
            lateral_jitter=profile.macro_lateral_jitter,
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue

        if float(np.mean(canvas[mask] == IDX_OLIVE)) > 0.45:
            continue

        zc = macro_zone_count(mask)
        if zc < 2 and rng.random() < 0.75:
            continue

        max_center_overlap = MAX_CENTER_TORSO_OVERLAP_MACRO * profile.center_torso_overlap_scale
        if angle == 0:
            max_center_overlap = min(0.42, max_center_overlap + 0.05)
        if zone_overlap_ratio(mask, ANATOMY_ZONES["center_torso"]) > max_center_overlap:
            continue

        apply_mask(canvas, origin_map, mask, IDX_OLIVE, ORIGIN_MACRO)
        macros.append(MacroRecord(IDX_OLIVE, poly, angle, (cx, cy), mask, zc))

    terre_attempts = 0
    while int(np.sum((canvas == IDX_TERRE) & (origin_map == ORIGIN_MACRO))) < target_macro_terre_pixels:
        terre_attempts += 1
        if terre_attempts > MAX_MACRO_PLACEMENT_ATTEMPTS_TERRE:
            break
        cx, cy = choose_biased_center(rng, profile.zone_weight_boosts)
        angle = pick_macro_angle(macros, profile, rng, force_vertical_floor=False)

        if local_parallel_conflict(macros, (cx, cy), angle, dist_threshold_px=220, angle_threshold_deg=6):
            continue

        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(40, 66)),
            width_px=cm_to_px(rng.uniform(12, 24)),
            angle_from_vertical_deg=angle,
            segments=rng.randint(6, 9),
            width_variation=max(0.18, profile.macro_width_variation - 0.03),
            lateral_jitter=max(0.12, profile.macro_lateral_jitter - 0.02),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue

        cur = canvas[mask]
        if float(np.mean(np.isin(cur, [IDX_COYOTE, IDX_OLIVE]))) < 0.65:
            continue

        max_center_overlap = MAX_CENTER_TORSO_OVERLAP_TERRAIN_MACRO * profile.center_torso_overlap_scale
        if zone_overlap_ratio(mask, ANATOMY_ZONES["center_torso"]) > max_center_overlap:
            continue

        zc = macro_zone_count(mask)
        if zc < 1 and rng.random() < 0.65:
            continue

        apply_mask(canvas, origin_map, mask, IDX_TERRE, ORIGIN_MACRO)
        macros.append(MacroRecord(IDX_TERRE, poly, angle, (cx, cy), mask, zc))

    gris_attempts = 0
    while int(np.sum((canvas == IDX_GRIS) & (origin_map == ORIGIN_MACRO))) < target_macro_gris_pixels:
        gris_attempts += 1
        if gris_attempts > MAX_MACRO_PLACEMENT_ATTEMPTS_GRIS:
            break
        cx, cy = choose_biased_center(rng, profile.zone_weight_boosts)
        angle = pick_macro_angle(macros, profile, rng, force_vertical_floor=False)

        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(34, 46)),
            width_px=cm_to_px(rng.uniform(9, 14)),
            angle_from_vertical_deg=angle,
            segments=rng.randint(5, 7),
            width_variation=max(0.15, profile.macro_width_variation - 0.08),
            lateral_jitter=max(0.10, profile.macro_lateral_jitter - 0.05),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue

        cur = canvas[mask]
        if float(np.mean(np.isin(cur, [IDX_OLIVE, IDX_TERRE]))) < 0.55:
            continue

        if zone_overlap_ratio(mask, ANATOMY_ZONES["center_torso"]) > 0.18:
            continue

        zc = macro_zone_count(mask)
        apply_mask(canvas, origin_map, mask, IDX_GRIS, ORIGIN_MACRO)
        macros.append(MacroRecord(IDX_GRIS, poly, angle, (cx, cy), mask, zc))

    return macros

def add_transitions(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: Sequence[MacroRecord],
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    if not macros:
        return

    placement_attempts = 0
    while True:
        placement_attempts += 1
        if placement_attempts > MAX_TRANSITION_PLACEMENTS:
            break
        rs = compute_ratios(canvas)
        visible = visible_origin_shares(canvas, origin_map)
        spatial = spatial_discipline_metrics(canvas)

        enough_terre = rs[IDX_TERRE] >= VISIBLE_TOTAL_TERRE_TARGET
        enough_olive = rs[IDX_OLIVE] >= VISIBLE_TOTAL_OLIVE_TARGET * 0.985
        enough_trans_share = visible["terre_de_france_transition_share"] >= MIN_VISIBLE_TERRE_TRANS_SHARE
        enough_periphery = (
            spatial["periphery_boundary_density_ratio"] >= MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO
            and spatial["periphery_non_coyote_ratio"] >= MIN_PERIPHERY_NON_COYOTE_RATIO
        )

        if enough_terre and enough_olive and enough_trans_share and enough_periphery:
            break

        prefer_periphery = not enough_periphery
        parent_pool = [m for m in macros if m.color_idx in (IDX_OLIVE, IDX_TERRE)]
        if prefer_periphery:
            periphery_pool = [m for m in parent_pool if float(np.mean(HIGH_DENSITY_ZONE_MASK[m.mask])) >= 0.40]
            if periphery_pool:
                parent_pool = periphery_pool

        parent = rng.choice(parent_pool or list(macros))

        poly = attached_transition(
            rng=rng,
            parent=parent.poly,
            length_px=cm_to_px(rng.uniform(*TRANSITION_LENGTH_CM)),
            width_px=cm_to_px(rng.uniform(*TRANSITION_WIDTH_CM)),
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue

        if prefer_periphery and float(np.mean(HIGH_DENSITY_ZONE_MASK[mask])) < 0.35:
            continue

        if not transition_is_attached(parent.mask, mask):
            continue

        cur = canvas[mask]
        deficits = TARGET - rs
        terre_need = max(0.0, deficits[IDX_TERRE])
        olive_need = max(0.0, deficits[IDX_OLIVE])
        coyote_need = max(0.0, deficits[IDX_COYOTE])

        if terre_need >= olive_need:
            p_terre, p_olive, p_coyote = 0.74, 0.18, 0.08
        else:
            p_terre, p_olive, p_coyote = 0.66, 0.24, 0.10

        if prefer_periphery:
            p_terre += 0.04
            p_olive += 0.03
            p_coyote -= 0.07

        p_terre += profile.transition_terre_bias
        p_olive += profile.transition_olive_bias
        p_coyote += profile.transition_coyote_bias

        if coyote_need > max(terre_need, olive_need) * 1.4:
            p_coyote += 0.06
            p_terre -= 0.04
            p_olive -= 0.02

        p_terre = max(0.05, p_terre)
        p_olive = max(0.05, p_olive)
        p_coyote = max(0.03, p_coyote)
        total_p = p_terre + p_olive + p_coyote
        p_terre, p_olive, p_coyote = (p_terre / total_p, p_olive / total_p, p_coyote / total_p)

        r = rng.random()
        if r < p_terre:
            color = IDX_TERRE
        elif r < p_terre + p_olive:
            color = IDX_OLIVE
        else:
            color = IDX_COYOTE

        if color == IDX_TERRE and float(np.mean(cur != IDX_COYOTE)) < 0.18:
            continue

        if creates_new_mass(canvas, mask, color, local_radius=38, max_local_area_ratio=0.74):
            continue

        if color == IDX_COYOTE and float(np.mean(cur == IDX_COYOTE)) > 0.90:
            continue

        apply_mask(canvas, origin_map, mask, color, ORIGIN_TRANSITION)

def add_micro_clusters(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    placement_attempts = 0
    stalled_rounds = 0

    while True:
        placement_attempts += 1
        if placement_attempts > MAX_MICRO_CLUSTER_PLACEMENTS:
            break

        rs = compute_ratios(canvas)
        visible = visible_origin_shares(canvas, origin_map)
        spatial = spatial_discipline_metrics(canvas)
        center_empty = center_empty_ratio(canvas)

        enough_gris = rs[IDX_GRIS] >= VISIBLE_TOTAL_GRIS_TARGET
        enough_terre = rs[IDX_TERRE] >= VISIBLE_TOTAL_TERRE_TARGET
        enough_micro_share = visible["vert_de_gris_micro_share"] >= MIN_VISIBLE_GRIS_MICRO_SHARE
        enough_periphery = (
            spatial["periphery_boundary_density_ratio"] >= MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO
            and spatial["periphery_non_coyote_ratio"] >= MIN_PERIPHERY_NON_COYOTE_RATIO
        )

        if enough_gris and enough_terre and enough_micro_share and enough_periphery and center_empty <= MAX_COYOTE_CENTER_EMPTY_RATIO:
            break

        boundary = compute_boundary_mask(canvas)
        prefer_periphery = not enough_periphery
        prefer_center = (not prefer_periphery) and (center_empty > MAX_COYOTE_CENTER_EMPTY_RATIO)

        preferred_mask = None
        if prefer_periphery:
            preferred_mask = HIGH_DENSITY_ZONE_MASK
        elif prefer_center:
            preferred_mask = CENTER_TORSO_MASK

        ys, xs = preferred_boundary_coordinates(boundary, preferred_mask=preferred_mask)
        if len(xs) == 0:
            break

        idx = rng.randint(0, len(xs) - 1)
        bx, by = int(xs[idx]), int(ys[idx])

        cluster_count = rng.randint(profile.micro_cluster_min, max(profile.micro_cluster_min, profile.micro_cluster_max))
        placed_any = False

        for _ in range(cluster_count):
            ox = bx + rng.randint(-8, 8)
            oy = by + rng.randint(-8, 8)

            if not (0 <= ox < WIDTH and 0 <= oy < HEIGHT):
                continue

            size = cm_to_px(rng.uniform(*MICRO_SIZE_CM))
            poly = jagged_spine_poly(
                rng=rng,
                cx=ox,
                cy=oy,
                length_px=size * rng.uniform(1.2, 2.0),
                width_px=size * rng.uniform(0.45, 0.90),
                angle_from_vertical_deg=rng.choice(profile.angle_pool or tuple(profile.allowed_angles)),
                segments=rng.randint(4, 6),
                width_variation=profile.micro_width_variation,
                lateral_jitter=profile.micro_lateral_jitter,
                tip_taper=profile.micro_tip_taper,
                edge_break=profile.micro_edge_break,
            )
            mask = polygon_mask(poly)
            if mask.sum() == 0:
                continue

            if preferred_mask is not None and float(np.mean(preferred_mask[mask])) < 0.30:
                continue

            cur = canvas[mask]
            if len(np.unique(cur)) < 2:
                continue

            if not micro_is_on_boundary(boundary, mask):
                continue

            rs_now = compute_ratios(canvas)
            deficits = TARGET - rs_now
            gris_need = max(0.0, deficits[IDX_GRIS])
            terre_need = max(0.0, deficits[IDX_TERRE])

            p_gris = 0.88 if gris_need >= terre_need else 0.82
            p_gris += profile.micro_gris_bias
            p_terre = (1.0 - p_gris) + profile.micro_terre_bias

            if prefer_periphery:
                p_gris += 0.03
                p_terre += 0.02
            if prefer_center:
                p_terre += 0.05

            total_p = max(0.10, p_gris) + max(0.10, p_terre)
            p_gris = max(0.10, p_gris) / total_p

            color = IDX_GRIS if rng.random() < p_gris else IDX_TERRE

            apply_mask(canvas, origin_map, mask, color, ORIGIN_MICRO)
            placed_any = True

        if placed_any:
            stalled_rounds = 0
        else:
            stalled_rounds += 1
            if stalled_rounds >= 20 and enough_gris and enough_micro_share:
                break

def nudge_proportions(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    profile_or_rng: VariantProfile | random.Random,
    rng: Optional[random.Random] = None,
) -> None:
    if isinstance(profile_or_rng, VariantProfile):
        profile = profile_or_rng
        local_rng = rng if rng is not None else random.Random(profile.seed ^ 0x5A17)
    else:
        profile = VariantProfile(
            seed=0,
            allowed_angles=list(BASE_ANGLES),
            micro_cluster_min=2,
            micro_cluster_max=4,
            macro_width_variation=0.24,
            macro_lateral_jitter=0.16,
            macro_tip_taper=0.40,
            macro_edge_break=0.12,
            micro_width_variation=0.20,
            micro_lateral_jitter=0.14,
            micro_tip_taper=0.48,
            micro_edge_break=0.14,
        )
        local_rng = profile_or_rng

    h, w = canvas.shape
    nudge_passes = max(1, int(BOUNDARY_NUDGE_PASSES + profile.extra_boundary_nudge_passes))
    sample_ratio = _clip_float(BOUNDARY_NUDGE_SAMPLE_RATIO * profile.boundary_sample_ratio_scale, 0.0015, 0.0085)

    for _ in range(nudge_passes):
        rs = compute_ratios(canvas)
        deficits = TARGET - rs

        if np.max(np.abs(deficits)) < 0.0055:
            break

        boundary = compute_boundary_mask(canvas)
        ys, xs = np.where(boundary)
        if len(xs) == 0:
            break

        sample_count = min(len(xs), int(canvas.size * sample_ratio))
        if sample_count <= 0:
            break

        picks = local_rng.sample(range(len(xs)), k=sample_count)

        for j in picks:
            x = int(xs[j])
            y = int(ys[j])
            current = int(canvas[y, x])
            origin = int(origin_map[y, x])

            allowed = SEMANTIC_ALLOWED_COLORS.get(origin, (current,))
            if current not in allowed:
                continue

            chosen = choose_semantic_color_for_origin(deficits, origin, current)
            if chosen is None:
                continue

            if deficits[current] > -0.002 and semantic_color_allowed(origin, current):
                continue

            y1, y2 = max(0, y - 2), min(h, y + 3)
            x1, x2 = max(0, x - 2), min(w, x + 3)
            neigh = canvas[y1:y2, x1:x2]

            if chosen not in neigh and local_rng.random() < 0.60:
                continue

            if origin == ORIGIN_MACRO and chosen == IDX_TERRE and local_color_variety(canvas, x, y, radius=2) < 2:
                continue
            if origin == ORIGIN_MICRO and chosen == IDX_GRIS and local_color_variety(canvas, x, y, radius=2) < 2:
                continue

            if origin == ORIGIN_BACKGROUND and chosen not in (IDX_COYOTE, IDX_OLIVE):
                continue
            if origin == ORIGIN_MACRO and chosen not in (IDX_OLIVE, IDX_TERRE):
                continue
            if origin == ORIGIN_TRANSITION and chosen not in (IDX_COYOTE, IDX_OLIVE, IDX_TERRE):
                continue
            if origin == ORIGIN_MICRO and chosen not in (IDX_TERRE, IDX_GRIS):
                continue

            canvas[y, x] = chosen

# ============================================================
# ============================================================
# MÉTRIQUES MULTI-ÉCHELLE
# ============================================================

def multiscale_metrics(canvas: np.ndarray) -> Dict[str, float]:
    small = downsample_nearest(canvas, 4)
    tiny = downsample_nearest(canvas, 8)

    return {
        "boundary_density_small": boundary_density(small),
        "boundary_density_tiny": boundary_density(tiny),
        "center_empty_ratio_small": center_empty_ratio_upscaled_proxy(small),
        "largest_olive_component_ratio_small": largest_component_ratio(small == IDX_OLIVE),
    }


def center_empty_ratio_upscaled_proxy(canvas_small: np.ndarray) -> float:
    h, w = canvas_small.shape
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.18)
    y2 = int(h * 0.62)
    zone = canvas_small[y1:y2, x1:x2]
    return float(np.mean(zone == IDX_COYOTE))


# ============================================================
# VALIDATION VISUELLE / PERCEPTIVE
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
        clamp01(1.0 - metrics["center_empty_ratio"] / 0.60),
        clamp01(1.0 - metrics["mirror_similarity"] / 0.90),
        clamp01((metrics["olive_multizone_share"] - 0.25) / 0.45),
        clamp01(1.0 - abs(metrics["boundary_density"] - 0.14) / 0.12),
        clamp01((metrics["vert_olive_macro_share"] - 0.45) / 0.30),
        clamp01((metrics["terre_de_france_transition_share"] - 0.20) / 0.30),
        clamp01((metrics["vert_de_gris_micro_share"] - 0.50) / 0.25),
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


def absolute_origin_color_ratios(canvas: np.ndarray, origin_map: np.ndarray) -> Dict[str, float]:
    total = float(canvas.size)
    return {
        "macro_olive_visible_ratio": float(np.sum((canvas == IDX_OLIVE) & (origin_map == ORIGIN_MACRO)) / total),
        "macro_terre_visible_ratio": float(np.sum((canvas == IDX_TERRE) & (origin_map == ORIGIN_MACRO)) / total),
        "macro_gris_visible_ratio": float(np.sum((canvas == IDX_GRIS) & (origin_map == ORIGIN_MACRO)) / total),
        "transition_terre_visible_ratio": float(np.sum((canvas == IDX_TERRE) & (origin_map == ORIGIN_TRANSITION)) / total),
        "micro_gris_visible_ratio": float(np.sum((canvas == IDX_GRIS) & (origin_map == ORIGIN_MICRO)) / total),
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


def military_visual_discipline_score(metrics: Dict[str, float]) -> Dict[str, float]:
    parts = [
        clamp01((metrics.get("visual_score_final", 0.0) - 0.45) / 0.30),
        clamp01((metrics.get("periphery_boundary_density_ratio", 0.0) - 0.95) / 0.45),
        clamp01((metrics.get("periphery_non_coyote_ratio", 0.0) - 0.95) / 0.45),
        clamp01((metrics.get("macro_olive_visible_ratio", 0.0) - 0.10) / 0.12),
        clamp01(1.0 - metrics.get("macro_terre_visible_ratio", 1.0) / 0.10),
        clamp01(1.0 - metrics.get("macro_gris_visible_ratio", 1.0) / 0.03),
        clamp01((metrics.get("oblique_share", 0.0) - 0.55) / 0.25),
        clamp01(1.0 - metrics.get("mirror_similarity", 1.0) / 0.85),
    ]
    score = float(np.mean(parts))
    return {
        "visual_military_score": score,
        "visual_military_passed": 1.0 if score >= VISUAL_MIN_MILITARY_SCORE else 0.0,
    }


# ============================================================
# DIAGNOSTIC DES REJETS + ADAPTATION EN DIRECT
# ============================================================

_ADAPTIVE_PROFILE_OVERRIDES: Dict[int, Dict[str, Any]] = {}
_LOG_MODULE_CACHE: Any = None
_LOG_MODULE_ATTEMPTED = False


def _register_profile_override(seed: int, adaptive_hint: Optional[Dict[str, Any]]) -> None:
    if adaptive_hint:
        _ADAPTIVE_PROFILE_OVERRIDES[int(seed)] = dict(adaptive_hint)


def _consume_profile_override(seed: int) -> Optional[Dict[str, Any]]:
    return _ADAPTIVE_PROFILE_OVERRIDES.pop(int(seed), None)


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
    log_event = getattr(mod, "log_event", None) if mod is not None else None
    if callable(log_event):
        try:
            log_event(level, source, message, **payload)
        except Exception:
            pass


def _failure_payload(rule: str, actual: float, delta: float, min_value: Optional[float] = None, max_value: Optional[float] = None, target: Optional[float] = None) -> Dict[str, Any]:
    return {
        "rule": str(rule),
        "actual": float(actual),
        "delta": float(delta),
        "min_value": None if min_value is None else float(min_value),
        "max_value": None if max_value is None else float(max_value),
        "target": None if target is None else float(target),
    }


def _append_failure_range(failures: List[Dict[str, Any]], rule: str, actual: float, min_value: float, max_value: float) -> None:
    if actual < min_value:
        failures.append(_failure_payload(rule, actual, min_value - actual, min_value=min_value, max_value=max_value))
    elif actual > max_value:
        failures.append(_failure_payload(rule, actual, actual - max_value, min_value=min_value, max_value=max_value))


def _append_failure_min(failures: List[Dict[str, Any]], rule: str, actual: float, min_value: float) -> None:
    if actual < min_value:
        failures.append(_failure_payload(rule, actual, min_value - actual, min_value=min_value))


def _append_failure_max(failures: List[Dict[str, Any]], rule: str, actual: float, max_value: float) -> None:
    if actual > max_value:
        failures.append(_failure_payload(rule, actual, actual - max_value, max_value=max_value))


def _append_failure_abs_target(failures: List[Dict[str, Any]], rule: str, actual: float, target: float, max_abs_error: float) -> None:
    delta = abs(actual - target)
    if delta > max_abs_error:
        failures.append(_failure_payload(rule, actual, delta - max_abs_error, max_value=max_abs_error, target=target))


def _fallback_analyze_candidate_failures(candidate: CandidateResult) -> List[Dict[str, Any]]:
    rs = np.asarray(candidate.ratios, dtype=float)
    m = dict(candidate.metrics)
    failures: List[Dict[str, Any]] = []

    abs_err = np.abs(rs - TARGET)
    mean_abs_err = float(np.mean(abs_err))

    _append_failure_abs_target(failures, "abs_err_coyote", float(rs[IDX_COYOTE]), float(TARGET[IDX_COYOTE]), float(MAX_ABS_ERROR_PER_COLOR[IDX_COYOTE]))
    _append_failure_abs_target(failures, "abs_err_olive", float(rs[IDX_OLIVE]), float(TARGET[IDX_OLIVE]), float(MAX_ABS_ERROR_PER_COLOR[IDX_OLIVE]))
    _append_failure_abs_target(failures, "abs_err_terre", float(rs[IDX_TERRE]), float(TARGET[IDX_TERRE]), float(MAX_ABS_ERROR_PER_COLOR[IDX_TERRE]))
    _append_failure_abs_target(failures, "abs_err_gris", float(rs[IDX_GRIS]), float(TARGET[IDX_GRIS]), float(MAX_ABS_ERROR_PER_COLOR[IDX_GRIS]))
    _append_failure_max(failures, "mean_abs_error", mean_abs_err, float(MAX_MEAN_ABS_ERROR))

    _append_failure_range(failures, "ratio_coyote", float(rs[IDX_COYOTE]), 0.27, 0.37)
    _append_failure_range(failures, "ratio_olive", float(rs[IDX_OLIVE]), 0.24, 0.33)
    _append_failure_range(failures, "ratio_terre", float(rs[IDX_TERRE]), 0.19, 0.26)
    _append_failure_range(failures, "ratio_gris", float(rs[IDX_GRIS]), 0.14, 0.21)

    _append_failure_min(failures, "largest_olive_component_ratio", _safe_failure_float(m.get("largest_olive_component_ratio")), float(MIN_OLIVE_CONNECTED_COMPONENT_RATIO))
    _append_failure_min(failures, "largest_olive_component_ratio_small", _safe_failure_float(m.get("largest_olive_component_ratio_small")), 0.12)
    _append_failure_min(failures, "olive_multizone_share", _safe_failure_float(m.get("olive_multizone_share")), float(MIN_OLIVE_MULTIZONE_SHARE))
    _append_failure_min(failures, "boundary_density", _safe_failure_float(m.get("boundary_density")), float(MIN_BOUNDARY_DENSITY))
    _append_failure_min(failures, "boundary_density_small", _safe_failure_float(m.get("boundary_density_small")), float(MIN_BOUNDARY_DENSITY_SMALL))
    _append_failure_min(failures, "oblique_share", _safe_failure_float(m.get("oblique_share")), float(MIN_OBLIQUE_SHARE))
    _append_failure_min(failures, "vert_olive_macro_share", _safe_failure_float(m.get("vert_olive_macro_share")), float(MIN_VISIBLE_OLIVE_MACRO_SHARE))
    _append_failure_min(failures, "terre_de_france_transition_share", _safe_failure_float(m.get("terre_de_france_transition_share")), float(MIN_VISIBLE_TERRE_TRANS_SHARE))
    _append_failure_min(failures, "vert_de_gris_micro_share", _safe_failure_float(m.get("vert_de_gris_micro_share")), float(MIN_VISIBLE_GRIS_MICRO_SHARE))

    _append_failure_max(failures, "center_empty_ratio", _safe_failure_float(m.get("center_empty_ratio")), float(MAX_COYOTE_CENTER_EMPTY_RATIO))
    _append_failure_max(failures, "center_empty_ratio_small", _safe_failure_float(m.get("center_empty_ratio_small")), float(MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL))
    _append_failure_max(failures, "boundary_density", _safe_failure_float(m.get("boundary_density")), float(MAX_BOUNDARY_DENSITY))
    _append_failure_max(failures, "boundary_density_small", _safe_failure_float(m.get("boundary_density_small")), float(MAX_BOUNDARY_DENSITY_SMALL))
    _append_failure_max(failures, "mirror_similarity", _safe_failure_float(m.get("mirror_similarity")), float(MAX_MIRROR_SIMILARITY))
    _append_failure_max(failures, "angle_dominance_ratio", _safe_failure_float(m.get("angle_dominance_ratio")), float(MAX_ANGLE_DOMINANCE_RATIO))
    _append_failure_max(failures, "vert_de_gris_macro_share", _safe_failure_float(m.get("vert_de_gris_macro_share")), float(MAX_VISIBLE_GRIS_MACRO_SHARE))
    _append_failure_range(failures, "vertical_share", _safe_failure_float(m.get("vertical_share")), float(MIN_VERTICAL_SHARE), float(MAX_VERTICAL_SHARE))

    if "macro_olive_visible_ratio" in m:
        _append_failure_min(failures, "macro_olive_visible_ratio", _safe_failure_float(m.get("macro_olive_visible_ratio")), float(MIN_MACRO_OLIVE_VISIBLE_RATIO))
    if "macro_gris_visible_ratio" in m:
        _append_failure_max(failures, "macro_gris_visible_ratio", _safe_failure_float(m.get("macro_gris_visible_ratio")), float(MAX_MACRO_GRIS_VISIBLE_RATIO))

    return failures


def extract_rejection_failures(candidate: CandidateResult, target_index: int, local_attempt: int) -> List[Dict[str, Any]]:
    mod = _get_log_module()
    analyze = getattr(mod, "analyze_candidate", None) if mod is not None else None

    if callable(analyze):
        try:
            diagnostic = analyze(candidate, target_index=target_index, local_attempt=local_attempt)
            failures = []
            for failure in getattr(diagnostic, "failures", []) or []:
                failures.append(
                    {
                        "rule": str(getattr(failure, "rule", "")),
                        "actual": _safe_failure_float(getattr(failure, "actual", 0.0)),
                        "target": getattr(failure, "target", None),
                        "min_value": getattr(failure, "min_value", None),
                        "max_value": getattr(failure, "max_value", None),
                        "delta": _safe_failure_float(getattr(failure, "delta", 0.0)),
                    }
                )
            return failures
        except Exception:
            pass

    return _fallback_analyze_candidate_failures(candidate)


def build_adaptive_hint(state: AdaptiveGenerationState) -> Dict[str, Any]:
    return state.to_hint()


def generate_candidate_from_seed_with_hint(seed: int, adaptive_hint: Optional[Dict[str, Any]] = None) -> CandidateResult:
    profile = make_profile(seed, adaptive_hint=adaptive_hint)
    image, ratios, metrics = generate_one_variant(profile)
    return CandidateResult(
        seed=seed,
        profile=profile,
        image=image,
        ratios=ratios,
        metrics=metrics,
    )


def generate_and_validate_from_seed_with_hint(seed: int, adaptive_hint: Optional[Dict[str, Any]] = None) -> Tuple[CandidateResult, bool]:
    candidate = generate_candidate_from_seed_with_hint(seed, adaptive_hint=adaptive_hint)
    accepted = validate_candidate_result(candidate)
    return candidate, accepted


def update_adaptive_state_after_attempt(
    state: Optional[AdaptiveGenerationState],
    candidate: CandidateResult,
    accepted: bool,
    target_index: int,
    local_attempt: int,
) -> List[Dict[str, Any]]:
    if state is None:
        return []

    if accepted:
        state.register_success()
        return []

    failures = extract_rejection_failures(candidate, target_index=target_index, local_attempt=local_attempt)
    state.register_failures(failures)

    hint = state.to_hint()
    _runtime_log(
        "WARNING",
        "main_adaptive",
        "Rejet analysé et réglages ajustés",
        target_index=target_index,
        local_attempt=local_attempt,
        seed=int(candidate.seed),
        fail_rules=[item.get("rule") for item in failures],
        consecutive_rejections=state.consecutive_rejections,
        adaptive_hint={
            "prefer_oblique": hint.get("prefer_oblique"),
            "prefer_vertical": hint.get("prefer_vertical"),
            "avoid_vertical": hint.get("avoid_vertical"),
            "olive_macro_target_scale": hint.get("olive_macro_target_scale"),
            "terre_macro_target_scale": hint.get("terre_macro_target_scale"),
            "micro_gris_bias": hint.get("micro_gris_bias"),
            "extra_boundary_nudge_passes": hint.get("extra_boundary_nudge_passes"),
        },
    )
    return failures


# ============================================================
# GÉNÉRATION D'UNE VARIANTE
# ============================================================

def repair_spatial_discipline(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    profile: VariantProfile,
    rng: random.Random,
) -> None:
    for _ in range(TARGET_PERIPHERY_REPAIR_STEPS):
        spatial = spatial_discipline_metrics(canvas)
        center_empty = center_empty_ratio(canvas)

        need_periphery = (
            spatial["periphery_boundary_density_ratio"] < MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO
            or spatial["periphery_non_coyote_ratio"] < MIN_PERIPHERY_NON_COYOTE_RATIO
        )
        need_center_fill = center_empty > MAX_COYOTE_CENTER_EMPTY_RATIO

        if not need_periphery and not need_center_fill:
            break

        boundary = compute_boundary_mask(canvas)
        if need_periphery:
            ys, xs = preferred_boundary_coordinates(boundary, HIGH_DENSITY_ZONE_MASK)
            origin_code = ORIGIN_TRANSITION if rng.random() < 0.70 else ORIGIN_MICRO
            color = IDX_TERRE if rng.random() < 0.55 else IDX_OLIVE
        else:
            ys, xs = preferred_boundary_coordinates(boundary, CENTER_TORSO_MASK)
            origin_code = ORIGIN_TRANSITION
            color = IDX_OLIVE if rng.random() < 0.65 else IDX_TERRE

        if len(xs) == 0:
            break

        idx = rng.randint(0, len(xs) - 1)
        cx, cy = int(xs[idx]), int(ys[idx])
        size = cm_to_px(rng.uniform(2.0, 6.0))
        angle = 0 if need_center_fill and 0 in profile.allowed_angles else rng.choice(profile.angle_pool or tuple(profile.allowed_angles))
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=size * rng.uniform(1.6, 2.6),
            width_px=size * rng.uniform(0.55, 0.95),
            angle_from_vertical_deg=angle,
            segments=rng.randint(4, 6),
            width_variation=profile.micro_width_variation,
            lateral_jitter=profile.micro_lateral_jitter,
            tip_taper=profile.micro_tip_taper,
            edge_break=profile.micro_edge_break,
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue

        if need_periphery and float(np.mean(HIGH_DENSITY_ZONE_MASK[mask])) < 0.30:
            continue
        if need_center_fill and float(np.mean(CENTER_TORSO_MASK[mask])) < 0.30:
            continue

        apply_mask(canvas, origin_map, mask, color, origin_code)


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

        cx, cy = choose_biased_center(rng, profile.zone_weight_boosts)
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(22, 42)),
            width_px=cm_to_px(rng.uniform(8, 16)),
            angle_from_vertical_deg=angle,
            segments=rng.randint(5, 7),
            width_variation=max(0.16, profile.macro_width_variation - 0.04),
            lateral_jitter=max(0.11, profile.macro_lateral_jitter - 0.02),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue
        if zone_overlap_ratio(mask, ANATOMY_ZONES["center_torso"]) > 0.46:
            continue
        if float(np.mean(canvas[mask] == IDX_OLIVE)) > 0.65:
            continue

        color = IDX_OLIVE if rng.random() < 0.78 else IDX_TERRE
        apply_mask(canvas, origin_map, mask, color, ORIGIN_MACRO)
        macros.append(MacroRecord(color, poly, angle, (cx, cy), mask, macro_zone_count(mask)))

def generate_one_variant(profile: VariantProfile) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
    rng = random.Random(profile.seed)

    canvas = np.full((HEIGHT, WIDTH), IDX_COYOTE, dtype=np.uint8)
    origin_map = np.full((HEIGHT, WIDTH), ORIGIN_BACKGROUND, dtype=np.uint8)

    macros = add_macros(canvas, origin_map, profile, rng)
    enforce_macro_angle_discipline(canvas, origin_map, macros, profile, rng)
    add_transitions(canvas, origin_map, macros, profile, rng)
    add_micro_clusters(canvas, origin_map, profile, rng)
    repair_spatial_discipline(canvas, origin_map, profile, rng)
    nudge_proportions(canvas, origin_map, profile, rng)
    repair_spatial_discipline(canvas, origin_map, profile, rng)
    nudge_proportions(canvas, origin_map, profile, rng)
    rs = compute_ratios(canvas)
    orient = orientation_score(macros)
    visible_shares = visible_origin_shares(canvas, origin_map)
    multi = multiscale_metrics(canvas)

    olive_macros = [m for m in macros if m.color_idx == IDX_OLIVE]
    multizone_share = 0.0
    if olive_macros:
        multizone_share = float(np.mean([m.zone_count >= 2 for m in olive_macros]))

    metrics = {
        "largest_olive_component_ratio": largest_component_ratio(canvas == IDX_OLIVE),
        "center_empty_ratio": center_empty_ratio(canvas),
        "boundary_density": boundary_density(canvas),
        "mirror_similarity": mirror_similarity_score(canvas),
        "oblique_share": orient["oblique_share"],
        "vertical_share": orient["vertical_share"],
        "angle_dominance_ratio": orient["dominance_ratio"],
        "olive_multizone_share": multizone_share,
        **visible_shares,
        **absolute_origin_color_ratios(canvas, origin_map),
        **spatial_discipline_metrics(canvas),
        **multi,
    }
    metrics.update(evaluate_visual_metrics(canvas, rs, metrics))
    metrics.update(military_visual_discipline_score(metrics))

    return render_canvas(canvas), rs, metrics


def generate_candidate_from_seed(seed: int) -> CandidateResult:
    adaptive_hint = _consume_profile_override(seed)
    return generate_candidate_from_seed_with_hint(seed, adaptive_hint=adaptive_hint)


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

    if metrics["boundary_density"] < MIN_BOUNDARY_DENSITY:
        return False
    if metrics["boundary_density"] > MAX_BOUNDARY_DENSITY:
        return False

    if metrics["boundary_density_small"] < MIN_BOUNDARY_DENSITY_SMALL:
        return False
    if metrics["boundary_density_small"] > MAX_BOUNDARY_DENSITY_SMALL:
        return False

    if metrics["mirror_similarity"] > MAX_MIRROR_SIMILARITY:
        return False

    if metrics["oblique_share"] < MIN_OBLIQUE_SHARE:
        return False
    if metrics["vertical_share"] < MIN_VERTICAL_SHARE or metrics["vertical_share"] > MAX_VERTICAL_SHARE:
        return False
    if metrics["angle_dominance_ratio"] > MAX_ANGLE_DOMINANCE_RATIO:
        return False

    if metrics["vert_olive_macro_share"] < MIN_VISIBLE_OLIVE_MACRO_SHARE:
        return False
    if metrics["terre_de_france_transition_share"] < MIN_VISIBLE_TERRE_TRANS_SHARE:
        return False
    if metrics["vert_de_gris_micro_share"] < MIN_VISIBLE_GRIS_MICRO_SHARE:
        return False
    if metrics["vert_de_gris_macro_share"] > MAX_VISIBLE_GRIS_MACRO_SHARE:
        return False

    # Validation perceptive optionnelle mais activée automatiquement quand les scores sont présents.
    if "visual_silhouette_color_diversity" in metrics:
        if metrics["visual_silhouette_color_diversity"] < VISUAL_MIN_SILHOUETTE_COLOR_DIVERSITY:
            return False
    if "visual_contour_break_score" in metrics:
        if metrics["visual_contour_break_score"] < VISUAL_MIN_CONTOUR_BREAK_SCORE:
            return False
    if "visual_outline_band_diversity" in metrics:
        if metrics["visual_outline_band_diversity"] < VISUAL_MIN_OUTLINE_BAND_DIVERSITY:
            return False
    if "visual_small_scale_structural_score" in metrics:
        if metrics["visual_small_scale_structural_score"] < VISUAL_MIN_SMALL_SCALE_STRUCTURAL_SCORE:
            return False
    if "visual_score_final" in metrics:
        if metrics["visual_score_final"] < VISUAL_MIN_FINAL_SCORE:
            return False
    if "periphery_boundary_density_ratio" in metrics:
        if metrics["periphery_boundary_density_ratio"] < MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO:
            return False
    if "periphery_non_coyote_ratio" in metrics:
        if metrics["periphery_non_coyote_ratio"] < MIN_PERIPHERY_NON_COYOTE_RATIO:
            return False
    if "macro_olive_visible_ratio" in metrics:
        if metrics["macro_olive_visible_ratio"] < MIN_MACRO_OLIVE_VISIBLE_RATIO:
            return False
    if "macro_terre_visible_ratio" in metrics:
        if metrics["macro_terre_visible_ratio"] > MAX_MACRO_TERRE_VISIBLE_RATIO:
            return False
    if "macro_gris_visible_ratio" in metrics:
        if metrics["macro_gris_visible_ratio"] > MAX_MACRO_GRIS_VISIBLE_RATIO:
            return False
    if "visual_military_score" in metrics:
        if metrics["visual_military_score"] < VISUAL_MIN_MILITARY_SCORE:
            return False

    return True


def validate_candidate_result(candidate: CandidateResult) -> bool:
    return variant_is_valid(candidate.ratios, candidate.metrics)


async def async_validate_candidate_result(candidate: CandidateResult) -> bool:
    return await asyncio.to_thread(validate_candidate_result, candidate)


# ============================================================
# EXPORT / API
# ============================================================

def save_candidate_image(candidate: CandidateResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate.image.save(path)
    return path


async def async_save_candidate_image(candidate: CandidateResult, path: Path) -> Path:
    return await asyncio.to_thread(save_candidate_image, candidate, path)


def candidate_row(
    target_index: int,
    local_attempt: int,
    global_attempt: int,
    candidate: CandidateResult,
) -> Dict[str, object]:
    rs = candidate.ratios
    metrics = candidate.metrics

    row = {
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
        "oblique_share": round(metrics["oblique_share"], 5),
        "vertical_share": round(metrics["vertical_share"], 5),
        "angle_dominance_ratio": round(metrics["angle_dominance_ratio"], 5),
        "olive_macro_share": round(metrics["vert_olive_macro_share"], 5),
        "terre_transition_share": round(metrics["terre_de_france_transition_share"], 5),
        "gris_micro_share": round(metrics["vert_de_gris_micro_share"], 5),
        "gris_macro_share": round(metrics["vert_de_gris_macro_share"], 5),
        "angles": " ".join(map(str, candidate.profile.allowed_angles)),
    }
    for key in (
        "visual_score_final",
        "visual_score_ratio",
        "visual_score_silhouette",
        "visual_score_contour",
        "visual_score_main",
        "visual_silhouette_color_diversity",
        "visual_contour_break_score",
        "visual_outline_band_diversity",
        "visual_small_scale_structural_score",
        "visual_validation_passed",
        "macro_olive_visible_ratio",
        "macro_terre_visible_ratio",
        "macro_gris_visible_ratio",
        "transition_terre_visible_ratio",
        "micro_gris_visible_ratio",
        "periphery_boundary_density",
        "center_boundary_density",
        "periphery_boundary_density_ratio",
        "periphery_non_coyote_density",
        "center_non_coyote_density",
        "periphery_non_coyote_ratio",
        "visual_military_score",
        "visual_military_passed",
    ):
        if key in metrics:
            row[key] = round(float(metrics[key]), 5)
    return row


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


async def async_write_report(
    rows: List[Dict[str, object]],
    output_dir: Path,
    filename: str = "rapport_camouflages.csv",
) -> Path:
    return await asyncio.to_thread(write_report, rows, output_dir, filename)


# ============================================================
# AIDES PARALLÉLISATION DES TENTATIVES
# ============================================================

def generate_and_validate_from_seed(seed: int) -> Tuple[CandidateResult, bool]:
    candidate = generate_candidate_from_seed(seed)
    accepted = validate_candidate_result(candidate)
    return candidate, accepted


def _batch_attempt_seeds(
    target_index: int,
    start_attempt: int,
    batch_size: int,
    base_seed: int,
) -> List[Tuple[int, int]]:
    return [
        (local_attempt, build_seed(target_index, local_attempt, base_seed=base_seed))
        for local_attempt in range(start_attempt, start_attempt + batch_size)
    ]


# ============================================================
# GÉNÉRATION SÉQUENTIELLE SYNCHRONE
# ============================================================

def generate_all(
    target_count: int = N_VARIANTS_REQUIRED,
    output_dir: Path = OUTPUT_DIR,
    base_seed: int = DEFAULT_BASE_SEED,
    progress_callback: Optional[
        Callable[[int, int, int, int, CandidateResult, bool], None]
    ] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    attempt_batch_size: int = DEFAULT_ATTEMPT_BATCH_SIZE,
    parallel_attempts: bool = True,
    adaptive_rejection_correction: bool = True,
) -> List[Dict[str, object]]:
    output_dir = ensure_output_dir(output_dir)

    rows: List[Dict[str, object]] = []
    total_attempts = 0
    max_workers = max(1, int(max_workers))
    attempt_batch_size = max(1, int(attempt_batch_size))

    use_parallel = parallel_attempts and max_workers > 1 and attempt_batch_size > 1 and not adaptive_rejection_correction
    pool = get_process_pool(max_workers) if use_parallel else None

    for target_index in range(1, target_count + 1):
        local_attempt = 1
        adaptive_state = AdaptiveGenerationState() if adaptive_rejection_correction else None
        _runtime_log(
            "INFO",
            "main_adaptive",
            "Initialisation du réglage adaptatif",
            target_index=target_index,
            enabled=bool(adaptive_state),
        )

        while True:
            if stop_requested is not None and stop_requested():
                write_report(rows, output_dir)
                return rows

            if use_parallel and pool is not None:
                batch = _batch_attempt_seeds(
                    target_index=target_index,
                    start_attempt=local_attempt,
                    batch_size=attempt_batch_size,
                    base_seed=base_seed,
                )
                adaptive_hint = build_adaptive_hint(adaptive_state) if adaptive_state is not None else None

                futures = [
                    pool.submit(generate_and_validate_from_seed_with_hint, seed, adaptive_hint)
                    for _, seed in batch
                ]

                batch_results: List[Tuple[int, CandidateResult, bool]] = []
                for (attempt_no, _seed), fut in zip(batch, futures):
                    candidate, accepted = fut.result()
                    total_attempts += 1
                    batch_results.append((attempt_no, candidate, accepted))

                    if progress_callback is not None:
                        progress_callback(
                            target_index,
                            attempt_no,
                            total_attempts,
                            target_count,
                            candidate,
                            accepted,
                        )

                    update_adaptive_state_after_attempt(
                        adaptive_state,
                        candidate,
                        accepted,
                        target_index=target_index,
                        local_attempt=attempt_no,
                    )

                accepted_result = next(((a, c) for a, c, ok in batch_results if ok), None)
                if accepted_result is None:
                    local_attempt += attempt_batch_size
                    continue

                accepted_attempt, accepted_candidate = accepted_result

            else:
                seed = build_seed(target_index, local_attempt, base_seed=base_seed)
                adaptive_hint = build_adaptive_hint(adaptive_state) if adaptive_state is not None else None
                _register_profile_override(seed, adaptive_hint)
                accepted_candidate = generate_candidate_from_seed(seed)
                accepted = validate_candidate_result(accepted_candidate)
                total_attempts += 1

                if progress_callback is not None:
                    progress_callback(
                        target_index,
                        local_attempt,
                        total_attempts,
                        target_count,
                        accepted_candidate,
                        accepted,
                    )

                update_adaptive_state_after_attempt(
                    adaptive_state,
                    accepted_candidate,
                    accepted,
                    target_index=target_index,
                    local_attempt=local_attempt,
                )

                if not accepted:
                    local_attempt += 1
                    continue

                accepted_attempt = local_attempt

            filename = output_dir / f"camouflage_{target_index:03d}.png"
            save_candidate_image(accepted_candidate, filename)

            rows.append(
                candidate_row(
                    target_index=target_index,
                    local_attempt=accepted_attempt,
                    global_attempt=total_attempts,
                    candidate=accepted_candidate,
                )
            )
            break

    write_report(rows, output_dir)
    return rows


# ============================================================
# GÉNÉRATION SÉQUENTIELLE ASYNCHRONE
# ============================================================

AsyncProgressCallback = Callable[[int, int, int, int, CandidateResult, bool], Awaitable[None]]
AsyncStopCallable = Callable[[], Awaitable[bool]]


async def async_generate_all(
    target_count: int = N_VARIANTS_REQUIRED,
    output_dir: Path = OUTPUT_DIR,
    base_seed: int = DEFAULT_BASE_SEED,
    progress_callback: Optional[AsyncProgressCallback] = None,
    stop_requested: Optional[AsyncStopCallable] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    attempt_batch_size: int = DEFAULT_ATTEMPT_BATCH_SIZE,
    parallel_attempts: bool = True,
    adaptive_rejection_correction: bool = True,
) -> List[Dict[str, object]]:
    """
    Version asynchrone séquentielle stricte.

    Important :
    - l'image N+1 ne commence jamais tant que N n'est pas validée ;
    - chaque image courante peut toutefois tester plusieurs seeds en parallèle ;
    - la première tentative valide dans l'ordre logique est conservée.
    """
    output_dir = ensure_output_dir(output_dir)

    rows: List[Dict[str, object]] = []
    total_attempts = 0
    max_workers = max(1, int(max_workers))
    attempt_batch_size = max(1, int(attempt_batch_size))

    use_parallel = parallel_attempts and max_workers > 1 and attempt_batch_size > 1 and not adaptive_rejection_correction
    loop = asyncio.get_running_loop()
    pool = get_process_pool(max_workers) if use_parallel else None

    for target_index in range(1, target_count + 1):
        local_attempt = 1
        adaptive_state = AdaptiveGenerationState() if adaptive_rejection_correction else None
        _runtime_log(
            "INFO",
            "main_adaptive",
            "Initialisation du réglage adaptatif",
            target_index=target_index,
            enabled=bool(adaptive_state),
        )

        while True:
            if stop_requested is not None and await stop_requested():
                await async_write_report(rows, output_dir)
                return rows

            if use_parallel and pool is not None:
                batch = _batch_attempt_seeds(
                    target_index=target_index,
                    start_attempt=local_attempt,
                    batch_size=attempt_batch_size,
                    base_seed=base_seed,
                )
                adaptive_hint = build_adaptive_hint(adaptive_state) if adaptive_state is not None else None

                tasks = [
                    loop.run_in_executor(pool, generate_and_validate_from_seed_with_hint, seed, adaptive_hint)
                    for _, seed in batch
                ]
                raw_results = await asyncio.gather(*tasks)

                batch_results: List[Tuple[int, CandidateResult, bool]] = []
                for (attempt_no, _seed), (candidate, accepted) in zip(batch, raw_results):
                    total_attempts += 1
                    batch_results.append((attempt_no, candidate, accepted))

                    if progress_callback is not None:
                        await progress_callback(
                            target_index,
                            attempt_no,
                            total_attempts,
                            target_count,
                            candidate,
                            accepted,
                        )

                    update_adaptive_state_after_attempt(
                        adaptive_state,
                        candidate,
                        accepted,
                        target_index=target_index,
                        local_attempt=attempt_no,
                    )

                accepted_result = next(((a, c) for a, c, ok in batch_results if ok), None)
                if accepted_result is None:
                    local_attempt += attempt_batch_size
                    continue

                accepted_attempt, accepted_candidate = accepted_result

            else:
                seed = build_seed(target_index, local_attempt, base_seed=base_seed)
                adaptive_hint = build_adaptive_hint(adaptive_state) if adaptive_state is not None else None
                _register_profile_override(seed, adaptive_hint)
                accepted_candidate = await async_generate_candidate_from_seed(seed)
                accepted = await async_validate_candidate_result(accepted_candidate)
                total_attempts += 1

                if progress_callback is not None:
                    await progress_callback(
                        target_index,
                        local_attempt,
                        total_attempts,
                        target_count,
                        accepted_candidate,
                        accepted,
                    )

                update_adaptive_state_after_attempt(
                    adaptive_state,
                    accepted_candidate,
                    accepted,
                    target_index=target_index,
                    local_attempt=local_attempt,
                )

                if not accepted:
                    local_attempt += 1
                    continue

                accepted_attempt = local_attempt

            filename = output_dir / f"camouflage_{target_index:03d}.png"
            await async_save_candidate_image(accepted_candidate, filename)

            rows.append(
                candidate_row(
                    target_index=target_index,
                    local_attempt=accepted_attempt,
                    global_attempt=total_attempts,
                    candidate=accepted_candidate,
                )
            )
            break

    await async_write_report(rows, output_dir)
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
            max_workers=DEFAULT_MAX_WORKERS,
            attempt_batch_size=DEFAULT_ATTEMPT_BATCH_SIZE,
            parallel_attempts=True,
        )

        csv_path = OUTPUT_DIR / "rapport_camouflages.csv"

        print("\nTerminé.")
        print(f"Images validées : {len(rows)}/{N_VARIANTS_REQUIRED}")
        print(f"Dossier : {OUTPUT_DIR.resolve()}")
        print(f"CSV : {csv_path.resolve()}")
        print(f"Workers : {DEFAULT_MAX_WORKERS}")
        print(f"Batch size : {DEFAULT_ATTEMPT_BATCH_SIZE}")
    finally:
        shutdown_process_pool()
