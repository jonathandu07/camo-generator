# -*- coding: utf-8 -*-
"""
main.py
Camouflage Armée Fédérale Europe
Version module-friendly + async-friendly.

Points clés :
- API synchrone conservée
- API asynchrone ajoutée
- génération séquentielle stricte
- aucun passage à l'image suivante tant que la précédente n'est pas validée
"""

from __future__ import annotations

import asyncio
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

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

VISIBLE_MACRO_OLIVE_TARGET = 0.215
VISIBLE_MACRO_TERRE_TARGET = 0.075
VISIBLE_MACRO_GRIS_TARGET = 0.015

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

MAX_MIRROR_SIMILARITY = 0.79

MIN_VISIBLE_OLIVE_MACRO_SHARE = 0.60
MIN_VISIBLE_TERRE_TRANS_SHARE = 0.36
MIN_VISIBLE_GRIS_MICRO_SHARE = 0.70
MAX_VISIBLE_GRIS_MACRO_SHARE = 0.18

MIN_OBLIQUE_SHARE = 0.58
MIN_VERTICAL_SHARE = 0.08
MAX_VERTICAL_SHARE = 0.34
MAX_ANGLE_DOMINANCE_RATIO = 0.34

BOUNDARY_NUDGE_PASSES = 10
BOUNDARY_NUDGE_SAMPLE_RATIO = 0.0035

MIN_TRANSITION_TOUCH_PIXELS = 22
MIN_MICRO_BOUNDARY_COVERAGE = 0.24
MAX_LOCAL_MASS_RATIO_TRANSITION = 0.72


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

def ensure_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# PROFIL
# ============================================================

def build_seed(target_index: int, local_attempt: int, base_seed: int = DEFAULT_BASE_SEED) -> int:
    return int(base_seed + target_index * 100000 + local_attempt)


def make_profile(seed: int) -> VariantProfile:
    rng = random.Random(seed)
    angles = BASE_ANGLES[:]
    rng.shuffle(angles)
    allowed = sorted(set([0] + angles[:rng.randint(8, len(BASE_ANGLES))]))

    return VariantProfile(
        seed=seed,
        allowed_angles=allowed,
        micro_cluster_min=2,
        micro_cluster_max=rng.randint(4, 5),
        macro_width_variation=rng.uniform(0.22, 0.30),
        macro_lateral_jitter=rng.uniform(0.14, 0.21),
        macro_tip_taper=rng.uniform(0.34, 0.43),
        macro_edge_break=rng.uniform(0.10, 0.15),
        micro_width_variation=rng.uniform(0.18, 0.25),
        micro_lateral_jitter=rng.uniform(0.12, 0.18),
        micro_tip_taper=rng.uniform(0.42, 0.52),
        micro_edge_break=rng.uniform(0.12, 0.18),
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


def choose_biased_center(rng: random.Random) -> Tuple[int, int]:
    weights = [z[4] for z in DENSITY_ZONES]
    z = rng.choices(DENSITY_ZONES, weights=weights, k=1)[0]
    x = int(rng.uniform(z[0], z[1]) * WIDTH)
    y = int(rng.uniform(z[2], z[3]) * HEIGHT)
    x = min(max(x, 60), WIDTH - 60)
    y = min(max(y, 60), HEIGHT - 60)
    return x, y


def polygon_mask(poly: Sequence[Tuple[float, float]]) -> np.ndarray:
    img = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(img).polygon(poly, fill=255)
    return np.array(img, dtype=np.uint8) > 0


def compute_boundary_mask(canvas: np.ndarray) -> np.ndarray:
    diff = np.zeros((HEIGHT, WIDTH), dtype=bool)
    diff[1:, :] |= (canvas[1:, :] != canvas[:-1, :])
    diff[:-1, :] |= (canvas[:-1, :] != canvas[1:, :])
    diff[:, 1:] |= (canvas[:, 1:] != canvas[:, :-1])
    diff[:, :-1] |= (canvas[:, :-1] != canvas[:, 1:])
    return diff


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
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


def local_color_variety(canvas: np.ndarray, x: int, y: int, radius: int = 2) -> int:
    y1, y2 = max(0, y - radius), min(HEIGHT, y + radius + 1)
    x1, x2 = max(0, x - radius), min(WIDTH, x + radius + 1)
    return len(np.unique(canvas[y1:y2, x1:x2]))


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


def infer_origin_from_neighbors(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    x: int,
    y: int,
    chosen_color: int,
    fallback_origin: int,
) -> int:
    y1, y2 = max(0, y - 2), min(canvas.shape[0], y + 3)
    x1, x2 = max(0, x - 2), min(canvas.shape[1], x + 3)

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


def macro_zone_count(mask: np.ndarray) -> int:
    count = 0
    for zone_mask in ANATOMY_ZONES.values():
        overlap = int((mask & zone_mask).sum())
        if overlap >= 600:
            count += 1
    return count


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
    rot = [rotate(x, y, angle_from_vertical_deg - 90) for x, y in poly]
    return [(cx + x, cy + y) for x, y in rot]


def attached_transition(
    rng: random.Random,
    parent: Sequence[Tuple[float, float]],
    length_px: float,
    width_px: float
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


def transition_is_attached(parent_mask: np.ndarray, transition_mask: np.ndarray, min_touch_pixels: int = MIN_TRANSITION_TOUCH_PIXELS) -> bool:
    contact = dilate_mask(parent_mask, radius=1) & transition_mask
    return int(contact.sum()) >= min_touch_pixels


def micro_is_on_boundary(boundary: np.ndarray, micro_mask: np.ndarray, min_boundary_coverage: float = MIN_MICRO_BOUNDARY_COVERAGE) -> bool:
    area = int(micro_mask.sum())
    if area == 0:
        return False
    cov = float(np.mean(boundary[micro_mask]))
    return cov >= min_boundary_coverage


def creates_new_mass(canvas: np.ndarray, new_mask: np.ndarray, color_idx: int, local_radius: int = 45, max_local_area_ratio: float = MAX_LOCAL_MASS_RATIO_TRANSITION) -> bool:
    ys, xs = np.where(new_mask)
    if len(xs) == 0:
        return False

    x1 = max(0, int(xs.min()) - local_radius)
    x2 = min(WIDTH, int(xs.max()) + local_radius + 1)
    y1 = max(0, int(ys.min()) - local_radius)
    y2 = min(HEIGHT, int(ys.max()) + local_radius + 1)

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

    target_macro_olive_pixels = int(VISIBLE_MACRO_OLIVE_TARGET * canvas.size)
    target_macro_terre_pixels = int(VISIBLE_MACRO_TERRE_TARGET * canvas.size)
    target_macro_gris_pixels = int(VISIBLE_MACRO_GRIS_TARGET * canvas.size)

    while int(np.sum((canvas == IDX_OLIVE) & (origin_map == ORIGIN_MACRO))) < target_macro_olive_pixels:
        cx, cy = choose_biased_center(rng)
        angle = rng.choice(profile.allowed_angles)

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

        apply_mask(canvas, origin_map, mask, IDX_OLIVE, ORIGIN_MACRO)
        macros.append(MacroRecord(IDX_OLIVE, poly, angle, (cx, cy), mask, zc))

    while int(np.sum((canvas == IDX_TERRE) & (origin_map == ORIGIN_MACRO))) < target_macro_terre_pixels:
        cx, cy = choose_biased_center(rng)
        angle = rng.choice(profile.allowed_angles)

        if local_parallel_conflict(macros, (cx, cy), angle, dist_threshold_px=220, angle_threshold_deg=6):
            continue

        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(42, 70)),
            width_px=cm_to_px(rng.uniform(12, 26)),
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
        if float(np.mean(np.isin(cur, [IDX_COYOTE, IDX_OLIVE]))) < 0.60:
            continue

        zc = macro_zone_count(mask)
        apply_mask(canvas, origin_map, mask, IDX_TERRE, ORIGIN_MACRO)
        macros.append(MacroRecord(IDX_TERRE, poly, angle, (cx, cy), mask, zc))

    while int(np.sum((canvas == IDX_GRIS) & (origin_map == ORIGIN_MACRO))) < target_macro_gris_pixels:
        cx, cy = choose_biased_center(rng)
        angle = rng.choice(profile.allowed_angles)

        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(35, 55)),
            width_px=cm_to_px(rng.uniform(10, 18)),
            angle_from_vertical_deg=angle,
            segments=rng.randint(5, 8),
            width_variation=max(0.16, profile.macro_width_variation - 0.08),
            lateral_jitter=max(0.10, profile.macro_lateral_jitter - 0.05),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue

        cur = canvas[mask]
        if float(np.mean(np.isin(cur, [IDX_OLIVE, IDX_TERRE]))) < 0.48:
            continue

        zc = macro_zone_count(mask)
        apply_mask(canvas, origin_map, mask, IDX_GRIS, ORIGIN_MACRO)
        macros.append(MacroRecord(IDX_GRIS, poly, angle, (cx, cy), mask, zc))

    return macros


def add_transitions(
    canvas: np.ndarray,
    origin_map: np.ndarray,
    macros: Sequence[MacroRecord],
    rng: random.Random,
) -> None:
    while True:
        rs = compute_ratios(canvas)
        visible = visible_origin_shares(canvas, origin_map)

        enough_terre = rs[IDX_TERRE] >= VISIBLE_TOTAL_TERRE_TARGET
        enough_olive = rs[IDX_OLIVE] >= VISIBLE_TOTAL_OLIVE_TARGET * 0.985
        enough_trans_share = visible["terre_de_france_transition_share"] >= 0.30

        if enough_terre and enough_olive and enough_trans_share:
            break

        parent = rng.choice(macros)

        poly = attached_transition(
            rng=rng,
            parent=parent.poly,
            length_px=cm_to_px(rng.uniform(*TRANSITION_LENGTH_CM)),
            width_px=cm_to_px(rng.uniform(*TRANSITION_WIDTH_CM)),
        )
        mask = polygon_mask(poly)
        if mask.sum() == 0:
            continue

        if not transition_is_attached(parent.mask, mask):
            continue

        cur = canvas[mask]
        deficits = TARGET - rs
        terre_need = max(0.0, deficits[IDX_TERRE])
        olive_need = max(0.0, deficits[IDX_OLIVE])

        if terre_need >= olive_need:
            p_terre, p_olive, p_coyote = 0.74, 0.18, 0.08
        else:
            p_terre, p_olive, p_coyote = 0.62, 0.28, 0.10

        r = rng.random()
        if r < p_terre:
            color = IDX_TERRE
        elif r < p_terre + p_olive:
            color = IDX_OLIVE
        else:
            color = IDX_COYOTE

        if color == IDX_TERRE and float(np.mean(cur != IDX_COYOTE)) < 0.18:
            continue

        if color == IDX_OLIVE and creates_new_mass(canvas, mask, color, local_radius=38, max_local_area_ratio=0.76):
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
    while True:
        rs = compute_ratios(canvas)
        visible = visible_origin_shares(canvas, origin_map)

        enough_gris = rs[IDX_GRIS] >= VISIBLE_TOTAL_GRIS_TARGET
        enough_terre = rs[IDX_TERRE] >= VISIBLE_TOTAL_TERRE_TARGET
        enough_micro_share = visible["vert_de_gris_micro_share"] >= 0.64

        if enough_gris and enough_terre and enough_micro_share:
            break

        boundary = compute_boundary_mask(canvas)
        ys, xs = np.where(boundary)
        if len(xs) == 0:
            break

        idx = rng.randint(0, len(xs) - 1)
        bx, by = int(xs[idx]), int(ys[idx])

        cluster_count = rng.randint(profile.micro_cluster_min, profile.micro_cluster_max)
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
                angle_from_vertical_deg=rng.choice(profile.allowed_angles),
                segments=rng.randint(4, 6),
                width_variation=profile.micro_width_variation,
                lateral_jitter=profile.micro_lateral_jitter,
                tip_taper=profile.micro_tip_taper,
                edge_break=profile.micro_edge_break,
            )
            mask = polygon_mask(poly)
            if mask.sum() == 0:
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

            if gris_need >= terre_need:
                color = IDX_GRIS if rng.random() < 0.82 else IDX_TERRE
            else:
                color = IDX_GRIS if rng.random() < 0.70 else IDX_TERRE

            apply_mask(canvas, origin_map, mask, color, ORIGIN_MICRO)
            placed_any = True

        if not placed_any and rs[IDX_GRIS] >= VISIBLE_TOTAL_GRIS_TARGET * 0.96:
            break


def nudge_proportions(canvas: np.ndarray, origin_map: np.ndarray, rng: random.Random) -> None:
    for _ in range(BOUNDARY_NUDGE_PASSES):
        rs = compute_ratios(canvas)
        deficits = TARGET - rs

        if np.max(np.abs(deficits)) < 0.0055:
            break

        boundary = compute_boundary_mask(canvas)
        ys, xs = np.where(boundary)
        if len(xs) == 0:
            break

        picks = np.random.permutation(len(xs))[: min(len(xs), int(canvas.size * BOUNDARY_NUDGE_SAMPLE_RATIO))]

        for j in picks:
            x = int(xs[j])
            y = int(ys[j])

            current = int(canvas[y, x])
            if deficits[current] >= 0:
                continue

            wanted = np.where(deficits > 0)[0]
            if len(wanted) == 0:
                break

            chosen = int(wanted[np.argmax(deficits[wanted])])

            y1, y2 = max(0, y - 2), min(HEIGHT, y + 3)
            x1, x2 = max(0, x - 2), min(WIDTH, x + 3)
            neigh = canvas[y1:y2, x1:x2]

            if chosen not in neigh and rng.random() < 0.72:
                continue

            if chosen in (IDX_TERRE, IDX_GRIS) and local_color_variety(canvas, x, y, radius=2) < 2:
                continue

            fallback_origin = int(origin_map[y, x])
            new_origin = infer_origin_from_neighbors(canvas, origin_map, x, y, chosen, fallback_origin)

            canvas[y, x] = chosen
            origin_map[y, x] = new_origin


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
# GÉNÉRATION D'UNE VARIANTE
# ============================================================

def generate_one_variant(profile: VariantProfile) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
    rng = random.Random(profile.seed)

    canvas = np.full((HEIGHT, WIDTH), IDX_COYOTE, dtype=np.uint8)
    origin_map = np.full((HEIGHT, WIDTH), ORIGIN_BACKGROUND, dtype=np.uint8)

    macros = add_macros(canvas, origin_map, profile, rng)
    add_transitions(canvas, origin_map, macros, rng)
    add_micro_clusters(canvas, origin_map, profile, rng)
    nudge_proportions(canvas, origin_map, rng)

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
        **multi,
    }

    return render_canvas(canvas), rs, metrics


def generate_candidate_from_seed(seed: int) -> CandidateResult:
    profile = make_profile(seed)
    image, ratios, metrics = generate_one_variant(profile)
    return CandidateResult(
        seed=seed,
        profile=profile,
        image=image,
        ratios=ratios,
        metrics=metrics,
    )


async def async_generate_candidate_from_seed(seed: int) -> CandidateResult:
    """
    Version asynchrone correcte pour travail CPU-bound :
    exécution dans un thread.
    """
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
        "oblique_share": round(metrics["oblique_share"], 5),
        "vertical_share": round(metrics["vertical_share"], 5),
        "angle_dominance_ratio": round(metrics["angle_dominance_ratio"], 5),
        "olive_macro_share": round(metrics["vert_olive_macro_share"], 5),
        "terre_transition_share": round(metrics["terre_de_france_transition_share"], 5),
        "gris_micro_share": round(metrics["vert_de_gris_micro_share"], 5),
        "gris_macro_share": round(metrics["vert_de_gris_macro_share"], 5),
        "angles": " ".join(map(str, candidate.profile.allowed_angles)),
    }


def write_report(rows: List[Dict[str, object]], output_dir: Path, filename: str = "rapport_camouflages.csv") -> Path:
    output_dir = ensure_output_dir(output_dir)
    csv_path = output_dir / filename
    if not rows:
        return csv_path

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


async def async_write_report(rows: List[Dict[str, object]], output_dir: Path, filename: str = "rapport_camouflages.csv") -> Path:
    return await asyncio.to_thread(write_report, rows, output_dir, filename)


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
) -> List[Dict[str, object]]:
    output_dir = ensure_output_dir(output_dir)

    rows: List[Dict[str, object]] = []
    total_attempts = 0

    for target_index in range(1, target_count + 1):
        local_attempt = 0

        while True:
            if stop_requested is not None and stop_requested():
                write_report(rows, output_dir)
                return rows

            total_attempts += 1
            local_attempt += 1

            seed = build_seed(target_index, local_attempt, base_seed=base_seed)
            candidate = generate_candidate_from_seed(seed)
            accepted = validate_candidate_result(candidate)

            if progress_callback is not None:
                progress_callback(
                    target_index,
                    local_attempt,
                    total_attempts,
                    target_count,
                    candidate,
                    accepted,
                )

            if not accepted:
                continue

            filename = output_dir / f"camouflage_{target_index:03d}.png"
            save_candidate_image(candidate, filename)

            rows.append(
                candidate_row(
                    target_index=target_index,
                    local_attempt=local_attempt,
                    global_attempt=total_attempts,
                    candidate=candidate,
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
) -> List[Dict[str, object]]:
    """
    Version asynchrone séquentielle stricte.

    Important :
    - l'image N+1 ne commence jamais tant que N n'est pas validée ;
    - chaque tentative CPU-bound est déportée avec asyncio.to_thread(...).
    """
    output_dir = ensure_output_dir(output_dir)

    rows: List[Dict[str, object]] = []
    total_attempts = 0

    for target_index in range(1, target_count + 1):
        local_attempt = 0

        while True:
            if stop_requested is not None:
                if await stop_requested():
                    await async_write_report(rows, output_dir)
                    return rows

            total_attempts += 1
            local_attempt += 1

            seed = build_seed(target_index, local_attempt, base_seed=base_seed)
            candidate = await async_generate_candidate_from_seed(seed)
            accepted = await async_validate_candidate_result(candidate)

            if progress_callback is not None:
                await progress_callback(
                    target_index,
                    local_attempt,
                    total_attempts,
                    target_count,
                    candidate,
                    accepted,
                )

            if not accepted:
                continue

            filename = output_dir / f"camouflage_{target_index:03d}.png"
            await async_save_candidate_image(candidate, filename)

            rows.append(
                candidate_row(
                    target_index=target_index,
                    local_attempt=local_attempt,
                    global_attempt=total_attempts,
                    candidate=candidate,
                )
            )
            break

    await async_write_report(rows, output_dir)
    return rows


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    rows = generate_all(
        target_count=N_VARIANTS_REQUIRED,
        output_dir=OUTPUT_DIR,
        base_seed=DEFAULT_BASE_SEED,
    )

    csv_path = OUTPUT_DIR / "rapport_camouflages.csv"

    print("\nTerminé.")
    print(f"Images validées : {len(rows)}/{N_VARIANTS_REQUIRED}")
    print(f"Dossier : {OUTPUT_DIR.resolve()}")
    print(f"CSV : {csv_path.resolve()}")