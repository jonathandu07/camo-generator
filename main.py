# -*- coding: utf-8 -*-
"""
Camouflage Armée Fédérale Europe
Générateur séquentiel filtré jusqu'à 100 images validées

Doctrine respectée :
- fond continu Coyote Brown
- macro-formes Olive dominantes
- Terre de France comme liaison / transition
- Vert-de-gris surtout en rupture fine
- trois niveaux hiérarchiques : Macro / Transition / Micro
- formes anguleuses uniquement
- orientation verticale / oblique
- micro-formes uniquement sur frontières
- densité asymétrique (épaules / flancs / cuisses)
- validation stricte avant passage à l'image suivante
- aucune limite de tentatives par image

Dépendances :
    pip install pillow numpy

Sorties :
    ./camouflages_federale_europe/
        camouflage_001.png
        ...
        camouflage_100.png
        rapport_camouflages.csv
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

COLOR_NAMES = [
    "coyote_brown",
    "vert_olive",
    "terre_de_france",
    "vert_de_gris",
]

# Indexation stricte
IDX_COYOTE = 0
IDX_OLIVE = 1
IDX_TERRE = 2
IDX_GRIS = 3

RGB = np.array([
    (0x81, 0x61, 0x3C),  # coyote brown
    (0x55, 0x54, 0x3F),  # vert olive
    (0x7C, 0x6D, 0x66),  # terre de france
    (0x57, 0x5D, 0x57),  # vert-de-gris
], dtype=np.uint8)

TARGET = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)

# Tolérances plus rigoureuses
MAX_ABS_ERROR_PER_COLOR = np.array([0.06, 0.06, 0.055, 0.055], dtype=float)
MAX_MEAN_ABS_ERROR = 0.038

# Angles autorisés : vertical + oblique 15° à 35°
BASE_ANGLES = [-35, -30, -25, -20, -15, 0, 15, 20, 25, 30, 35]

# Densité asymétrique :
# épaules / flancs / cuisses plus denses, centre plus calme
DENSITY_ZONES = [
    (0.02, 0.26, 0.02, 0.18, 1.75),  # épaule gauche
    (0.74, 0.98, 0.02, 0.18, 1.75),  # épaule droite
    (0.00, 0.22, 0.18, 0.72, 1.60),  # flanc gauche
    (0.78, 1.00, 0.18, 0.72, 1.60),  # flanc droit
    (0.20, 0.42, 0.62, 0.96, 1.50),  # cuisse gauche
    (0.58, 0.80, 0.62, 0.96, 1.50),  # cuisse droite
    (0.30, 0.70, 0.18, 0.62, 0.70),  # centre calme
]

# Réglages géométriques
MACRO_LENGTH_CM = (40, 90)
MACRO_WIDTH_CM = (15, 35)

TRANSITION_LENGTH_CM = (10, 30)
TRANSITION_WIDTH_CM = (5, 15)

MICRO_SIZE_CM = (2, 8)

# Réglages de correction finale
BOUNDARY_NUDGE_PASSES = 8
BOUNDARY_NUDGE_SAMPLE_RATIO = 0.0032

# Validation perceptive
MIN_OLIVE_CONNECTED_COMPONENT_AREA_RATIO = 0.035
MAX_COYOTE_CENTER_EMPTY_RATIO = 0.62
MIN_BOUNDARY_DENSITY = 0.08
MAX_BOUNDARY_DENSITY = 0.34

# Rapports internes désirés par niveau
# Une partie seulement de TERRE et GRIS doit venir du niveau macro
MACRO_TERRE_TARGET_FACTOR = 0.58
MACRO_GRIS_TARGET_FACTOR = 0.12


# ============================================================
# PROFIL DE VARIANTE
# ============================================================

@dataclass
class VariantProfile:
    seed: int

    allowed_angles: List[int]

    n_macro_olive: int
    n_macro_terre: int
    n_macro_gris: int

    n_transition_total: int
    n_micro_clusters: int

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


def make_profile(seed: int) -> VariantProfile:
    """
    Variations contrôlées, même famille visuelle, jamais anarchique.
    """
    rng = random.Random(seed)

    angles = BASE_ANGLES[:]
    rng.shuffle(angles)
    allowed = sorted(angles[:rng.randint(8, len(BASE_ANGLES))])

    return VariantProfile(
        seed=seed,
        allowed_angles=allowed,

        # Olive doit structurellement dominer
        n_macro_olive=rng.randint(34, 48),

        # Terre macro présente, mais inférieure à olive
        n_macro_terre=rng.randint(16, 24),

        # Gris rare en macro
        n_macro_gris=rng.randint(2, 4),

        # Transitions nombreuses mais pas autonomes
        n_transition_total=rng.randint(210, 320),

        # Micro présents mais subordonnés aux interfaces
        n_micro_clusters=rng.randint(520, 860),

        micro_cluster_min=2,
        micro_cluster_max=rng.randint(4, 5),

        macro_width_variation=rng.uniform(0.22, 0.31),
        macro_lateral_jitter=rng.uniform(0.14, 0.22),
        macro_tip_taper=rng.uniform(0.34, 0.44),
        macro_edge_break=rng.uniform(0.10, 0.15),

        micro_width_variation=rng.uniform(0.18, 0.26),
        micro_lateral_jitter=rng.uniform(0.12, 0.18),
        micro_tip_taper=rng.uniform(0.42, 0.52),
        micro_edge_break=rng.uniform(0.12, 0.18),
    )


# ============================================================
# OUTILS DE BASE
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


def boundary_mask(canvas: np.ndarray) -> np.ndarray:
    diff = np.zeros((HEIGHT, WIDTH), dtype=bool)
    diff[1:, :] |= (canvas[1:, :] != canvas[:-1, :])
    diff[:-1, :] |= (canvas[:-1, :] != canvas[1:, :])
    diff[:, 1:] |= (canvas[:, 1:] != canvas[:, :-1])
    diff[:, :-1] |= (canvas[:, :-1] != canvas[:, 1:])
    return diff


def local_color_variety(canvas: np.ndarray, x: int, y: int, radius: int = 2) -> int:
    y1, y2 = max(0, y - radius), min(HEIGHT, y + radius + 1)
    x1, x2 = max(0, x - radius), min(WIDTH, x + radius + 1)
    return len(np.unique(canvas[y1:y2, x1:x2]))


# ============================================================
# ANALYSE MORPHOLOGIQUE SIMPLE
# ============================================================

def largest_component_ratio(mask: np.ndarray) -> float:
    """
    Taille relative de la plus grande composante connexe 4-voisins.
    Sans dépendance externe.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = 0
    total = int(mask.sum())
    if total == 0:
        return 0.0

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


def center_empty_ratio(canvas: np.ndarray) -> float:
    """
    Mesure la part de coyote dans la zone centrale calme.
    Elle ne doit pas devenir un grand vide brun uniforme.
    """
    x1 = int(WIDTH * 0.30)
    x2 = int(WIDTH * 0.70)
    y1 = int(HEIGHT * 0.18)
    y2 = int(HEIGHT * 0.62)
    zone = canvas[y1:y2, x1:x2]
    return float(np.mean(zone == IDX_COYOTE))


def boundary_density(canvas: np.ndarray) -> float:
    return float(np.mean(boundary_mask(canvas)))


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
    """
    Forme allongée, anguleuse, irrégulière, jamais ronde.
    """
    half_len = length_px / 2.0
    half_w = width_px / 2.0
    ys = np.linspace(-half_len, half_len, segments)

    left, right = [], []

    for y in ys:
        t = abs(y) / half_len
        taper = max(0.35, 1.0 - tip_taper * t)

        local_half_w = half_w * rng.uniform(
            1.0 - width_variation,
            1.0 + width_variation
        ) * taper

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
    """
    Forme de transition attachée à une macro.
    """
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
# COUCHES
# ============================================================

def add_macros(canvas: np.ndarray, profile: VariantProfile, rng: random.Random) -> List[Tuple[int, List[Tuple[float, float]]]]:
    macro_polys: List[Tuple[int, List[Tuple[float, float]]]] = []

    # 1) Olive : masses structurantes dominantes
    for _ in range(profile.n_macro_olive):
        cx, cy = choose_biased_center(rng)
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(*MACRO_LENGTH_CM)),
            width_px=cm_to_px(rng.uniform(*MACRO_WIDTH_CM)),
            angle_from_vertical_deg=rng.choice(profile.allowed_angles),
            segments=rng.randint(7, 10),
            width_variation=profile.macro_width_variation,
            lateral_jitter=profile.macro_lateral_jitter,
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        canvas[polygon_mask(poly)] = IDX_OLIVE
        macro_polys.append((IDX_OLIVE, poly))

    # 2) Terre : liaisons macro
    for _ in range(profile.n_macro_terre):
        cx, cy = choose_biased_center(rng)
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(42, 70)),
            width_px=cm_to_px(rng.uniform(12, 26)),
            angle_from_vertical_deg=rng.choice(profile.allowed_angles),
            segments=rng.randint(6, 9),
            width_variation=max(0.18, profile.macro_width_variation - 0.03),
            lateral_jitter=max(0.12, profile.macro_lateral_jitter - 0.02),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        canvas[polygon_mask(poly)] = IDX_TERRE
        macro_polys.append((IDX_TERRE, poly))

    # 3) Gris : rare en macro
    for _ in range(profile.n_macro_gris):
        cx, cy = choose_biased_center(rng)
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(35, 55)),
            width_px=cm_to_px(rng.uniform(10, 18)),
            angle_from_vertical_deg=rng.choice(profile.allowed_angles),
            segments=rng.randint(5, 8),
            width_variation=max(0.16, profile.macro_width_variation - 0.08),
            lateral_jitter=max(0.10, profile.macro_lateral_jitter - 0.05),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        canvas[polygon_mask(poly)] = IDX_GRIS
        macro_polys.append((IDX_GRIS, poly))

    return macro_polys


def add_transitions(canvas: np.ndarray, macro_polys: Sequence[Tuple[int, List[Tuple[float, float]]]], profile: VariantProfile, rng: random.Random) -> None:
    """
    Transitions attachées aux macros.
    Terre dominante, olive secondaire, coyote ponctuel.
    """
    for _ in range(profile.n_transition_total):
        parent_color, parent = rng.choice(macro_polys)

        poly = attached_transition(
            rng=rng,
            parent=parent,
            length_px=cm_to_px(rng.uniform(*TRANSITION_LENGTH_CM)),
            width_px=cm_to_px(rng.uniform(*TRANSITION_WIDTH_CM)),
        )
        mask = polygon_mask(poly)
        cur = canvas[mask]

        # Interdit transition isolée totalement libre
        if cur.size == 0:
            continue

        # Terre dominante
        r = rng.random()
        if r < 0.70:
            color = IDX_TERRE
        elif r < 0.90:
            color = IDX_OLIVE
        else:
            color = IDX_COYOTE

        # Terre ne doit pas créer une masse flottante en zone totalement vide
        if color == IDX_TERRE and np.mean(cur != IDX_COYOTE) < 0.18:
            continue

        # Olive transitionnaire ne doit pas écraser partout le coyote
        if color == IDX_OLIVE and np.mean(cur == IDX_COYOTE) > 0.92 and rng.random() < 0.65:
            continue

        canvas[mask] = color


def add_micro_clusters(canvas: np.ndarray, profile: VariantProfile, rng: random.Random) -> None:
    """
    Micro-formes uniquement sur frontières.
    Gris dominant, terre secondaire.
    """
    for _ in range(profile.n_micro_clusters):
        b = boundary_mask(canvas)
        ys, xs = np.where(b)
        if len(xs) == 0:
            break

        idx = rng.randint(0, len(xs) - 1)
        bx, by = int(xs[idx]), int(ys[idx])

        for _ in range(rng.randint(profile.micro_cluster_min, profile.micro_cluster_max)):
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
            cur = canvas[mask]

            # Interdit en champ libre
            if len(np.unique(cur)) < 2:
                continue

            color = IDX_GRIS if rng.random() < 0.76 else IDX_TERRE
            canvas[mask] = color


def nudge_proportions(canvas: np.ndarray, rng: random.Random) -> None:
    """
    Ajustement doux sur frontières uniquement.
    """
    for _ in range(BOUNDARY_NUDGE_PASSES):
        rs = compute_ratios(canvas)
        deficits = TARGET - rs

        if np.max(np.abs(deficits)) < 0.008:
            break

        b = boundary_mask(canvas)
        ys, xs = np.where(b)
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

            # Cohérence locale stricte
            if chosen not in neigh and rng.random() < 0.72:
                continue

            # Gris et terre ne doivent pas apparaître en quasi champ libre
            if chosen in (IDX_TERRE, IDX_GRIS) and local_color_variety(canvas, x, y, radius=2) < 2:
                continue

            canvas[y, x] = chosen


# ============================================================
# GÉNÉRATION D'UNE VARIANTE
# ============================================================

def generate_one_variant(profile: VariantProfile) -> Tuple[Image.Image, np.ndarray, Dict[str, float]]:
    rng = random.Random(profile.seed)
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)  # fond coyote continu

    macro_polys = add_macros(canvas, profile, rng)
    add_transitions(canvas, macro_polys, profile, rng)
    add_micro_clusters(canvas, profile, rng)
    nudge_proportions(canvas, rng)

    rs = compute_ratios(canvas)

    metrics = {
        "largest_olive_component_ratio": largest_component_ratio(canvas == IDX_OLIVE),
        "center_empty_ratio": center_empty_ratio(canvas),
        "boundary_density": boundary_density(canvas),
    }

    return render_canvas(canvas), rs, metrics


# ============================================================
# VALIDATION RIGOUREUSE
# ============================================================

def variant_is_valid(rs: np.ndarray, metrics: Dict[str, float]) -> bool:
    abs_err = np.abs(rs - TARGET)

    # 1) Proportions
    if np.any(abs_err > MAX_ABS_ERROR_PER_COLOR):
        return False

    if float(np.mean(abs_err)) > MAX_MEAN_ABS_ERROR:
        return False

    # 2) Contraintes doctrinales
    # Fond coyote continu mais non écrasant
    if rs[IDX_COYOTE] < 0.22 or rs[IDX_COYOTE] > 0.43:
        return False

    # Olive doit être une vraie masse principale
    if rs[IDX_OLIVE] < 0.22:
        return False

    # Terre bien visible, mais pas dominante sur l'olive
    if rs[IDX_TERRE] < 0.16 or rs[IDX_TERRE] > 0.30:
        return False

    # Gris visible, mais secondaire
    if rs[IDX_GRIS] < 0.11 or rs[IDX_GRIS] > 0.24:
        return False

    # 3) Cohérence perceptive opérationnelle
    # Olive doit avoir au moins une grande masse connectée
    if metrics["largest_olive_component_ratio"] < MIN_OLIVE_CONNECTED_COMPONENT_AREA_RATIO:
        return False

    # Le centre ne doit pas devenir un grand vide brun
    if metrics["center_empty_ratio"] > MAX_COYOTE_CENTER_EMPTY_RATIO:
        return False

    # Il faut assez de frontières pour casser la lecture
    if metrics["boundary_density"] < MIN_BOUNDARY_DENSITY:
        return False

    # ...mais pas au point de devenir du bruit isotrope
    if metrics["boundary_density"] > MAX_BOUNDARY_DENSITY:
        return False

    return True


# ============================================================
# GÉNÉRATION STRICTEMENT SÉQUENTIELLE
# ============================================================

def generate_all() -> None:
    rows = []
    total_attempts = 0

    for target_index in range(1, N_VARIANTS_REQUIRED + 1):
        local_attempt = 0

        while True:
            total_attempts += 1
            local_attempt += 1

            # seed unique par image + tentative
            seed = 202603120000 + target_index * 100000 + local_attempt
            profile = make_profile(seed)

            img, rs, metrics = generate_one_variant(profile)

            if not variant_is_valid(rs, metrics):
                print(
                    f"[global={total_attempts:06d}] "
                    f"[image={target_index:03d}] "
                    f"[essai={local_attempt:04d}] rejeté | "
                    f"C={rs[0]*100:.1f} O={rs[1]*100:.1f} T={rs[2]*100:.1f} G={rs[3]*100:.1f}"
                )
                continue

            filename = OUTPUT_DIR / f"camouflage_{target_index:03d}.png"
            img.save(filename)

            rows.append({
                "index": target_index,
                "seed": profile.seed,
                "attempts_for_this_image": local_attempt,
                "global_attempt": total_attempts,
                "coyote_brown_pct": round(float(rs[IDX_COYOTE] * 100), 2),
                "vert_olive_pct": round(float(rs[IDX_OLIVE] * 100), 2),
                "terre_de_france_pct": round(float(rs[IDX_TERRE] * 100), 2),
                "vert_de_gris_pct": round(float(rs[IDX_GRIS] * 100), 2),
                "largest_olive_component_ratio": round(metrics["largest_olive_component_ratio"], 5),
                "center_empty_ratio": round(metrics["center_empty_ratio"], 5),
                "boundary_density": round(metrics["boundary_density"], 5),
                "macro_olive": profile.n_macro_olive,
                "macro_terre": profile.n_macro_terre,
                "macro_gris": profile.n_macro_gris,
                "transitions": profile.n_transition_total,
                "micro_clusters": profile.n_micro_clusters,
                "angles": " ".join(map(str, profile.allowed_angles)),
            })

            print(
                f"[global={total_attempts:06d}] "
                f"[image={target_index:03d}] "
                f"[essai={local_attempt:04d}] accepté -> {filename.name} | "
                f"C={rs[0]*100:.1f} O={rs[1]*100:.1f} T={rs[2]*100:.1f} G={rs[3]*100:.1f} | "
                f"olive_conn={metrics['largest_olive_component_ratio']:.3f} "
                f"center={metrics['center_empty_ratio']:.3f} "
                f"boundary={metrics['boundary_density']:.3f}"
            )

            # On ne passe à l'image suivante qu'après validation
            break

    csv_path = OUTPUT_DIR / "rapport_camouflages.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTerminé.")
    print(f"Images validées : {N_VARIANTS_REQUIRED}/{N_VARIANTS_REQUIRED}")
    print(f"Tentatives globales : {total_attempts}")
    print(f"Dossier : {OUTPUT_DIR.resolve()}")
    print(f"CSV : {csv_path.resolve()}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_all()