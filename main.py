# -*- coding: utf-8 -*-
"""
Camouflage Armée Fédérale Europe
Générateur de 100 variantes cohérentes

Objectif :
- produire 100 camouflages différents
- tout en respectant la même doctrine visuelle :
  * fond coyote dominant
  * grandes masses olive verticales/obliques
  * terre de transition
  * gris de rupture
  * micro-clusters uniquement sur interfaces
  * pas de formes rondes
  * pas de répétition périodique
  * pas de symétrie

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
# CONFIGURATION GÉNÉRALE
# ============================================================

OUTPUT_DIR = Path("camouflages_federale_europe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 1400
HEIGHT = 2000
PX_PER_CM = 4.6

N_VARIANTS = 100

COLOR_NAMES = [
    "coyote_brown",
    "vert_olive",
    "terre_de_france",
    "vert_de_gris",
]

RGB = np.array([
    (0x81, 0x61, 0x3C),  # coyote brown
    (0x55, 0x54, 0x3F),  # olive
    (0x7C, 0x6D, 0x66),  # terre
    (0x57, 0x5D, 0x57),  # vert-de-gris
], dtype=np.uint8)

TARGET = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)

BASE_ANGLES = [-35, -30, -25, -20, -15, 0, 15, 20, 25, 30, 35]

DENSITY_ZONES = [
    # x1, x2, y1, y2, poids
    (0.02, 0.26, 0.02, 0.18, 1.70),  # épaule gauche
    (0.74, 0.98, 0.02, 0.18, 1.70),  # épaule droite
    (0.00, 0.22, 0.18, 0.72, 1.55),  # flanc gauche
    (0.78, 1.00, 0.18, 0.72, 1.55),  # flanc droit
    (0.20, 0.42, 0.62, 0.96, 1.45),  # cuisse gauche
    (0.58, 0.80, 0.62, 0.96, 1.45),  # cuisse droite
    (0.30, 0.70, 0.18, 0.62, 0.75),  # centre calme
]


# ============================================================
# PROFIL DE VARIANTE
# ============================================================

@dataclass
class VariantProfile:
    seed: int
    n_macro_olive: int
    n_macro_terre: int
    n_macro_gris: int
    n_transitions_min: int
    n_transitions_max: int
    n_micro_clusters: int
    micro_cluster_min: int
    micro_cluster_max: int
    width_variation_macro: float
    lateral_jitter_macro: float
    edge_break_macro: float
    tip_taper_macro: float
    width_variation_micro: float
    lateral_jitter_micro: float
    edge_break_micro: float
    tip_taper_micro: float
    allowed_angles: List[int]


def make_variant_profile(index: int) -> VariantProfile:
    """
    Génère une variante différente mais cohérente avec la doctrine.
    """
    seed = 2026031200 + index
    rng = random.Random(seed)

    # Variation contrôlée autour d'un même style
    angle_pool = BASE_ANGLES[:]
    rng.shuffle(angle_pool)
    allowed_angles = sorted(angle_pool[: rng.randint(7, len(BASE_ANGLES))])

    return VariantProfile(
        seed=seed,
        n_macro_olive=rng.randint(16, 24),
        n_macro_terre=rng.randint(9, 15),
        n_macro_gris=rng.randint(3, 6),
        n_transitions_min=rng.randint(2, 4),
        n_transitions_max=rng.randint(5, 8),
        n_micro_clusters=rng.randint(700, 1300),
        micro_cluster_min=2,
        micro_cluster_max=rng.randint(4, 5),
        width_variation_macro=rng.uniform(0.24, 0.34),
        lateral_jitter_macro=rng.uniform(0.16, 0.24),
        edge_break_macro=rng.uniform(0.09, 0.15),
        tip_taper_macro=rng.uniform(0.36, 0.48),
        width_variation_micro=rng.uniform(0.20, 0.28),
        lateral_jitter_micro=rng.uniform(0.14, 0.20),
        edge_break_micro=rng.uniform(0.12, 0.18),
        tip_taper_micro=rng.uniform(0.42, 0.54),
        allowed_angles=allowed_angles,
    )


# ============================================================
# OUTILS
# ============================================================

def cm_to_px(cm: float) -> int:
    return max(1, int(round(cm * PX_PER_CM)))


def ratios(canvas: np.ndarray) -> np.ndarray:
    counts = np.bincount(canvas.ravel(), minlength=4).astype(float)
    return counts / canvas.size


def render_canvas(canvas: np.ndarray) -> Image.Image:
    arr = RGB[canvas]
    return Image.fromarray(arr, "RGB")


def rotate(x: float, y: float, deg: float) -> Tuple[float, float]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def choose_biased_center(rng: random.Random) -> Tuple[int, int]:
    weights = [z[4] for z in DENSITY_ZONES]
    z = rng.choices(DENSITY_ZONES, weights=weights, k=1)[0]
    x = int(rng.uniform(z[0], z[1]) * WIDTH)
    y = int(rng.uniform(z[2], z[3]) * HEIGHT)
    x = min(max(x, 50), WIDTH - 50)
    y = min(max(y, 50), HEIGHT - 50)
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
# GÉNÉRATION D'UNE VARIANTE
# ============================================================

def generate_one_variant(profile: VariantProfile) -> Tuple[Image.Image, np.ndarray]:
    rng = random.Random(profile.seed)
    np.random.seed(profile.seed % (2**32 - 1))

    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)  # fond coyote
    macro_polys: List[Tuple[int, List[Tuple[float, float]]]] = []

    # --------------------------------------------------------
    # 1) Macro olive
    # --------------------------------------------------------
    for _ in range(profile.n_macro_olive):
        cx, cy = choose_biased_center(rng)
        poly = jagged_spine_poly(
            rng=rng,
            cx=cx,
            cy=cy,
            length_px=cm_to_px(rng.uniform(40, 90)),
            width_px=cm_to_px(rng.uniform(15, 35)),
            angle_from_vertical_deg=rng.choice(profile.allowed_angles),
            segments=rng.randint(7, 10),
            width_variation=profile.width_variation_macro,
            lateral_jitter=profile.lateral_jitter_macro,
            tip_taper=profile.tip_taper_macro,
            edge_break=profile.edge_break_macro,
        )
        mask = polygon_mask(poly)
        canvas[mask] = 1
        macro_polys.append((1, poly))

    # --------------------------------------------------------
    # 2) Macro terre
    # --------------------------------------------------------
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
            width_variation=max(0.18, profile.width_variation_macro - 0.03),
            lateral_jitter=max(0.12, profile.lateral_jitter_macro - 0.02),
            tip_taper=profile.tip_taper_macro,
            edge_break=profile.edge_break_macro,
        )
        mask = polygon_mask(poly)
        canvas[mask] = 2
        macro_polys.append((2, poly))

    # --------------------------------------------------------
    # 3) Macro gris rare
    # --------------------------------------------------------
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
            width_variation=max(0.16, profile.width_variation_macro - 0.08),
            lateral_jitter=max(0.10, profile.lateral_jitter_macro - 0.05),
            tip_taper=profile.tip_taper_macro,
            edge_break=profile.edge_break_macro,
        )
        mask = polygon_mask(poly)
        canvas[mask] = 3
        macro_polys.append((3, poly))

    # --------------------------------------------------------
    # 4) Transitions attachées
    # --------------------------------------------------------
    for color_idx, parent in macro_polys:
        n_trans = rng.randint(profile.n_transitions_min, profile.n_transitions_max)
        for _ in range(n_trans):
            poly = attached_transition(
                rng=rng,
                parent=parent,
                length_px=cm_to_px(rng.uniform(10, 30)),
                width_px=cm_to_px(rng.uniform(5, 15)),
            )
            mask = polygon_mask(poly)

            r = rng.random()
            if r < 0.66:
                canvas[mask] = 2  # terre dominante
            elif r < 0.87:
                canvas[mask] = 1  # olive secondaire
            else:
                canvas[mask] = 0  # coyote secondaire

    # --------------------------------------------------------
    # 5) Micro-clusters sur frontières uniquement
    # --------------------------------------------------------
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

            size = cm_to_px(rng.uniform(2, 8))
            poly = jagged_spine_poly(
                rng=rng,
                cx=ox,
                cy=oy,
                length_px=size * rng.uniform(1.2, 2.0),
                width_px=size * rng.uniform(0.45, 0.90),
                angle_from_vertical_deg=rng.choice(profile.allowed_angles),
                segments=rng.randint(4, 6),
                width_variation=profile.width_variation_micro,
                lateral_jitter=profile.lateral_jitter_micro,
                tip_taper=profile.tip_taper_micro,
                edge_break=profile.edge_break_micro,
            )
            mask = polygon_mask(poly)
            cur = canvas[mask]

            # interdit en champ libre
            if len(np.unique(cur)) < 2:
                continue

            canvas[mask] = 3 if rng.random() < 0.74 else 2

    # --------------------------------------------------------
    # 6) Ajustement léger sur frontières
    # --------------------------------------------------------
    for _ in range(4):
        rs = ratios(canvas)
        deficits = TARGET - rs

        if np.max(np.abs(deficits)) < 0.012:
            break

        b = boundary_mask(canvas)
        ys, xs = np.where(b)
        if len(xs) == 0:
            break

        picks = np.random.permutation(len(xs))[: min(len(xs), int(canvas.size * 0.002))]

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

            if chosen not in neigh and rng.random() < 0.70:
                continue

            canvas[y, x] = chosen

    return render_canvas(canvas), ratios(canvas)


# ============================================================
# GÉNÉRATION DES 100 VARIANTES
# ============================================================

def generate_100_camouflages() -> None:
    report_rows = []

    for i in range(1, N_VARIANTS + 1):
        profile = make_variant_profile(i)
        img, rs = generate_one_variant(profile)

        filename = OUTPUT_DIR / f"camouflage_{i:03d}.png"
        img.save(filename)

        report_rows.append({
            "index": i,
            "seed": profile.seed,
            "coyote_brown_pct": round(rs[0] * 100, 2),
            "vert_olive_pct": round(rs[1] * 100, 2),
            "terre_de_france_pct": round(rs[2] * 100, 2),
            "vert_de_gris_pct": round(rs[3] * 100, 2),
            "n_macro_olive": profile.n_macro_olive,
            "n_macro_terre": profile.n_macro_terre,
            "n_macro_gris": profile.n_macro_gris,
            "n_micro_clusters": profile.n_micro_clusters,
            "angles": " ".join(map(str, profile.allowed_angles)),
        })

        print(f"[{i:03d}/{N_VARIANTS}] {filename.name} généré")

    csv_path = OUTPUT_DIR / "rapport_camouflages.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nTerminé.")
    print(f"Dossier : {OUTPUT_DIR.resolve()}")
    print(f"Rapport : {csv_path.resolve()}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_100_camouflages()