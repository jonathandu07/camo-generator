# -*- coding: utf-8 -*-
"""
Camouflage Armée Fédérale Europe
Générateur filtré de 100 variantes cohérentes

Objectif :
- 100 camouflages différents mais de la même famille visuelle
- respect de la hiérarchie Macro / Transition / Micro
- contrôle des proportions
- rejet automatique des variantes trop éloignées de la cible

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
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

OUTPUT_DIR = Path("camouflages_federale_europe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 1200
HEIGHT = 1700
PX_PER_CM = 4.2

N_VARIANTS_REQUIRED = 100
MAX_ATTEMPTS = 3000

COLOR_NAMES = [
    "coyote_brown",
    "vert_olive",
    "terre_de_france",
    "vert_de_gris",
]

RGB = np.array([
    (0x81, 0x61, 0x3C),  # coyote brown
    (0x55, 0x54, 0x3F),  # vert olive
    (0x7C, 0x6D, 0x66),  # terre de france
    (0x57, 0x5D, 0x57),  # vert-de-gris
], dtype=np.uint8)

TARGET = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)

# Tolérances d'acceptation
# On impose une proximité réelle à la cible.
MAX_ABS_ERROR_PER_COLOR = np.array([0.09, 0.09, 0.08, 0.08], dtype=float)
MAX_MEAN_ABS_ERROR = 0.055

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
    rng = random.Random(seed)

    allowed = BASE_ANGLES[:]
    rng.shuffle(allowed)
    allowed = sorted(allowed[:rng.randint(8, len(BASE_ANGLES))])

    return VariantProfile(
        seed=seed,
        allowed_angles=allowed,
        n_macro_olive=rng.randint(24, 34),
        n_macro_terre=rng.randint(16, 24),
        n_macro_gris=rng.randint(4, 7),
        n_transition_total=rng.randint(180, 320),
        n_micro_clusters=rng.randint(420, 780),
        micro_cluster_min=2,
        micro_cluster_max=rng.randint(4, 5),
        macro_width_variation=rng.uniform(0.24, 0.34),
        macro_lateral_jitter=rng.uniform(0.16, 0.24),
        macro_tip_taper=rng.uniform(0.34, 0.46),
        macro_edge_break=rng.uniform(0.09, 0.15),
        micro_width_variation=rng.uniform(0.20, 0.28),
        micro_lateral_jitter=rng.uniform(0.14, 0.20),
        micro_tip_taper=rng.uniform(0.42, 0.54),
        micro_edge_break=rng.uniform(0.12, 0.18),
    )


# ============================================================
# OUTILS
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
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)  # fond coyote

    macro_polys: List[Tuple[int, List[Tuple[float, float]]]] = []

    # --------------------------------------------------------
    # 1) Macros Olive
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
            width_variation=profile.macro_width_variation,
            lateral_jitter=profile.macro_lateral_jitter,
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        canvas[polygon_mask(poly)] = 1
        macro_polys.append((1, poly))

    # --------------------------------------------------------
    # 2) Macros Terre
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
            width_variation=max(0.18, profile.macro_width_variation - 0.03),
            lateral_jitter=max(0.12, profile.macro_lateral_jitter - 0.02),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        canvas[polygon_mask(poly)] = 2
        macro_polys.append((2, poly))

    # --------------------------------------------------------
    # 3) Intrusions rares gris
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
            width_variation=max(0.16, profile.macro_width_variation - 0.08),
            lateral_jitter=max(0.10, profile.macro_lateral_jitter - 0.05),
            tip_taper=profile.macro_tip_taper,
            edge_break=profile.macro_edge_break,
        )
        canvas[polygon_mask(poly)] = 3
        macro_polys.append((3, poly))

    # --------------------------------------------------------
    # 4) Transitions attachées
    # --------------------------------------------------------
    for _ in range(profile.n_transition_total):
        parent_color, parent = rng.choice(macro_polys)
        poly = attached_transition(
            rng=rng,
            parent=parent,
            length_px=cm_to_px(rng.uniform(10, 30)),
            width_px=cm_to_px(rng.uniform(5, 15)),
        )
        mask = polygon_mask(poly)

        # Terre dominante en transition
        r = rng.random()
        if r < 0.68:
            canvas[mask] = 2
        elif r < 0.88:
            canvas[mask] = 1
        else:
            canvas[mask] = 0

    # --------------------------------------------------------
    # 5) Micro-clusters uniquement sur frontières
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
                width_variation=profile.micro_width_variation,
                lateral_jitter=profile.micro_lateral_jitter,
                tip_taper=profile.micro_tip_taper,
                edge_break=profile.micro_edge_break,
            )
            mask = polygon_mask(poly)
            cur = canvas[mask]

            # interdit en champ libre
            if len(np.unique(cur)) < 2:
                continue

            # gris dominant, terre secondaire
            canvas[mask] = 3 if rng.random() < 0.74 else 2

    # --------------------------------------------------------
    # 6) Ajustement léger sur frontières
    # --------------------------------------------------------
    for _ in range(5):
        rs = compute_ratios(canvas)
        deficits = TARGET - rs

        if np.max(np.abs(deficits)) < 0.010:
            break

        b = boundary_mask(canvas)
        ys, xs = np.where(b)
        if len(xs) == 0:
            break

        picks = np.random.permutation(len(xs))[: min(len(xs), int(canvas.size * 0.0025))]

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

            if chosen not in neigh and rng.random() < 0.65:
                continue

            canvas[y, x] = chosen

    return render_canvas(canvas), compute_ratios(canvas)


# ============================================================
# VALIDATION
# ============================================================

def variant_is_valid(rs: np.ndarray) -> bool:
    abs_err = np.abs(rs - TARGET)
    if np.any(abs_err > MAX_ABS_ERROR_PER_COLOR):
        return False
    if float(np.mean(abs_err)) > MAX_MEAN_ABS_ERROR:
        return False

    # Contraintes supplémentaires doctrinales
    # coyote doit rester dominant ou co-dominant, mais pas écrasant
    if rs[0] < 0.20 or rs[0] > 0.46:
        return False

    # olive doit rester forte
    if rs[1] < 0.18:
        return False

    # terre doit être réellement présente
    if rs[2] < 0.14:
        return False

    # gris doit être visible
    if rs[3] < 0.10:
        return False

    return True


# ============================================================
# GÉNÉRATION DES 100 VARIANTES
# ============================================================

def generate_all() -> None:
    rows = []
    accepted = 0
    attempts = 0

    while accepted < N_VARIANTS_REQUIRED and attempts < MAX_ATTEMPTS:
        attempts += 1
        seed = 2026031200 + attempts
        profile = make_profile(seed)

        img, rs = generate_one_variant(profile)

        if not variant_is_valid(rs):
            print(f"[{attempts:04d}] rejeté")
            continue

        accepted += 1
        filename = OUTPUT_DIR / f"camouflage_{accepted:03d}.png"
        img.save(filename)

        rows.append({
            "index": accepted,
            "seed": profile.seed,
            "coyote_brown_pct": round(float(rs[0] * 100), 2),
            "vert_olive_pct": round(float(rs[1] * 100), 2),
            "terre_de_france_pct": round(float(rs[2] * 100), 2),
            "vert_de_gris_pct": round(float(rs[3] * 100), 2),
            "macro_olive": profile.n_macro_olive,
            "macro_terre": profile.n_macro_terre,
            "macro_gris": profile.n_macro_gris,
            "transitions": profile.n_transition_total,
            "micro_clusters": profile.n_micro_clusters,
            "angles": " ".join(map(str, profile.allowed_angles)),
        })

        print(
            f"[{attempts:04d}] accepté -> camouflage_{accepted:03d}.png | "
            f"C={rs[0]*100:.1f} O={rs[1]*100:.1f} T={rs[2]*100:.1f} G={rs[3]*100:.1f}"
        )

    if not rows:
        print("Aucune variante valide n'a été produite.")
        return

    csv_path = OUTPUT_DIR / "rapport_camouflages.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTerminé.")
    print(f"Variantes acceptées : {accepted}/{N_VARIANTS_REQUIRED}")
    print(f"Tentatives : {attempts}")
    print(f"Dossier : {OUTPUT_DIR.resolve()}")
    print(f"CSV : {csv_path.resolve()}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_all()