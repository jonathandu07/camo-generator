#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur de camouflage "France tempéré" : macro hiérarchisée + micro agressif
- 4 couleurs EXACTES (hex) — corrigées selon ta référence "Terre de France"
- Non directionnel (orientations indépendantes)
- Macro 3 échelles (50–60 / 70–80 / 90–110 mm)
- Micro (8–25 mm) qui "mord" les contours et traverse parfois 2 macros
- Tile 640x640 mm, répétable bord à bord (wrap par duplication +/-W)
- Sorties : SVG (vectoriel) + PNG (aperçu raster)

Dépendances:
  pip install pillow
(SVG écrit à la main, pas de lib obligatoire)

NOTE FACTUELLE:
- "Terre de France" est ici verrouillé à #7C6D66 (RGB 124,109,102) d'après ta capture.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw  # type: ignore


# ============================================================
# Palette verrouillée (corrigée)
# ============================================================

C1_COYOTE = "#81613C"      # grandes masses seulement (macro)
C2_TERRE  = "#7C6D66"      # Terre de France (référence normée) : transition (micro secondaire)
C3_OLIVE  = "#4B5320"      # grandes + moyennes masses (macro)
C4_VERDIG = "#7A8B7A"      # micro + morsures (prioritaire)

PALETTE = {
    "C1": C1_COYOTE,
    "C2": C2_TERRE,
    "C3": C3_OLIVE,
    "C4": C4_VERDIG,
}

# Proportions cibles (indicatives: le calcul affiché reste "approx" car superpositions)
TARGET_RATIO = {
    "C1": 0.32,
    "C3": 0.28,
    "C2": 0.22,  # Terre de France = #7C6D66 (corrigé)
    "C4": 0.18,
}


# ============================================================
# Géométrie / utilitaires
# ============================================================

Point = Tuple[float, float]


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.strip().lower()
    if not h.startswith("#") or len(h) != 7:
        raise ValueError(f"Couleur invalide: {h!r}")
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


def polygon_area(poly: List[Point]) -> float:
    # Aire (shoelace) en unités²
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def rotate_points(poly: List[Point], angle_rad: float) -> List[Point]:
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    out: List[Point] = []
    for x, y in poly:
        out.append((x * ca - y * sa, x * sa + y * ca))
    return out


def translate_points(poly: List[Point], dx: float, dy: float) -> List[Point]:
    return [(x + dx, y + dy) for x, y in poly]


# ============================================================
# Forme "blob" dentelé (sans pixels, bords nets)
# ============================================================

def random_blob(
    rng: random.Random,
    r_min: float,
    r_max: float,
    *,
    n_base: Tuple[int, int] = (7, 13),
    jag_subdiv: Tuple[int, int] = (1, 3),
    jag_amp: Tuple[float, float] = (0.10, 0.30),
) -> List[Point]:
    """
    Génère un polygone organique dentelé autour de l'origine.
    r_min/r_max en "unités" (mm ou px selon ton espace).
    """
    n = rng.randint(n_base[0], n_base[1])
    base_r = rng.uniform(r_min, r_max)

    # Points radiaux de base
    pts: List[Point] = []
    for i in range(n):
        a = (2 * math.pi / n) * i + math.radians(rng.uniform(-18, 18))
        r = base_r * rng.uniform(0.62, 1.05)
        pts.append((math.cos(a) * r, math.sin(a) * r))

    # Dentelage: subdivise chaque arête et perturbe perpendiculairement
    dent: List[Point] = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dent.append((x1, y1))

        k = rng.randint(jag_subdiv[0], jag_subdiv[1])
        for j in range(1, k + 1):
            t = j / (k + 1)
            mx = x1 + (x2 - x1) * t
            my = y1 + (y2 - y1) * t

            # normale
            vx, vy = (x2 - x1), (y2 - y1)
            L = math.hypot(vx, vy) + 1e-9
            nx, ny = (-vy / L), (vx / L)

            amp = base_r * rng.uniform(jag_amp[0], jag_amp[1])
            sign = -1 if rng.random() < 0.5 else 1
            dent.append((mx + nx * amp * sign, my + ny * amp * sign))

    return dent


# ============================================================
# Shapes
# ============================================================

@dataclass
class Shape:
    poly: List[Point]   # points en "px" (coord canvas)
    color_key: str      # "C1"/"C2"/"C3"/"C4"
    z: int              # ordre de dessin


# ============================================================
# Génération du motif (tileable par wrap)
# ============================================================

def generate_camo(
    *,
    seed: int,
    tile_mm: int = 640,
    px_per_mm: float = 2.0,      # augmente pour plus de finesse (ex 3.0 ou 4.0)
    macro_count: int = 42,
    micro_count: int = 140,
) -> Tuple[int, List[Shape]]:
    """
    Retourne (tile_px, shapes)
    """
    rng = random.Random(seed)
    W = int(round(tile_mm * px_per_mm))
    H = W

    shapes: List[Shape] = []

    # --- Répartition des tailles macro (3 échelles distinctes) ---
    # moyennes 50–60mm (50%), grandes 70–80mm (30%), très grandes 90–110mm (20%)
    macro_sizes_mm: List[Tuple[float, float]] = []
    n_mid = int(round(macro_count * 0.50))
    n_big = int(round(macro_count * 0.30))
    n_huge = macro_count - n_mid - n_big

    macro_sizes_mm += [(50, 60)] * n_mid
    macro_sizes_mm += [(70, 80)] * n_big
    macro_sizes_mm += [(90, 110)] * n_huge
    rng.shuffle(macro_sizes_mm)

    # --- Placement macro : désordre contrôlé (orientations indépendantes) ---
    for (a_mm, b_mm) in macro_sizes_mm:
        r_min = a_mm * px_per_mm * 0.50
        r_max = b_mm * px_per_mm * 0.60

        poly0 = random_blob(rng, r_min, r_max)

        # Orientation indépendante (casse toute direction)
        ang = rng.uniform(0, 2 * math.pi)
        poly1 = rotate_points(poly0, ang)

        # Anisotropie aléatoire (compact / étiré)
        sx = rng.uniform(0.75, 1.35)
        sy = rng.uniform(0.75, 1.35)
        poly2 = [(x * sx, y * sy) for x, y in poly1]

        # Position
        cx = rng.uniform(0, W)
        cy = rng.uniform(0, H)
        poly = translate_points(poly2, cx, cy)

        # Couleur par rôle :
        # - C1 seulement macro (grandes masses)
        # - C3 macro (grandes + moyennes)
        size_class = "mid" if (a_mm, b_mm) == (50, 60) else "big" if (a_mm, b_mm) == (70, 80) else "huge"
        if size_class in ("big", "huge"):
            color_key = "C1" if rng.random() < 0.55 else "C3"
        else:
            color_key = "C3" if rng.random() < 0.70 else "C1"

        shapes.append(Shape(poly=poly, color_key=color_key, z=10))

    # --- Micro agressif : mord les contours et parfois traverse 2 macros ---
    for _ in range(micro_count):
        micro_mm = rng.uniform(8, 25)
        r_min = micro_mm * px_per_mm * 0.35
        r_max = micro_mm * px_per_mm * 0.55

        micro0 = random_blob(
            rng, r_min, r_max,
            n_base=(6, 10),
            jag_subdiv=(1, 2),
            jag_amp=(0.12, 0.28),
        )

        ang = rng.uniform(0, 2 * math.pi)
        micro1 = rotate_points(micro0, ang)

        # Choisit une macro cible (z=10)
        tgt = shapes[rng.randrange(0, len(shapes))]
        tgt_poly = tgt.poly

        # Segment du polygone macro (approx "bord")
        k = rng.randrange(0, len(tgt_poly))
        x1, y1 = tgt_poly[k]
        x2, y2 = tgt_poly[(k + 1) % len(tgt_poly)]

        # Point sur l'arête
        t = rng.uniform(0.15, 0.85)
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t

        # Normale pour morsure
        vx, vy = (x2 - x1), (y2 - y1)
        L = math.hypot(vx, vy) + 1e-9
        nx, ny = (-vy / L), (vx / L)

        # Micro-forme chevauche le bord (morsure visible)
        bite = rng.uniform(-1.1, 1.1) * (micro_mm * px_per_mm * 0.55)
        px2 = px + nx * bite
        py2 = py + ny * bite

        # 20% : traverse 2 macros
        if rng.random() < 0.20:
            push = rng.uniform(0.8, 1.4) * (micro_mm * px_per_mm)
            px2 += nx * push
            py2 += ny * push

        micro = translate_points(micro1, px2, py2)

        # Couleurs micro par rôle :
        # - C4 prioritaire (vert-de-gris)
        # - C2 secondaire (Terre de France, normée #7C6D66)
        color_key = "C4" if rng.random() < 0.68 else "C2"

        shapes.append(Shape(poly=micro, color_key=color_key, z=30))

    shapes.sort(key=lambda s: s.z)
    return W, shapes


# ============================================================
# Rendu TILEABLE (wrap) : on dessine chaque shape en 9 copies
# ============================================================

def wrapped_instances(poly: List[Point], W: int, H: int) -> List[List[Point]]:
    inst: List[List[Point]] = []
    for dx in (-W, 0, W):
        for dy in (-H, 0, H):
            inst.append([(x + dx, y + dy) for x, y in poly])
    return inst


def render_png(W: int, shapes: List[Shape], out_png: str) -> None:
    img = Image.new("RGB", (W, W), hex_to_rgb(C3_OLIVE))  # fond neutre olive
    draw = ImageDraw.Draw(img)

    for s in shapes:
        rgb = hex_to_rgb(PALETTE[s.color_key])
        for poly in wrapped_instances(s.poly, W, W):
            pts = [(int(round(x)), int(round(y))) for (x, y) in poly]
            draw.polygon(pts, fill=rgb)

    img.save(out_png, format="PNG")


def render_svg(W: int, shapes: List[Shape], out_svg: str) -> None:
    def pts_str(poly: List[Point]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in poly)

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W}" '
        f'viewBox="0 0 {W} {W}">'
    )

    # Fond
    lines.append(f'<rect x="0" y="0" width="{W}" height="{W}" fill="{C3_OLIVE}"/>')

    for s in shapes:
        fill = PALETTE[s.color_key]
        for poly in wrapped_instances(s.poly, W, W):
            lines.append(f'<polygon points="{pts_str(poly)}" fill="{fill}" />')

    lines.append("</svg>")

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Générateur camouflage tileable (macro hiérarchisé + micro agressif).")
    ap.add_argument("--seed", type=int, default=12345, help="Seed pour reproductibilité.")
    ap.add_argument("--tile-mm", type=int, default=640, help="Taille du tile en mm.")
    ap.add_argument("--px-per-mm", type=float, default=2.0, help="Résolution (px par mm).")
    ap.add_argument("--macro", type=int, default=42, help="Nombre de formes macro.")
    ap.add_argument("--micro", type=int, default=140, help="Nombre de formes micro.")
    ap.add_argument("--out", type=str, default="camo", help="Préfixe de sortie (sans extension).")
    args = ap.parse_args()

    W, shapes = generate_camo(
        seed=args.seed,
        tile_mm=args.tile_mm,
        px_per_mm=args.px_per_mm,
        macro_count=args.macro,
        micro_count=args.micro,
    )

    out_png = f"{args.out}.png"
    out_svg = f"{args.out}.svg"

    render_png(W, shapes, out_png)
    render_svg(W, shapes, out_svg)

    # Résumé minimal (approx, car superpositions possibles)
    total_area = sum(polygon_area(s.poly) for s in shapes)
    area_by = {"C1": 0.0, "C2": 0.0, "C3": 0.0, "C4": 0.0}
    for s in shapes:
        area_by[s.color_key] += polygon_area(s.poly)

    print(f"OK: {out_png} / {out_svg}")
    print(f"Tile: {W}x{W} px  | seed={args.seed} | px_per_mm={args.px_per_mm}")
    print("Aire (approx, superpositions possibles) :")
    for k in ("C1", "C3", "C2", "C4"):
        ratio = area_by[k] / total_area if total_area > 0 else 0.0
        print(f"  {k} {PALETTE[k]} : {ratio*100:.1f}% (cible {TARGET_RATIO[k]*100:.1f}%)")


if __name__ == "__main__":
    main()
