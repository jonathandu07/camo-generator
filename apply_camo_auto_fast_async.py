# -*- coding: utf-8 -*-
"""
apply_camo_auto_fast_async.py

Version plus rapide et plus agressive sur les ressources :
- prend automatiquement 1774949910078.png
- utilise OpenCV optimisé + tous les threads CPU
- active OpenCL si disponible
- exécute en parallèle les étapes indépendantes
- réduit le coût d'I/O PNG
- ne demande que le motif de camouflage

Dépendances :
    pip install opencv-python numpy
"""

from __future__ import annotations

import os

# ============================================================
# Variables d'environnement AVANT numpy/cv2
# ============================================================

CPU_COUNT = max(1, os.cpu_count() or 1)

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_COUNT))

import asyncio
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Optimisations globales OpenCV
# ============================================================

cv2.setUseOptimized(True)
cv2.setNumThreads(CPU_COUNT)

if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)

FAST_POOL = ThreadPoolExecutor(
    max_workers=max(4, CPU_COUNT * 2),
    thread_name_prefix="camo_fast"
)


# ============================================================
# Réglages
# ============================================================

DEFAULT_IMAGE_NAME = "1774949910078.png"

DEFAULT_SOLDIER_CANDIDATES = [
    Path(DEFAULT_IMAGE_NAME),
    Path(__file__).with_name(DEFAULT_IMAGE_NAME),
    Path("/mnt/data") / DEFAULT_IMAGE_NAME,
]

INITIAL_GREEN_LOWER = np.array([35, 20, 20], dtype=np.uint8)
INITIAL_GREEN_UPPER = np.array([95, 255, 200], dtype=np.uint8)

MIN_SUBJECT_AREA = 5000
MIN_UNIFORM_AREA = 1500
MASK_FEATHER_SIGMA = 2.0

# PNG compression faible = plus rapide à écrire
PNG_COMPRESSION = 1


# ============================================================
# Dataclass
# ============================================================

@dataclass
class UniformAnalysis:
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    h_q05: float
    h_q95: float
    s_q05: float
    s_q95: float
    v_q05: float
    v_q95: float


# ============================================================
# Utilitaires async
# ============================================================

async def run_cpu(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    fn = partial(func, *args, **kwargs)
    return await loop.run_in_executor(FAST_POOL, fn)


# ============================================================
# Dialogues
# ============================================================

def _try_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update()
        return root, filedialog
    except Exception:
        return None, None


def ask_open_file(title: str, filetypes):
    root, filedialog = _try_gui()
    if root is not None:
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        if path:
            return path

    return input(f"{title} - chemin du fichier : ").strip().strip('"')


def ask_save_file(title: str, initial_name: str):
    root, filedialog = _try_gui()
    if root is not None:
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".png",
            initialfile=initial_name,
            filetypes=[("PNG", "*.png")]
        )
        root.destroy()
        if path:
            return path

    return input(f"{title} - chemin de sortie (.png) : ").strip().strip('"')


# ============================================================
# I/O image
# ============================================================

def find_default_soldier_image() -> Path:
    for candidate in DEFAULT_SOLDIER_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Impossible de trouver automatiquement {DEFAULT_IMAGE_NAME}."
    )


def load_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return img


def load_pattern_bgr(path: str | Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Impossible de lire le motif : {path}")

    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    if raw.shape[2] == 3:
        return raw

    # BGRA -> composition rapide sur noir
    bgr = raw[..., :3].astype(np.float32)
    alpha = raw[..., 3:4].astype(np.float32) / 255.0
    composed = bgr * alpha
    return np.clip(composed, 0, 255).astype(np.uint8)


def save_bgr(path: str | Path, img_bgr: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        img_bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
    )
    if not ok:
        raise IOError(f"Impossible d'écrire : {path}")


# ============================================================
# Masques
# ============================================================

def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255

    return out


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask), (0, 0, 0, 0), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))

    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    area = int(stats[best, cv2.CC_STAT_AREA])

    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out, (x, y, w, h), area


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def compute_subject_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    subject = ((gray < 245) | (hsv[..., 1] > 20)).astype(np.uint8) * 255

    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, kernel9, iterations=2)
    subject = cv2.morphologyEx(subject, cv2.MORPH_OPEN, kernel9, iterations=1)
    subject = remove_small_components(subject, MIN_SUBJECT_AREA)
    subject = fill_holes(subject)
    return subject


def analyze_uniform_signature(img_bgr: np.ndarray) -> UniformAnalysis:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    subject_mask = compute_subject_mask(img_bgr)

    broad_green = cv2.inRange(hsv, INITIAL_GREEN_LOWER, INITIAL_GREEN_UPPER)
    broad_green = cv2.bitwise_and(broad_green, subject_mask)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    broad_green = cv2.morphologyEx(broad_green, cv2.MORPH_OPEN, kernel5, iterations=1)
    broad_green = cv2.morphologyEx(broad_green, cv2.MORPH_CLOSE, kernel5, iterations=2)

    main_green, bbox, area = largest_component(broad_green)
    if area <= 0:
        raise RuntimeError("Impossible d'isoler l'uniforme.")

    pixels = hsv[main_green > 0]
    if len(pixels) < 1000:
        raise RuntimeError("Pas assez de pixels pour analyser l'uniforme.")

    h_vals = pixels[:, 0].astype(np.float32)
    s_vals = pixels[:, 1].astype(np.float32)
    v_vals = pixels[:, 2].astype(np.float32)

    h_q05 = float(np.quantile(h_vals, 0.05))
    h_q95 = float(np.quantile(h_vals, 0.95))
    s_q05 = float(np.quantile(s_vals, 0.05))
    s_q95 = float(np.quantile(s_vals, 0.95))
    v_q05 = float(np.quantile(v_vals, 0.05))
    v_q95 = float(np.quantile(v_vals, 0.95))

    lower = np.array([
        max(35, int(np.quantile(h_vals, 0.02)) - 4),
        max(18, int(np.quantile(s_vals, 0.03)) - 10),
        max(18, int(np.quantile(v_vals, 0.03)) - 10),
    ], dtype=np.uint8)

    upper = np.array([
        min(95, int(np.quantile(h_vals, 0.98)) + 4),
        255,
        min(175, int(np.quantile(v_vals, 0.98)) + 25),
    ], dtype=np.uint8)

    return UniformAnalysis(
        hsv_lower=lower,
        hsv_upper=upper,
        bbox=bbox,
        area=area,
        h_q05=h_q05,
        h_q95=h_q95,
        s_q05=s_q05,
        s_q95=s_q95,
        v_q05=v_q05,
        v_q95=v_q95,
    )


def compute_uniform_mask(img_bgr: np.ndarray, analysis: UniformAnalysis) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    subject_mask = compute_subject_mask(img_bgr)

    mask = cv2.inRange(hsv, analysis.hsv_lower, analysis.hsv_upper)
    mask = cv2.bitwise_and(mask, subject_mask)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    mask, _, area = largest_component(mask)
    if area < MIN_UNIFORM_AREA:
        raise RuntimeError("Masque uniforme vide ou trop petit.")

    mask = fill_holes(mask)
    mask = cv2.GaussianBlur(mask, (0, 0), MASK_FEATHER_SIGMA)
    return mask


# ============================================================
# Camouflage
# ============================================================

def tile_pattern(pattern_bgr: np.ndarray, target_h: int, target_w: int, scale: float = 1.0) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale doit être > 0.")

    ph, pw = pattern_bgr.shape[:2]
    new_w = max(1, int(pw * scale))
    new_h = max(1, int(ph * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(pattern_bgr, (new_w, new_h), interpolation=interp)

    rep_x = math.ceil(target_w / new_w)
    rep_y = math.ceil(target_h / new_h)

    tiled = np.tile(resized, (rep_y, rep_x, 1))
    return tiled[:target_h, :target_w]


def make_shading_map(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    base = cv2.GaussianBlur(gray, (0, 0), 11)
    detail = gray / (base + 1e-6)
    detail = np.clip(detail, 0.80, 1.20)

    shade = 0.55 + 0.90 * gray
    shade = np.clip(shade, 0.55, 1.35)

    return np.clip(shade * detail, 0.45, 1.45)


def compose_camo_with_precomputed(
    img_bgr: np.ndarray,
    tiled_pattern_bgr: np.ndarray,
    shading: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    camo = tiled_pattern_bgr.astype(np.float32) / 255.0
    camo *= shading[..., None]
    camo = np.clip(camo, 0.0, 1.0)

    original = img_bgr.astype(np.float32) / 255.0
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    result = original * (1.0 - alpha) + camo * alpha
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# Main async
# ============================================================

async def main_async():
    soldier_path = find_default_soldier_image()
    print(f"Image automatique : {soldier_path}")
    print(f"CPU logiques      : {CPU_COUNT}")
    print(f"OpenCL disponible : {cv2.ocl.haveOpenCL()}")
    print(f"OpenCL actif      : {cv2.ocl.useOpenCL()}")

    camo_path = ask_open_file(
        "Sélectionne le motif de camouflage PNG",
        [("PNG", "*.png"), ("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("Tous les fichiers", "*.*")]
    )
    if not camo_path:
        print("Aucun motif sélectionné.")
        return

    scale_str = input("Échelle du motif [1] : ").strip()
    try:
        pattern_scale = float(scale_str) if scale_str else 1.0
    except ValueError:
        pattern_scale = 1.0

    # Chargement parallèle
    img_task = run_cpu(load_image_bgr, soldier_path)
    pattern_task = run_cpu(load_pattern_bgr, camo_path)
    img_bgr, pattern_bgr = await asyncio.gather(img_task, pattern_task)

    h, w = img_bgr.shape[:2]

    # Étapes indépendantes parallélisées au maximum
    analysis_task = asyncio.create_task(run_cpu(analyze_uniform_signature, img_bgr))
    shading_task = asyncio.create_task(run_cpu(make_shading_map, img_bgr))
    tiled_task = asyncio.create_task(run_cpu(tile_pattern, pattern_bgr, h, w, pattern_scale))

    analysis = await analysis_task
    print("Analyse automatique :")
    print(f"  HSV lower : {analysis.hsv_lower.tolist()}")
    print(f"  HSV upper : {analysis.hsv_upper.tolist()}")
    print(f"  bbox      : {analysis.bbox}")
    print(f"  aire      : {analysis.area}")

    # Le masque dépend de l'analyse
    mask_task = asyncio.create_task(run_cpu(compute_uniform_mask, img_bgr, analysis))

    shading, tiled_pattern_bgr, mask = await asyncio.gather(
        shading_task,
        tiled_task,
        mask_task,
    )

    # Composition finale
    result = await run_cpu(
        compose_camo_with_precomputed,
        img_bgr,
        tiled_pattern_bgr,
        shading,
        mask,
    )

    out_path = ask_save_file(
        "Choisis où enregistrer le résultat",
        f"{soldier_path.stem}_uniforme_camo.png"
    )
    if not out_path:
        print("Annulé.")
        return

    out_path = Path(out_path)
    mask_path = out_path.with_name(out_path.stem + "_mask.png")
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Sauvegarde parallèle
    await asyncio.gather(
        run_cpu(save_bgr, out_path, result),
        run_cpu(save_bgr, mask_path, mask_bgr),
    )

    print(f"Résultat : {out_path}")
    print(f"Masque   : {mask_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrompu.")
    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)
    finally:
        FAST_POOL.shutdown(wait=False, cancel_futures=True)