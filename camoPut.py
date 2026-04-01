# -*- coding: utf-8 -*-
"""
apply_camo_on_uniform.py

Applique un camouflage uniquement sur l'uniforme d'un soldat à partir :
- d'une image du soldat
- d'un motif de camouflage PNG

Fonctionne particulièrement bien si :
- le fond est clair ou propre,
- l'uniforme a une couleur bien distincte,
- les accessoires (arme, gilet, bottes) sont d'une autre couleur.

Dépendances :
    pip install opencv-python pillow numpy
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ============================================================
# Réglages principaux
# ============================================================

# Seuils HSV pour détecter le vert de l'uniforme
# À ajuster si nécessaire sur d'autres images.
GREEN_LOWER = np.array([35, 35, 25], dtype=np.uint8)
GREEN_UPPER = np.array([95, 255, 255], dtype=np.uint8)

# Surface minimale pour garder un composant du masque
MIN_COMPONENT_AREA = 1200

# Flou de bord pour une intégration plus naturelle
MASK_FEATHER_SIGMA = 2.2

# Taille de noyau morphologique
MORPH_KERNEL = 5

# ============================================================
# Sélection de fichiers
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

    path = input(f"{title} - entre le chemin du fichier : ").strip().strip('"')
    return path


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

    path = input(f"{title} - entre le chemin de sortie (.png) : ").strip().strip('"')
    return path


# ============================================================
# Chargement / sauvegarde
# ============================================================

def load_image_bgr(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return img


def load_pattern_bgr(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    # On passe par PIL pour gérer proprement PNG/RGBA
    pil = Image.open(path).convert("RGBA")
    rgba = np.array(pil)

    rgb = rgba[..., :3].astype(np.uint8)
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    # Si le PNG a de la transparence, on le compose sur lui-même
    # au lieu de laisser des trous noirs.
    if np.any(alpha < 1.0):
        bg = np.zeros_like(rgb, dtype=np.float32)
        composed = rgb.astype(np.float32) * alpha[..., None] + bg * (1.0 - alpha[..., None])
        rgb = np.clip(composed, 0, 255).astype(np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_bgr(path: str, img_bgr: np.ndarray):
    out_dir = Path(path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(path, img_bgr)
    if not ok:
        raise IOError(f"Impossible d'écrire le fichier : {path}")


# ============================================================
# Outils de masque
# ============================================================

def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255

    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def compute_subject_mask(img_bgr: np.ndarray) -> np.ndarray:
    # Fond très clair => on écarte les zones presque blanches
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Sujet = pixels ni trop blancs ni totalement désaturés
    subject = ((gray < 245) | (hsv[..., 1] > 20)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, kernel, iterations=2)
    subject = cv2.morphologyEx(subject, cv2.MORPH_OPEN, kernel, iterations=1)
    subject = remove_small_components(subject, 5000)
    subject = fill_holes(subject)
    return subject


def compute_uniform_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    subject_mask = compute_subject_mask(img_bgr)

    # Détection du vert principal de l'uniforme
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # On contraint à l'intérieur du sujet
    mask = cv2.bitwise_and(green_mask, subject_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = remove_small_components(mask, MIN_COMPONENT_AREA)
    mask = fill_holes(mask)

    # Adoucissement final
    mask = cv2.GaussianBlur(mask, (0, 0), MASK_FEATHER_SIGMA)
    return mask


# ============================================================
# Motif de camouflage
# ============================================================

def tile_pattern(pattern_bgr: np.ndarray, target_h: int, target_w: int, scale: float = 1.0) -> np.ndarray:
    if scale <= 0:
        raise ValueError("La valeur de scale doit être > 0.")

    ph, pw = pattern_bgr.shape[:2]
    new_w = max(1, int(pw * scale))
    new_h = max(1, int(ph * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(pattern_bgr, (new_w, new_h), interpolation=interp)

    rep_x = math.ceil(target_w / new_w)
    rep_y = math.ceil(target_h / new_h)

    tiled = np.tile(resized, (rep_y, rep_x, 1))
    return tiled[:target_h, :target_w].copy()


# ============================================================
# Fusion
# ============================================================

def make_shading_map(img_bgr: np.ndarray) -> np.ndarray:
    """
    Produit une carte d'ombrage pour conserver les plis et le volume du tissu.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Ombre globale
    base = cv2.GaussianBlur(gray, (0, 0), 11)

    # Détail local des plis
    detail = gray / (base + 1e-6)
    detail = np.clip(detail, 0.80, 1.20)

    # Échelle de luminosité
    shade = 0.55 + 0.90 * gray
    shade = np.clip(shade, 0.55, 1.35)

    return np.clip(shade * detail, 0.45, 1.45)


def apply_camo_on_uniform(img_bgr: np.ndarray, pattern_bgr: np.ndarray, mask: np.ndarray, pattern_scale: float = 1.0) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    tiled = tile_pattern(pattern_bgr, h, w, scale=pattern_scale)

    shading = make_shading_map(img_bgr)

    camo = tiled.astype(np.float32) / 255.0
    camo = camo * shading[..., None]
    camo = np.clip(camo, 0.0, 1.0)

    original = img_bgr.astype(np.float32) / 255.0
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    # Fusion
    result = original * (1.0 - alpha) + camo * alpha
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


# ============================================================
# Programme principal
# ============================================================

def main():
    print("=== Application de camouflage sur uniforme ===")

    soldier_path = ask_open_file(
        "Sélectionne l'image du soldat",
        [("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("Tous les fichiers", "*.*")]
    )
    if not soldier_path:
        print("Aucune image soldat sélectionnée.")
        return

    camo_path = ask_open_file(
        "Sélectionne le motif de camouflage PNG",
        [("PNG", "*.png"), ("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("Tous les fichiers", "*.*")]
    )
    if not camo_path:
        print("Aucun motif de camouflage sélectionné.")
        return

    try:
        scale_str = input("Échelle du motif (1.0 = taille d'origine, ex: 0.75) [1.0] : ").strip()
        pattern_scale = float(scale_str) if scale_str else 1.0
    except ValueError:
        pattern_scale = 1.0

    img_bgr = load_image_bgr(soldier_path)
    pattern_bgr = load_pattern_bgr(camo_path)

    mask = compute_uniform_mask(img_bgr)
    result = apply_camo_on_uniform(img_bgr, pattern_bgr, mask, pattern_scale=pattern_scale)

    soldier_name = Path(soldier_path).stem
    out_path = ask_save_file(
        "Choisis où enregistrer le résultat",
        f"{soldier_name}_uniforme_camo.png"
    )
    if not out_path:
        print("Annulé.")
        return

    mask_path = str(Path(out_path).with_name(Path(out_path).stem + "_mask.png"))

    save_bgr(out_path, result)
    save_bgr(mask_path, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    print(f"Image enregistrée : {out_path}")
    print(f"Masque enregistré : {mask_path}")
    print("Terminé.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)