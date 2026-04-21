# -*- coding: utf-8 -*-
"""
camouflage_ml_dl.py

Pipeline hybride ML + DL compatible avec le main.py réel.

Extensions de cette version :
- guidage ML/DL de la génération des seeds ;
- guidage optionnel de la projection mannequin si un backend de projection
  compatible (start_corrected.py ou start.py) est disponible ;
- sélection conjointe seed + projection_scale ;
- reward combiné backend + projection.

Le motif_scale, la palette et les proportions couleur restent fixés par
l'utilisateur et/ou le backend principal. Le ML/DL ne modifie pas ces choix.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    random_split = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def _resolve_camo_module():
    for name in ("main", "__main__"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "generate_and_validate_from_seed") and hasattr(mod, "validate_with_reasons"):
            return mod
    return importlib.import_module("main")


camo = _resolve_camo_module()


# ============================================================
# PROJECTION BACKEND OPTIONNEL
# ============================================================


def _resolve_projection_module() -> Optional[Any]:
    for name in ("start_corrected", "start"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "projection_preview_with_report"):
            return mod

    for name in ("start_corrected", "start"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "projection_preview_with_report"):
            return mod
    return None


projection_mod = _resolve_projection_module()
PROJECTION_AVAILABLE = bool(
    projection_mod is not None
    and hasattr(projection_mod, "projection_preview_with_report")
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_KEYS: tuple[str, ...] = (
    "ratio_coyote",
    "ratio_olive",
    "ratio_terre",
    "ratio_gris",
    "abs_err_coyote",
    "abs_err_olive",
    "abs_err_terre",
    "abs_err_gris",
    "mean_abs_error",
    "largest_component_ratio_class_1",
    "boundary_density",
    "boundary_density_small",
    "boundary_density_tiny",
    "mirror_similarity",
    "edge_contact_ratio",
    "overscan",
    "shift_strength",
    "weak_ratio",
    "micro_components_per_mp",
    "orphan_ratio",
    "mode_filtered_ratio",
    "macro_prior_agreement",
    "macro_guide_agreement",
    "anti_pixel_enabled",
)

PROJECTION_FEATURE_KEYS: tuple[str, ...] = (
    "projection_scale",
    "projection_valid",
    "projection_uniform_pixels_norm",
    "projection_residual_pixels_norm",
    "projection_still_green_pixels_norm",
    "projection_residual_ratio",
    "projection_mean_lab_distance",
    "projection_mean_rgb_delta",
)

FAILURE_KEYS: tuple[str, ...] = (
    "abs_err_coyote",
    "abs_err_olive",
    "abs_err_terre",
    "abs_err_gris",
    "mean_abs_error",
    "boundary_density_low",
    "boundary_density_high",
    "boundary_density_small_low",
    "boundary_density_small_high",
    "boundary_density_tiny_low",
    "boundary_density_tiny_high",
    "mirror_similarity_high",
    "largest_olive_component_ratio_low",
    "edge_contact_ratio_high",
    "weak_ratio_high",
    "micro_components_high",
    "orphan_ratio_high",
)

ACTION_LIBRARY: tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("linear_step_1", {"mode": "linear", "offset": 0, "step": 1}),
    ("linear_step_2", {"mode": "linear", "offset": 17, "step": 2}),
    ("linear_step_3", {"mode": "linear", "offset": 53, "step": 3}),
    ("offset_97", {"mode": "offset", "offset": 97}),
    ("offset_997", {"mode": "offset", "offset": 997}),
    ("offset_7919", {"mode": "offset", "offset": 7919}),
    ("affine_small", {"mode": "affine", "mul": 3, "add": 101}),
    ("affine_medium", {"mode": "affine", "mul": 5, "add": 1009}),
    ("xor_low", {"mode": "xor", "mask": 0x9E37}),
    ("xor_high", {"mode": "xor", "mask": 0x5A5A5A}),
)


@dataclass
class MLDLConfig:
    target_count: int = 20
    warmup_samples: int = 128
    candidate_pool_size: int = 8
    validate_top_k: int = 3
    max_attempts_per_target: int = 120
    train_epochs: int = 24
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    device: str = "auto"
    base_seed: int = camo.DEFAULT_BASE_SEED
    output_dir: str = "camouflages_ml_dl"
    checkpoint_name: str = "surrogate_camouflage.pt"
    dataset_name: str = "dataset_camouflage_ml_dl.npz"
    report_name: str = "rapport_camouflages_ml_dl.csv"
    alpha_ucb: float = 1.25
    min_train_size: int = 12
    retrain_every: int = 8
    val_split: float = 0.15
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    random_seed: int = 12345
    parallel_train_enabled: bool = True
    parallel_train_min_interval_s: float = 3.0

    pretrain_relax_level: float = 0.0
    pretrain_max_orphan_ratio: Optional[float] = None
    pretrain_max_micro_islands_per_mp: Optional[float] = None
    warmup_persist_every: int = 8
    tolerance_state_name: str = "adaptive_tolerance_state.json"
    bootstrap_first_candidate: bool = True
    bootstrap_image_name: str = "bootstrap_reference.png"

    max_repair_rounds: int = getattr(camo, "MAX_REPAIR_ROUNDS", 3)
    anti_pixel: bool = bool(getattr(camo, "DEFAULT_ENABLE_ANTI_PIXEL", True))

    # Projection ML/DL
    projection_ml_enabled: bool = True
    projection_base_scale: float = 1.0
    projection_scale_candidates: Tuple[float, ...] = (0.82, 0.92, 1.00, 1.08, 1.18)
    projection_preview_mode: str = "fast"
    projection_final_mode: str = "quality"
    projection_reward_weight: float = 0.65


@dataclass
class RejectionAnalysis:
    target_index: int
    local_attempt: int
    seed: int
    fail_count: int
    severity: float
    failure_names: List[str]
    notes: List[str]


@dataclass
class ProjectionStats:
    scale: float = 1.0
    valid: bool = False
    uniform_pixels: int = 0
    residual_pixels: int = 0
    still_green_pixels: int = 0
    residual_ratio: float = 1.0
    mean_lab_distance: float = 0.0
    mean_rgb_delta: float = 0.0

    def to_feature_dict(self) -> Dict[str, float]:
        uniform = max(1, int(self.uniform_pixels))
        return {
            "projection_scale": float(self.scale),
            "projection_valid": float(1.0 if self.valid else 0.0),
            "projection_uniform_pixels_norm": float(min(1.0, uniform / 500000.0)),
            "projection_residual_pixels_norm": float(min(1.0, self.residual_pixels / 50000.0)),
            "projection_still_green_pixels_norm": float(min(1.0, self.still_green_pixels / 50000.0)),
            "projection_residual_ratio": float(self.residual_ratio),
            "projection_mean_lab_distance": float(min(5.0, self.mean_lab_distance / 30.0)),
            "projection_mean_rgb_delta": float(min(5.0, self.mean_rgb_delta / 25.0)),
        }


@dataclass
class Proposal:
    seed: int
    action_idx: int
    action_name: str
    candidate: Any
    pred_valid: float
    pred_reward: float
    projection_scale: float = 1.0
    projection_stats: Optional[ProjectionStats] = None


# ============================================================
# OUTILS FEATURES / REWARD
# ============================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def candidate_to_feature_dict(candidate: Any) -> Dict[str, float]:
    rs = np.asarray(candidate.ratios, dtype=float)
    abs_err = np.abs(rs - camo.TARGET)
    m = dict(candidate.metrics)
    pixel_count = max(1, int(candidate.label_map.size))
    mode_filtered_pixels = _safe_float(m.get("mode_filtered_pixels", 0.0))
    return {
        "ratio_coyote": float(rs[camo.IDX_0]),
        "ratio_olive": float(rs[camo.IDX_1]),
        "ratio_terre": float(rs[camo.IDX_2]),
        "ratio_gris": float(rs[camo.IDX_3]),
        "abs_err_coyote": float(abs_err[camo.IDX_0]),
        "abs_err_olive": float(abs_err[camo.IDX_1]),
        "abs_err_terre": float(abs_err[camo.IDX_2]),
        "abs_err_gris": float(abs_err[camo.IDX_3]),
        "mean_abs_error": float(np.mean(abs_err)),
        "largest_component_ratio_class_1": _safe_float(m.get("largest_component_ratio_class_1", 0.0)),
        "boundary_density": _safe_float(m.get("boundary_density", 0.0)),
        "boundary_density_small": _safe_float(m.get("boundary_density_small", 0.0)),
        "boundary_density_tiny": _safe_float(m.get("boundary_density_tiny", 0.0)),
        "mirror_similarity": _safe_float(m.get("mirror_similarity", 0.0)),
        "edge_contact_ratio": _safe_float(m.get("edge_contact_ratio", 0.0)),
        "overscan": _safe_float(m.get("overscan", 0.0)),
        "shift_strength": _safe_float(m.get("shift_strength", 0.0)),
        "weak_ratio": _safe_float(m.get("weak_ratio", 0.0)),
        "micro_components_per_mp": _safe_float(m.get("micro_components_per_mp", 0.0)),
        "orphan_ratio": _safe_float(m.get("orphan_ratio", 0.0)),
        "mode_filtered_ratio": float(mode_filtered_pixels / pixel_count),
        "macro_prior_agreement": _safe_float(m.get("macro_prior_agreement", 0.0)),
        "macro_guide_agreement": _safe_float(m.get("macro_guide_agreement", 0.0)),
        "anti_pixel_enabled": _safe_float(m.get("anti_pixel_enabled", 0.0)),
    }


def projection_stats_to_feature_dict(stats: Optional[ProjectionStats]) -> Dict[str, float]:
    if stats is None:
        return ProjectionStats().to_feature_dict()
    return stats.to_feature_dict()


def candidate_to_feature_vector(candidate: Any, projection_stats: Optional[ProjectionStats] = None) -> np.ndarray:
    feat = candidate_to_feature_dict(candidate)
    pfeat = projection_stats_to_feature_dict(projection_stats)
    ordered = [feat.get(name, 0.0) for name in FEATURE_KEYS] + [pfeat.get(name, 0.0) for name in PROJECTION_FEATURE_KEYS]
    return np.array(ordered, dtype=np.float32)


def analyze_rejection(candidate: Any, target_index: int, local_attempt: int) -> RejectionAnalysis:
    feat = candidate_to_feature_dict(candidate)
    failures: List[str] = []
    notes: List[str] = []

    per_color_names = ("abs_err_coyote", "abs_err_olive", "abs_err_terre", "abs_err_gris")
    for idx, name in enumerate(per_color_names):
        if feat[name] > float(camo.MAX_ABS_ERROR_PER_COLOR[idx]):
            failures.append(name)

    if feat["mean_abs_error"] > float(camo.MAX_MEAN_ABS_ERROR):
        failures.append("mean_abs_error")
    if feat["boundary_density"] < float(camo.MIN_BOUNDARY_DENSITY):
        failures.append("boundary_density_low")
    if feat["boundary_density"] > float(camo.MAX_BOUNDARY_DENSITY):
        failures.append("boundary_density_high")
    if feat["boundary_density_small"] < float(camo.MIN_BOUNDARY_DENSITY_SMALL):
        failures.append("boundary_density_small_low")
    if feat["boundary_density_small"] > float(camo.MAX_BOUNDARY_DENSITY_SMALL):
        failures.append("boundary_density_small_high")
    if feat["boundary_density_tiny"] < float(camo.MIN_BOUNDARY_DENSITY_TINY):
        failures.append("boundary_density_tiny_low")
    if feat["boundary_density_tiny"] > float(camo.MAX_BOUNDARY_DENSITY_TINY):
        failures.append("boundary_density_tiny_high")
    if feat["mirror_similarity"] > float(camo.MAX_MIRROR_SIMILARITY):
        failures.append("mirror_similarity_high")
    if feat["largest_component_ratio_class_1"] < float(camo.MIN_LARGEST_COMPONENT_RATIO_CLASS_1):
        failures.append("largest_olive_component_ratio_low")
    if feat["edge_contact_ratio"] > float(camo.MAX_EDGE_CONTACT_RATIO):
        failures.append("edge_contact_ratio_high")
    if feat["weak_ratio"] > 0.010:
        failures.append("weak_ratio_high")
    if feat["micro_components_per_mp"] > float(getattr(camo, "MAX_MICRO_ISLANDS_PER_MP", 0.0)):
        failures.append("micro_components_high")
    if feat["orphan_ratio"] > float(getattr(camo, "MAX_ORPHAN_RATIO", 0.0)):
        failures.append("orphan_ratio_high")

    severity = 0.0
    severity += 100.0 * feat["mean_abs_error"]
    severity += max(0.0, float(camo.MIN_BOUNDARY_DENSITY) - feat["boundary_density"]) * 10.0
    severity += max(0.0, feat["boundary_density"] - float(camo.MAX_BOUNDARY_DENSITY)) * 10.0
    severity += max(0.0, feat["mirror_similarity"] - float(camo.MAX_MIRROR_SIMILARITY)) * 5.0
    severity += max(0.0, float(camo.MIN_LARGEST_COMPONENT_RATIO_CLASS_1) - feat["largest_component_ratio_class_1"]) * 10.0
    severity += max(0.0, feat["edge_contact_ratio"] - float(camo.MAX_EDGE_CONTACT_RATIO)) * 10.0
    severity += feat["weak_ratio"] * 40.0
    severity += feat["micro_components_per_mp"] * 0.30
    severity += feat["orphan_ratio"] * 1000.0

    if not failures:
        notes.append("Validation négative sans règle identifiée ; vérifier les données.")
    else:
        notes.append(", ".join(failures))

    return RejectionAnalysis(
        target_index=int(target_index),
        local_attempt=int(local_attempt),
        seed=int(candidate.seed),
        fail_count=len(failures),
        severity=float(severity),
        failure_names=failures,
        notes=notes,
    )


def analysis_to_failure_vector(analysis: Optional[RejectionAnalysis]) -> np.ndarray:
    names = set() if analysis is None else set(str(x) for x in analysis.failure_names)
    return np.array([1.0 if name in names else 0.0 for name in FAILURE_KEYS], dtype=np.float32)


def build_context_vector(candidate: Any, analysis: Optional[RejectionAnalysis], projection_stats: Optional[ProjectionStats] = None) -> np.ndarray:
    feat = candidate_to_feature_vector(candidate, projection_stats=projection_stats)
    fail = analysis_to_failure_vector(analysis)
    return np.concatenate([feat, fail], axis=0)


def candidate_reward(candidate: Any, accepted: bool) -> float:
    feat = candidate_to_feature_dict(candidate)
    score = 0.0
    score += 2.0 if accepted else 0.0
    score -= 140.0 * feat["mean_abs_error"]

    def interval_score(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        mid = 0.5 * (low + high)
        radius = 0.5 * (high - low)
        if radius <= 1e-9:
            return 0.0
        return max(0.0, 1.0 - abs(value - mid) / radius)

    score += 0.50 * interval_score(feat["boundary_density"], float(camo.MIN_BOUNDARY_DENSITY), float(camo.MAX_BOUNDARY_DENSITY))
    score += 0.35 * interval_score(feat["boundary_density_small"], float(camo.MIN_BOUNDARY_DENSITY_SMALL), float(camo.MAX_BOUNDARY_DENSITY_SMALL))
    score += 0.25 * interval_score(feat["boundary_density_tiny"], float(camo.MIN_BOUNDARY_DENSITY_TINY), float(camo.MAX_BOUNDARY_DENSITY_TINY))
    score += 0.45 * feat["largest_component_ratio_class_1"]
    score += 0.20 * (1.0 - min(1.0, feat["mirror_similarity"]))
    score += 0.15 * (1.0 - min(1.0, feat["edge_contact_ratio"]))
    score += 0.20 * feat["macro_prior_agreement"]
    score += 0.15 * feat["macro_guide_agreement"]
    score -= 8.0 * feat["weak_ratio"]
    score -= 0.25 * feat["micro_components_per_mp"]
    score -= 25.0 * feat["orphan_ratio"]
    score -= 12.0 * feat["mode_filtered_ratio"]
    return float(score)


def projection_reward(stats: Optional[ProjectionStats]) -> float:
    if stats is None:
        return 0.0
    uniform = max(1, int(stats.uniform_pixels))
    residual_ratio = float(stats.residual_ratio)
    still_ratio = float(stats.still_green_pixels / uniform)
    reward = 0.0
    reward += 1.0 if stats.valid else -0.25
    reward -= 18.0 * residual_ratio
    reward -= 10.0 * still_ratio
    reward -= 2.5 * min(1.0, stats.residual_pixels / 5000.0)
    reward += 0.20 * min(3.0, stats.mean_lab_distance / 20.0)
    reward += 0.10 * min(3.0, stats.mean_rgb_delta / 20.0)
    return float(reward)


def combined_reward(candidate: Any, accepted: bool, projection_stats: Optional[ProjectionStats], weight: float) -> float:
    return float(candidate_reward(candidate, accepted) + float(weight) * projection_reward(projection_stats))


# ============================================================
# NORMALISATION
# ============================================================

class Standardizer:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.std = np.ones(self.dim, dtype=np.float32)
        self.fitted = False

    def fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        self.fitted = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if not self.fitted:
            return x
        return (x - self.mean) / self.std

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fitted": bool(self.fitted)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.std = np.array(state["std"], dtype=np.float32)
        self.fitted = bool(state.get("fitted", True))


# ============================================================
# DEEP LEARNING : SURROGATE
# ============================================================

if TORCH_AVAILABLE:
    class SurrogateNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            half = max(8, hidden_dim // 2)
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, half),
                nn.GELU(),
            )
            self.head_valid = nn.Linear(half, 1)
            self.head_reward = nn.Linear(half, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            z = self.backbone(x)
            valid_logit = self.head_valid(z).squeeze(-1)
            reward = self.head_reward(z).squeeze(-1)
            return valid_logit, reward


class DeepSurrogate:
    def __init__(self, input_dim: int, hidden_dim: int = 128, lr: float = 1e-3, device: str = "cpu") -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch est requis pour ce script : le deep learning doit être réellement entraîné, "
                f"or l'import de torch a échoué: {TORCH_IMPORT_ERROR!r}"
            )

        self.input_dim = int(input_dim)
        self.scaler = Standardizer(input_dim)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.trained = False
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(self.device)

        self.model = SurrogateNet(input_dim=input_dim, hidden_dim=hidden_dim).to(self._torch_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_mse = nn.MSELoss()

    def fit(
        self,
        features: np.ndarray,
        valid: np.ndarray,
        rewards: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        val_split: float = 0.15,
        patience: int = 6,
        min_delta: float = 1e-4,
        random_seed: int = 12345,
    ) -> Dict[str, float]:
        x = np.asarray(features, dtype=np.float32)
        y_valid = np.asarray(valid, dtype=np.float32)
        y_reward = np.asarray(rewards, dtype=np.float32)

        if len(x) < 2:
            raise ValueError("Le deep learning nécessite au moins 2 échantillons.")

        self.scaler.fit(x)
        self.reward_mean = float(np.mean(y_reward)) if len(y_reward) else 0.0
        self.reward_std = float(np.std(y_reward)) if len(y_reward) else 1.0
        if self.reward_std < 1e-6:
            self.reward_std = 1.0

        x_norm = self.scaler.transform(x)
        y_reward_norm = (y_reward - self.reward_mean) / self.reward_std

        ds = TensorDataset(
            torch.from_numpy(x_norm),
            torch.from_numpy(y_valid),
            torch.from_numpy(y_reward_norm.astype(np.float32)),
        )

        n_total = len(ds)
        n_val = max(1, int(round(n_total * max(0.0, min(0.40, float(val_split))))))
        n_train = max(1, n_total - n_val)
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        if n_val <= 0:
            n_val = 1
            n_train = max(1, n_total - 1)

        generator = torch.Generator().manual_seed(int(random_seed))
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)

        train_dl = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

        best_state = None
        best_val_loss = float("inf")
        best_epoch = 0
        no_improve = 0
        last = {
            "epoch": 0.0,
            "train_loss": 0.0,
            "train_loss_valid": 0.0,
            "train_loss_reward": 0.0,
            "val_loss": 0.0,
            "val_loss_valid": 0.0,
            "val_loss_reward": 0.0,
            "best_epoch": 0.0,
        }

        for epoch in range(1, max(1, int(epochs)) + 1):
            self.model.train()
            agg_train_loss = 0.0
            agg_train_bce = 0.0
            agg_train_mse = 0.0
            agg_train_count = 0

            for xb, yb_valid, yb_reward in train_dl:
                xb = xb.to(self._torch_device)
                yb_valid = yb_valid.to(self._torch_device)
                yb_reward = yb_reward.to(self._torch_device)

                self.optimizer.zero_grad(set_to_none=True)
                logit_valid, pred_reward = self.model(xb)
                loss_valid = self.loss_bce(logit_valid, yb_valid)
                loss_reward = self.loss_mse(pred_reward, yb_reward)
                loss = loss_valid + 0.7 * loss_reward
                loss.backward()
                self.optimizer.step()

                bs = int(xb.shape[0])
                agg_train_count += bs
                agg_train_loss += float(loss.item()) * bs
                agg_train_bce += float(loss_valid.item()) * bs
                agg_train_mse += float(loss_reward.item()) * bs

            self.model.eval()
            agg_val_loss = 0.0
            agg_val_bce = 0.0
            agg_val_mse = 0.0
            agg_val_count = 0

            with torch.no_grad():
                for xb, yb_valid, yb_reward in val_dl:
                    xb = xb.to(self._torch_device)
                    yb_valid = yb_valid.to(self._torch_device)
                    yb_reward = yb_reward.to(self._torch_device)

                    logit_valid, pred_reward = self.model(xb)
                    loss_valid = self.loss_bce(logit_valid, yb_valid)
                    loss_reward = self.loss_mse(pred_reward, yb_reward)
                    loss = loss_valid + 0.7 * loss_reward

                    bs = int(xb.shape[0])
                    agg_val_count += bs
                    agg_val_loss += float(loss.item()) * bs
                    agg_val_bce += float(loss_valid.item()) * bs
                    agg_val_mse += float(loss_reward.item()) * bs

            train_loss = agg_train_loss / max(1, agg_train_count)
            train_bce = agg_train_bce / max(1, agg_train_count)
            train_mse = agg_train_mse / max(1, agg_train_count)
            val_loss = agg_val_loss / max(1, agg_val_count)
            val_bce = agg_val_bce / max(1, agg_val_count)
            val_mse = agg_val_mse / max(1, agg_val_count)

            last = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_loss_valid": float(train_bce),
                "train_loss_reward": float(train_mse),
                "val_loss": float(val_loss),
                "val_loss_valid": float(val_bce),
                "val_loss_reward": float(val_mse),
                "best_epoch": float(best_epoch),
            }

            if val_loss + float(min_delta) < best_val_loss:
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= max(1, int(patience)):
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.trained = True
        last["best_epoch"] = float(best_epoch)
        last["best_val_loss"] = float(best_val_loss if best_val_loss < float("inf") else last["val_loss"])
        return last

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Le surrogate DL n'est pas encore entraîné.")

        x = np.asarray(features, dtype=np.float32)
        one = False
        if x.ndim == 1:
            x = x[None, :]
            one = True
        x_norm = self.scaler.transform(x)

        with torch.no_grad():
            xt = torch.from_numpy(x_norm).to(self._torch_device)
            self.model.eval()
            logit_valid, pred_reward = self.model(xt)
            prob_valid = torch.sigmoid(logit_valid).cpu().numpy()
            reward = pred_reward.cpu().numpy() * self.reward_std + self.reward_mean

        if one:
            return prob_valid[0:1], reward[0:1]
        return prob_valid, reward

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self.scaler.state_dict(),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "trained": self.trained,
            "model": self.model.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = torch.load(path, map_location=self._torch_device)
        if "model" in payload:
            self.model.load_state_dict(payload["model"])
        self.scaler.load_state_dict(payload["scaler"])
        self.reward_mean = float(payload.get("reward_mean", 0.0))
        self.reward_std = float(payload.get("reward_std", 1.0))
        self.trained = bool(payload.get("trained", True))


# ============================================================
# MACHINE LEARNING : BANDIT CONTEXTUEL
# ============================================================

class LinUCBBandit:
    def __init__(self, n_actions: int, context_dim: int, alpha: float = 1.25) -> None:
        self.n_actions = int(n_actions)
        self.context_dim = int(context_dim) + 1
        self.alpha = float(alpha)
        self.A = [np.eye(self.context_dim, dtype=np.float64) for _ in range(self.n_actions)]
        self.b = [np.zeros((self.context_dim,), dtype=np.float64) for _ in range(self.n_actions)]

    def _phi(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.float64)
        return np.concatenate([context, np.array([1.0], dtype=np.float64)], axis=0)

    def scores(self, context: np.ndarray) -> np.ndarray:
        phi = self._phi(context)
        out = np.zeros((self.n_actions,), dtype=np.float64)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(theta @ phi)
            unc = float(np.sqrt(phi @ A_inv @ phi))
            out[a] = mean + self.alpha * unc
        return out

    def select_top_k(self, context: np.ndarray, k: int) -> List[int]:
        scores = self.scores(context)
        order = list(np.argsort(scores)[::-1])
        return [int(i) for i in order[:max(1, int(k))]]

    def update(self, action_idx: int, context: np.ndarray, reward: float) -> None:
        phi = self._phi(context)
        a = int(action_idx)
        self.A[a] += np.outer(phi, phi)
        self.b[a] += float(reward) * phi


# ============================================================
# DATASET / BUFFERS
# ============================================================

class ExperienceBuffer:
    def __init__(self) -> None:
        self.features: List[np.ndarray] = []
        self.valid: List[float] = []
        self.rewards: List[float] = []

    @staticmethod
    def expected_dim() -> int:
        return int(len(FEATURE_KEYS) + len(PROJECTION_FEATURE_KEYS))

    @classmethod
    def _normalize_feature_row(cls, row: Any) -> np.ndarray:
        dim = cls.expected_dim()
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if arr.size == dim:
            return arr.astype(np.float32, copy=True)
        if arr.size > dim:
            return arr[:dim].astype(np.float32, copy=True)
        out = np.zeros((dim,), dtype=np.float32)
        out[:arr.size] = arr
        return out

    def normalize_in_place(self) -> None:
        dim = self.expected_dim()
        self.features = [self._normalize_feature_row(row) for row in self.features]
        n = len(self.features)
        if len(self.valid) < n:
            self.valid.extend([0.0] * (n - len(self.valid)))
        if len(self.rewards) < n:
            self.rewards.extend([0.0] * (n - len(self.rewards)))
        self.valid = [float(v) for v in self.valid[:n]]
        self.rewards = [float(v) for v in self.rewards[:n]]
        if n == 0:
            self.features = []
            self.valid = []
            self.rewards = []

    def add(self, candidate: Any, accepted: bool, projection_stats: Optional[ProjectionStats], projection_weight: float) -> float:
        feat = self._normalize_feature_row(candidate_to_feature_vector(candidate, projection_stats=projection_stats))
        reward = combined_reward(candidate, accepted, projection_stats, projection_weight)
        self.features.append(feat)
        self.valid.append(float(1.0 if accepted else 0.0))
        self.rewards.append(float(reward))
        return reward

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.normalize_in_place()
        dim = self.expected_dim()
        x = np.stack(self.features, axis=0).astype(np.float32) if self.features else np.zeros((0, dim), dtype=np.float32)
        y_valid = np.array(self.valid, dtype=np.float32)
        y_reward = np.array(self.rewards, dtype=np.float32)
        return x, y_valid, y_reward

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        x, y_valid, y_reward = self.as_arrays()
        np.savez_compressed(path, x=x, y_valid=y_valid, y_reward=y_reward)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = np.load(path, allow_pickle=False)
        x = np.asarray(data["x"], dtype=np.float32)
        y_valid = np.asarray(data["y_valid"], dtype=np.float32).reshape(-1)
        y_reward = np.asarray(data["y_reward"], dtype=np.float32).reshape(-1)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 0:
            x = np.zeros((0, self.expected_dim()), dtype=np.float32)

        self.features = [self._normalize_feature_row(row) for row in x]
        n = len(self.features)
        self.valid = [float(v) for v in y_valid[:n].tolist()]
        self.rewards = [float(v) for v in y_reward[:n].tolist()]
        self.normalize_in_place()


# ============================================================
# POLITIQUES DE SEED
# ============================================================

def propose_seed(base_seed: int, action: Dict[str, Any]) -> int:
    mode = str(action.get("mode", "linear"))
    seed = int(base_seed)

    if mode == "linear":
        seed = seed + int(action.get("offset", 0)) + int(action.get("step", 1))
    elif mode == "offset":
        seed = seed + int(action.get("offset", 0))
    elif mode == "affine":
        seed = seed * int(action.get("mul", 1)) + int(action.get("add", 0))
    elif mode == "xor":
        seed = seed ^ int(action.get("mask", 0))

    return int(seed & 0x7FFF_FFFF_FFFF_FFFF)


# ============================================================
# PROJECTION HELPERS
# ============================================================


def _extract_projection_stats(report: Any, scale: float) -> ProjectionStats:
    if report is None:
        return ProjectionStats(scale=float(scale))
    return ProjectionStats(
        scale=float(scale),
        valid=bool(getattr(report, "valid", False)),
        uniform_pixels=int(getattr(report, "uniform_pixels", 0) or 0),
        residual_pixels=int(getattr(report, "residual_pixels", 0) or 0),
        still_green_pixels=int(getattr(report, "still_green_pixels", 0) or 0),
        residual_ratio=float(getattr(report, "residual_ratio", 1.0) or 1.0),
        mean_lab_distance=float(getattr(report, "mean_lab_distance", 0.0) or 0.0),
        mean_rgb_delta=float(getattr(report, "mean_rgb_delta", 0.0) or 0.0),
    )


def evaluate_projection(candidate: Any, scale: float, mode: str = "fast") -> ProjectionStats:
    if not PROJECTION_AVAILABLE or projection_mod is None:
        return ProjectionStats(scale=float(scale), valid=False)
    try:
        _, report = projection_mod.projection_preview_with_report(
            candidate.image,
            cfg=getattr(projection_mod, "PROJECTION_CFG"),
            user_scale=float(scale),
            preview_mode=str(mode),
        )
        return _extract_projection_stats(report, scale=float(scale))
    except Exception:
        return ProjectionStats(scale=float(scale), valid=False)


# ============================================================
# ORCHESTRATEUR HYBRIDE ML + DL
# ============================================================

class CamouflageMLDLGenerator:
    def __init__(self, config: MLDLConfig) -> None:
        self.cfg = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(config.device)
        self.rng = random.Random(config.random_seed)

        self.buffer = ExperienceBuffer()
        input_dim = len(FEATURE_KEYS) + len(PROJECTION_FEATURE_KEYS)
        self.surrogate = DeepSurrogate(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            lr=config.learning_rate,
            device=self.device,
        )
        context_dim = input_dim + len(FAILURE_KEYS)
        self.bandit = LinUCBBandit(n_actions=len(ACTION_LIBRARY), context_dim=context_dim, alpha=config.alpha_ucb)

        self.rows: List[Dict[str, object]] = []
        self.total_attempts = 0
        self.training_log: List[Dict[str, Any]] = []
        self.last_rejected_candidate: Optional[Any] = None
        self.last_analysis: Optional[RejectionAnalysis] = None
        self.last_projection_stats: Optional[ProjectionStats] = None

        base_relax = float(getattr(config, "pretrain_relax_level", 0.0) or 0.0)
        self.tolerance_profile = (
            camo.build_validation_tolerance_profile(base_relax)
            if hasattr(camo, "build_validation_tolerance_profile")
            else None
        )

        self._buffer_lock = threading.RLock()
        self._trainer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camo-mldl-trainer")
        self._train_future: Optional[Future] = None
        self._last_train_request_ts = 0.0
        self._last_train_request_samples = 0
        self._latest_train_stats: Optional[Dict[str, float]] = None
        self._latest_train_error: Optional[str] = None

        self._maybe_load_existing_state()

    def _maybe_load_existing_state(self) -> None:
        checkpoint = self.output_dir / self.cfg.checkpoint_name
        dataset = self.output_dir / self.cfg.dataset_name
        try:
            if dataset.exists():
                self.buffer.load(dataset)
        except Exception:
            pass
        try:
            if checkpoint.exists():
                self.surrogate.load(checkpoint)
        except Exception:
            pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        return device

    def _buffer_arrays_copy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._buffer_lock:
            return self.buffer.as_arrays()

    def _write_parallel_training_log(self) -> None:
        path = self.output_dir / "training_log_ml_dl.json"
        payload = {
            "training_log": self.training_log,
            "latest_stats": self._latest_train_stats,
            "latest_error": self._latest_train_error,
            "projection_available": bool(PROJECTION_AVAILABLE),
            "projection_ml_enabled": bool(self.cfg.projection_ml_enabled),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_buffer_now(self) -> None:
        with self._buffer_lock:
            self.buffer.save(self.output_dir / self.cfg.dataset_name)

    def _train_snapshot_worker(
        self,
        x: np.ndarray,
        y_valid: np.ndarray,
        y_reward: np.ndarray,
        sample_count: int,
    ) -> Dict[str, Any]:
        surrogate = DeepSurrogate(
            input_dim=len(FEATURE_KEYS) + len(PROJECTION_FEATURE_KEYS),
            hidden_dim=self.cfg.hidden_dim,
            lr=self.cfg.learning_rate,
            device=self.device,
        )
        stats = surrogate.fit(
            x,
            y_valid,
            y_reward,
            epochs=self.cfg.train_epochs,
            batch_size=self.cfg.batch_size,
            val_split=self.cfg.val_split,
            patience=self.cfg.early_stopping_patience,
            min_delta=self.cfg.early_stopping_min_delta,
            random_seed=self.cfg.random_seed,
        )

        checkpoint_tmp = self.output_dir / f"{self.cfg.checkpoint_name}.tmp"
        dataset_tmp = self.output_dir / f"{self.cfg.dataset_name}.tmp.npz"
        checkpoint_final = self.output_dir / self.cfg.checkpoint_name
        dataset_final = self.output_dir / self.cfg.dataset_name

        surrogate.save(checkpoint_tmp)
        np.savez_compressed(dataset_tmp, x=x, y_valid=y_valid, y_reward=y_reward)

        checkpoint_tmp.replace(checkpoint_final)
        dataset_tmp.replace(dataset_final)

        return {
            "n_samples": int(sample_count),
            "stats": stats,
            "ts": time.time(),
            "checkpoint": str(checkpoint_final),
            "dataset": str(dataset_final),
        }

    def _schedule_background_train(self, force: bool = False) -> Optional[Future]:
        if not self.cfg.parallel_train_enabled:
            return None
        self._poll_background_train()
        if self._train_future is not None and not self._train_future.done():
            return self._train_future

        x, y_valid, y_reward = self._buffer_arrays_copy()
        sample_count = int(len(x))
        if sample_count < self.cfg.min_train_size:
            return None
        if not force and (sample_count % self.cfg.retrain_every != 0):
            return None

        now = time.time()
        if not force and (now - self._last_train_request_ts) < float(self.cfg.parallel_train_min_interval_s):
            return None

        self._last_train_request_ts = now
        self._last_train_request_samples = sample_count
        self._latest_train_error = None
        self._train_future = self._trainer_pool.submit(
            self._train_snapshot_worker,
            np.asarray(x, dtype=np.float32).copy(),
            np.asarray(y_valid, dtype=np.float32).copy(),
            np.asarray(y_reward, dtype=np.float32).copy(),
            sample_count,
        )
        return self._train_future

    def _apply_latest_checkpoint(self) -> None:
        checkpoint = self.output_dir / self.cfg.checkpoint_name
        if checkpoint.exists():
            self.surrogate.load(checkpoint)

    def _poll_background_train(self) -> Optional[Dict[str, Any]]:
        fut = self._train_future
        if fut is None or not fut.done():
            return None
        self._train_future = None
        try:
            payload = fut.result()
        except Exception as exc:
            self._latest_train_error = str(exc)
            self.training_log.append({"ts": time.time(), "error": self._latest_train_error, "n_samples": int(self._last_train_request_samples)})
            self._write_parallel_training_log()
            return None

        self._apply_latest_checkpoint()
        self._latest_train_stats = dict(payload.get("stats", {}))
        self.training_log.append(payload)
        self._write_parallel_training_log()
        return payload

    def _flush_background_train(self) -> None:
        fut = self._train_future
        if fut is None:
            return
        self._train_future = None
        payload = fut.result()
        self._apply_latest_checkpoint()
        self._latest_train_stats = dict(payload.get("stats", {}))
        self.training_log.append(payload)
        self._write_parallel_training_log()

    def _shutdown_trainer_pool(self) -> None:
        try:
            self._trainer_pool.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    def _persist_warmup_if_needed(self, sample_count: int) -> None:
        every = max(1, int(self.cfg.warmup_persist_every))
        if sample_count > 0 and (sample_count % every == 0):
            self._save_buffer_now()

    def _generate_candidate(self, seed: int) -> Any:
        return camo.generate_candidate_from_seed(seed, anti_pixel=self.cfg.anti_pixel)

    def _generate_and_validate(self, seed: int) -> Tuple[Any, Any]:
        return camo.generate_and_validate_from_seed(
            seed,
            max_repair_rounds=self.cfg.max_repair_rounds,
            tolerance_profile=self.tolerance_profile,
            anti_pixel=self.cfg.anti_pixel,
        )

    def _projection_scales(self) -> Tuple[float, ...]:
        base = max(0.05, float(self.cfg.projection_base_scale))
        vals = []
        for mult in self.cfg.projection_scale_candidates:
            vals.append(float(np.clip(base * float(mult), 0.05, 2.0)))
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        return tuple(uniq)

    def _choose_projection_for_candidate(self, candidate: Any) -> ProjectionStats:
        if not (self.cfg.projection_ml_enabled and PROJECTION_AVAILABLE):
            return ProjectionStats(scale=float(self.cfg.projection_base_scale), valid=False)

        best_stats: Optional[ProjectionStats] = None
        best_key = (-1e18, -1e18)
        for scale in self._projection_scales():
            stats = evaluate_projection(candidate, scale=scale, mode=self.cfg.projection_preview_mode)
            if self.surrogate.trained:
                feat = candidate_to_feature_vector(candidate, projection_stats=stats)
                pred_valid, pred_reward = self.surrogate.predict(feat)
                key = (float(pred_valid[0]), float(pred_reward[0]))
            else:
                key = (1.0 if stats.valid else 0.0, projection_reward(stats))
            if key > best_key:
                best_key = key
                best_stats = stats

        return best_stats or ProjectionStats(scale=float(self.cfg.projection_base_scale), valid=False)

    def warmup(self) -> None:
        for i in range(self.cfg.warmup_samples):
            seed = camo.build_seed(0, i + 1, self.cfg.base_seed)
            candidate, outcome = self._generate_and_validate(seed)
            projection_stats = self._choose_projection_for_candidate(candidate)
            backend_ok = bool(getattr(outcome, "accepted", False))
            accepted = bool(backend_ok and ((projection_stats.valid) if (self.cfg.projection_ml_enabled and PROJECTION_AVAILABLE) else True))

            with self._buffer_lock:
                self.buffer.add(candidate, accepted, projection_stats, self.cfg.projection_reward_weight)
                sample_count = len(self.buffer.features)
            self._persist_warmup_if_needed(sample_count)

            if not accepted:
                self.last_rejected_candidate = candidate
                self.last_analysis = analyze_rejection(candidate, target_index=0, local_attempt=i + 1)
                self.last_projection_stats = projection_stats

        self._save_buffer_now()

    def maybe_train(self, force: bool = False) -> Optional[Dict[str, float]]:
        x, y_valid, y_reward = self._buffer_arrays_copy()
        if len(x) < self.cfg.min_train_size:
            return None
        if not force and (len(x) % self.cfg.retrain_every != 0):
            return None

        stats = self.surrogate.fit(
            x,
            y_valid,
            y_reward,
            epochs=self.cfg.train_epochs,
            batch_size=self.cfg.batch_size,
            val_split=self.cfg.val_split,
            patience=self.cfg.early_stopping_patience,
            min_delta=self.cfg.early_stopping_min_delta,
            random_seed=self.cfg.random_seed,
        )
        self.training_log.append({"n_samples": int(len(x)), "stats": stats, "ts": time.time()})
        self.surrogate.save(self.output_dir / self.cfg.checkpoint_name)
        self._save_buffer_now()
        self._latest_train_stats = dict(stats)
        self._write_parallel_training_log()
        return stats

    def _select_action_indexes(self, analysis: Optional[RejectionAnalysis]) -> List[int]:
        if analysis is None or self.last_rejected_candidate is None:
            action_indexes = list(range(len(ACTION_LIBRARY)))
            self.rng.shuffle(action_indexes)
            return action_indexes[: self.cfg.candidate_pool_size]

        context = build_context_vector(self.last_rejected_candidate, analysis, projection_stats=self.last_projection_stats)
        ranked = self.bandit.select_top_k(context, k=max(1, self.cfg.candidate_pool_size - 2))
        others = [i for i in range(len(ACTION_LIBRARY)) if i not in ranked]
        self.rng.shuffle(others)
        return ranked + others[: max(0, self.cfg.candidate_pool_size - len(ranked))]

    def _propose_candidates(self, target_index: int, local_attempt: int, analysis: Optional[RejectionAnalysis]) -> List[Proposal]:
        self._poll_background_train()
        proposals: List[Proposal] = []
        action_indexes = self._select_action_indexes(analysis)

        for offset, action_idx in enumerate(action_indexes, start=0):
            action_name, action = ACTION_LIBRARY[action_idx]
            base_seed = camo.build_seed(target_index, local_attempt + offset, self.cfg.base_seed)
            seed = propose_seed(base_seed, action)
            candidate = self._generate_candidate(seed)
            projection_stats = self._choose_projection_for_candidate(candidate)

            if self.surrogate.trained:
                feat = candidate_to_feature_vector(candidate, projection_stats=projection_stats)
                prob_valid, pred_reward = self.surrogate.predict(feat)
                pred_valid_f = float(prob_valid[0])
                pred_reward_f = float(pred_reward[0])
            else:
                pred_valid_f = 0.5 + (0.10 if projection_stats.valid else 0.0)
                pred_reward_f = combined_reward(candidate, False, projection_stats, self.cfg.projection_reward_weight)

            proposals.append(Proposal(
                seed=int(seed),
                action_idx=int(action_idx),
                action_name=str(action_name),
                candidate=candidate,
                pred_valid=pred_valid_f,
                pred_reward=pred_reward_f,
                projection_scale=float(projection_stats.scale),
                projection_stats=projection_stats,
            ))

        proposals.sort(key=lambda p: (p.pred_valid, p.pred_reward), reverse=True)
        return proposals

    def _final_projection_refine(self, candidate: Any, initial_stats: Optional[ProjectionStats]) -> ProjectionStats:
        if not (self.cfg.projection_ml_enabled and PROJECTION_AVAILABLE):
            return initial_stats or ProjectionStats(scale=float(self.cfg.projection_base_scale), valid=False)

        candidate_scales = sorted(set([
            float(self.cfg.projection_base_scale),
            float(initial_stats.scale if initial_stats is not None else self.cfg.projection_base_scale),
            float((initial_stats.scale if initial_stats is not None else self.cfg.projection_base_scale) * 0.94),
            float((initial_stats.scale if initial_stats is not None else self.cfg.projection_base_scale) * 1.06),
        ]))

        best_stats: Optional[ProjectionStats] = None
        best_key = (-1e18, -1e18)
        for scale in candidate_scales:
            stats = evaluate_projection(candidate, scale=scale, mode=self.cfg.projection_final_mode)
            score = projection_reward(stats)
            key = (1.0 if stats.valid else 0.0, score)
            if key > best_key:
                best_key = key
                best_stats = stats
        return best_stats or ProjectionStats(scale=float(self.cfg.projection_base_scale), valid=False)

    def _validate_top_candidates(self, proposals: Sequence[Proposal], target_index: int, local_attempt: int) -> Tuple[Optional[Proposal], Optional[RejectionAnalysis]]:
        accepted: Optional[Proposal] = None
        best_analysis: Optional[RejectionAnalysis] = None
        best_reward = -1e18

        for rank, proposal in enumerate(proposals[: self.cfg.validate_top_k], start=1):
            self.total_attempts += 1
            final_candidate, final_outcome = self._generate_and_validate(proposal.seed)
            projection_stats = self._final_projection_refine(final_candidate, proposal.projection_stats)

            backend_ok = bool(getattr(final_outcome, "accepted", False))
            projection_ok = bool(projection_stats.valid) if (self.cfg.projection_ml_enabled and PROJECTION_AVAILABLE) else True
            real_ok = bool(backend_ok and projection_ok)

            proposal = Proposal(
                seed=proposal.seed,
                action_idx=proposal.action_idx,
                action_name=proposal.action_name,
                candidate=final_candidate,
                pred_valid=proposal.pred_valid,
                pred_reward=proposal.pred_reward,
                projection_scale=float(projection_stats.scale),
                projection_stats=projection_stats,
            )

            with self._buffer_lock:
                reward = self.buffer.add(proposal.candidate, real_ok, projection_stats, self.cfg.projection_reward_weight)
            self._save_buffer_now()
            self._schedule_background_train(force=False)
            self._poll_background_train()

            if real_ok:
                accepted = proposal
                context = build_context_vector(proposal.candidate, None, projection_stats=projection_stats)
                self.bandit.update(proposal.action_idx, context, reward)
                break

            analysis = analyze_rejection(proposal.candidate, target_index=target_index, local_attempt=local_attempt + rank - 1)
            if reward > best_reward:
                best_reward = reward
                best_analysis = analysis

            context = build_context_vector(proposal.candidate, analysis, projection_stats=projection_stats)
            self.bandit.update(proposal.action_idx, context, reward)
            self.last_rejected_candidate = proposal.candidate
            self.last_analysis = analysis
            self.last_projection_stats = projection_stats

        return accepted, best_analysis

    def generate(self) -> List[Dict[str, object]]:
        try:
            self.warmup()
            self.maybe_train(force=True)
            self._schedule_background_train(force=True)

            if self.last_rejected_candidate is None:
                self.last_rejected_candidate = self._generate_candidate(self.cfg.base_seed)
                self.last_analysis = None
                self.last_projection_stats = self._choose_projection_for_candidate(self.last_rejected_candidate)

            for target_index in range(1, self.cfg.target_count + 1):
                local_attempt = 1
                accepted_proposal: Optional[Proposal] = None
                current_analysis = self.last_analysis

                while local_attempt <= self.cfg.max_attempts_per_target:
                    self._poll_background_train()
                    proposals = self._propose_candidates(target_index=target_index, local_attempt=local_attempt, analysis=current_analysis)
                    accepted_proposal, best_analysis = self._validate_top_candidates(proposals, target_index=target_index, local_attempt=local_attempt)
                    current_analysis = best_analysis

                    if accepted_proposal is not None:
                        filename = self.output_dir / f"camouflage_{target_index:03d}.png"
                        saved_path = camo.save_candidate_image(accepted_proposal.candidate, filename)
                        backend_outcome = camo.validate_with_reasons(accepted_proposal.candidate, tolerance_profile=self.tolerance_profile)
                        row = camo.candidate_row(
                            target_index,
                            local_attempt,
                            self.total_attempts,
                            accepted_proposal.candidate,
                            backend_outcome,
                            image_name=saved_path.name,
                            image_path=str(saved_path),
                            tolerance_profile=self.tolerance_profile,
                        )
                        row["projection_ml_enabled"] = int(bool(self.cfg.projection_ml_enabled and PROJECTION_AVAILABLE))
                        row["projection_scale"] = float(accepted_proposal.projection_scale)
                        if accepted_proposal.projection_stats is not None:
                            row["projection_valid"] = int(bool(accepted_proposal.projection_stats.valid))
                            row["projection_uniform_pixels"] = int(accepted_proposal.projection_stats.uniform_pixels)
                            row["projection_residual_pixels"] = int(accepted_proposal.projection_stats.residual_pixels)
                            row["projection_still_green_pixels"] = int(accepted_proposal.projection_stats.still_green_pixels)
                            row["projection_residual_ratio"] = float(accepted_proposal.projection_stats.residual_ratio)
                            row["projection_mean_lab_distance"] = float(accepted_proposal.projection_stats.mean_lab_distance)
                            row["projection_mean_rgb_delta"] = float(accepted_proposal.projection_stats.mean_rgb_delta)
                        row["mldl_action_name"] = accepted_proposal.action_name
                        row["mldl_pred_valid"] = float(accepted_proposal.pred_valid)
                        row["mldl_pred_reward"] = float(accepted_proposal.pred_reward)
                        self.rows.append(row)
                        break

                    local_attempt += max(1, self.cfg.candidate_pool_size)

                if accepted_proposal is None:
                    raise RuntimeError(
                        f"Impossible d'obtenir un camouflage valide pour target_index={target_index} "
                        f"dans la limite de {self.cfg.max_attempts_per_target} tentatives locales."
                    )

            self._flush_background_train()
            camo.write_report(self.rows, self.output_dir, filename=self.cfg.report_name)
            self._write_summary()
            return self.rows
        finally:
            self._shutdown_trainer_pool()

    def _write_summary(self) -> None:
        summary = {
            "config": asdict(self.cfg),
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "projection_available": bool(PROJECTION_AVAILABLE),
            "projection_ml_enabled": bool(self.cfg.projection_ml_enabled),
            "total_rows": len(self.rows),
            "total_attempts": self.total_attempts,
            "training_log": self.training_log,
            "latest_train_stats": self._latest_train_stats,
            "latest_train_error": self._latest_train_error,
            "parallel_train_enabled": bool(self.cfg.parallel_train_enabled),
            "report": str((self.output_dir / self.cfg.report_name).resolve()),
            "checkpoint": str((self.output_dir / self.cfg.checkpoint_name).resolve()),
            "dataset": str((self.output_dir / self.cfg.dataset_name).resolve()),
            "actions": [name for name, _ in ACTION_LIBRARY],
        }
        (self.output_dir / "run_summary_ml_dl.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# CLI
# ============================================================

def parse_args() -> MLDLConfig:
    parser = argparse.ArgumentParser(description="Générateur de camouflage guidé par ML + DL")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--warmup-samples", type=int, default=128)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--validate-top-k", type=int, default=3)
    parser.add_argument("--max-attempts-per-target", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-seed", type=int, default=camo.DEFAULT_BASE_SEED)
    parser.add_argument("--output-dir", type=str, default="camouflages_ml_dl")
    parser.add_argument("--alpha-ucb", type=float, default=1.25)
    parser.add_argument("--min-train-size", type=int, default=12)
    parser.add_argument("--retrain-every", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--parallel-train-min-interval-s", type=float, default=3.0)
    parser.add_argument("--parallel-train", dest="parallel_train_enabled", action="store_true")
    parser.add_argument("--no-parallel-train", dest="parallel_train_enabled", action="store_false")
    parser.set_defaults(parallel_train_enabled=True)
    parser.add_argument("--pretrain-relax-level", type=float, default=0.0)
    parser.add_argument("--max-repair-rounds", type=int, default=getattr(camo, "MAX_REPAIR_ROUNDS", 3))
    parser.add_argument("--disable-anti-pixel", action="store_true")

    parser.add_argument("--projection-ml", dest="projection_ml_enabled", action="store_true")
    parser.add_argument("--no-projection-ml", dest="projection_ml_enabled", action="store_false")
    parser.set_defaults(projection_ml_enabled=True)
    parser.add_argument("--projection-base-scale", type=float, default=1.0)
    parser.add_argument("--projection-scales", type=str, default="0.82,0.92,1.00,1.08,1.18")
    parser.add_argument("--projection-preview-mode", type=str, default="fast")
    parser.add_argument("--projection-final-mode", type=str, default="quality")
    parser.add_argument("--projection-reward-weight", type=float, default=0.65)

    args = parser.parse_args()
    try:
        proj_scales = tuple(float(x.strip()) for x in str(args.projection_scales).split(",") if x.strip())
    except Exception:
        proj_scales = (0.82, 0.92, 1.00, 1.08, 1.18)

    return MLDLConfig(
        target_count=args.target_count,
        warmup_samples=args.warmup_samples,
        candidate_pool_size=args.candidate_pool_size,
        validate_top_k=args.validate_top_k,
        max_attempts_per_target=args.max_attempts_per_target,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        device=args.device,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        alpha_ucb=args.alpha_ucb,
        min_train_size=args.min_train_size,
        retrain_every=args.retrain_every,
        val_split=args.val_split,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        random_seed=args.random_seed,
        parallel_train_enabled=bool(args.parallel_train_enabled),
        parallel_train_min_interval_s=float(args.parallel_train_min_interval_s),
        pretrain_relax_level=float(args.pretrain_relax_level),
        max_repair_rounds=int(args.max_repair_rounds),
        anti_pixel=not bool(args.disable_anti_pixel),
        projection_ml_enabled=bool(args.projection_ml_enabled),
        projection_base_scale=float(args.projection_base_scale),
        projection_scale_candidates=tuple(proj_scales),
        projection_preview_mode=str(args.projection_preview_mode),
        projection_final_mode=str(args.projection_final_mode),
        projection_reward_weight=float(args.projection_reward_weight),
    )


def build_config_from_main_args(args: Any) -> MLDLConfig:
    projection_scales = getattr(args, "mldl_projection_scales", (0.82, 0.92, 1.00, 1.08, 1.18))
    if isinstance(projection_scales, str):
        try:
            projection_scales = tuple(float(x.strip()) for x in projection_scales.split(",") if x.strip())
        except Exception:
            projection_scales = (0.82, 0.92, 1.00, 1.08, 1.18)
    return MLDLConfig(
        target_count=int(getattr(args, "target_count", 20)),
        warmup_samples=int(getattr(args, "mldl_warmup_samples", 128)),
        candidate_pool_size=int(getattr(args, "mldl_candidate_pool_size", 8)),
        validate_top_k=int(getattr(args, "mldl_validate_top_k", 3)),
        max_attempts_per_target=int(getattr(args, "mldl_max_attempts_per_target", 120)),
        train_epochs=int(getattr(args, "mldl_train_epochs", 24)),
        batch_size=int(getattr(args, "mldl_batch_size", 32)),
        learning_rate=float(getattr(args, "mldl_learning_rate", 1e-3)),
        hidden_dim=int(getattr(args, "mldl_hidden_dim", 128)),
        device=str(getattr(args, "mldl_device", "auto")),
        base_seed=int(getattr(args, "base_seed", camo.DEFAULT_BASE_SEED)),
        output_dir=str(getattr(args, "output_dir", "camouflages_ml_dl")),
        alpha_ucb=float(getattr(args, "mldl_alpha_ucb", 1.25)),
        min_train_size=int(getattr(args, "mldl_min_train_size", 12)),
        retrain_every=int(getattr(args, "mldl_retrain_every", 8)),
        val_split=float(getattr(args, "mldl_val_split", 0.15)),
        early_stopping_patience=int(getattr(args, "mldl_early_stopping_patience", 6)),
        early_stopping_min_delta=float(getattr(args, "mldl_early_stopping_min_delta", 1e-4)),
        random_seed=int(getattr(args, "random_seed", 12345)),
        parallel_train_enabled=bool(getattr(args, "mldl_parallel_train_enabled", True)),
        parallel_train_min_interval_s=float(getattr(args, "mldl_parallel_train_min_interval_s", 3.0)),
        pretrain_relax_level=float(getattr(args, "mldl_pretrain_relax_level", 0.0)),
        max_repair_rounds=int(getattr(args, "mldl_max_repair_rounds", getattr(camo, "MAX_REPAIR_ROUNDS", 3))),
        anti_pixel=bool(getattr(args, "mldl_anti_pixel", getattr(camo, "DEFAULT_ENABLE_ANTI_PIXEL", True))),
        projection_ml_enabled=bool(getattr(args, "mldl_projection_ml_enabled", True)),
        projection_base_scale=float(getattr(args, "mldl_projection_base_scale", 1.0)),
        projection_scale_candidates=tuple(float(x) for x in projection_scales),
        projection_preview_mode=str(getattr(args, "mldl_projection_preview_mode", "fast")),
        projection_final_mode=str(getattr(args, "mldl_projection_final_mode", "quality")),
        projection_reward_weight=float(getattr(args, "mldl_projection_reward_weight", 0.65)),
    )


def run_guided_generation(config: MLDLConfig) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

    runner = CamouflageMLDLGenerator(config)
    rows = runner.generate()

    summary_path = Path(config.output_dir) / "run_summary_ml_dl.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    summary.setdefault("target_count", config.target_count)
    summary.setdefault("total_attempts", runner.total_attempts)
    summary.setdefault("output_dir", str(Path(config.output_dir).resolve()))
    summary.setdefault("report", str((Path(config.output_dir) / config.report_name).resolve()))
    return rows, summary


def main() -> None:
    cfg = parse_args()
    rows, summary = run_guided_generation(cfg)
    print("Terminé.")
    print(f"Camouflages validés : {len(rows)}/{cfg.target_count}")
    print(f"Tentatives totales : {int(summary.get('total_attempts', 0))}")
    print(f"Projection ML dispo : {bool(summary.get('projection_available', False))}")
    print(f"Dossier : {Path(cfg.output_dir).resolve()}")
    print(f"Rapport : {(Path(cfg.output_dir) / cfg.report_name).resolve()}")


if __name__ == "__main__":
    main()
