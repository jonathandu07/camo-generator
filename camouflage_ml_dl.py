# -*- coding: utf-8 -*-
"""
camouflage_ml_dl.py

Pipeline hybride Machine Learning + Deep Learning pour guider la génération
vers des camouflages acceptés plus rapidement, en s'appuyant sur main.py.

Principe :
1) Phase warmup : génération de candidats, mesure des métriques et constitution
   d'un dataset supervisé.
2) Deep Learning : entraînement d'un réseau PyTorch à double tête qui prédit
   la probabilité de validation et un score de qualité.
3) Machine Learning en ligne : bandit contextuel LinUCB pour choisir les
   corrections à appliquer après rejet.
4) Recherche guidée : pour chaque image cible, création d'un petit pool de
   propositions (seed + corrections), classement par le réseau profond, puis
   validation réelle des meilleures propositions.

Le script réutilise le backend de main.py :
- generate_candidate_from_seed(..., correction_state=...)
- validate_candidate_result(...)
- deep_rejection_analysis(...)
- _guided_state_init(), _merge_guided_generation_state(...)
- save_candidate_image(...), candidate_row(...), write_report(...)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import main as camo


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_KEYS: tuple[str, ...] = (
    "ratio_coyote",
    "ratio_olive",
    "ratio_terre",
    "ratio_gris",
    "largest_olive_component_ratio",
    "largest_olive_component_ratio_small",
    "olive_multizone_share",
    "center_empty_ratio",
    "center_empty_ratio_small",
    "boundary_density",
    "boundary_density_small",
    "boundary_density_tiny",
    "mirror_similarity",
    "central_brown_continuity",
    "oblique_share",
    "vertical_share",
    "angle_dominance_ratio",
    "macro_olive_visible_ratio",
    "macro_terre_visible_ratio",
    "macro_gris_visible_ratio",
    "macro_total_count",
    "macro_olive_count",
    "macro_terre_count",
    "macro_gris_count",
    "macro_multizone_ratio",
    "largest_macro_mask_ratio",
    "periphery_boundary_density_ratio",
    "periphery_non_coyote_ratio",
    "visual_score_final",
    "visual_score_ratio",
    "visual_score_silhouette",
    "visual_score_contour",
    "visual_score_main",
    "visual_silhouette_color_diversity",
    "visual_contour_break_score",
    "visual_outline_band_diversity",
    "visual_small_scale_structural_score",
    "visual_military_score",
)

FAILURE_KEYS: tuple[str, ...] = (
    "abs_err_coyote",
    "abs_err_olive",
    "abs_err_terre",
    "abs_err_gris",
    "mean_abs_error",
    "ratio_coyote",
    "ratio_olive",
    "ratio_terre",
    "ratio_gris",
    "largest_olive_component_ratio",
    "largest_olive_component_ratio_small",
    "olive_multizone_share",
    "center_empty_ratio",
    "center_empty_ratio_small",
    "boundary_density",
    "boundary_density_small",
    "mirror_similarity",
    "central_brown_continuity",
    "oblique_share",
    "vertical_share",
    "angle_dominance_ratio",
    "macro_olive_visible_ratio",
    "macro_terre_visible_ratio",
    "macro_gris_visible_ratio",
    "macro_total_count",
    "macro_olive_count",
    "macro_terre_count",
    "macro_gris_count",
    "macro_multizone_ratio",
    "largest_macro_mask_ratio",
    "periphery_boundary_density_ratio",
    "periphery_non_coyote_ratio",
    "visual_silhouette_color_diversity",
    "visual_contour_break_score",
    "visual_outline_band_diversity",
    "visual_small_scale_structural_score",
    "visual_score_final",
    "visual_military_score",
)

ACTION_LIBRARY: tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("boost_olive", {
        "olive_scale_delta": 0.10,
        "extra_macro_attempts": 40,
        "zone_boost_deltas": [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.00],
    }),
    ("boost_terre", {
        "terre_scale_delta": 0.08,
        "extra_macro_attempts": 30,
        "zone_boost_deltas": [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.00],
    }),
    ("boost_gris_asym_left", {
        "gris_scale_delta": 0.07,
        "zone_boost_deltas": [0.12, 0.00, 0.10, 0.00, 0.08, 0.00, 0.00],
    }),
    ("boost_gris_asym_right", {
        "gris_scale_delta": 0.07,
        "zone_boost_deltas": [0.00, 0.12, 0.00, 0.10, 0.00, 0.08, 0.00],
    }),
    ("fill_center", {
        "center_overlap_delta": 0.10,
        "extra_macro_attempts": 35,
        "zone_boost_deltas": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.28],
    }),
    ("strengthen_periphery", {
        "extra_macro_attempts": 35,
        "zone_boost_deltas": [0.12, 0.12, 0.10, 0.10, 0.08, 0.08, -0.04],
    }),
    ("diversify_angles", {
        "expand_angle_pool": True,
        "avoid_vertical": True,
    }),
    ("force_vertical", {
        "force_vertical": True,
    }),
    ("rougher_edges", {
        "edge_break_delta": 0.05,
        "lateral_jitter_delta": 0.03,
        "width_variation_delta": 0.02,
    }),
    ("smoother_edges", {
        "edge_break_delta": -0.02,
        "width_variation_delta": -0.01,
    }),
    ("heavy_repair", {
        "olive_scale_delta": 0.08,
        "terre_scale_delta": 0.05,
        "gris_scale_delta": 0.04,
        "center_overlap_delta": 0.08,
        "extra_macro_attempts": 80,
        "expand_angle_pool": True,
        "prefer_sequential_repair": True,
        "zone_boost_deltas": [0.08, 0.08, 0.08, 0.08, 0.06, 0.06, 0.10],
    }),
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
    min_train_size: int = 32
    retrain_every: int = 24
    random_seed: int = 12345


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


def candidate_to_feature_dict(candidate: camo.CandidateResult) -> Dict[str, float]:
    rs = np.asarray(candidate.ratios, dtype=float)
    m = dict(candidate.metrics)
    return {
        "ratio_coyote": float(rs[camo.IDX_COYOTE]),
        "ratio_olive": float(rs[camo.IDX_OLIVE]),
        "ratio_terre": float(rs[camo.IDX_TERRE]),
        "ratio_gris": float(rs[camo.IDX_GRIS]),
        **{k: _safe_float(m.get(k, 0.0), 0.0) for k in FEATURE_KEYS if not k.startswith("ratio_")},
    }


def candidate_to_feature_vector(candidate: camo.CandidateResult) -> np.ndarray:
    feat = candidate_to_feature_dict(candidate)
    return np.array([feat.get(name, 0.0) for name in FEATURE_KEYS], dtype=np.float32)


def analysis_to_failure_vector(analysis: camo.RejectionAnalysis) -> np.ndarray:
    names = set(str(x) for x in analysis.failure_names)
    return np.array([1.0 if name in names else 0.0 for name in FAILURE_KEYS], dtype=np.float32)


def build_context_vector(candidate: camo.CandidateResult, analysis: Optional[camo.RejectionAnalysis]) -> np.ndarray:
    feat = candidate_to_feature_vector(candidate)
    if analysis is None:
        fail = np.zeros(len(FAILURE_KEYS), dtype=np.float32)
    else:
        fail = analysis_to_failure_vector(analysis)
    return np.concatenate([feat, fail], axis=0)


def candidate_reward(candidate: camo.CandidateResult, accepted: bool) -> float:
    m = candidate.metrics
    score = 0.0
    score += 2.00 if accepted else 0.0
    score += 1.20 * _safe_float(m.get("visual_score_final", 0.0))
    score += 1.10 * _safe_float(m.get("visual_military_score", 0.0))
    score += 0.60 * _safe_float(m.get("visual_silhouette_color_diversity", 0.0))
    score += 0.50 * _safe_float(m.get("visual_contour_break_score", 0.0))
    score += 0.45 * _safe_float(m.get("periphery_non_coyote_ratio", 0.0))
    score += 0.45 * _safe_float(m.get("periphery_boundary_density_ratio", 0.0))
    score -= 0.80 * max(0.0, _safe_float(m.get("center_empty_ratio", 0.0)) - camo.MAX_COYOTE_CENTER_EMPTY_RATIO)
    score -= 0.50 * max(0.0, _safe_float(m.get("mirror_similarity", 0.0)) - 0.55)
    return float(score)


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
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fitted": bool(self.fitted),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.std = np.array(state["std"], dtype=np.float32)
        self.fitted = bool(state.get("fitted", True))


# ============================================================
# DEEP LEARNING : SURROGATE
# ============================================================

class SurrogateNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.head_valid = nn.Linear(hidden_dim // 2, 1)
        self.head_reward = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        valid_logit = self.head_valid(z).squeeze(-1)
        reward = self.head_reward(z).squeeze(-1)
        return valid_logit, reward


class DeepSurrogate:
    def __init__(self, input_dim: int, hidden_dim: int = 128, lr: float = 1e-3, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = SurrogateNet(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_mse = nn.MSELoss()
        self.scaler = Standardizer(input_dim)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.trained = False

    def fit(self, features: np.ndarray, valid: np.ndarray, rewards: np.ndarray, epochs: int = 20, batch_size: int = 32) -> Dict[str, float]:
        x = np.asarray(features, dtype=np.float32)
        y_valid = np.asarray(valid, dtype=np.float32)
        y_reward = np.asarray(rewards, dtype=np.float32)
        self.scaler.fit(x)
        x_norm = self.scaler.transform(x)

        self.reward_mean = float(np.mean(y_reward))
        self.reward_std = float(np.std(y_reward))
        if self.reward_std < 1e-6:
            self.reward_std = 1.0
        y_reward_norm = (y_reward - self.reward_mean) / self.reward_std

        ds = TensorDataset(
            torch.from_numpy(x_norm),
            torch.from_numpy(y_valid),
            torch.from_numpy(y_reward_norm.astype(np.float32)),
        )
        dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

        self.model.train()
        last = {"loss": 0.0, "loss_valid": 0.0, "loss_reward": 0.0}
        for _ in range(max(1, int(epochs))):
            agg_loss = 0.0
            agg_bce = 0.0
            agg_mse = 0.0
            agg_count = 0
            for xb, yb_valid, yb_reward in dl:
                xb = xb.to(self.device)
                yb_valid = yb_valid.to(self.device)
                yb_reward = yb_reward.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logit_valid, pred_reward = self.model(xb)
                loss_valid = self.loss_bce(logit_valid, yb_valid)
                loss_reward = self.loss_mse(pred_reward, yb_reward)
                loss = loss_valid + 0.7 * loss_reward
                loss.backward()
                self.optimizer.step()

                bs = int(xb.shape[0])
                agg_count += bs
                agg_loss += float(loss.item()) * bs
                agg_bce += float(loss_valid.item()) * bs
                agg_mse += float(loss_reward.item()) * bs
            last = {
                "loss": agg_loss / max(1, agg_count),
                "loss_valid": agg_bce / max(1, agg_count),
                "loss_reward": agg_mse / max(1, agg_count),
            }
        self.trained = True
        return last

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float32)
        one = False
        if x.ndim == 1:
            x = x[None, :]
            one = True
        x_norm = self.scaler.transform(x)
        xt = torch.from_numpy(x_norm).to(self.device)
        self.model.eval()
        logit_valid, pred_reward = self.model(xt)
        prob_valid = torch.sigmoid(logit_valid).cpu().numpy()
        reward = pred_reward.cpu().numpy() * self.reward_std + self.reward_mean
        if one:
            return prob_valid[0:1], reward[0:1]
        return prob_valid, reward

    def save(self, path: Path) -> None:
        payload = {
            "model": self.model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "trained": self.trained,
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
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
        self.context_dim = int(context_dim) + 1  # biais
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
# ÉTAT GUIDÉ / ACTIONS
# ============================================================

def neutral_guided_state() -> Dict[str, Any]:
    return camo._guided_state_init()


def merge_guided_delta(state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    out.setdefault("zone_boost_deltas", [0.0 for _ in camo.DENSITY_ZONES])
    incoming = dict(delta)

    for name in (
        "olive_scale_delta",
        "terre_scale_delta",
        "gris_scale_delta",
        "center_overlap_delta",
        "width_variation_delta",
        "lateral_jitter_delta",
        "tip_taper_delta",
        "edge_break_delta",
    ):
        out[name] = float(out.get(name, 0.0)) + float(incoming.get(name, 0.0))

    out["extra_macro_attempts"] = int(out.get("extra_macro_attempts", 0)) + int(incoming.get("extra_macro_attempts", 0))

    for name in ("force_vertical", "avoid_vertical", "expand_angle_pool", "prefer_sequential_repair"):
        out[name] = bool(out.get(name, False) or incoming.get(name, False))

    base_boosts = list(out.get("zone_boost_deltas", [0.0 for _ in camo.DENSITY_ZONES]))
    extra_boosts = list(incoming.get("zone_boost_deltas", [0.0 for _ in camo.DENSITY_ZONES]))
    if len(base_boosts) < len(camo.DENSITY_ZONES):
        base_boosts.extend([0.0] * (len(camo.DENSITY_ZONES) - len(base_boosts)))
    if len(extra_boosts) < len(camo.DENSITY_ZONES):
        extra_boosts.extend([0.0] * (len(camo.DENSITY_ZONES) - len(extra_boosts)))
    out["zone_boost_deltas"] = [float(a) + float(b) for a, b in zip(base_boosts, extra_boosts)]
    return out


# ============================================================
# DATASET / RUN SUMMARY
# ============================================================

class ExperienceBuffer:
    def __init__(self) -> None:
        self.features: List[np.ndarray] = []
        self.valid: List[float] = []
        self.rewards: List[float] = []

    def add(self, candidate: camo.CandidateResult, accepted: bool) -> float:
        feat = candidate_to_feature_vector(candidate)
        reward = candidate_reward(candidate, accepted)
        self.features.append(feat)
        self.valid.append(float(1.0 if accepted else 0.0))
        self.rewards.append(float(reward))
        return reward

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack(self.features, axis=0).astype(np.float32) if self.features else np.zeros((0, len(FEATURE_KEYS)), dtype=np.float32)
        y_valid = np.array(self.valid, dtype=np.float32)
        y_reward = np.array(self.rewards, dtype=np.float32)
        return x, y_valid, y_reward

    def save(self, path: Path) -> None:
        x, y_valid, y_reward = self.as_arrays()
        np.savez_compressed(path, x=x, y_valid=y_valid, y_reward=y_reward)


@dataclass
class Proposal:
    seed: int
    action_idx: int
    action_name: str
    guided_state: Dict[str, Any]
    candidate: camo.CandidateResult
    pred_valid: float
    pred_reward: float


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
        self.surrogate = DeepSurrogate(
            input_dim=len(FEATURE_KEYS),
            hidden_dim=config.hidden_dim,
            lr=config.learning_rate,
            device=self.device,
        )
        context_dim = len(FEATURE_KEYS) + len(FAILURE_KEYS)
        self.bandit = LinUCBBandit(n_actions=len(ACTION_LIBRARY), context_dim=context_dim, alpha=config.alpha_ucb)
        self.rows: List[Dict[str, object]] = []
        self.total_attempts = 0
        self.training_log: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def warmup(self) -> None:
        for i in range(self.cfg.warmup_samples):
            seed = camo.build_seed(0, i + 1, self.cfg.base_seed)
            candidate = camo.generate_candidate_from_seed(seed)
            accepted = camo.validate_candidate_result(candidate)
            self.buffer.add(candidate, accepted)

    def maybe_train(self, force: bool = False) -> Optional[Dict[str, float]]:
        x, y_valid, y_reward = self.buffer.as_arrays()
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
        )
        self.training_log.append({
            "n_samples": int(len(x)),
            "stats": stats,
            "ts": time.time(),
        })
        self.surrogate.save(self.output_dir / self.cfg.checkpoint_name)
        self.buffer.save(self.output_dir / self.cfg.dataset_name)
        return stats

    def _build_action_state(self, base_state: Dict[str, Any], action_idx: int) -> Dict[str, Any]:
        _name, delta = ACTION_LIBRARY[action_idx]
        return merge_guided_delta(base_state, delta)

    def _propose_candidates(
        self,
        target_index: int,
        local_attempt: int,
        base_state: Dict[str, Any],
        analysis: Optional[camo.RejectionAnalysis],
    ) -> List[Proposal]:
        if analysis is None:
            action_indexes = list(range(min(self.cfg.candidate_pool_size, len(ACTION_LIBRARY))))
            self.rng.shuffle(action_indexes)
            action_indexes = action_indexes[:self.cfg.candidate_pool_size]
        else:
            context = build_context_vector(self.last_rejected_candidate, analysis)
            ranked = self.bandit.select_top_k(context, k=max(1, self.cfg.candidate_pool_size - 2))
            others = [i for i in range(len(ACTION_LIBRARY)) if i not in ranked]
            self.rng.shuffle(others)
            action_indexes = ranked + others[:2]

        proposals: List[Proposal] = []
        for offset, action_idx in enumerate(action_indexes, start=0):
            action_name, _delta = ACTION_LIBRARY[action_idx]
            seed = camo.build_seed(target_index, local_attempt + offset, self.cfg.base_seed)
            guided_state = self._build_action_state(base_state, action_idx)
            candidate = camo.generate_candidate_from_seed(
                seed,
                correction_state=guided_state if camo._guided_state_has_effects(guided_state) else None,
            )
            if self.surrogate.trained:
                prob_valid, pred_reward = self.surrogate.predict(candidate_to_feature_vector(candidate))
                pred_valid_f = float(prob_valid[0])
                pred_reward_f = float(pred_reward[0])
            else:
                pred_valid_f = 0.5
                pred_reward_f = 0.0
            proposals.append(Proposal(
                seed=seed,
                action_idx=action_idx,
                action_name=action_name,
                guided_state=guided_state,
                candidate=candidate,
                pred_valid=pred_valid_f,
                pred_reward=pred_reward_f,
            ))
        proposals.sort(key=lambda p: (p.pred_valid, p.pred_reward), reverse=True)
        return proposals

    def _validate_top_candidates(
        self,
        proposals: Sequence[Proposal],
        target_index: int,
        local_attempt: int,
    ) -> Tuple[Optional[Proposal], Optional[camo.RejectionAnalysis], Dict[str, Any]]:
        base_state = neutral_guided_state()
        accepted: Optional[Proposal] = None
        best_analysis: Optional[camo.RejectionAnalysis] = None
        best_reward = -1e18

        for rank, proposal in enumerate(proposals[: self.cfg.validate_top_k], start=1):
            self.total_attempts += 1
            real_ok = camo.validate_candidate_result(proposal.candidate)
            reward = self.buffer.add(proposal.candidate, real_ok)
            self.maybe_train(force=False)

            analysis = None
            if not real_ok:
                analysis = camo.deep_rejection_analysis(
                    proposal.candidate,
                    target_index=target_index,
                    local_attempt=local_attempt + rank - 1,
                    reject_streak=int(base_state.get("reject_streak", 0)) + 1,
                )
                base_state = camo._merge_guided_generation_state(base_state, analysis)
                if reward > best_reward:
                    best_reward = reward
                    best_analysis = analysis
                context = build_context_vector(proposal.candidate, analysis)
                self.bandit.update(proposal.action_idx, context, reward)
                self.last_rejected_candidate = proposal.candidate
            else:
                accepted = proposal
                if best_analysis is None:
                    best_analysis = None
                break

        return accepted, best_analysis, base_state

    def generate(self) -> List[Dict[str, object]]:
        self.last_rejected_candidate = camo.generate_candidate_from_seed(self.cfg.base_seed)
        self.warmup()
        self.maybe_train(force=True)

        for target_index in range(1, self.cfg.target_count + 1):
            guided_state = neutral_guided_state()
            last_analysis: Optional[camo.RejectionAnalysis] = None
            local_attempt = 1
            accepted_proposal: Optional[Proposal] = None

            while local_attempt <= self.cfg.max_attempts_per_target:
                proposals = self._propose_candidates(
                    target_index=target_index,
                    local_attempt=local_attempt,
                    base_state=guided_state,
                    analysis=last_analysis,
                )
                accepted_proposal, best_analysis, merged_state = self._validate_top_candidates(
                    proposals,
                    target_index=target_index,
                    local_attempt=local_attempt,
                )
                guided_state = merge_guided_delta(guided_state, merged_state)
                last_analysis = best_analysis

                if accepted_proposal is not None:
                    filename = self.output_dir / f"camouflage_{target_index:03d}.png"
                    camo.save_candidate_image(accepted_proposal.candidate, filename)
                    self.rows.append(camo.candidate_row(
                        target_index,
                        local_attempt,
                        self.total_attempts,
                        accepted_proposal.candidate,
                    ))
                    break

                if last_analysis is not None:
                    guided_state = camo._merge_guided_generation_state(guided_state, last_analysis)

                local_attempt += max(1, self.cfg.validate_top_k)

            if accepted_proposal is None:
                raise RuntimeError(
                    f"Impossible d'obtenir un camouflage valide pour target_index={target_index} "
                    f"dans la limite de {self.cfg.max_attempts_per_target} tentatives locales."
                )

        camo.write_report(self.rows, self.output_dir, filename=self.cfg.report_name)
        self._write_summary()
        return self.rows

    def _write_summary(self) -> None:
        summary = {
            "config": asdict(self.cfg),
            "device": self.device,
            "total_rows": len(self.rows),
            "total_attempts": self.total_attempts,
            "training_log": self.training_log,
            "report": str((self.output_dir / self.cfg.report_name).resolve()),
            "checkpoint": str((self.output_dir / self.cfg.checkpoint_name).resolve()),
            "dataset": str((self.output_dir / self.cfg.dataset_name).resolve()),
            "actions": [name for name, _ in ACTION_LIBRARY],
        }
        (self.output_dir / "run_summary_ml_dl.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
    parser.add_argument("--min-train-size", type=int, default=32)
    parser.add_argument("--retrain-every", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=12345)
    args = parser.parse_args()
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
        random_seed=args.random_seed,
    )


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)

    runner = CamouflageMLDLGenerator(cfg)
    rows = runner.generate()
    print("Terminé.")
    print(f"Camouflages validés : {len(rows)}/{cfg.target_count}")
    print(f"Tentatives totales : {runner.total_attempts}")
    print(f"Dossier : {Path(cfg.output_dir).resolve()}")
    print(f"Rapport : {(Path(cfg.output_dir) / cfg.report_name).resolve()}")


if __name__ == "__main__":
    main()
