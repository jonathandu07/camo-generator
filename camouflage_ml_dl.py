# -*- coding: utf-8 -*-
"""
camouflage_ml_dl.py

Pipeline hybride ML + DL compatible avec le main.py réel.

Ce script ne suppose plus l'existence d'API absentes dans main.py
(comme correction_state, deep_rejection_analysis, RejectionAnalysis,
_guided_state_init, _merge_guided_generation_state, etc.).

Approche :
1) Warmup : génération de candidats par seed, extraction des features réelles,
   validation et constitution du dataset supervisé.
2) Deep Learning : surrogate léger qui prédit probabilité de validation et reward.
3) Bandit contextuel LinUCB : choisit des politiques de recherche de seed.
4) Recherche guidée : pour chaque image cible, on génère un petit pool de seeds,
   on classe les propositions avec le surrogate, puis on valide réellement les
   meilleures candidates via main.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

import main as camo


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
    "largest_olive_component_ratio",
    "boundary_density",
    "boundary_density_small",
    "boundary_density_tiny",
    "mirror_similarity",
    "edge_contact_ratio",
    "overscan",
    "shift_strength",
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
)

# Chaque action représente une politique déterministe de dérivation de seed.
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
    min_train_size: int = 32
    retrain_every: int = 24
    random_seed: int = 12345


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
class Proposal:
    seed: int
    action_idx: int
    action_name: str
    candidate: camo.CandidateResult
    pred_valid: float
    pred_reward: float


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
    abs_err = np.abs(rs - camo.TARGET)
    m = dict(candidate.metrics)
    return {
        "ratio_coyote": float(rs[camo.IDX_COYOTE]),
        "ratio_olive": float(rs[camo.IDX_OLIVE]),
        "ratio_terre": float(rs[camo.IDX_TERRE]),
        "ratio_gris": float(rs[camo.IDX_GRIS]),
        "abs_err_coyote": float(abs_err[camo.IDX_COYOTE]),
        "abs_err_olive": float(abs_err[camo.IDX_OLIVE]),
        "abs_err_terre": float(abs_err[camo.IDX_TERRE]),
        "abs_err_gris": float(abs_err[camo.IDX_GRIS]),
        "mean_abs_error": float(np.mean(abs_err)),
        "largest_olive_component_ratio": _safe_float(m.get("largest_olive_component_ratio", 0.0)),
        "boundary_density": _safe_float(m.get("boundary_density", 0.0)),
        "boundary_density_small": _safe_float(m.get("boundary_density_small", 0.0)),
        "boundary_density_tiny": _safe_float(m.get("boundary_density_tiny", 0.0)),
        "mirror_similarity": _safe_float(m.get("mirror_similarity", 0.0)),
        "edge_contact_ratio": _safe_float(m.get("edge_contact_ratio", 0.0)),
        "overscan": _safe_float(m.get("overscan", 0.0)),
        "shift_strength": _safe_float(m.get("shift_strength", 0.0)),
    }


def candidate_to_feature_vector(candidate: camo.CandidateResult) -> np.ndarray:
    feat = candidate_to_feature_dict(candidate)
    return np.array([feat.get(name, 0.0) for name in FEATURE_KEYS], dtype=np.float32)


def analyze_rejection(candidate: camo.CandidateResult, target_index: int, local_attempt: int) -> RejectionAnalysis:
    feat = candidate_to_feature_dict(candidate)
    failures: List[str] = []
    notes: List[str] = []

    for name in ("abs_err_coyote", "abs_err_olive", "abs_err_terre", "abs_err_gris"):
        if feat[name] > float(camo.MAX_ABS_ERROR_PER_COLOR[["abs_err_coyote", "abs_err_olive", "abs_err_terre", "abs_err_gris"].index(name)]):
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
    if feat["largest_olive_component_ratio"] < float(camo.MIN_LARGEST_OLIVE_COMPONENT_RATIO):
        failures.append("largest_olive_component_ratio_low")
    if feat["edge_contact_ratio"] > float(camo.MAX_EDGE_CONTACT_RATIO):
        failures.append("edge_contact_ratio_high")

    severity = 0.0
    severity += 100.0 * feat["mean_abs_error"]
    severity += max(0.0, float(camo.MIN_BOUNDARY_DENSITY) - feat["boundary_density"]) * 10.0
    severity += max(0.0, feat["boundary_density"] - float(camo.MAX_BOUNDARY_DENSITY)) * 10.0
    severity += max(0.0, feat["mirror_similarity"] - float(camo.MAX_MIRROR_SIMILARITY)) * 5.0
    severity += max(0.0, float(camo.MIN_LARGEST_OLIVE_COMPONENT_RATIO) - feat["largest_olive_component_ratio"]) * 10.0
    severity += max(0.0, feat["edge_contact_ratio"] - float(camo.MAX_EDGE_CONTACT_RATIO)) * 10.0

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


def build_context_vector(candidate: camo.CandidateResult, analysis: Optional[RejectionAnalysis]) -> np.ndarray:
    feat = candidate_to_feature_vector(candidate)
    fail = analysis_to_failure_vector(analysis)
    return np.concatenate([feat, fail], axis=0)


def candidate_reward(candidate: camo.CandidateResult, accepted: bool) -> float:
    feat = candidate_to_feature_dict(candidate)
    score = 0.0
    score += 2.0 if accepted else 0.0
    score -= 140.0 * feat["mean_abs_error"]

    # Bonus si les métriques sont proches du centre de leur intervalle valide.
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
    score += 0.45 * feat["largest_olive_component_ratio"]
    score += 0.20 * (1.0 - min(1.0, feat["mirror_similarity"]))
    score += 0.15 * (1.0 - min(1.0, feat["edge_contact_ratio"]))
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
        self.input_dim = int(input_dim)
        self.scaler = Standardizer(input_dim)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.trained = False
        self.torch_enabled = TORCH_AVAILABLE
        self.device = device if device != "auto" else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

        if self.torch_enabled:
            self._torch_device = torch.device(self.device)
            self.model = SurrogateNet(input_dim=input_dim, hidden_dim=hidden_dim).to(self._torch_device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.loss_bce = nn.BCEWithLogitsLoss()
            self.loss_mse = nn.MSELoss()
        else:
            self.model = None

    def fit(self, features: np.ndarray, valid: np.ndarray, rewards: np.ndarray, epochs: int = 20, batch_size: int = 32) -> Dict[str, float]:
        x = np.asarray(features, dtype=np.float32)
        y_valid = np.asarray(valid, dtype=np.float32)
        y_reward = np.asarray(rewards, dtype=np.float32)
        self.scaler.fit(x)
        self.reward_mean = float(np.mean(y_reward)) if len(y_reward) else 0.0
        self.reward_std = float(np.std(y_reward)) if len(y_reward) else 1.0
        if self.reward_std < 1e-6:
            self.reward_std = 1.0

        if not self.torch_enabled:
            self.trained = True
            return {"loss": 0.0, "loss_valid": 0.0, "loss_reward": 0.0}

        x_norm = self.scaler.transform(x)
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

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float32)
        one = False
        if x.ndim == 1:
            x = x[None, :]
            one = True
        x_norm = self.scaler.transform(x)

        if not self.torch_enabled or self.model is None:
            # Fallback déterministe sans torch : heuristique sur les features.
            prob_valid = np.clip(1.0 - x[:, FEATURE_KEYS.index("mean_abs_error")] * 200.0, 0.0, 1.0).astype(np.float32)
            reward = (
                1.0
                - 100.0 * x[:, FEATURE_KEYS.index("mean_abs_error")]
                + 0.5 * x[:, FEATURE_KEYS.index("largest_olive_component_ratio")]
                - 0.3 * x[:, FEATURE_KEYS.index("mirror_similarity")]
                - 0.2 * x[:, FEATURE_KEYS.index("edge_contact_ratio")]
            ).astype(np.float32)
            if one:
                return prob_valid[:1], reward[:1]
            return prob_valid, reward

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
        payload = {
            "scaler": self.scaler.state_dict(),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "trained": self.trained,
            "torch_enabled": self.torch_enabled,
        }
        if self.torch_enabled and self.model is not None:
            payload["model"] = self.model.state_dict()
        if TORCH_AVAILABLE:
            torch.save(payload, path)
        else:  # pragma: no cover
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        if self.torch_enabled and TORCH_AVAILABLE:
            payload = torch.load(path, map_location=self._torch_device)
            if "model" in payload and self.model is not None:
                self.model.load_state_dict(payload["model"])
        else:  # pragma: no cover
            payload = json.loads(path.read_text(encoding="utf-8"))
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
    else:
        seed = seed

    return int(seed & 0x7FFF_FFFF_FFFF_FFFF)


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
        self.last_rejected_candidate: Optional[camo.CandidateResult] = None
        self.last_analysis: Optional[RejectionAnalysis] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        return device

    def warmup(self) -> None:
        for i in range(self.cfg.warmup_samples):
            seed = camo.build_seed(0, i + 1, self.cfg.base_seed)
            candidate = camo.generate_candidate_from_seed(seed)
            accepted = camo.validate_candidate_result(candidate)
            self.buffer.add(candidate, accepted)
            if not accepted:
                self.last_rejected_candidate = candidate
                self.last_analysis = analyze_rejection(candidate, target_index=0, local_attempt=i + 1)

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

    def _select_action_indexes(self, analysis: Optional[RejectionAnalysis]) -> List[int]:
        if analysis is None or self.last_rejected_candidate is None:
            action_indexes = list(range(len(ACTION_LIBRARY)))
            self.rng.shuffle(action_indexes)
            return action_indexes[: self.cfg.candidate_pool_size]
        context = build_context_vector(self.last_rejected_candidate, analysis)
        ranked = self.bandit.select_top_k(context, k=max(1, self.cfg.candidate_pool_size - 2))
        others = [i for i in range(len(ACTION_LIBRARY)) if i not in ranked]
        self.rng.shuffle(others)
        return ranked + others[: max(0, self.cfg.candidate_pool_size - len(ranked))]

    def _propose_candidates(self, target_index: int, local_attempt: int, analysis: Optional[RejectionAnalysis]) -> List[Proposal]:
        proposals: List[Proposal] = []
        action_indexes = self._select_action_indexes(analysis)
        for offset, action_idx in enumerate(action_indexes, start=0):
            action_name, action = ACTION_LIBRARY[action_idx]
            base_seed = camo.build_seed(target_index, local_attempt + offset, self.cfg.base_seed)
            seed = propose_seed(base_seed, action)
            candidate = camo.generate_candidate_from_seed(seed)
            if self.surrogate.trained:
                prob_valid, pred_reward = self.surrogate.predict(candidate_to_feature_vector(candidate))
                pred_valid_f = float(prob_valid[0])
                pred_reward_f = float(pred_reward[0])
            else:
                pred_valid_f = max(0.0, min(1.0, 1.0 - candidate_to_feature_dict(candidate)["mean_abs_error"] * 200.0))
                pred_reward_f = candidate_reward(candidate, accepted=False)
            proposals.append(Proposal(
                seed=seed,
                action_idx=action_idx,
                action_name=action_name,
                candidate=candidate,
                pred_valid=pred_valid_f,
                pred_reward=pred_reward_f,
            ))
        proposals.sort(key=lambda p: (p.pred_valid, p.pred_reward), reverse=True)
        return proposals

    def _validate_top_candidates(self, proposals: Sequence[Proposal], target_index: int, local_attempt: int) -> Tuple[Optional[Proposal], Optional[RejectionAnalysis]]:
        accepted: Optional[Proposal] = None
        best_analysis: Optional[RejectionAnalysis] = None
        best_reward = -1e18

        for rank, proposal in enumerate(proposals[: self.cfg.validate_top_k], start=1):
            self.total_attempts += 1
            real_ok = camo.validate_candidate_result(proposal.candidate)
            reward = self.buffer.add(proposal.candidate, real_ok)
            self.maybe_train(force=False)

            if real_ok:
                accepted = proposal
                context = build_context_vector(proposal.candidate, None)
                self.bandit.update(proposal.action_idx, context, reward)
                break

            analysis = analyze_rejection(
                proposal.candidate,
                target_index=target_index,
                local_attempt=local_attempt + rank - 1,
            )
            if reward > best_reward:
                best_reward = reward
                best_analysis = analysis
            context = build_context_vector(proposal.candidate, analysis)
            self.bandit.update(proposal.action_idx, context, reward)
            self.last_rejected_candidate = proposal.candidate
            self.last_analysis = analysis

        return accepted, best_analysis

    def generate(self) -> List[Dict[str, object]]:
        self.warmup()
        self.maybe_train(force=True)

        if self.last_rejected_candidate is None:
            self.last_rejected_candidate = camo.generate_candidate_from_seed(self.cfg.base_seed)
            self.last_analysis = None

        for target_index in range(1, self.cfg.target_count + 1):
            local_attempt = 1
            accepted_proposal: Optional[Proposal] = None
            current_analysis = self.last_analysis

            while local_attempt <= self.cfg.max_attempts_per_target:
                proposals = self._propose_candidates(
                    target_index=target_index,
                    local_attempt=local_attempt,
                    analysis=current_analysis,
                )
                accepted_proposal, best_analysis = self._validate_top_candidates(
                    proposals,
                    target_index=target_index,
                    local_attempt=local_attempt,
                )
                current_analysis = best_analysis

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

                local_attempt += max(1, self.cfg.candidate_pool_size)

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
            "torch_available": TORCH_AVAILABLE,
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
    if TORCH_AVAILABLE:
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
