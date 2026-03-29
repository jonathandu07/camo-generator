# -*- coding: utf-8 -*-
"""
Suite de tests pour camouflage_ml_dl.py.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

MODULE_NAME = os.getenv("CAMO_MLDL_MODULE", "camouflage_ml_dl")
mut = importlib.import_module(MODULE_NAME)

LOG_DIR = Path(os.getenv("LOG_OUTPUT_DIR", Path(__file__).resolve().parent / "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_camouflage_ml_dl.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_camouflage_ml_dl")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    logger.addHandler(fh)
    logger.propagate = False
    return logger


LOGGER = configure_logger()


class TempDirMixin:
    def setUp(self) -> None:
        super().setUp()
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_camouflage_ml_dl_")
        self.tmpdir = Path(self._tmpdir_obj.name)

    def tearDown(self) -> None:
        self._tmpdir_obj.cleanup()
        super().tearDown()


def valid_metrics() -> Dict[str, float]:
    return {
        "largest_olive_component_ratio": 0.24,
        "largest_olive_component_ratio_small": 0.18,
        "olive_multizone_share": 0.62,
        "center_empty_ratio": 0.36,
        "center_empty_ratio_small": 0.41,
        "boundary_density": 0.145,
        "boundary_density_small": 0.11,
        "boundary_density_tiny": 0.11,
        "mirror_similarity": 0.44,
        "central_brown_continuity": 0.20,
        "oblique_share": 0.72,
        "vertical_share": 0.16,
        "angle_dominance_ratio": 0.20,
        "macro_olive_visible_ratio": 0.24,
        "macro_terre_visible_ratio": 0.18,
        "macro_gris_visible_ratio": 0.14,
        "macro_total_count": 18.0,
        "macro_olive_count": 8.0,
        "macro_terre_count": 6.0,
        "macro_gris_count": 4.0,
        "macro_multizone_ratio": 0.60,
        "largest_macro_mask_ratio": 0.06,
        "periphery_boundary_density_ratio": 1.22,
        "periphery_non_coyote_ratio": 1.17,
        "visual_score_final": 0.72,
        "visual_score_ratio": 0.91,
        "visual_score_silhouette": 0.79,
        "visual_score_contour": 0.73,
        "visual_score_main": 0.68,
        "visual_silhouette_color_diversity": 0.74,
        "visual_contour_break_score": 0.58,
        "visual_outline_band_diversity": 0.66,
        "visual_small_scale_structural_score": 0.54,
        "visual_military_score": 0.74,
    }


def make_candidate(seed: int = 1234, ratios: np.ndarray | None = None, metrics: Dict[str, float] | None = None) -> Any:
    ratios = np.array([0.32, 0.28, 0.22, 0.18], dtype=float) if ratios is None else ratios
    metrics = valid_metrics() if metrics is None else metrics
    return types.SimpleNamespace(
        seed=seed,
        ratios=np.asarray(ratios, dtype=float),
        metrics=dict(metrics),
        image=Image.new("RGB", (32, 32), (0, 0, 0)),
    )


def make_analysis(names: List[str] | None = None) -> Any:
    return types.SimpleNamespace(failure_names=list(names or ["ratio_olive", "center_empty_ratio"]))


class TestFeatureHelpers(unittest.TestCase):
    def test_candidate_to_feature_vector(self):
        candidate = make_candidate()
        vec = mut.candidate_to_feature_vector(candidate)
        self.assertEqual(vec.shape, (len(mut.FEATURE_KEYS),))
        self.assertAlmostEqual(float(vec[0]), 0.32)

    def test_analysis_to_failure_vector_and_context(self):
        analysis = make_analysis(["ratio_olive", "visual_military_score"])
        fail = mut.analysis_to_failure_vector(analysis)
        self.assertEqual(fail.shape, (len(mut.FAILURE_KEYS),))
        ctx = mut.build_context_vector(make_candidate(), analysis)
        self.assertEqual(ctx.shape, (len(mut.FEATURE_KEYS) + len(mut.FAILURE_KEYS),))

    def test_candidate_reward(self):
        c = make_candidate()
        self.assertGreater(mut.candidate_reward(c, True), mut.candidate_reward(c, False))


class TestStandardizer(unittest.TestCase):
    def test_fit_transform_state(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        s = mut.Standardizer(2)
        s.fit(x)
        out = s.transform(x)
        self.assertEqual(out.shape, x.shape)
        s2 = mut.Standardizer(2)
        s2.load_state_dict(s.state_dict())
        np.testing.assert_allclose(out, s2.transform(x))


class TestDeepSurrogate(TempDirMixin, unittest.TestCase):
    def test_fit_predict_save_load(self):
        torch.set_num_threads(1)
        dim = len(mut.FEATURE_KEYS)
        x = np.random.default_rng(123).normal(size=(8, dim)).astype(np.float32)
        y_valid = np.array([0, 1] * 4, dtype=np.float32)
        y_reward = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        model = mut.DeepSurrogate(input_dim=dim, hidden_dim=16, lr=1e-3, device="cpu")
        stats = model.fit(x, y_valid, y_reward, epochs=1, batch_size=4)
        self.assertIn("loss", stats)
        p_valid, p_reward = model.predict(x[0])
        self.assertEqual(p_valid.shape, (1,))
        self.assertEqual(p_reward.shape, (1,))
        path = self.tmpdir / "surrogate.pt"
        model.save(path)
        model2 = mut.DeepSurrogate(input_dim=dim, hidden_dim=16, lr=1e-3, device="cpu")
        model2.load(path)
        p2_valid, _ = model2.predict(x[0])
        self.assertEqual(p2_valid.shape, (1,))


class TestLinUCBBandit(unittest.TestCase):
    def test_scores_select_update(self):
        bandit = mut.LinUCBBandit(n_actions=3, context_dim=5, alpha=1.0)
        ctx = np.ones((5,), dtype=np.float32)
        top = bandit.select_top_k(ctx, 2)
        self.assertEqual(len(top), 2)
        bandit.update(top[0], ctx, 1.5)
        self.assertEqual(bandit.scores(ctx).shape, (3,))


class TestGuidedStateHelpers(unittest.TestCase):
    def test_merge_guided_delta(self):
        base = {"olive_scale_delta": 0.1, "zone_boost_deltas": [0.0 for _ in range(len(mut.camo.DENSITY_ZONES))]}
        delta = {"olive_scale_delta": 0.2, "extra_macro_attempts": 10, "zone_boost_deltas": [0.1 for _ in range(len(mut.camo.DENSITY_ZONES))], "expand_angle_pool": True}
        out = mut.merge_guided_delta(base, delta)
        self.assertAlmostEqual(out["olive_scale_delta"], 0.3)
        self.assertEqual(out["extra_macro_attempts"], 10)
        self.assertTrue(out["expand_angle_pool"])


class TestExperienceBuffer(TempDirMixin, unittest.TestCase):
    def test_add_as_arrays_save(self):
        buf = mut.ExperienceBuffer()
        reward = buf.add(make_candidate(), True)
        self.assertIsInstance(reward, float)
        x, y_valid, y_reward = buf.as_arrays()
        self.assertEqual(x.shape[0], 1)
        path = self.tmpdir / "dataset.npz"
        buf.save(path)
        self.assertTrue(path.exists())


class TestGenerator(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = mut.MLDLConfig(target_count=1, warmup_samples=2, candidate_pool_size=3, validate_top_k=2, max_attempts_per_target=4, train_epochs=1, batch_size=2, hidden_dim=8, output_dir=str(self.tmpdir), min_train_size=2, retrain_every=2, device="cpu")
        self.gen = mut.CamouflageMLDLGenerator(self.cfg)

    def test_warmup(self):
        with patch.object(mut.camo, "build_seed", side_effect=lambda target_index, local_attempt, base_seed: 1000 + local_attempt), \
             patch.object(mut.camo, "generate_candidate_from_seed", side_effect=lambda seed: make_candidate(seed=seed)), \
             patch.object(mut.camo, "validate_candidate_result", side_effect=[False, True]):
            self.gen.warmup()
        x, y_valid, _ = self.gen.buffer.as_arrays()
        self.assertEqual(len(x), 2)
        self.assertEqual(int(np.sum(y_valid)), 1)

    def test_maybe_train(self):
        for i in range(2):
            self.gen.buffer.add(make_candidate(seed=i), accepted=bool(i % 2))
        with patch.object(self.gen.surrogate, "fit", return_value={"loss": 0.1}) as mock_fit, \
             patch.object(self.gen.surrogate, "save") as mock_save, \
             patch.object(self.gen.buffer, "save") as mock_buf_save:
            out = self.gen.maybe_train(force=True)
        self.assertEqual(out, {"loss": 0.1})
        mock_fit.assert_called_once()
        mock_save.assert_called_once()
        mock_buf_save.assert_called_once()

    def test_propose_candidates(self):
        base_state = {"zone_boost_deltas": [0.0 for _ in range(len(mut.camo.DENSITY_ZONES))]}
        with patch.object(mut.camo, "build_seed", side_effect=lambda target_index, local_attempt, base_seed: 2000 + local_attempt), \
             patch.object(mut.camo, "generate_candidate_from_seed", side_effect=lambda seed, correction_state=None: make_candidate(seed=seed)), \
             patch.object(mut.camo, "_guided_state_has_effects", return_value=False, create=True):
            proposals = self.gen._propose_candidates(1, 1, base_state, analysis=None)
        self.assertTrue(proposals)
        self.assertTrue(all(isinstance(p, mut.Proposal) for p in proposals))

    def test_validate_top_candidates(self):
        proposals = [
            mut.Proposal(seed=1, action_idx=0, action_name="a0", guided_state={}, candidate=make_candidate(seed=1), pred_valid=0.8, pred_reward=1.0),
            mut.Proposal(seed=2, action_idx=1, action_name="a1", guided_state={}, candidate=make_candidate(seed=2), pred_valid=0.6, pred_reward=0.5),
        ]
        fake_analysis = types.SimpleNamespace(failure_names=["ratio_olive"])
        self.gen.last_rejected_candidate = make_candidate(seed=0)
        with patch.object(mut, "neutral_guided_state", return_value={"zone_boost_deltas": [0.0 for _ in range(len(mut.camo.DENSITY_ZONES))]}), \
             patch.object(mut.camo, "validate_candidate_result", side_effect=[False, True]), \
             patch.object(mut.camo, "deep_rejection_analysis", return_value=fake_analysis, create=True), \
             patch.object(mut.camo, "_merge_guided_generation_state", side_effect=lambda state, analysis: {"zone_boost_deltas": [0.0 for _ in range(len(mut.camo.DENSITY_ZONES))], "reject_streak": 1}, create=True), \
             patch.object(self.gen, "maybe_train", return_value=None), \
             patch.object(self.gen.bandit, "update") as mock_update:
            accepted, analysis, state = self.gen._validate_top_candidates(proposals, 1, 1)
        self.assertIsNotNone(accepted)
        self.assertIsNotNone(state)
        mock_update.assert_called_once()

    def test_generate(self):
        self.gen.last_rejected_candidate = make_candidate(seed=0)
        accepted = mut.Proposal(seed=10, action_idx=0, action_name="boost_olive", guided_state={}, candidate=make_candidate(seed=10), pred_valid=0.9, pred_reward=2.0)
        with patch.object(mut.camo, "generate_candidate_from_seed", return_value=make_candidate(seed=0)), \
             patch.object(mut, "neutral_guided_state", return_value={"zone_boost_deltas": [0.0 for _ in range(len(mut.camo.DENSITY_ZONES))]}), \
             patch.object(self.gen, "warmup", return_value=None), \
             patch.object(self.gen, "maybe_train", return_value={"loss": 0.1}), \
             patch.object(self.gen, "_propose_candidates", return_value=[accepted]), \
             patch.object(self.gen, "_validate_top_candidates", return_value=(accepted, None, {"zone_boost_deltas": [0.0 for _ in range(len(mut.camo.DENSITY_ZONES))]})), \
             patch.object(mut.camo, "save_candidate_image") as mock_save, \
             patch.object(mut.camo, "candidate_row", return_value={"index": 1, "seed": 10}) as mock_row, \
             patch.object(mut.camo, "write_report") as mock_report, \
             patch.object(self.gen, "_write_summary") as mock_summary:
            rows = self.gen.generate()
        self.assertEqual(rows, [{"index": 1, "seed": 10}])
        mock_save.assert_called_once()
        mock_row.assert_called_once()
        mock_report.assert_called_once()
        mock_summary.assert_called_once()

    def test_write_summary(self):
        self.gen.rows = [{"index": 1}]
        self.gen.total_attempts = 3
        self.gen._write_summary()
        path = self.tmpdir / "run_summary_ml_dl.json"
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["total_rows"], 1)


class TestCLI(unittest.TestCase):
    def test_parse_args(self):
        argv = ["camouflage_ml_dl.py", "--target-count", "5", "--warmup-samples", "16", "--candidate-pool-size", "4", "--validate-top-k", "2", "--output-dir", "out_dir"]
        with patch.object(sys, "argv", argv):
            cfg = mut.parse_args()
        self.assertEqual(cfg.target_count, 5)
        self.assertEqual(cfg.output_dir, "out_dir")

    def test_main(self):
        cfg = mut.MLDLConfig(target_count=2, output_dir="out_dir_test")
        fake_runner = types.SimpleNamespace(generate=Mock(return_value=[{"index": 1}, {"index": 2}]), total_attempts=7)
        with patch.object(mut, "parse_args", return_value=cfg), \
             patch.object(mut, "CamouflageMLDLGenerator", return_value=fake_runner), \
             patch.object(mut.random, "seed") as mock_rseed, \
             patch.object(mut.np.random, "seed") as mock_nseed, \
             patch.object(mut.torch, "manual_seed") as mock_tseed, \
             patch.object(mut.torch.cuda, "is_available", return_value=False):
            mut.main()
        fake_runner.generate.assert_called_once()
        mock_rseed.assert_called_once_with(cfg.random_seed)
        mock_nseed.assert_called_once_with(cfg.random_seed)
        mock_tseed.assert_called_once_with(cfg.random_seed)


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_camouflage_ml_dl.py ==========")
    unittest.main(verbosity=2)
