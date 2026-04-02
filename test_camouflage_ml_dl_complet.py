# -*- coding: utf-8 -*-
"""
Suite de tests pour camouflage_ml_dl_guided.py.

Objectifs :
- couvrir les helpers de features, rejet et reward ;
- couvrir standardisation, surrogate DL, bandit et buffer ;
- couvrir l'orchestrateur guidé ML/DL, le résumé et la CLI ;
- rester rapide et déterministe.
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
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

MODULE_NAME = os.getenv("CAMO_MLDL_MODULE", "camouflage_ml_dl")
mut = importlib.import_module(MODULE_NAME)

LOG_DIR = Path(os.getenv("LOG_OUTPUT_DIR", TEST_DIR / "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_camouflage_ml_dl_guided.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_camouflage_ml_dl_precise")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


LOGGER = configure_logger()
LOGGER.info("Module ML/DL sous test: %s", getattr(mut, "__name__", type(mut).__name__))
LOGGER.info("Fichier de log ML/DL: %s", LOG_FILE)


class LoggedTestMixin:
    def setUp(self) -> None:
        super().setUp()
        import time as _time
        self._started_at = _time.perf_counter()
        LOGGER.info("▶ START %s", self.id())

    def tearDown(self) -> None:
        import time as _time
        elapsed = _time.perf_counter() - getattr(self, "_started_at", _time.perf_counter())
        LOGGER.info("■ END   %s | %.3fs", self.id(), elapsed)
        super().tearDown()

    def log_state(self, message: str, **payload: Any) -> None:
        if payload:
            LOGGER.info("%s | %s", message, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        else:
            LOGGER.info("%s", message)


class TempDirMixin:
    def setUp(self) -> None:
        super().setUp()
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_camouflage_ml_dl_precise_")
        self.tmpdir = Path(self._tmpdir_obj.name)
        LOGGER.info("tmpdir ML/DL prêt: %s", self.tmpdir)

    def tearDown(self) -> None:
        LOGGER.info("tmpdir ML/DL cleanup: %s", getattr(self, "tmpdir", "<absent>"))
        self._tmpdir_obj.cleanup()
        super().tearDown()


def valid_metrics() -> Dict[str, float]:
    return {
        "largest_olive_component_ratio": 0.24,
        "boundary_density": 0.05,
        "boundary_density_small": 0.06,
        "boundary_density_tiny": 0.07,
        "mirror_similarity": 0.44,
        "edge_contact_ratio": 0.30,
        "overscan": 1.11,
        "shift_strength": 0.92,
        "width": 256.0,
        "height": 144.0,
        "physical_width_cm": 240.0,
        "physical_height_cm": 135.0,
        "px_per_cm": 1.066,
        "motif_scale": 0.58,
    }


def make_candidate(seed: int = 1234, ratios: np.ndarray | None = None, metrics: Dict[str, float] | None = None) -> Any:
    ratios = np.array([0.32, 0.28, 0.22, 0.18], dtype=float) if ratios is None else np.asarray(ratios, dtype=float)
    metrics = valid_metrics() if metrics is None else dict(metrics)
    return types.SimpleNamespace(
        seed=seed,
        ratios=ratios,
        metrics=metrics,
        image=Image.new("RGB", (32, 32), (0, 0, 0)),
    )


def make_invalid_candidate(seed: int = 1234) -> Any:
    m = valid_metrics()
    m["mirror_similarity"] = 0.99
    ratios = np.array([0.32, 0.28, 0.22, 0.18], dtype=float)
    return make_candidate(seed=seed, ratios=ratios, metrics=m)


def log_json_preview(path: Path) -> None:
    if not path.exists():
        LOGGER.info("json absent: %s", path)
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.info("json illisible: %s | %s", path, exc)
        return
    if isinstance(payload, dict):
        LOGGER.info("json %s | keys=%s", path.name, sorted(payload.keys()))
    else:
        LOGGER.info("json %s | type=%s", path.name, type(payload).__name__)


class TestHelpers(LoggedTestMixin, unittest.TestCase):
    def test_safe_float(self):
        self.assertEqual(mut._safe_float(1.2), 1.2)
        self.assertEqual(mut._safe_float("x", 3.4), 3.4)
        self.assertEqual(mut._safe_float(float("inf"), 5.6), 5.6)

    def test_candidate_to_feature_dict_and_vector(self):
        candidate = make_candidate()
        feat = mut.candidate_to_feature_dict(candidate)
        vec = mut.candidate_to_feature_vector(candidate)
        self.assertEqual(set(mut.FEATURE_KEYS), set(feat.keys()))
        self.assertEqual(vec.shape, (len(mut.FEATURE_KEYS),))
        self.assertAlmostEqual(float(feat["ratio_coyote"]), 0.32)

    def test_analyze_rejection_and_failure_vector(self):
        candidate = make_invalid_candidate()
        analysis = mut.analyze_rejection(candidate, target_index=1, local_attempt=2)
        self.assertIsInstance(analysis, mut.RejectionAnalysis)
        self.assertGreaterEqual(analysis.fail_count, 1)
        vec = mut.analysis_to_failure_vector(analysis)
        self.assertEqual(vec.shape, (len(mut.FAILURE_KEYS),))
        ctx = mut.build_context_vector(candidate, analysis)
        self.assertEqual(ctx.shape, (len(mut.FEATURE_KEYS) + len(mut.FAILURE_KEYS),))

    def test_candidate_reward(self):
        c = make_candidate()
        self.assertGreater(mut.candidate_reward(c, True), mut.candidate_reward(c, False))

    def test_propose_seed(self):
        self.assertEqual(mut.propose_seed(10, {"mode": "linear", "offset": 2, "step": 3}), 15)
        self.assertEqual(mut.propose_seed(10, {"mode": "offset", "offset": 7}), 17)
        self.assertEqual(mut.propose_seed(10, {"mode": "affine", "mul": 3, "add": 1}), 31)
        self.assertEqual(mut.propose_seed(10, {"mode": "xor", "mask": 5}), 15)


class TestStandardizer(LoggedTestMixin, unittest.TestCase):
    def test_fit_transform_state(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        s = mut.Standardizer(2)
        s.fit(x)
        out = s.transform(x)
        self.assertEqual(out.shape, x.shape)
        s2 = mut.Standardizer(2)
        s2.load_state_dict(s.state_dict())
        np.testing.assert_allclose(out, s2.transform(x))


class TestDeepSurrogate(LoggedTestMixin, TempDirMixin, unittest.TestCase):
    def test_fit_predict_save_load(self):
        dim = len(mut.FEATURE_KEYS)
        x = np.random.default_rng(123).normal(size=(12, dim)).astype(np.float32)
        y_valid = np.array([0, 1] * 6, dtype=np.float32)
        y_reward = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        model = mut.DeepSurrogate(input_dim=dim, hidden_dim=16, lr=1e-3, device="cpu")
        stats = model.fit(x, y_valid, y_reward, epochs=1, batch_size=4)
        self.assertIn("loss", stats)
        p_valid, p_reward = model.predict(x[0])
        self.assertEqual(p_valid.shape, (1,))
        self.assertEqual(p_reward.shape, (1,))
        path = self.tmpdir / "surrogate.pt"
        model.save(path)
        self.assertTrue(path.exists())
        model2 = mut.DeepSurrogate(input_dim=dim, hidden_dim=16, lr=1e-3, device="cpu")
        model2.load(path)
        p2_valid, _ = model2.predict(x[0])
        self.assertEqual(p2_valid.shape, (1,))


class TestLinUCBBandit(LoggedTestMixin, unittest.TestCase):
    def test_scores_select_update(self):
        bandit = mut.LinUCBBandit(n_actions=3, context_dim=5, alpha=1.0)
        ctx = np.ones((5,), dtype=np.float32)
        top = bandit.select_top_k(ctx, 2)
        self.assertEqual(len(top), 2)
        bandit.update(top[0], ctx, 1.5)
        self.assertEqual(bandit.scores(ctx).shape, (3,))


class TestExperienceBuffer(LoggedTestMixin, TempDirMixin, unittest.TestCase):
    def test_add_as_arrays_save(self):
        buf = mut.ExperienceBuffer()
        reward = buf.add(make_candidate(), True)
        self.assertIsInstance(reward, float)
        x, y_valid, y_reward = buf.as_arrays()
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(y_valid.tolist(), [1.0])
        self.assertEqual(y_reward.shape[0], 1)
        path = self.tmpdir / "dataset.npz"
        buf.save(path)
        self.assertTrue(path.exists())


class TestGenerator(LoggedTestMixin, TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = mut.MLDLConfig(
            target_count=1,
            warmup_samples=2,
            candidate_pool_size=3,
            validate_top_k=2,
            max_attempts_per_target=4,
            train_epochs=1,
            batch_size=2,
            hidden_dim=8,
            output_dir=str(self.tmpdir),
            min_train_size=2,
            retrain_every=2,
            device="cpu",
        )
        self.gen = mut.CamouflageMLDLGenerator(self.cfg)

    def test_resolve_device(self):
        self.assertIn(self.gen._resolve_device("auto"), {"cpu", "cuda"})
        self.assertEqual(self.gen._resolve_device("cpu"), "cpu")

    def test_warmup(self):
        with patch.object(mut.camo, "build_seed", side_effect=lambda target_index, local_attempt, base_seed: 1000 + local_attempt), \
             patch.object(mut.camo, "generate_candidate_from_seed", side_effect=lambda seed: make_candidate(seed=seed)), \
             patch.object(mut.camo, "validate_candidate_result", side_effect=[False, True]):
            self.gen.warmup()
        x, y_valid, _ = self.gen.buffer.as_arrays()
        self.assertEqual(len(x), 2)
        self.assertEqual(int(np.sum(y_valid)), 1)
        self.assertIsNotNone(self.gen.last_rejected_candidate)

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

    def test_select_action_indexes(self):
        out = self.gen._select_action_indexes(None)
        self.assertTrue(out)
        self.assertLessEqual(len(out), self.cfg.candidate_pool_size)

    def test_propose_candidates(self):
        self.gen.last_rejected_candidate = make_candidate(seed=999)
        self.gen.last_analysis = mut.RejectionAnalysis(1, 1, 999, 1, 1.0, ["mirror_similarity_high"], ["x"])
        with patch.object(mut.camo, "build_seed", side_effect=lambda target_index, local_attempt, base_seed: 2000 + local_attempt), \
             patch.object(mut.camo, "generate_candidate_from_seed", side_effect=lambda seed: make_candidate(seed=seed)):
            proposals = self.gen._propose_candidates(1, 1, self.gen.last_analysis)
        self.assertTrue(proposals)
        self.assertTrue(all(isinstance(p, mut.Proposal) for p in proposals))
        self.assertLessEqual(len(proposals), self.cfg.candidate_pool_size)

    def test_validate_top_candidates_accepts(self):
        proposal = mut.Proposal(seed=1, action_idx=0, action_name="a0", candidate=make_candidate(seed=1), pred_valid=0.7, pred_reward=0.1)
        with patch.object(mut.camo, "validate_candidate_result", return_value=True), \
             patch.object(self.gen, "maybe_train", return_value=None) as maybe_train, \
             patch.object(self.gen.bandit, "update") as bandit_update:
            accepted, best_analysis = self.gen._validate_top_candidates([proposal], target_index=1, local_attempt=1)
        self.assertIs(accepted, proposal)
        self.assertIsNone(best_analysis)
        maybe_train.assert_called_once()
        bandit_update.assert_called_once()

    def test_validate_top_candidates_rejects(self):
        proposal = mut.Proposal(seed=1, action_idx=0, action_name="a0", candidate=make_invalid_candidate(seed=1), pred_valid=0.2, pred_reward=-0.1)
        with patch.object(mut.camo, "validate_candidate_result", return_value=False), \
             patch.object(self.gen, "maybe_train", return_value=None), \
             patch.object(self.gen.bandit, "update") as bandit_update:
            accepted, best_analysis = self.gen._validate_top_candidates([proposal], target_index=1, local_attempt=1)
        self.assertIsNone(accepted)
        self.assertIsInstance(best_analysis, mut.RejectionAnalysis)
        bandit_update.assert_called_once()
        self.assertIsNotNone(self.gen.last_rejected_candidate)

    def test_write_summary(self):
        self.gen.rows = [{"index": 1}]
        self.gen.total_attempts = 3
        self.gen._write_summary()
        path = self.tmpdir / "run_summary_ml_dl.json"
        self.assertTrue(path.exists())
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["total_rows"], 1)
        self.assertEqual(summary["total_attempts"], 3)

    def test_generate(self):
        candidate = make_candidate(seed=7)
        proposal = mut.Proposal(seed=7, action_idx=0, action_name="a0", candidate=candidate, pred_valid=0.8, pred_reward=1.0)
        row = {"index": 1, "seed": 7}

        def fake_warmup():
            self.gen.last_rejected_candidate = make_invalid_candidate(seed=5)
            self.gen.last_analysis = mut.RejectionAnalysis(0, 1, 5, 1, 1.0, ["mirror_similarity_high"], ["x"])

        with patch.object(self.gen, "warmup", side_effect=fake_warmup), \
             patch.object(self.gen, "maybe_train", return_value=None), \
             patch.object(self.gen, "_propose_candidates", return_value=[proposal]), \
             patch.object(self.gen, "_validate_top_candidates", return_value=(proposal, None)), \
             patch.object(mut.camo, "save_candidate_image") as save_img, \
             patch.object(mut.camo, "candidate_row", return_value=row), \
             patch.object(mut.camo, "write_report") as write_report:
            rows = self.gen.generate()
        self.assertEqual(rows, [row])
        save_img.assert_called_once()
        write_report.assert_called_once()
        self.assertTrue((self.tmpdir / "run_summary_ml_dl.json").exists())


class TestCliAndPublicAPI(LoggedTestMixin, TempDirMixin, unittest.TestCase):
    def test_parse_args(self):
        with patch.object(sys, "argv", [
            "prog",
            "--target-count", "2",
            "--warmup-samples", "4",
            "--candidate-pool-size", "5",
            "--validate-top-k", "2",
            "--max-attempts-per-target", "11",
            "--train-epochs", "3",
            "--batch-size", "7",
            "--learning-rate", "0.02",
            "--hidden-dim", "16",
            "--device", "cpu",
            "--base-seed", "999",
            "--output-dir", str(self.tmpdir),
            "--alpha-ucb", "1.7",
            "--min-train-size", "8",
            "--retrain-every", "9",
            "--random-seed", "111",
        ]):
            cfg = mut.parse_args()
        self.assertEqual(cfg.target_count, 2)
        self.assertEqual(cfg.warmup_samples, 4)
        self.assertEqual(cfg.output_dir, str(self.tmpdir))
        self.assertEqual(cfg.random_seed, 111)

    def test_build_config_from_main_args(self):
        args = types.SimpleNamespace(
            target_count=3,
            mldl_warmup_samples=4,
            mldl_candidate_pool_size=5,
            mldl_validate_top_k=2,
            mldl_max_attempts_per_target=10,
            mldl_train_epochs=6,
            mldl_batch_size=7,
            mldl_learning_rate=0.01,
            mldl_hidden_dim=16,
            mldl_device="cpu",
            base_seed=999,
            output_dir=str(self.tmpdir),
            mldl_alpha_ucb=1.7,
            mldl_min_train_size=8,
            mldl_retrain_every=9,
            random_seed=111,
        )
        cfg = mut.build_config_from_main_args(args)
        self.assertEqual(cfg.target_count, 3)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(cfg.output_dir, str(self.tmpdir))

    def test_run_guided_generation(self):
        cfg = mut.MLDLConfig(target_count=1, output_dir=str(self.tmpdir), random_seed=123)
        summary_path = self.tmpdir / "run_summary_ml_dl.json"
        summary_path.write_text(json.dumps({"total_attempts": 5}), encoding="utf-8")
        with patch.object(mut, "CamouflageMLDLGenerator") as runner_cls:
            runner = runner_cls.return_value
            runner.generate.return_value = [{"index": 1}]
            rows, summary = mut.run_guided_generation(cfg)
        self.assertEqual(rows, [{"index": 1}])
        self.assertEqual(summary["total_attempts"], 5)

    def test_main(self):
        cfg = mut.MLDLConfig(target_count=1, output_dir=str(self.tmpdir), random_seed=123)
        with patch.object(mut, "parse_args", return_value=cfg), \
             patch.object(mut, "run_guided_generation", return_value=([{"index": 1}], {"total_attempts": 7})) as run_mock:
            mut.main()
        run_mock.assert_called_once_with(cfg)


class TestLoggingArtifacts(LoggedTestMixin, TempDirMixin, unittest.TestCase):
    def test_log_helpers_and_summary_preview(self):
        payload = {"ok": True, "items": [1, 2, 3]}
        path = self.tmpdir / "summary.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        log_json_preview(path)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_camouflage_ml_dl_precise.py ==========")
    unittest.main(verbosity=2)
