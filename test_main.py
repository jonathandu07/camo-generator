# -*- coding: utf-8 -*-
"""
Suite de tests robuste pour main.py / main_fused.py.

Objectifs :
- couvrir les helpers purs, la géométrie, les métriques et la validation ;
- couvrir le guidage adaptatif et les helpers ML/DL ;
- tester les exports ;
- tester les orchestrateurs sync / async sans vrai ProcessPool ;
- rester déterministe et rapide.

Exécution :
    MUT_MODULE=main python -m unittest -v test_main_corrected.py
ou :
    MUT_MODULE=main_fused python -m unittest -v test_main_corrected.py
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import logging
import os
import random
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
from typing import Any, Dict, Iterable, List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from PIL import Image


def _import_mut():
    candidates = []
    env_name = os.getenv("MUT_MODULE", "").strip()
    if env_name:
        candidates.append(env_name)
    candidates.extend(["main", "main_fused"])
    last_exc = None
    for name in dict.fromkeys(candidates):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    raise RuntimeError(f"Impossible d'importer le module cible parmi {candidates!r}") from last_exc


mut = _import_mut()

LOG_DIR = Path(os.getenv("LOG_OUTPUT_DIR", Path(__file__).resolve().parent / "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_main.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_main")
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


class TempDirMixin:
    def setUp(self) -> None:
        super().setUp()
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_")
        self.tmpdir = Path(self._tmpdir_obj.name)

    def tearDown(self) -> None:
        self._tmpdir_obj.cleanup()
        super().tearDown()


class GlobalStateMixin:
    def setUp(self) -> None:
        super().setUp()
        self._orig_log_cache = getattr(mut, "_LOG_MODULE_CACHE", None)
        self._orig_log_attempted = getattr(mut, "_LOG_MODULE_ATTEMPTED", False)
        self._orig_pool = getattr(mut, "_PROCESS_POOL", None)
        self._orig_pool_workers = getattr(mut, "_PROCESS_POOL_WORKERS", None)
        if hasattr(mut, "_LOG_MODULE_CACHE"):
            mut._LOG_MODULE_CACHE = None
        if hasattr(mut, "_LOG_MODULE_ATTEMPTED"):
            mut._LOG_MODULE_ATTEMPTED = False
        if hasattr(mut, "_PROCESS_POOL"):
            mut._PROCESS_POOL = None
        if hasattr(mut, "_PROCESS_POOL_WORKERS"):
            mut._PROCESS_POOL_WORKERS = None

    def tearDown(self) -> None:
        try:
            if getattr(mut, "_PROCESS_POOL", None) is not None:
                mut.shutdown_process_pool()
        except Exception:
            pass
        if hasattr(mut, "_LOG_MODULE_CACHE"):
            mut._LOG_MODULE_CACHE = self._orig_log_cache
        if hasattr(mut, "_LOG_MODULE_ATTEMPTED"):
            mut._LOG_MODULE_ATTEMPTED = self._orig_log_attempted
        if hasattr(mut, "_PROCESS_POOL"):
            mut._PROCESS_POOL = self._orig_pool
        if hasattr(mut, "_PROCESS_POOL_WORKERS"):
            mut._PROCESS_POOL_WORKERS = self._orig_pool_workers
        super().tearDown()


class AssertionsMixin:
    def assertFloatClose(self, a: float, b: float, places: int = 7, msg: str | None = None) -> None:
        self.assertAlmostEqual(float(a), float(b), places=places, msg=msg)

    def assertArrayClose(self, a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> None:
        np.testing.assert_allclose(a, b, atol=atol, rtol=0)

    def assertCandidateLooksConsistent(self, candidate: Any) -> None:
        self.assertIsInstance(candidate, mut.CandidateResult)
        self.assertIsInstance(candidate.profile, mut.VariantProfile)
        self.assertIsInstance(candidate.image, Image.Image)
        self.assertEqual(candidate.image.size, (mut.WIDTH, mut.HEIGHT))
        self.assertEqual(candidate.ratios.shape, (4,))
        self.assertAlmostEqual(float(np.sum(candidate.ratios)), 1.0, places=5)
        self.assertIsInstance(candidate.metrics, dict)


REQUIRED_METRIC_KEYS = {
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
    "visual_silhouette_color_diversity",
    "visual_contour_break_score",
    "visual_outline_band_diversity",
    "visual_small_scale_structural_score",
    "visual_military_score",
}


def valid_ratios() -> np.ndarray:
    return np.array([0.32, 0.28, 0.22, 0.18], dtype=float)


def invalid_ratios_far() -> np.ndarray:
    return np.array([0.50, 0.20, 0.20, 0.10], dtype=float)


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
        "visual_silhouette_color_diversity": 0.74,
        "visual_contour_break_score": 0.58,
        "visual_outline_band_diversity": 0.66,
        "visual_small_scale_structural_score": 0.54,
        "visual_military_score": 0.74,
    }


def make_candidate(seed: int = 123456, ratios: np.ndarray | None = None, metrics: Dict[str, float] | None = None):
    ratios = valid_ratios() if ratios is None else ratios
    metrics = valid_metrics() if metrics is None else metrics
    return mut.CandidateResult(
        seed=seed,
        profile=mut.make_profile(seed),
        image=Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0)),
        ratios=ratios,
        metrics=metrics,
    )


def fake_snapshot(*, machine_intensity: float = 0.94, available_mb: float = 8192.0, disk_free_mb: float = 4096.0):
    return mut.ResourceSnapshot(
        ts=1.0,
        cpu_count=max(1, mut.CPU_COUNT),
        process_cpu_percent=10.0,
        system_cpu_percent=20.0,
        process_rss_mb=128.0,
        system_available_mb=available_mb,
        system_total_mb=32768.0,
        disk_free_mb=disk_free_mb,
        machine_intensity=machine_intensity,
    )


def iter_metric_failure_cases() -> Iterable[tuple[str, float]]:
    yield "largest_olive_component_ratio", mut.MIN_OLIVE_CONNECTED_COMPONENT_RATIO - 0.001
    yield "largest_olive_component_ratio_small", 0.119
    yield "olive_multizone_share", mut.MIN_OLIVE_MULTIZONE_SHARE - 0.001
    yield "center_empty_ratio", mut.MAX_COYOTE_CENTER_EMPTY_RATIO + 0.001
    yield "center_empty_ratio_small", mut.MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL + 0.001
    yield "boundary_density", mut.MIN_BOUNDARY_DENSITY - 0.001
    yield "boundary_density", mut.MAX_BOUNDARY_DENSITY + 0.001
    yield "boundary_density_small", mut.MIN_BOUNDARY_DENSITY_SMALL - 0.001
    yield "boundary_density_small", mut.MAX_BOUNDARY_DENSITY_SMALL + 0.001
    yield "mirror_similarity", mut.MAX_MIRROR_SIMILARITY + 0.001
    yield "central_brown_continuity", mut.MAX_CENTRAL_BROWN_CONTINUITY + 0.001
    yield "oblique_share", mut.MIN_OBLIQUE_SHARE - 0.001
    yield "vertical_share", mut.MIN_VERTICAL_SHARE - 0.001
    yield "vertical_share", mut.MAX_VERTICAL_SHARE + 0.001
    yield "angle_dominance_ratio", mut.MAX_ANGLE_DOMINANCE_RATIO + 0.001
    yield "macro_olive_visible_ratio", mut.MIN_MACRO_OLIVE_VISIBLE_RATIO - 0.001
    yield "macro_terre_visible_ratio", mut.MIN_MACRO_TERRE_VISIBLE_RATIO - 0.001
    yield "macro_gris_visible_ratio", mut.MIN_MACRO_GRIS_VISIBLE_RATIO - 0.001
    yield "macro_total_count", mut.MIN_TOTAL_MACRO_COUNT - 1
    yield "macro_olive_count", mut.MIN_OLIVE_MACRO_COUNT - 1
    yield "macro_terre_count", mut.MIN_TERRE_MACRO_COUNT - 1
    yield "macro_gris_count", mut.MIN_GRIS_MACRO_COUNT - 1
    yield "macro_multizone_ratio", mut.MIN_GLOBAL_MACRO_MULTIZONE_RATIO - 0.001
    yield "largest_macro_mask_ratio", mut.MAX_SINGLE_MACRO_MASK_RATIO + 0.001
    yield "periphery_boundary_density_ratio", mut.MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO - 0.001
    yield "periphery_non_coyote_ratio", mut.MIN_PERIPHERY_NON_COYOTE_RATIO - 0.001
    yield "visual_silhouette_color_diversity", mut.VISUAL_MIN_SILHOUETTE_COLOR_DIVERSITY - 0.001
    yield "visual_contour_break_score", mut.VISUAL_MIN_CONTOUR_BREAK_SCORE - 0.001
    yield "visual_outline_band_diversity", mut.VISUAL_MIN_OUTLINE_BAND_DIVERSITY - 0.001
    yield "visual_small_scale_structural_score", mut.VISUAL_MIN_SMALL_SCALE_STRUCTURAL_SCORE - 0.001
    yield "visual_score_final", mut.VISUAL_MIN_FINAL_SCORE - 0.001
    yield "visual_military_score", mut.VISUAL_MIN_MILITARY_SCORE - 0.001


class TestConstantsAndDataclasses(AssertionsMixin, unittest.TestCase):
    def test_global_constants_are_coherent(self) -> None:
        self.assertEqual(mut.COLOR_NAMES, ["coyote_brown", "vert_olive", "terre_de_france", "vert_de_gris"])
        self.assertEqual(tuple(mut.RGB.shape), (4, 3))
        self.assertEqual(tuple(mut.TARGET.shape), (4,))
        self.assertEqual(float(np.sum(mut.TARGET)), 1.0)
        self.assertGreater(mut.WIDTH, 0)
        self.assertGreater(mut.HEIGHT, 0)
        self.assertGreater(mut.PX_PER_CM, 0)

    def test_resource_snapshot_to_dict(self) -> None:
        snap = fake_snapshot()
        out = snap.to_dict()
        self.assertEqual(set(out.keys()), {
            "ts", "cpu_count", "process_cpu_percent", "system_cpu_percent", "process_rss_mb",
            "system_available_mb", "system_total_mb", "disk_free_mb", "machine_intensity",
        })
        self.assertEqual(out["cpu_count"], float(mut.CPU_COUNT))

    def test_runtime_tuning_normalized(self) -> None:
        rt = mut.RuntimeTuning(max_workers=0, attempt_batch_size=0, parallel_attempts=True, machine_intensity=2.0).normalized()
        self.assertEqual(rt.max_workers, 1)
        self.assertEqual(rt.attempt_batch_size, 1)
        self.assertFalse(rt.parallel_attempts)
        self.assertEqual(rt.machine_intensity, 1.0)


class TestLoggingHelpers(GlobalStateMixin, TempDirMixin, unittest.TestCase):
    def test_get_log_module_from_sys_modules(self) -> None:
        fake = types.SimpleNamespace()
        with patch.dict(sys.modules, {"log": fake}, clear=False):
            mod = mut._get_log_module()
        self.assertIs(mod, fake)

    def test_runtime_log_calls_log_event(self) -> None:
        called: list[tuple[Any, ...]] = []

        def log_event(level: str, source: str, message: str, **payload: Any) -> None:
            called.append((level, source, message, payload))

        fake = types.SimpleNamespace(log_event=log_event)
        with patch.dict(sys.modules, {"log": fake}, clear=False):
            mut._LOG_MODULE_CACHE = None
            mut._LOG_MODULE_ATTEMPTED = False
            mut._runtime_log("INFO", "src", "hello", value=3)
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][0:3], ("INFO", "src", "hello"))
        self.assertEqual(called[0][3]["value"], 3)

    def test_run_log_preflight_can_raise(self) -> None:
        fake = types.SimpleNamespace(run_generation_preflight=lambda **_: {"ok": False, "message": "refused"})
        with patch.dict(sys.modules, {"log": fake}, clear=False):
            mut._LOG_MODULE_CACHE = None
            mut._LOG_MODULE_ATTEMPTED = False
            with self.assertRaises(RuntimeError):
                mut._run_log_preflight(strict=True, output_dir=self.tmpdir)


class TestSystemHelpers(GlobalStateMixin, TempDirMixin, AssertionsMixin, unittest.TestCase):
    def test_worker_initializer_can_limit_numeric_threads(self) -> None:
        with patch.dict(os.environ, {"CAMO_LIMIT_NUMERIC_THREADS": "1"}, clear=True):
            mut._worker_initializer()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "1")

    def test_shutdown_process_pool_calls_shutdown(self) -> None:
        pool = Mock()
        mut._PROCESS_POOL = pool
        mut._PROCESS_POOL_WORKERS = 4
        mut.shutdown_process_pool()
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        self.assertIsNone(mut._PROCESS_POOL)

    def test_get_process_pool_recreates_when_worker_count_changes(self) -> None:
        first = Mock(name="first_pool")
        second = Mock(name="second_pool")
        with patch.object(mut, "ProcessPoolExecutor", side_effect=[first, second]):
            p1 = mut.get_process_pool(2)
            p2 = mut.get_process_pool(4)
        self.assertIs(p1, first)
        self.assertIs(p2, second)
        first.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_compute_runtime_tuning_uses_memory_thresholds(self) -> None:
        low_mem = fake_snapshot(available_mb=900.0)
        out = mut.compute_runtime_tuning(machine_intensity=0.9, sample=low_mem)
        self.assertEqual(out.max_workers, 1)
        self.assertEqual(out.attempt_batch_size, 1)

    def test_validate_generation_request_rejects_low_disk(self) -> None:
        with patch.object(mut, "sample_process_resources", return_value=fake_snapshot(disk_free_mb=128.0)):
            with self.assertRaises(RuntimeError):
                mut.validate_generation_request(
                    target_count=1,
                    output_dir=self.tmpdir,
                    base_seed=1,
                    machine_intensity=0.5,
                    max_workers=1,
                    attempt_batch_size=1,
                )


class TestPureUtilities(TempDirMixin, AssertionsMixin, unittest.TestCase):
    def test_build_seed_is_deterministic(self) -> None:
        self.assertEqual(mut.build_seed(3, 7, 1000), mut.build_seed(3, 7, 1000))
        self.assertNotEqual(mut.build_seed(3, 7, 1000), mut.build_seed(4, 7, 1000))

    def test_make_profile_is_deterministic_for_same_seed(self) -> None:
        p1 = mut.make_profile(424242)
        p2 = mut.make_profile(424242)
        self.assertEqual(p1.allowed_angles, p2.allowed_angles)
        self.assertEqual(p1.angle_pool, p2.angle_pool)
        self.assertEqual(p1.zone_weight_boosts, p2.zone_weight_boosts)
        self.assertIn(0, p1.allowed_angles)

    def test_compute_ratios_and_render_canvas(self) -> None:
        canvas = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        ratios = mut.compute_ratios(canvas)
        self.assertArrayClose(ratios, np.array([0.25, 0.25, 0.25, 0.25]))
        self.assertEqual(mut.render_canvas(canvas).size, (2, 2))

    def test_geometric_helpers(self) -> None:
        x, y = mut.rotate(1.0, 0.0, 90.0)
        self.assertFloatClose(x, 0.0, places=6)
        self.assertFloatClose(y, 1.0, places=6)
        poly = [(100, 100), (120, 100), (120, 130), (100, 130)]
        mask = mut.polygon_mask(poly)
        self.assertGreater(int(mask.sum()), 0)

    def test_boundary_helpers(self) -> None:
        canvas = np.zeros((6, 6), dtype=np.uint8)
        canvas[:, 3:] = 1
        boundary = mut.compute_boundary_mask(canvas)
        self.assertTrue(boundary.any())
        dilated = mut.dilate_mask(boundary, radius=1)
        self.assertGreater(int(dilated.sum()), int(boundary.sum()))
        ds = mut.downsample_nearest(np.arange(16, dtype=np.uint8).reshape(4, 4), factor=2)
        self.assertArrayClose(ds, np.array([[0, 2], [8, 10]], dtype=np.uint8))


class TestMorphologyAndVisualMetrics(AssertionsMixin, unittest.TestCase):
    def make_canvas_quadrants(self, size: int = 64) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.uint8)
        half = size // 2
        canvas[:half, half:] = 1
        canvas[half:, :half] = 2
        canvas[half:, half:] = 3
        return canvas

    def test_zone_helpers(self) -> None:
        zones = mut.anatomy_zone_masks()
        self.assertIn("center_torso", zones)
        mask = mut.combine_zone_masks(["left_shoulder", "right_shoulder"])
        self.assertTrue(mask.any())
        self.assertGreaterEqual(mut.macro_zone_count(mut.combine_zone_masks(["left_shoulder", "left_flank"])), 2)

    def test_orientation_and_macro_helpers(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], -20, (10, 10), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 0, (20, 20), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_TERRE, [], 25, (30, 30), dummy_mask, 2),
        ]
        out = mut.orientation_score(macros)
        self.assertFloatClose(out["oblique_share"], 2 / 3)
        self.assertFloatClose(out["vertical_share"], 1 / 3)
        self.assertEqual(mut.macro_angle_histogram(macros)[0], 1)
        self.assertIn(mut.pick_macro_angle([], mut.make_profile(111), random.Random(42)), mut.make_profile(111).allowed_angles)

    def test_visual_metrics_ranges(self) -> None:
        canvas = self.make_canvas_quadrants(96)
        rs = mut.compute_ratios(canvas)
        base = {
            "largest_olive_component_ratio": 0.20,
            "center_empty_ratio": 0.30,
            "mirror_similarity": 0.40,
            "central_brown_continuity": 0.20,
            "olive_multizone_share": 0.60,
            "boundary_density": 0.12,
            "macro_total_count": 18.0,
            "macro_multizone_ratio": 0.55,
            "macro_olive_visible_ratio": 0.22,
            "macro_terre_visible_ratio": 0.18,
            "macro_gris_visible_ratio": 0.14,
            "periphery_boundary_density_ratio": 1.20,
            "periphery_non_coyote_ratio": 1.15,
            "oblique_share": 0.70,
        }
        visual = mut.evaluate_visual_metrics(canvas, rs, base)
        military = mut.military_visual_discipline_score({**base, **visual})
        self.assertIn("visual_score_final", visual)
        self.assertIn("visual_military_score", military)
        self.assertGreaterEqual(visual["visual_score_final"], 0.0)
        self.assertLessEqual(visual["visual_score_final"], 1.0)


class TestCandidateAndValidation(AssertionsMixin, unittest.TestCase):
    def test_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        seed = mut.DEFAULT_BASE_SEED + 7
        fake_image = Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0))
        with patch.object(mut, "generate_one_variant", return_value=(fake_image, valid_ratios(), valid_metrics())):
            cand = mut.generate_candidate_from_seed(seed)
        self.assertCandidateLooksConsistent(cand)
        self.assertEqual(set(REQUIRED_METRIC_KEYS).issubset(set(cand.metrics.keys())), True)

    def test_variant_is_valid_accepts_and_rejects(self) -> None:
        self.assertTrue(mut.variant_is_valid(valid_ratios(), valid_metrics()))
        self.assertFalse(mut.variant_is_valid(invalid_ratios_far(), valid_metrics()))
        for key, value in iter_metric_failure_cases():
            metrics = valid_metrics()
            metrics[key] = value
            with self.subTest(metric=key, value=value):
                self.assertFalse(mut.variant_is_valid(valid_ratios(), metrics))

    def test_candidate_row_contains_expected_fields(self) -> None:
        row = mut.candidate_row(1, 2, 3, make_candidate(seed=777))
        self.assertEqual(row["index"], 1)
        self.assertEqual(row["seed"], 777)
        self.assertIn("visual_military_score", row)
        self.assertIn("angles", row)


class TestGuidedAndMLDLHelpers(AssertionsMixin, unittest.TestCase):
    def test_guided_state_and_merge(self) -> None:
        state = mut._guided_state_init()
        self.assertFalse(mut._guided_state_has_effects(state))
        analysis = mut.RejectionAnalysis(
            target_index=1,
            local_attempt=1,
            seed=1,
            reject_streak=1,
            fail_count=1,
            severity=2.0,
            failure_names=["ratio_olive"],
            notes=["olive"],
            corrections={
                "olive_scale_delta": 0.1,
                "terre_scale_delta": 0.0,
                "gris_scale_delta": 0.0,
                "center_overlap_delta": 0.05,
                "extra_macro_attempts": 40,
                "zone_boost_deltas": [0.1 for _ in mut.DENSITY_ZONES],
                "width_variation_delta": 0.02,
                "lateral_jitter_delta": 0.01,
                "tip_taper_delta": 0.0,
                "edge_break_delta": 0.03,
                "force_vertical": False,
                "avoid_vertical": True,
                "expand_angle_pool": True,
                "prefer_sequential_repair": True,
            },
        )
        merged = mut._merge_guided_generation_state(state, analysis)
        self.assertTrue(mut._guided_state_has_effects(merged))
        self.assertTrue(merged["avoid_vertical"])
        self.assertTrue(merged["expand_angle_pool"])

    def test_deep_rejection_analysis_builds_corrections(self) -> None:
        candidate = make_candidate(metrics={**valid_metrics(), "vertical_share": 0.05, "boundary_density": 0.05})
        failures = [
            {"rule": "ratio_olive"},
            {"rule": "center_empty_ratio"},
            {"rule": "periphery_non_coyote_ratio"},
            {"rule": "vertical_share"},
            {"rule": "visual_contour_break_score"},
        ]
        with patch.object(mut, "extract_rejection_failures", return_value=failures):
            analysis = mut.deep_rejection_analysis(candidate, 1, 2, reject_streak=2)
        self.assertGreater(analysis.corrections["olive_scale_delta"], 0.0)
        self.assertGreater(analysis.corrections["center_overlap_delta"], 0.0)
        self.assertTrue(analysis.corrections["prefer_sequential_repair"])

    def test_mldl_feature_and_context_helpers(self) -> None:
        if not hasattr(mut, "candidate_to_feature_vector"):
            self.skipTest("Helpers ML/DL absents")
        candidate = make_candidate()
        analysis = mut.RejectionAnalysis(
            target_index=1,
            local_attempt=1,
            seed=1,
            reject_streak=1,
            fail_count=2,
            severity=2.0,
            failure_names=["ratio_olive", "center_empty_ratio"],
            notes=[],
            corrections={},
        )
        feat = mut.candidate_to_feature_vector(candidate)
        ctx = mut.build_context_vector(candidate, analysis)
        reward = mut.candidate_reward(candidate, True)
        self.assertEqual(feat.ndim, 1)
        self.assertEqual(ctx.ndim, 1)
        self.assertGreater(reward, 0.0)

    def test_standardizer_roundtrip(self) -> None:
        if not hasattr(mut, "Standardizer"):
            self.skipTest("Standardizer absent")
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        std = mut.Standardizer(2)
        std.fit(x)
        out = std.transform(x)
        self.assertEqual(out.shape, x.shape)
        state = std.state_dict()
        std2 = mut.Standardizer(2)
        std2.load_state_dict(state)
        self.assertArrayClose(std2.transform(x), out)


class TestAsyncWrappersAndExports(TempDirMixin, AssertionsMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_candidate_wrappers(self) -> None:
        fake_candidate = make_candidate(seed=42)
        with patch.object(mut, "generate_candidate_from_seed", return_value=fake_candidate):
            cand = await mut.async_generate_candidate_from_seed(42)
        self.assertCandidateLooksConsistent(cand)
        self.assertTrue(await mut.async_validate_candidate_result(make_candidate()))

    async def test_wrap_async_attempt(self) -> None:
        loop = asyncio.get_running_loop()
        candidate = make_candidate(seed=42)
        fut = loop.create_future()
        fut.set_result((candidate, True))
        out = await mut._wrap_async_attempt(fut, 3, 42, 1.5)
        self.assertEqual(out[0], 3)
        self.assertEqual(out[1], 42)
        self.assertTrue(out[4])

    async def test_exports(self) -> None:
        candidate = make_candidate()
        out1 = mut.save_candidate_image(candidate, self.tmpdir / "x" / "img.png")
        out2 = await mut.async_save_candidate_image(candidate, self.tmpdir / "x" / "img_async.png")
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())
        row = mut.candidate_row(1, 1, 1, candidate)
        csv1 = mut.write_report([row], self.tmpdir)
        csv2 = await mut.async_write_report([row], self.tmpdir, filename="rapport_async.csv")
        self.assertTrue(csv1.exists())
        self.assertTrue(csv2.exists())
        with csv1.open("r", encoding="utf-8", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 1)


class TestOrchestratorsSync(TempDirMixin, AssertionsMixin, unittest.TestCase):
    def _unified_common_patches(self):
        return patch.multiple(
            mut,
            validate_generation_request=Mock(return_value=None),
            _run_log_preflight=Mock(return_value=None),
            compute_runtime_tuning=Mock(return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")),
            sample_process_resources=Mock(return_value=fake_snapshot()),
            TORCH_AVAILABLE=False,
        )

    def test_batch_attempt_seeds(self) -> None:
        batch = mut._batch_attempt_seeds(2, 3, 4, 1000)
        self.assertEqual(batch, [(3, 201003), (4, 201004), (5, 201005), (6, 201006)])

    def test_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=111)
        progress = Mock()
        with self._unified_common_patches(), \
             patch.object(mut, "generate_candidate_from_seed", return_value=candidate), \
             patch.object(mut, "validate_candidate_result", return_value=True):
            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                progress_callback=progress,
                enable_live_supervisor=False,
                strict_preflight=False,
                warmup_samples=0,
                candidate_pool_size=1,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 111)
        progress.assert_called_once()
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())

    def test_generate_all_retries_until_accept(self) -> None:
        candidates = [make_candidate(seed=101), make_candidate(seed=102)]
        with self._unified_common_patches(), \
             patch.object(mut, "generate_candidate_from_seed", side_effect=candidates), \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]):
            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                enable_live_supervisor=False,
                strict_preflight=False,
                warmup_samples=0,
                candidate_pool_size=1,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 102)
        self.assertEqual(rows[0]["global_attempt"], 2)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)

    def test_generate_all_parallel_path(self) -> None:
        def side_effect(seed: int, base_state: Dict[str, Any] | None = None, action_idx: int | None = None):
            return make_candidate(seed=seed)

        pool = ThreadPoolExecutor(max_workers=2)
        try:
            with patch.multiple(
                mut,
                validate_generation_request=Mock(return_value=None),
                _run_log_preflight=Mock(return_value=None),
                compute_runtime_tuning=Mock(return_value=mut.RuntimeTuning(2, 2, True, 0.9, "test")),
                sample_process_resources=Mock(return_value=fake_snapshot()),
                get_process_pool=Mock(return_value=pool),
                _generate_guided_candidate_task=Mock(side_effect=side_effect),
                validate_candidate_result=Mock(side_effect=lambda c: c.seed % 2 == 0),
                TORCH_AVAILABLE=False,
            ):
                rows = mut.generate_all(
                    target_count=1,
                    output_dir=self.tmpdir,
                    enable_live_supervisor=False,
                    strict_preflight=False,
                    warmup_samples=0,
                    candidate_pool_size=2,
                )
        finally:
            pool.shutdown(wait=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)

    def test_generate_all_applies_guided_corrections_after_reject(self) -> None:
        seen_states: List[Any] = []

        def fake_generate(seed: int, correction_state: Dict[str, Any] | None = None):
            seen_states.append(correction_state)
            return make_candidate(seed=seed)

        analysis = mut.RejectionAnalysis(
            target_index=1,
            local_attempt=1,
            seed=101,
            reject_streak=1,
            fail_count=2,
            severity=2.5,
            failure_names=["ratio_olive", "center_empty_ratio"],
            notes=["guided"],
            corrections={
                "olive_scale_delta": 0.1,
                "terre_scale_delta": 0.0,
                "gris_scale_delta": 0.0,
                "center_overlap_delta": 0.05,
                "extra_macro_attempts": 50,
                "zone_boost_deltas": [0.0 for _ in mut.DENSITY_ZONES],
                "width_variation_delta": 0.0,
                "lateral_jitter_delta": 0.0,
                "tip_taper_delta": 0.0,
                "edge_break_delta": 0.0,
                "force_vertical": False,
                "avoid_vertical": True,
                "expand_angle_pool": False,
                "prefer_sequential_repair": True,
            },
        )
        with self._unified_common_patches(), \
             patch.object(mut, "_select_action_indexes", side_effect=[[0], [0]]), \
             patch.object(mut, "generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]), \
             patch.object(mut, "deep_rejection_analysis", return_value=analysis), \
             patch.object(mut, "_runtime_log"):
            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                enable_live_supervisor=False,
                strict_preflight=False,
                warmup_samples=0,
                candidate_pool_size=1,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(seen_states), 2)
        self.assertTrue(mut._guided_state_has_effects(seen_states[0]))
        self.assertTrue(mut._guided_state_has_effects(seen_states[1]))
        self.assertNotEqual(seen_states[0], seen_states[1])
        self.assertGreater(float(seen_states[1].get("extra_macro_attempts", 0)), float(seen_states[0].get("extra_macro_attempts", 0)))
        self.assertTrue(bool(seen_states[1].get("prefer_sequential_repair", False)))


class TestOrchestratorsAsync(TempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=211)
        progress = AsyncMock()
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "generate_candidate_from_seed", return_value=candidate), \
             patch.object(mut, "validate_candidate_result", return_value=True):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                progress_callback=progress,
                enable_live_supervisor=False,
                strict_preflight=False,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 211)
        progress.assert_awaited()

    async def test_async_generate_all_stop_requested(self) -> None:
        stop = AsyncMock(side_effect=[True])
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()):
            rows = await mut.async_generate_all(
                target_count=2,
                output_dir=self.tmpdir,
                stop_requested=stop,
                enable_live_supervisor=False,
                strict_preflight=False,
            )
        self.assertEqual(rows, [])
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_main_corrected.py ==========")
    unittest.main(verbosity=2)
