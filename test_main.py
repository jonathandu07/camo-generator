# -*- coding: utf-8 -*-
"""
Suite de tests alignée sur l'API réelle de main.py.

Objectifs :
- couvrir les helpers purs, la géométrie et les métriques ;
- vérifier la validation métier ;
- tester les exports ;
- tester les orchestrateurs sync/async sans vrai ProcessPool ;
- rester rapide et déterministe.

Exécution :
    python -m unittest -v test_main.py
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import random
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from PIL import Image

import main as mut


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
        self._orig_log_cache = mut._LOG_MODULE_CACHE
        self._orig_log_attempted = mut._LOG_MODULE_ATTEMPTED
        self._orig_pool = mut._PROCESS_POOL
        self._orig_pool_workers = mut._PROCESS_POOL_WORKERS
        mut._LOG_MODULE_CACHE = None
        mut._LOG_MODULE_ATTEMPTED = False
        mut._PROCESS_POOL = None
        mut._PROCESS_POOL_WORKERS = None

    def tearDown(self) -> None:
        try:
            if mut._PROCESS_POOL is not None:
                mut.shutdown_process_pool()
        except Exception:
            pass
        mut._LOG_MODULE_CACHE = self._orig_log_cache
        mut._LOG_MODULE_ATTEMPTED = self._orig_log_attempted
        mut._PROCESS_POOL = self._orig_pool
        mut._PROCESS_POOL_WORKERS = self._orig_pool_workers
        super().tearDown()


class AssertionsMixin:
    def assertFloatClose(self, a: float, b: float, places: int = 7, msg: str | None = None) -> None:
        self.assertAlmostEqual(float(a), float(b), places=places, msg=msg)

    def assertArrayClose(self, a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> None:
        np.testing.assert_allclose(a, b, atol=atol, rtol=0)

    def assertCandidateLooksConsistent(self, candidate: mut.CandidateResult) -> None:
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


def make_candidate(
    seed: int = 123456,
    ratios: np.ndarray | None = None,
    metrics: Dict[str, float] | None = None,
) -> mut.CandidateResult:
    ratios = valid_ratios() if ratios is None else ratios
    metrics = valid_metrics() if metrics is None else metrics
    return mut.CandidateResult(
        seed=seed,
        profile=mut.make_profile(seed),
        image=Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0)),
        ratios=ratios,
        metrics=metrics,
    )


def fake_snapshot(
    *,
    machine_intensity: float = 0.94,
    available_mb: float = 8192.0,
    disk_free_mb: float = 4096.0,
) -> mut.ResourceSnapshot:
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
        fake = types.SimpleNamespace(
            run_generation_preflight=lambda **_: {"ok": False, "message": "refused"}
        )
        with patch.dict(sys.modules, {"log": fake}, clear=False):
            mut._LOG_MODULE_CACHE = None
            mut._LOG_MODULE_ATTEMPTED = False
            with self.assertRaises(RuntimeError):
                mut._run_log_preflight(strict=True, output_dir=self.tmpdir)

    def test_supervisor_feedback_returns_dict(self) -> None:
        fake = types.SimpleNamespace(
            feedback_runtime_event=lambda **kwargs: {"max_workers": 1, "reason": kwargs["event_type"]}
        )
        with patch.dict(sys.modules, {"log": fake}, clear=False):
            mut._LOG_MODULE_CACHE = None
            mut._LOG_MODULE_ATTEMPTED = False
            out = mut._supervisor_feedback("resource_snapshot", value=1)
        self.assertEqual(out, {"max_workers": 1, "reason": "resource_snapshot"})

    def test_merge_supervisor_tuning_applies_advice(self) -> None:
        tuning = mut.RuntimeTuning(4, 4, True, 0.9)
        out = mut._merge_supervisor_tuning(
            tuning,
            {"max_workers": 1, "attempt_batch_size": 1, "parallel_attempts": True, "machine_intensity": 0.3, "reason": "throttle"},
        )
        self.assertEqual(out.max_workers, 1)
        self.assertEqual(out.attempt_batch_size, 1)
        self.assertFalse(out.parallel_attempts)
        self.assertEqual(out.machine_intensity, 0.3)
        self.assertEqual(out.reason, "throttle")


class TestSystemHelpers(GlobalStateMixin, TempDirMixin, AssertionsMixin, unittest.TestCase):
    def test_worker_initializer_can_limit_numeric_threads(self) -> None:
        with patch.dict(os.environ, {"CAMO_LIMIT_NUMERIC_THREADS": "1"}, clear=True):
            mut._worker_initializer()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "1")

    def test_worker_initializer_can_leave_environment_unmodified(self) -> None:
        with patch.dict(os.environ, {"CAMO_LIMIT_NUMERIC_THREADS": "0"}, clear=True):
            mut._worker_initializer()
            self.assertNotIn("OMP_NUM_THREADS", os.environ)

    def test_shutdown_process_pool_calls_shutdown(self) -> None:
        pool = Mock()
        mut._PROCESS_POOL = pool
        mut._PROCESS_POOL_WORKERS = 4
        mut.shutdown_process_pool()
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        self.assertIsNone(mut._PROCESS_POOL)
        self.assertIsNone(mut._PROCESS_POOL_WORKERS)

    def test_get_process_pool_creates_and_reuses_pool(self) -> None:
        fake_pool = Mock()
        with patch.object(mut, "ProcessPoolExecutor", return_value=fake_pool) as mock_exec:
            p1 = mut.get_process_pool(3)
            p2 = mut.get_process_pool(3)
        self.assertIs(p1, fake_pool)
        self.assertIs(p2, fake_pool)
        mock_exec.assert_called_once()
        self.assertEqual(mut._PROCESS_POOL_WORKERS, 3)

    def test_get_process_pool_recreates_when_worker_count_changes(self) -> None:
        first = Mock(name="first_pool")
        second = Mock(name="second_pool")
        with patch.object(mut, "ProcessPoolExecutor", side_effect=[first, second]) as mock_exec:
            p1 = mut.get_process_pool(2)
            p2 = mut.get_process_pool(4)
        self.assertIs(p1, first)
        self.assertIs(p2, second)
        self.assertEqual(mock_exec.call_count, 2)
        first.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_clip_safe_and_clamp_helpers(self) -> None:
        self.assertEqual(mut._clip_float(5.0, 0.0, 3.0), 3.0)
        self.assertEqual(mut._clip_float(-1.0, 0.0, 3.0), 0.0)
        self.assertEqual(mut._safe_float("2.5"), 2.5)
        self.assertEqual(mut._safe_float("bad", default=7.0), 7.0)
        self.assertEqual(mut._safe_float(float("nan"), default=8.0), 8.0)
        self.assertEqual(mut.clamp01(-1.0), 0.0)
        self.assertEqual(mut.clamp01(2.0), 1.0)

    def test_sample_process_resources_without_psutil(self) -> None:
        with patch.object(mut, "psutil", None):
            snap = mut.sample_process_resources(output_dir=self.tmpdir)
        self.assertIsInstance(snap, mut.ResourceSnapshot)
        self.assertEqual(snap.process_cpu_percent, 0.0)
        self.assertGreater(snap.disk_free_mb, 0.0)

    def test_compute_runtime_tuning_uses_memory_thresholds(self) -> None:
        low_mem = fake_snapshot(available_mb=900.0)
        out = mut.compute_runtime_tuning(machine_intensity=0.9, sample=low_mem)
        self.assertEqual(out.max_workers, 1)
        self.assertEqual(out.attempt_batch_size, 1)

    def test_validate_generation_request_happy_path(self) -> None:
        with patch.object(mut, "sample_process_resources", return_value=fake_snapshot(disk_free_mb=1024.0)):
            mut.validate_generation_request(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=1,
                machine_intensity=0.5,
                max_workers=1,
                attempt_batch_size=1,
            )
        self.assertFalse((self.tmpdir / ".write_probe.tmp").exists())

    def test_validate_generation_request_rejects_bad_target_count(self) -> None:
        with self.assertRaises(ValueError):
            mut.validate_generation_request(
                target_count=0,
                output_dir=self.tmpdir,
                base_seed=1,
                machine_intensity=0.5,
                max_workers=1,
                attempt_batch_size=1,
            )

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
    def test_ensure_output_dir_creates_directory(self) -> None:
        path = self.tmpdir / "a" / "b" / "c"
        out = mut.ensure_output_dir(path)
        self.assertEqual(out, path)
        self.assertTrue(path.exists())

    def test_build_seed_is_deterministic(self) -> None:
        self.assertEqual(mut.build_seed(3, 7, 1000), mut.build_seed(3, 7, 1000))
        self.assertNotEqual(mut.build_seed(3, 7, 1000), mut.build_seed(4, 7, 1000))

    def test_make_profile_is_deterministic_for_same_seed(self) -> None:
        p1 = mut.make_profile(424242)
        p2 = mut.make_profile(424242)
        self.assertEqual(p1.seed, p2.seed)
        self.assertEqual(p1.allowed_angles, p2.allowed_angles)
        self.assertEqual(p1.angle_pool, p2.angle_pool)
        self.assertEqual(p1.zone_weight_boosts, p2.zone_weight_boosts)
        self.assertIn(0, p1.allowed_angles)
        self.assertEqual(sorted(set(p1.allowed_angles)), p1.allowed_angles)

    def test_cm_to_px_returns_at_least_one(self) -> None:
        self.assertEqual(mut.cm_to_px(0.0), 1)
        self.assertGreaterEqual(mut.cm_to_px(0.01), 1)
        self.assertEqual(mut.cm_to_px(10.0), int(round(10.0 * mut.PX_PER_CM)))

    def test_compute_ratios_sums_to_one(self) -> None:
        canvas = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        ratios = mut.compute_ratios(canvas)
        self.assertFloatClose(float(np.sum(ratios)), 1.0)
        self.assertArrayClose(ratios, np.array([0.25, 0.25, 0.25, 0.25]))

    def test_render_canvas_returns_pil_image(self) -> None:
        canvas = np.zeros((5, 7), dtype=np.uint8)
        img = mut.render_canvas(canvas)
        self.assertEqual(img.size, (7, 5))
        self.assertIsInstance(img, Image.Image)

    def test_rotate_90_deg(self) -> None:
        x, y = mut.rotate(1.0, 0.0, 90.0)
        self.assertFloatClose(x, 0.0, places=6)
        self.assertFloatClose(y, 1.0, places=6)

    def test_choose_biased_center_is_in_bounds(self) -> None:
        rng = random.Random(123)
        for _ in range(20):
            x, y = mut.choose_biased_center(rng)
            self.assertGreaterEqual(x, 60)
            self.assertLessEqual(x, mut.WIDTH - 60)
            self.assertGreaterEqual(y, 60)
            self.assertLessEqual(y, mut.HEIGHT - 60)

    def test_polygon_mask_non_empty(self) -> None:
        poly = [(100, 100), (120, 100), (120, 130), (100, 130)]
        mask = mut.polygon_mask(poly)
        self.assertEqual(mask.shape, (mut.HEIGHT, mut.WIDTH))
        self.assertGreater(int(mask.sum()), 0)

    def test_compute_boundary_mask_detects_changes(self) -> None:
        canvas = np.zeros((6, 6), dtype=np.uint8)
        canvas[:, 3:] = 1
        boundary = mut.compute_boundary_mask(canvas)
        self.assertTrue(boundary.any())

    def test_dilate_mask_expands_true_area(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        dilated = mut.dilate_mask(mask, radius=1)
        self.assertGreater(int(dilated.sum()), int(mask.sum()))

    def test_downsample_nearest(self) -> None:
        canvas = np.arange(16, dtype=np.uint8).reshape(4, 4)
        ds = mut.downsample_nearest(canvas, factor=2)
        self.assertArrayClose(ds, np.array([[0, 2], [8, 10]], dtype=np.uint8))

    def test_boundary_density(self) -> None:
        canvas = np.zeros((10, 10), dtype=np.uint8)
        d0 = mut.boundary_density(canvas)
        canvas[:, 5:] = 1
        d1 = mut.boundary_density(canvas)
        self.assertEqual(d0, 0.0)
        self.assertGreater(d1, 0.0)

    def test_mirror_similarity_score(self) -> None:
        canvas = np.array([[0, 1, 1, 0], [2, 3, 3, 2]], dtype=np.uint8)
        score = mut.mirror_similarity_score(canvas)
        self.assertFloatClose(score, 1.0)


class TestMorphologyAndMacros(AssertionsMixin, unittest.TestCase):
    def test_rect_mask_shape_and_population(self) -> None:
        mask = mut.rect_mask(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(mask.shape, (mut.HEIGHT, mut.WIDTH))
        self.assertTrue(mask.any())

    def test_anatomy_zone_masks_contains_expected_keys(self) -> None:
        zones = mut.anatomy_zone_masks()
        self.assertEqual(
            set(zones.keys()),
            {
                "left_shoulder", "right_shoulder", "left_flank", "right_flank",
                "left_thigh", "right_thigh", "center_torso",
            },
        )

    def test_combine_zone_masks(self) -> None:
        mask = mut.combine_zone_masks(["left_shoulder", "right_shoulder"])
        self.assertTrue(mask.any())
        self.assertGreater(int(mask.sum()), 1000)

    def test_macro_zone_count_and_overlap_ratio(self) -> None:
        mask = mut.combine_zone_masks(["left_shoulder", "left_flank"])
        self.assertGreaterEqual(mut.macro_zone_count(mask), 2)
        self.assertFloatClose(mut.zone_overlap_ratio(mut.ANATOMY_ZONES["center_torso"], mut.ANATOMY_ZONES["center_torso"]), 1.0)

    def test_center_empty_ratio(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        self.assertEqual(mut.center_empty_ratio(canvas), 1.0)

    def test_largest_component_ratio(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[3:6, 3:6] = True
        self.assertFloatClose(mut.largest_component_ratio(mask), 9 / 13)

    def test_orientation_score_and_histogram(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], -20, (10, 10), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 0, (20, 20), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_TERRE, [], 25, (30, 30), dummy_mask, 2),
        ]
        out = mut.orientation_score(macros)
        hist = mut.macro_angle_histogram(macros)
        self.assertFloatClose(out["oblique_share"], 2 / 3)
        self.assertFloatClose(out["vertical_share"], 1 / 3)
        self.assertEqual(hist[0], 1)
        self.assertEqual(hist[25], 1)

    def test_pick_macro_angle_returns_allowed_angle(self) -> None:
        profile = mut.make_profile(111)
        angle = mut.pick_macro_angle([], profile, random.Random(42))
        self.assertIn(angle, profile.allowed_angles)

    def test_jagged_spine_poly_returns_points(self) -> None:
        rng = random.Random(42)
        poly = mut.jagged_spine_poly(
            rng=rng,
            cx=100,
            cy=100,
            length_px=80,
            width_px=30,
            angle_from_vertical_deg=20,
            segments=8,
            width_variation=0.2,
            lateral_jitter=0.15,
            tip_taper=0.4,
            edge_break=0.12,
        )
        self.assertEqual(len(poly), 16)

    def test_apply_mask_changes_canvas_and_origin(self) -> None:
        canvas = np.zeros((20, 20), dtype=np.uint8)
        origin_map = np.zeros((20, 20), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 6:11] = True
        mut.apply_mask(canvas, origin_map, mask, mut.IDX_OLIVE)
        self.assertTrue(np.all(canvas[mask] == mut.IDX_OLIVE))
        self.assertTrue(np.all(origin_map[mask] == mut.ORIGIN_MACRO))

    def test_macro_candidate_diagnostics_empty(self) -> None:
        diag = mut.macro_candidate_diagnostics(np.zeros((mut.HEIGHT, mut.WIDTH), dtype=bool))
        self.assertEqual(diag["zone_count"], 0.0)
        self.assertEqual(diag["edge_lock_ratio"], 1.0)

    def test_macro_candidate_is_valid_rejects_empty_and_oversized(self) -> None:
        canvas = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=np.uint8)
        self.assertFalse(mut.macro_candidate_is_valid(np.zeros_like(canvas, dtype=bool), mut.IDX_OLIVE, 0, canvas, []))
        oversized = mut.combine_zone_masks(["left_shoulder", "left_flank", "left_thigh"])
        self.assertFalse(mut.macro_candidate_is_valid(oversized, mut.IDX_OLIVE, 20, canvas, []))

    def test_try_place_validated_macro_can_place_real_macro(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        origin_map = np.full((mut.HEIGHT, mut.WIDTH), mut.ORIGIN_BACKGROUND, dtype=np.uint8)
        macros: List[mut.MacroRecord] = []
        profile = mut.make_profile(123)
        rng = random.Random(123)

        placed = False
        for _ in range(60):
            if mut.try_place_validated_macro(canvas, origin_map, macros, mut.IDX_OLIVE, profile, rng, long_mode=True):
                placed = True
                break

        self.assertTrue(placed)
        self.assertEqual(len(macros), 1)
        record = macros[0]
        self.assertTrue(mut.macro_candidate_is_valid(record.mask, record.color_idx, record.angle_deg, np.full_like(canvas, mut.IDX_COYOTE), []))

    def test_zone_center_angle_and_nearest_allowed_angle(self) -> None:
        p1 = mut.zone_center("left_shoulder")
        p2 = mut.zone_center("left_flank")
        angle = mut.angle_from_points_vertical(p1, p2)
        nearest = mut.nearest_allowed_angle(angle, mut.BASE_ANGLES)
        self.assertIsInstance(angle, float)
        self.assertIn(nearest, mut.BASE_ANGLES)

    def test_macro_counter_helpers(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (0, 0), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 25, (0, 0), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_TERRE, [], 0, (0, 0), dummy_mask, 1),
        ]
        self.assertEqual(mut.macro_color_count(macros, mut.IDX_OLIVE), 2)
        counts = mut.macro_counts(macros)
        self.assertEqual(counts[mut.IDX_OLIVE], 2)
        self.assertEqual(counts[mut.IDX_TERRE], 1)
        self.assertEqual(counts[mut.IDX_GRIS], 0)

    def test_macro_metrics_and_visible_pixels(self) -> None:
        canvas = np.zeros((4, 4), dtype=np.uint8)
        origin = np.zeros((4, 4), dtype=np.uint8)
        canvas[0:2, 0:2] = mut.IDX_OLIVE
        origin[0:2, 0:2] = mut.ORIGIN_MACRO
        mask = np.zeros((4, 4), dtype=bool)
        mask[0:2, 0:2] = True
        macros = [mut.MacroRecord(mut.IDX_OLIVE, [], 20, (1, 1), mask, 2)]
        self.assertEqual(mut.macro_visible_pixels(canvas, origin, mut.IDX_OLIVE), 4)
        out = mut.macro_system_metrics(macros, canvas, origin)
        self.assertEqual(out["macro_total_count"], 1.0)
        self.assertEqual(out["macro_olive_count"], 1.0)
        self.assertGreater(out["macro_visible_total_ratio"], 0.0)


class TestVisualMetrics(AssertionsMixin, unittest.TestCase):
    def make_canvas_quadrants(self, size: int = 64) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.uint8)
        half = size // 2
        canvas[:half, half:] = 1
        canvas[half:, :half] = 2
        canvas[half:, half:] = 3
        return canvas

    def test_absolute_origin_color_ratios_returns_macro_ratios_only(self) -> None:
        canvas = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        origin = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        out = mut.absolute_origin_color_ratios(canvas, origin)
        self.assertEqual(set(out.keys()), {"macro_olive_visible_ratio", "macro_terre_visible_ratio", "macro_gris_visible_ratio"})
        self.assertFloatClose(out["macro_olive_visible_ratio"], 0.25)
        self.assertFloatClose(out["macro_terre_visible_ratio"], 0.25)
        self.assertFloatClose(out["macro_gris_visible_ratio"], 0.25)

    def test_spatial_discipline_metrics_periphery_heavier_than_center(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        canvas[mut.HIGH_DENSITY_ZONE_MASK] = mut.IDX_OLIVE
        out = mut.spatial_discipline_metrics(canvas)
        self.assertGreater(out["periphery_non_coyote_ratio"], 1.0)

    def test_central_brown_continuity(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        score = mut.central_brown_continuity(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_multiscale_metrics(self) -> None:
        canvas = self.make_canvas_quadrants(64)
        out = mut.multiscale_metrics(canvas)
        self.assertEqual(set(out.keys()), {
            "boundary_density_small", "boundary_density_tiny", "center_empty_ratio_small", "largest_olive_component_ratio_small",
        })

    def test_build_silhouette_mask_and_boundary(self) -> None:
        mask = mut.build_silhouette_mask(64, 96)
        boundary = mut.silhouette_boundary(mask)
        self.assertEqual(mask.shape, (96, 64))
        self.assertTrue(mask.any())
        self.assertTrue(boundary.any())
        self.assertTrue(np.all(boundary <= mask))

    def test_silhouette_color_diversity_score_range(self) -> None:
        canvas = self.make_canvas_quadrants(96)
        score = mut.silhouette_color_diversity_score(canvas)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_contour_break_score_range(self) -> None:
        canvas = self.make_canvas_quadrants(96)
        score, entropy = mut.contour_break_score(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)

    def test_small_scale_structural_score_range(self) -> None:
        canvas = self.make_canvas_quadrants(96)
        score = mut.small_scale_structural_score(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ratio_score_and_main_metrics_score(self) -> None:
        self.assertGreater(mut.ratio_score(valid_ratios()), 0.9)
        self.assertGreater(mut.main_metrics_score(valid_metrics()), 0.0)

    def test_evaluate_visual_metrics_and_military_score(self) -> None:
        canvas = self.make_canvas_quadrants(96)
        rs = mut.compute_ratios(canvas)
        base_metrics = {
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
        visual = mut.evaluate_visual_metrics(canvas, rs, base_metrics)
        military = mut.military_visual_discipline_score({**base_metrics, **visual})
        self.assertIn("visual_score_final", visual)
        self.assertIn("visual_military_score", military)
        self.assertGreaterEqual(visual["visual_score_final"], 0.0)
        self.assertLessEqual(visual["visual_score_final"], 1.0)


class TestCandidateAndValidation(AssertionsMixin, unittest.TestCase):
    def test_generate_one_variant_structure_with_patched_pipeline(self) -> None:
        profile = mut.make_profile(12345)
        with patch.object(mut, "add_forced_structural_macros", return_value=None), \
             patch.object(mut, "add_macros", return_value=None), \
             patch.object(mut, "enforce_macro_population", return_value=None), \
             patch.object(mut, "enforce_macro_angle_discipline", return_value=None), \
             patch.object(mut, "repair_center_and_periphery", return_value=None):
            img, ratios, metrics = mut.generate_one_variant(profile)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (mut.WIDTH, mut.HEIGHT))
        self.assertEqual(ratios.shape, (4,))
        self.assertIn("visual_score_final", metrics)
        self.assertIn("macro_total_count", metrics)

    def test_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        seed = mut.DEFAULT_BASE_SEED + 7
        fake_image = Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0))
        fake_ratios = valid_ratios()
        fake_metrics = valid_metrics()
        with patch.object(mut, "generate_one_variant", return_value=(fake_image, fake_ratios, fake_metrics)) as mock_generate:
            cand = mut.generate_candidate_from_seed(seed)
        self.assertCandidateLooksConsistent(cand)
        self.assertEqual(cand.seed, seed)
        self.assertEqual(cand.profile.allowed_angles, mut.make_profile(seed).allowed_angles)
        mock_generate.assert_called_once()

    def test_generate_and_validate_from_seed_wrapper(self) -> None:
        candidate = make_candidate(seed=999)
        with patch.object(mut, "generate_candidate_from_seed", return_value=candidate) as mock_gen, \
             patch.object(mut, "validate_candidate_result", return_value=True) as mock_val:
            out_candidate, accepted = mut.generate_and_validate_from_seed(999)
        self.assertIs(out_candidate, candidate)
        self.assertTrue(accepted)
        mock_gen.assert_called_once_with(999)
        mock_val.assert_called_once_with(candidate)

    def test_variant_is_valid_accepts_valid_candidate(self) -> None:
        self.assertTrue(mut.variant_is_valid(valid_ratios(), valid_metrics()))

    def test_variant_is_valid_rejects_bad_ratios(self) -> None:
        self.assertFalse(mut.variant_is_valid(invalid_ratios_far(), valid_metrics()))

    def test_variant_is_valid_rejects_each_metric_failure(self) -> None:
        for key, value in iter_metric_failure_cases():
            metrics = valid_metrics()
            metrics[key] = value
            with self.subTest(metric=key, value=value):
                self.assertFalse(mut.variant_is_valid(valid_ratios(), metrics))

    def test_validate_candidate_result_wrapper(self) -> None:
        self.assertTrue(mut.validate_candidate_result(make_candidate()))

    def test_candidate_row_contains_expected_fields(self) -> None:
        row = mut.candidate_row(1, 2, 3, make_candidate(seed=777))
        self.assertEqual(row["index"], 1)
        self.assertEqual(row["seed"], 777)
        self.assertEqual(row["attempts_for_this_image"], 2)
        self.assertEqual(row["global_attempt"], 3)
        self.assertIn("coyote_brown_pct", row)
        self.assertIn("visual_military_score", row)
        self.assertIn("angles", row)


class TestGuidedRejectionAnalysis(AssertionsMixin, unittest.TestCase):
    def test_rejection_analysis_dataclass(self) -> None:
        analysis = mut.RejectionAnalysis(
            target_index=1,
            local_attempt=2,
            seed=3,
            reject_streak=1,
            fail_count=2,
            severity=3.5,
            failure_names=["ratio_olive", "center_empty_ratio"],
            notes=["note"],
            corrections={"olive_scale_delta": 0.1},
        )
        self.assertEqual(analysis.target_index, 1)
        self.assertEqual(analysis.fail_count, 2)
        self.assertIn("ratio_olive", analysis.failure_names)

    def test_guided_state_init_and_has_effects(self) -> None:
        state = mut._guided_state_init()
        self.assertFalse(mut._guided_state_has_effects(state))
        state["olive_scale_delta"] = 0.1
        self.assertTrue(mut._guided_state_has_effects(state))

    def test_extract_rejection_failures_uses_log_analyzer(self) -> None:
        class FakeFail:
            def __init__(self, rule: str):
                self.rule = rule

            def to_dict(self) -> Dict[str, Any]:
                return {"rule": self.rule, "delta": 0.1}

        fake_diag = types.SimpleNamespace(failures=[FakeFail("ratio_olive"), {"rule": "center_empty_ratio"}])
        fake_log = types.SimpleNamespace(analyze_candidate=lambda candidate, target_index, local_attempt: fake_diag)
        with patch.object(mut, "_get_log_module", return_value=fake_log):
            out = mut.extract_rejection_failures(make_candidate(), 1, 2)
        self.assertEqual([x["rule"] for x in out], ["ratio_olive", "center_empty_ratio"])

    def test_deep_rejection_analysis_builds_corrections(self) -> None:
        candidate = make_candidate(metrics={
            **valid_metrics(),
            "vertical_share": 0.05,
            "boundary_density": 0.05,
        })
        failures = [
            {"rule": "ratio_olive"},
            {"rule": "center_empty_ratio"},
            {"rule": "periphery_non_coyote_ratio"},
            {"rule": "vertical_share"},
            {"rule": "visual_contour_break_score"},
        ]
        with patch.object(mut, "extract_rejection_failures", return_value=failures):
            analysis = mut.deep_rejection_analysis(candidate, 1, 2, reject_streak=2)
        self.assertGreater(analysis.fail_count, 0)
        self.assertGreater(analysis.corrections["olive_scale_delta"], 0.0)
        self.assertGreater(analysis.corrections["center_overlap_delta"], 0.0)
        self.assertTrue(analysis.corrections["prefer_sequential_repair"])
        self.assertTrue(analysis.notes)

    def test_merge_guided_generation_state_accumulates(self) -> None:
        state = mut._guided_state_init()
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
        self.assertEqual(merged["reject_streak"], 1)
        self.assertGreater(merged["olive_scale_delta"], 0.0)
        self.assertEqual(merged["extra_macro_attempts"], 40)
        self.assertTrue(merged["avoid_vertical"])
        self.assertTrue(merged["expand_angle_pool"])
        self.assertTrue(mut._guided_state_has_effects(merged))

    def test_apply_guided_generation_state_changes_profile(self) -> None:
        profile = mut.make_profile(123)
        guided = {
            "olive_scale_delta": 0.1,
            "terre_scale_delta": 0.05,
            "gris_scale_delta": 0.02,
            "center_overlap_delta": 0.08,
            "extra_macro_attempts": 60,
            "zone_boost_deltas": [0.1 for _ in mut.DENSITY_ZONES],
            "width_variation_delta": 0.03,
            "lateral_jitter_delta": 0.02,
            "tip_taper_delta": 0.01,
            "edge_break_delta": 0.02,
            "force_vertical": True,
            "avoid_vertical": False,
            "expand_angle_pool": True,
            "prefer_sequential_repair": True,
        }
        out = mut._apply_guided_generation_state(profile, guided)
        self.assertGreater(out.olive_macro_target_scale, 1.0)
        self.assertGreater(out.terre_macro_target_scale, 1.0)
        self.assertGreater(out.extra_macro_attempts, 0)
        self.assertIn(0, out.allowed_angles)
        self.assertEqual(set(out.allowed_angles), set(mut.BASE_ANGLES))

    def test_generate_candidate_from_seed_applies_correction_state(self) -> None:
        seed = 123456
        fake_image = Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0))
        fake_ratios = valid_ratios()
        fake_metrics = valid_metrics()
        seen: List[mut.VariantProfile] = []

        def fake_generate_one_variant(profile: mut.VariantProfile):
            seen.append(profile)
            return fake_image, fake_ratios, fake_metrics

        guided = {
            "olive_scale_delta": 0.15,
            "terre_scale_delta": 0.0,
            "gris_scale_delta": 0.0,
            "center_overlap_delta": 0.05,
            "extra_macro_attempts": 50,
            "zone_boost_deltas": [0.0 for _ in mut.DENSITY_ZONES],
            "width_variation_delta": 0.01,
            "lateral_jitter_delta": 0.0,
            "tip_taper_delta": 0.0,
            "edge_break_delta": 0.0,
            "force_vertical": False,
            "avoid_vertical": True,
            "expand_angle_pool": False,
            "prefer_sequential_repair": True,
        }
        with patch.object(mut, "generate_one_variant", side_effect=fake_generate_one_variant):
            cand = mut.generate_candidate_from_seed(seed, correction_state=guided)
        self.assertCandidateLooksConsistent(cand)
        self.assertGreater(seen[0].olive_macro_target_scale, 1.0)
        self.assertNotIn(0, seen[0].allowed_angles)


class TestAsyncCandidateWrappers(AssertionsMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        fake_candidate = make_candidate(seed=mut.DEFAULT_BASE_SEED + 1)
        with patch.object(mut, "generate_candidate_from_seed", return_value=fake_candidate) as mock_generate:
            cand = await mut.async_generate_candidate_from_seed(mut.DEFAULT_BASE_SEED + 1)
        self.assertCandidateLooksConsistent(cand)
        mock_generate.assert_called_once()

    async def test_async_validate_candidate_result(self) -> None:
        self.assertTrue(await mut.async_validate_candidate_result(make_candidate()))

    async def test_wrap_async_attempt(self) -> None:
        loop = asyncio.get_running_loop()
        candidate = make_candidate(seed=42)
        fut = loop.create_future()
        fut.set_result((candidate, True))
        out = await mut._wrap_async_attempt(fut, 3, 42, 1.5)
        self.assertEqual(out[0], 3)
        self.assertEqual(out[1], 42)
        self.assertEqual(out[2], 1.5)
        self.assertIs(out[3], candidate)
        self.assertTrue(out[4])


class TestExports(TempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_save_and_async_save_candidate_image(self) -> None:
        candidate = make_candidate()
        out1 = mut.save_candidate_image(candidate, self.tmpdir / "x" / "img.png")
        out2 = await mut.async_save_candidate_image(candidate, self.tmpdir / "x" / "img_async.png")
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())

    async def test_write_report_and_async_write_report(self) -> None:
        row = mut.candidate_row(1, 1, 1, make_candidate())
        out1 = mut.write_report([row], self.tmpdir)
        out2 = await mut.async_write_report([row], self.tmpdir, filename="rapport_async.csv")
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())
        with out1.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    async def test_write_report_empty(self) -> None:
        out = mut.write_report([], self.tmpdir)
        out_async = await mut.async_write_report([], self.tmpdir, filename="empty_async.csv")
        self.assertEqual(out.read_text(encoding="utf-8"), "")
        self.assertEqual(out_async.read_text(encoding="utf-8"), "")


class TestOrchestratorsSync(TempDirMixin, AssertionsMixin, unittest.TestCase):
    def test_batch_attempt_seeds(self) -> None:
        batch = mut._batch_attempt_seeds(2, 3, 4, 1000)
        self.assertEqual(batch, [(3, 201003), (4, 201004), (5, 201005), (6, 201006)])

    def test_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=111)
        progress = Mock()

        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "generate_candidate_from_seed", return_value=candidate), \
             patch.object(mut, "validate_candidate_result", return_value=True):
            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                progress_callback=progress,
                enable_live_supervisor=False,
                strict_preflight=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())
        progress.assert_called_once()
        self.assertEqual(rows[0]["seed"], 111)

    def test_generate_all_retries_until_accept(self) -> None:
        candidates = [make_candidate(seed=101), make_candidate(seed=102)]
        accepted = [False, True]

        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "generate_candidate_from_seed", side_effect=candidates), \
             patch.object(mut, "validate_candidate_result", side_effect=accepted):
            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                enable_live_supervisor=False,
                strict_preflight=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 102)
        self.assertEqual(rows[0]["global_attempt"], 2)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)

    def test_generate_all_parallel_path(self) -> None:
        def side_effect(seed: int) -> tuple[mut.CandidateResult, bool]:
            candidate = make_candidate(seed=seed)
            return candidate, seed % 2 == 0

        pool = ThreadPoolExecutor(max_workers=2)
        try:
            progress = Mock()
            with patch.object(mut, "validate_generation_request", return_value=None), \
                 patch.object(mut, "_run_log_preflight", return_value=None), \
                 patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(2, 2, True, 0.9, "test")), \
                 patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
                 patch.object(mut, "get_process_pool", return_value=pool), \
                 patch.object(mut, "generate_and_validate_from_seed", side_effect=side_effect):
                rows = mut.generate_all(
                    target_count=1,
                    output_dir=self.tmpdir,
                    progress_callback=progress,
                    enable_live_supervisor=False,
                    strict_preflight=False,
                )
        finally:
            pool.shutdown(wait=True)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)
        self.assertEqual(progress.call_count, 2)

    def test_generate_all_applies_guided_corrections_after_reject(self) -> None:
        seen_states: List[Any] = []

        def fake_generate(seed: int, correction_state: Dict[str, Any] | None = None) -> mut.CandidateResult:
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

        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]), \
             patch.object(mut, "deep_rejection_analysis", return_value=analysis), \
             patch.object(mut, "_runtime_log"):
            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                enable_live_supervisor=False,
                strict_preflight=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(seen_states), 2)
        self.assertFalse(mut._guided_state_has_effects(seen_states[0]))
        self.assertTrue(mut._guided_state_has_effects(seen_states[1]))

    def test_generate_all_stop_requested_writes_partial_report(self) -> None:
        stop = Mock(side_effect=[True])
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()):
            rows = mut.generate_all(
                target_count=2,
                output_dir=self.tmpdir,
                stop_requested=stop,
                enable_live_supervisor=False,
                strict_preflight=False,
            )
        self.assertEqual(rows, [])
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())


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
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    async def test_async_generate_all_parallel_path(self) -> None:
        def side_effect(seed: int) -> tuple[mut.CandidateResult, bool]:
            return make_candidate(seed=seed), seed % 2 == 0

        pool = ThreadPoolExecutor(max_workers=2)
        try:
            progress = AsyncMock()
            with patch.object(mut, "validate_generation_request", return_value=None), \
                 patch.object(mut, "_run_log_preflight", return_value=None), \
                 patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(2, 2, True, 0.9, "test")), \
                 patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
                 patch.object(mut, "get_process_pool", return_value=pool), \
                 patch.object(mut, "generate_and_validate_from_seed", side_effect=side_effect):
                rows = await mut.async_generate_all(
                    target_count=1,
                    output_dir=self.tmpdir,
                    progress_callback=progress,
                    enable_live_supervisor=False,
                    strict_preflight=False,
                )
        finally:
            pool.shutdown(wait=True)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)
        self.assertEqual(progress.await_count, 2)

    async def test_async_generate_all_applies_guided_corrections_after_reject(self) -> None:
        seen_states: List[Any] = []

        def fake_generate(seed: int, correction_state: Dict[str, Any] | None = None) -> mut.CandidateResult:
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

        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]), \
             patch.object(mut, "deep_rejection_analysis", return_value=analysis), \
             patch.object(mut, "_runtime_log"):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                enable_live_supervisor=False,
                strict_preflight=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(seen_states), 2)
        self.assertFalse(mut._guided_state_has_effects(seen_states[0]))
        self.assertTrue(mut._guided_state_has_effects(seen_states[1]))

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
    LOGGER.info("========== DÉBUT DES TESTS test_main.py ==========")
    unittest.main(verbosity=2)
