# -*- coding: utf-8 -*-
"""
test_main.py
Suite de tests robuste et rapide pour main.py.

Objectifs :
- couvrir les helpers purs et la géométrie ;
- couvrir les diagnostics macro et l'adaptation des rejets ;
- couvrir la validation métier ;
- couvrir les exports ;
- couvrir les orchestrateurs sync/async ;
- couvrir les chemins séquentiels et parallèles sans lancer de vrai ProcessPool.

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
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
from PIL import Image

import main as mut


# ============================================================
# LOGGING
# ============================================================

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

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


LOGGER = configure_logger()


# ============================================================
# HELPERS / MIXINS
# ============================================================

class TempDirMixin:
    def setUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_")
        self.tmpdir = Path(self._tmpdir_obj.name)

    def tearDown(self) -> None:
        self._tmpdir_obj.cleanup()


class AsyncTempDirMixin:
    async def asyncSetUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_async_")
        self.tmpdir = Path(self._tmpdir_obj.name)

    async def asyncTearDown(self) -> None:
        self._tmpdir_obj.cleanup()


class GlobalStateMixin:
    def setUp(self) -> None:
        self._orig_overrides = dict(mut._ADAPTIVE_PROFILE_OVERRIDES)
        self._orig_log_cache = mut._LOG_MODULE_CACHE
        self._orig_log_attempted = mut._LOG_MODULE_ATTEMPTED
        self._orig_pool = mut._PROCESS_POOL
        self._orig_pool_workers = mut._PROCESS_POOL_WORKERS
        mut._ADAPTIVE_PROFILE_OVERRIDES.clear()
        mut._LOG_MODULE_CACHE = None
        mut._LOG_MODULE_ATTEMPTED = False
        mut._PROCESS_POOL = None
        mut._PROCESS_POOL_WORKERS = None

    def tearDown(self) -> None:
        if mut._PROCESS_POOL is not None:
            try:
                mut.shutdown_process_pool()
            except Exception:
                pass
        mut._ADAPTIVE_PROFILE_OVERRIDES.clear()
        mut._ADAPTIVE_PROFILE_OVERRIDES.update(self._orig_overrides)
        mut._LOG_MODULE_CACHE = self._orig_log_cache
        mut._LOG_MODULE_ATTEMPTED = self._orig_log_attempted
        mut._PROCESS_POOL = self._orig_pool
        mut._PROCESS_POOL_WORKERS = self._orig_pool_workers


class TestAssertionsMixin:
    def assertFloatClose(self, a: float, b: float, places: int = 7, msg: str | None = None) -> None:
        self.assertAlmostEqual(float(a), float(b), places=places, msg=msg)

    def assertArrayClose(self, a: np.ndarray, b: np.ndarray, atol: float = 1e-8, msg: str | None = None) -> None:
        try:
            np.testing.assert_allclose(a, b, atol=atol, rtol=0)
        except AssertionError as exc:
            self.fail(msg or str(exc))

    def assertImageSize(self, img: Image.Image, size: tuple[int, int]) -> None:
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, size)

    def assertCandidateLooksConsistent(self, candidate: mut.CandidateResult) -> None:
        self.assertIsInstance(candidate, mut.CandidateResult)
        self.assertIsInstance(candidate.profile, mut.VariantProfile)
        self.assertIsInstance(candidate.image, Image.Image)
        self.assertEqual(candidate.ratios.shape, (4,))
        self.assertAlmostEqual(float(np.sum(candidate.ratios)), 1.0, places=5)
        self.assertIsInstance(candidate.metrics, dict)


# ============================================================
# HELPERS CANDIDATS / MÉTRIQUES
# ============================================================

def valid_ratios() -> np.ndarray:
    return np.array([0.32, 0.28, 0.22, 0.18], dtype=float)


def invalid_ratios_far() -> np.ndarray:
    return np.array([0.50, 0.20, 0.20, 0.10], dtype=float)


def valid_metrics() -> Dict[str, float]:
    return {
        "largest_olive_component_ratio": 0.24,
        "largest_olive_component_ratio_small": 0.20,
        "olive_multizone_share": 0.62,
        "center_empty_ratio": 0.30,
        "center_empty_ratio_small": 0.34,
        "boundary_density": 0.145,
        "boundary_density_small": 0.11,
        "boundary_density_tiny": 0.11,
        "mirror_similarity": 0.44,
        "central_brown_continuity": 0.20,
        "oblique_share": 0.72,
        "vertical_share": 0.16,
        "angle_dominance_ratio": 0.20,
        "coyote_brown_macro_share": 0.00,
        "coyote_brown_transition_share": 0.05,
        "coyote_brown_micro_share": 0.00,
        "vert_olive_macro_share": 0.75,
        "vert_olive_transition_share": 0.10,
        "vert_olive_micro_share": 0.05,
        "terre_de_france_macro_share": 0.20,
        "terre_de_france_transition_share": 0.55,
        "terre_de_france_micro_share": 0.05,
        "vert_de_gris_macro_share": 0.05,
        "vert_de_gris_transition_share": 0.05,
        "vert_de_gris_micro_share": 0.83,
        "visual_score_final": 0.72,
        "visual_score_ratio": 0.90,
        "visual_score_silhouette": 0.74,
        "visual_score_contour": 0.70,
        "visual_score_main": 0.68,
        "visual_silhouette_color_diversity": 0.74,
        "visual_contour_break_score": 0.58,
        "visual_outline_band_diversity": 0.66,
        "visual_small_scale_structural_score": 0.54,
        "visual_validation_passed": 1.0,
        "macro_olive_visible_ratio": 0.19,
        "macro_terre_visible_ratio": 0.11,
        "macro_gris_visible_ratio": 0.08,
        "transition_terre_visible_ratio": 0.12,
        "micro_gris_visible_ratio": 0.12,
        "periphery_boundary_density": 0.17,
        "center_boundary_density": 0.12,
        "periphery_boundary_density_ratio": 1.22,
        "periphery_non_coyote_density": 0.68,
        "center_non_coyote_density": 0.58,
        "periphery_non_coyote_ratio": 1.17,
        "visual_military_score": 0.74,
        "visual_military_passed": 1.0,
    }


def make_candidate(
    seed: int = 123456,
    ratios: np.ndarray | None = None,
    metrics: Dict[str, float] | None = None,
) -> mut.CandidateResult:
    ratios = valid_ratios() if ratios is None else ratios
    metrics = valid_metrics() if metrics is None else metrics
    profile = mut.make_profile(seed)
    image = Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0))
    return mut.CandidateResult(
        seed=seed,
        profile=profile,
        image=image,
        ratios=ratios,
        metrics=metrics,
    )


def make_canvas_quadrants() -> np.ndarray:
    canvas = np.zeros((8, 8), dtype=np.uint8)
    canvas[:4, 4:] = 1
    canvas[4:, :4] = 2
    canvas[4:, 4:] = 3
    return canvas


def full_size_mask_from_zone_names(*names: str) -> np.ndarray:
    mask = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=bool)
    for name in names:
        mask |= mut.ANATOMY_ZONES[name]
    return mask


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
    yield "vert_olive_macro_share", mut.MIN_VISIBLE_OLIVE_MACRO_SHARE - 0.001
    yield "terre_de_france_transition_share", mut.MIN_VISIBLE_TERRE_TRANS_SHARE - 0.001
    yield "vert_de_gris_micro_share", mut.MIN_VISIBLE_GRIS_MICRO_SHARE - 0.001
    yield "vert_de_gris_macro_share", mut.MAX_VISIBLE_GRIS_MACRO_SHARE + 0.001
    yield "visual_silhouette_color_diversity", mut.VISUAL_MIN_SILHOUETTE_COLOR_DIVERSITY - 0.001
    yield "visual_contour_break_score", mut.VISUAL_MIN_CONTOUR_BREAK_SCORE - 0.001
    yield "visual_outline_band_diversity", mut.VISUAL_MIN_OUTLINE_BAND_DIVERSITY - 0.001
    yield "visual_small_scale_structural_score", mut.VISUAL_MIN_SMALL_SCALE_STRUCTURAL_SCORE - 0.001
    yield "visual_score_final", mut.VISUAL_MIN_FINAL_SCORE - 0.001
    yield "periphery_boundary_density_ratio", mut.MIN_PERIPHERY_BOUNDARY_DENSITY_RATIO - 0.001
    yield "periphery_non_coyote_ratio", mut.MIN_PERIPHERY_NON_COYOTE_RATIO - 0.001
    yield "macro_olive_visible_ratio", mut.MIN_MACRO_OLIVE_VISIBLE_RATIO - 0.001
    yield "macro_terre_visible_ratio", mut.MIN_MACRO_TERRE_VISIBLE_RATIO - 0.001
    yield "macro_terre_visible_ratio", mut.MAX_MACRO_TERRE_VISIBLE_RATIO + 0.001
    yield "macro_gris_visible_ratio", mut.MIN_MACRO_GRIS_VISIBLE_RATIO - 0.001
    yield "macro_gris_visible_ratio", mut.MAX_MACRO_GRIS_VISIBLE_RATIO + 0.001
    yield "visual_military_score", mut.VISUAL_MIN_MILITARY_SCORE - 0.001


# ============================================================
# TESTS UTILITAIRES / CONFIG / SYSTÈME
# ============================================================

class TestSystemHelpers(GlobalStateMixin, TestAssertionsMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_worker_initializer_can_limit_numeric_threads(self) -> None:
        with patch.dict(os.environ, {"CAMO_LIMIT_NUMERIC_THREADS": "1"}, clear=True):
            mut._worker_initializer()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")

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
        fake_pool = object()
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

    def test_clip_clamp_safe_helpers(self) -> None:
        self.assertEqual(mut.clamp01(-1.0), 0.0)
        self.assertEqual(mut.clamp01(2.0), 1.0)
        self.assertEqual(mut._clip_float(5.0, 0.0, 3.0), 3.0)
        self.assertEqual(mut._clip_float(-2.0, 0.0, 3.0), 0.0)
        self.assertEqual(mut._safe_failure_float("2.5"), 2.5)
        self.assertEqual(mut._safe_failure_float("bad", default=7.0), 7.0)
        self.assertEqual(mut._safe_failure_float(float("nan"), default=8.0), 8.0)


class TestPureUtilities(TempDirMixin, TestAssertionsMixin, unittest.TestCase):
    def test_ensure_output_dir_creates_directory(self) -> None:
        path = self.tmpdir / "a" / "b" / "c"
        out = mut.ensure_output_dir(path)
        self.assertEqual(out, path)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())

    def test_build_seed_is_deterministic(self) -> None:
        self.assertEqual(mut.build_seed(3, 7, 1000), mut.build_seed(3, 7, 1000))
        self.assertNotEqual(mut.build_seed(3, 7, 1000), mut.build_seed(4, 7, 1000))

    def test_make_profile_is_deterministic_for_same_seed(self) -> None:
        p1 = mut.make_profile(424242)
        p2 = mut.make_profile(424242)
        self.assertEqual(p1.allowed_angles, p2.allowed_angles)
        self.assertEqual(p1.micro_cluster_max, p2.micro_cluster_max)
        self.assertEqual(p1.seed, p2.seed)
        self.assertIn(0, p1.allowed_angles)
        self.assertEqual(sorted(set(p1.allowed_angles)), p1.allowed_angles)

    def test_make_profile_accepts_adaptive_hint(self) -> None:
        hint = {
            "prefer_oblique": 2.0,
            "prefer_vertical": 1.0,
            "diversify_angles": 1.5,
            "micro_cluster_bonus": 2,
            "olive_macro_target_scale": 1.18,
            "transition_terre_bias": 0.08,
            "micro_gris_bias": 0.06,
            "zone_weight_boosts": (1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 0.5),
        }
        profile = mut.make_profile(424242, adaptive_hint=hint)
        self.assertIsInstance(profile, mut.VariantProfile)
        self.assertGreater(len(profile.angle_pool), len(profile.allowed_angles))
        self.assertGreaterEqual(profile.micro_cluster_max, 6)
        self.assertGreater(profile.olive_macro_target_scale, 1.0)
        self.assertAlmostEqual(profile.transition_terre_bias, 0.08, places=6)
        self.assertAlmostEqual(profile.micro_gris_bias, 0.06, places=6)
        self.assertEqual(len(profile.zone_weight_boosts), len(mut.DENSITY_ZONES))

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
        self.assertImageSize(img, (7, 5))

    def test_rotate_90_deg(self) -> None:
        x, y = mut.rotate(1.0, 0.0, 90.0)
        self.assertFloatClose(x, 0.0, places=6)
        self.assertFloatClose(y, 1.0, places=6)

    def test_choose_biased_center_is_in_bounds(self) -> None:
        rng = random.Random(123)
        for _ in range(30):
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

    def test_local_color_variety_counts_unique_values(self) -> None:
        canvas = np.array([[0, 0, 1], [0, 2, 1], [3, 2, 1]], dtype=np.uint8)
        self.assertEqual(mut.local_color_variety(canvas, 1, 1, radius=1), 4)

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

    def test_infer_origin_from_neighbors(self) -> None:
        canvas = np.array([[0, 1, 1], [0, 1, 2], [3, 1, 2]], dtype=np.uint8)
        origin_map = np.array([[0, 1, 1], [0, 1, 2], [3, 1, 2]], dtype=np.uint8)
        origin = mut.infer_origin_from_neighbors(canvas, origin_map, 1, 1, chosen_color=1, fallback_origin=0)
        self.assertEqual(origin, 1)

    def test_pick_macro_angle_returns_allowed_angle(self) -> None:
        profile = mut.make_profile(111)
        angle = mut.pick_macro_angle([], profile, random.Random(42))
        self.assertIn(angle, profile.allowed_angles)

    def test_semantic_color_helpers(self) -> None:
        deficits = np.array([0.00, 0.04, -0.01, 0.02], dtype=float)
        chosen = mut.choose_semantic_color_for_origin(deficits, mut.ORIGIN_MICRO, mut.IDX_TERRE)
        self.assertEqual(chosen, mut.IDX_GRIS)
        self.assertTrue(mut.semantic_color_allowed(mut.ORIGIN_MACRO, mut.IDX_OLIVE))
        self.assertTrue(mut.semantic_color_allowed(mut.ORIGIN_MACRO, mut.IDX_GRIS))


# ============================================================
# TESTS ZONES / MORPHOLOGIE / MACROS
# ============================================================

class TestMorphologyAndZones(TestAssertionsMixin, unittest.TestCase):
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

    def test_macro_zone_count(self) -> None:
        mask = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=bool)
        mask |= mut.ANATOMY_ZONES["left_shoulder"]
        mask |= mut.ANATOMY_ZONES["left_flank"]
        self.assertGreaterEqual(mut.macro_zone_count(mask), 2)

    def test_zone_overlap_ratio(self) -> None:
        mask = mut.ANATOMY_ZONES["center_torso"].copy()
        ratio = mut.zone_overlap_ratio(mask, mut.ANATOMY_ZONES["center_torso"])
        self.assertFloatClose(ratio, 1.0)

    def test_center_empty_ratio(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        self.assertEqual(mut.center_empty_ratio(canvas), 1.0)

    def test_largest_component_ratio(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[3:6, 3:6] = True
        self.assertFloatClose(mut.largest_component_ratio(mask), 9 / 13)

    def test_orientation_score_non_empty(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], -20, (10, 10), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 0, (20, 20), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_TERRE, [], 25, (30, 30), dummy_mask, 2),
        ]
        out = mut.orientation_score(macros)
        self.assertFloatClose(out["oblique_share"], 2 / 3)
        self.assertFloatClose(out["vertical_share"], 1 / 3)
        self.assertFloatClose(out["dominance_ratio"], 1 / 3)

    def test_macro_angle_histogram(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (0, 0), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (0, 0), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_TERRE, [], 0, (0, 0), dummy_mask, 1),
        ]
        hist = mut.macro_angle_histogram(macros)
        self.assertEqual(hist[20], 2)
        self.assertEqual(hist[0], 1)

    def test_preferred_boundary_coordinates_prefers_mask(self) -> None:
        boundary = np.zeros((6, 6), dtype=bool)
        boundary[1, 1] = True
        boundary[4, 4] = True
        preferred = np.zeros((6, 6), dtype=bool)
        preferred[4, 4] = True
        ys, xs = mut.preferred_boundary_coordinates(boundary, preferred)
        self.assertEqual(list(zip(xs.tolist(), ys.tolist())), [(4, 4)])

    def test_visible_origin_shares(self) -> None:
        canvas = np.array([[0, 1, 1, 2], [0, 1, 3, 3]], dtype=np.uint8)
        origin_map = np.array([[0, 1, 3, 2], [0, 1, 3, 3]], dtype=np.uint8)
        out = mut.visible_origin_shares(canvas, origin_map)
        self.assertFloatClose(out["vert_olive_macro_share"], 2 / 3)
        self.assertFloatClose(out["vert_olive_micro_share"], 1 / 3)

    def test_macro_candidate_diagnostics_empty(self) -> None:
        diag = mut.macro_candidate_diagnostics(np.zeros((mut.HEIGHT, mut.WIDTH), dtype=bool))
        self.assertEqual(diag["zone_count"], 0.0)
        self.assertEqual(diag["edge_lock_ratio"], 1.0)

    def test_macro_candidate_diagnostics_non_empty(self) -> None:
        mask = full_size_mask_from_zone_names("left_shoulder", "left_flank")
        diag = mut.macro_candidate_diagnostics(mask)
        self.assertGreaterEqual(diag["zone_count"], 2.0)
        self.assertGreater(diag["height_span_ratio"], 0.0)
        self.assertGreater(diag["high_density_overlap"], 0.0)

    def test_macro_candidate_is_valid_rejects_empty(self) -> None:
        canvas = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=np.uint8)
        self.assertFalse(mut.macro_candidate_is_valid(np.zeros_like(canvas, dtype=bool), mut.IDX_OLIVE, 0, canvas, []))

    def test_macro_candidate_is_valid_accepts_good_olive(self) -> None:
        canvas = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=np.uint8)
        mask = full_size_mask_from_zone_names("left_shoulder", "left_flank")
        self.assertTrue(mut.macro_candidate_is_valid(mask, mut.IDX_OLIVE, 20, canvas, []))

    def test_macro_candidate_is_valid_rejects_local_parallel_conflict(self) -> None:
        canvas = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=np.uint8)
        mask = full_size_mask_from_zone_names("left_shoulder", "left_flank")
        dummy_mask = full_size_mask_from_zone_names("left_shoulder")
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (100, 100), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (120, 120), dummy_mask, 2),
        ]
        with patch.object(mut, "local_parallel_conflict", return_value=True):
            self.assertFalse(mut.macro_candidate_is_valid(mask, mut.IDX_OLIVE, 20, canvas, macros))

    def test_zone_center_angle_and_nearest_allowed_angle(self) -> None:
        p1 = mut.zone_center("left_shoulder")
        p2 = mut.zone_center("left_flank")
        angle = mut.angle_from_points_vertical(p1, p2)
        nearest = mut.nearest_allowed_angle(angle, mut.BASE_ANGLES)
        self.assertIsInstance(p1, tuple)
        self.assertIsInstance(angle, float)
        self.assertIn(nearest, mut.BASE_ANGLES)


# ============================================================
# TESTS GÉOMÉTRIE / CONTRÔLES / COUCHES
# ============================================================

class TestGeometryAndLayers(unittest.TestCase):
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

    def test_attached_transition_returns_polygon(self) -> None:
        rng = random.Random(42)
        parent = [(10, 10), (50, 10), (50, 50), (10, 50)]
        poly = mut.attached_transition(rng, parent, length_px=20, width_px=10)
        self.assertEqual(len(poly), 5)

    def test_angle_distance_deg(self) -> None:
        self.assertEqual(mut.angle_distance_deg(10, 20), 10)
        self.assertEqual(mut.angle_distance_deg(-35, 35), 70)

    def test_local_parallel_conflict(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (100, 100), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 22, (150, 110), dummy_mask, 2),
        ]
        self.assertTrue(
            mut.local_parallel_conflict(
                macros,
                center=(130, 105),
                angle_deg=18,
                dist_threshold_px=100,
                angle_threshold_deg=8,
            )
        )

    def test_transition_is_attached(self) -> None:
        parent = np.zeros((20, 20), dtype=bool)
        parent[5:10, 5:10] = True
        transition = np.zeros((20, 20), dtype=bool)
        transition[9:13, 8:12] = True
        self.assertTrue(mut.transition_is_attached(parent, transition, min_touch_pixels=1))

    def test_micro_is_on_boundary(self) -> None:
        boundary = np.zeros((10, 10), dtype=bool)
        boundary[3:7, 3:7] = True
        micro = np.zeros((10, 10), dtype=bool)
        micro[4:6, 4:6] = True
        self.assertTrue(mut.micro_is_on_boundary(boundary, micro, min_boundary_coverage=0.5))

    def test_creates_new_mass_false_when_sparse(self) -> None:
        canvas = np.zeros((100, 100), dtype=np.uint8)
        new_mask = np.zeros((100, 100), dtype=bool)
        new_mask[10:12, 10:12] = True
        self.assertFalse(mut.creates_new_mass(canvas, new_mask, mut.IDX_OLIVE, local_radius=10))

    def test_apply_mask_changes_canvas_and_origin(self) -> None:
        canvas = np.zeros((10, 10), dtype=np.uint8)
        origin_map = np.zeros((10, 10), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 2:4] = True
        mut.apply_mask(canvas, origin_map, mask, mut.IDX_TERRE, mut.ORIGIN_TRANSITION)
        self.assertTrue(np.all(canvas[mask] == mut.IDX_TERRE))
        self.assertTrue(np.all(origin_map[mask] == mut.ORIGIN_TRANSITION))

    def test_enforce_secondary_macro_budgets_smoke(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        origin_map = np.full((mut.HEIGHT, mut.WIDTH), mut.ORIGIN_BACKGROUND, dtype=np.uint8)
        macros: List[mut.MacroRecord] = []
        profile = mut.make_profile(123)
        rng = random.Random(42)
        simple_mask = full_size_mask_from_zone_names("right_shoulder")

        with patch.object(mut, "absolute_origin_color_ratios", side_effect=[
            {"macro_terre_visible_ratio": 0.00, "macro_gris_visible_ratio": 0.00, "macro_olive_visible_ratio": 0.20},
            {"macro_terre_visible_ratio": 0.10, "macro_gris_visible_ratio": 0.08, "macro_olive_visible_ratio": 0.20},
        ]), patch.object(mut, "polygon_mask", return_value=simple_mask), patch.object(mut, "macro_candidate_is_valid", return_value=True), patch.object(mut, "zone_overlap_ratio", return_value=0.0):
            mut.enforce_secondary_macro_budgets(canvas, origin_map, macros, profile, rng)

        self.assertGreaterEqual(len(macros), 1)

    def test_add_forced_structural_macros_smoke(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        origin_map = np.full((mut.HEIGHT, mut.WIDTH), mut.ORIGIN_BACKGROUND, dtype=np.uint8)
        macros: List[mut.MacroRecord] = []
        profile = mut.make_profile(123)
        rng = random.Random(42)
        simple_mask = full_size_mask_from_zone_names("left_shoulder", "left_flank")

        with patch.object(mut, "polygon_mask", return_value=simple_mask), patch.object(mut, "macro_candidate_is_valid", return_value=True):
            mut.add_forced_structural_macros(canvas, origin_map, macros, profile, rng)

        self.assertGreaterEqual(len(macros), 1)


# ============================================================
# TESTS VISUELS / MÉTRIQUES
# ============================================================

class TestVisualMetrics(TestAssertionsMixin, unittest.TestCase):
    def test_center_empty_ratio_upscaled_proxy(self) -> None:
        canvas = np.zeros((20, 20), dtype=np.uint8)
        self.assertEqual(mut.center_empty_ratio_upscaled_proxy(canvas), 1.0)

    def test_build_silhouette_mask_and_boundary(self) -> None:
        mask = mut.build_silhouette_mask(100, 200)
        boundary = mut.silhouette_boundary(mask)
        self.assertEqual(mask.shape, (200, 100))
        self.assertTrue(mask.any())
        self.assertTrue(boundary.any())

    def test_silhouette_color_diversity_score(self) -> None:
        canvas = np.zeros((120, 80), dtype=np.uint8)
        canvas[:, :20] = 1
        canvas[:, 20:40] = 2
        canvas[:, 40:60] = 3
        score = mut.silhouette_color_diversity_score(canvas)
        self.assertGreater(score, 0.0)

    def test_contour_break_score(self) -> None:
        canvas = np.zeros((120, 80), dtype=np.uint8)
        canvas[::2, :] = 1
        score, entropy = mut.contour_break_score(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertGreaterEqual(entropy, 0.0)

    def test_small_scale_structural_score_and_ratio_score(self) -> None:
        canvas = make_canvas_quadrants()
        self.assertGreaterEqual(mut.small_scale_structural_score(canvas), 0.0)
        self.assertGreater(mut.ratio_score(valid_ratios()), 0.9)
        self.assertLess(mut.ratio_score(invalid_ratios_far()), 0.5)

    def test_main_metrics_score(self) -> None:
        score = mut.main_metrics_score(valid_metrics())
        self.assertGreater(score, 0.0)

    def test_absolute_origin_color_ratios(self) -> None:
        canvas = np.array([[1, 1, 2], [3, 0, 2]], dtype=np.uint8)
        origin = np.array([[1, 1, 2], [3, 0, 2]], dtype=np.uint8)
        out = mut.absolute_origin_color_ratios(canvas, origin)
        self.assertIn("macro_olive_visible_ratio", out)
        self.assertIn("transition_terre_visible_ratio", out)

    def test_spatial_discipline_metrics(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        canvas[mut.HIGH_DENSITY_ZONE_MASK] = mut.IDX_OLIVE
        out = mut.spatial_discipline_metrics(canvas)
        self.assertIn("periphery_boundary_density_ratio", out)
        self.assertIn("periphery_non_coyote_ratio", out)

    def test_central_brown_continuity(self) -> None:
        canvas = np.zeros((100, 100), dtype=np.uint8)
        canvas[:, 45:55] = mut.IDX_COYOTE
        score = mut.central_brown_continuity(canvas)
        self.assertGreaterEqual(score, 0.0)

    def test_evaluate_visual_metrics(self) -> None:
        canvas = np.zeros((120, 80), dtype=np.uint8)
        canvas[:, :20] = 1
        canvas[:, 20:40] = 2
        canvas[:, 40:60] = 3
        metrics = valid_metrics()
        out = mut.evaluate_visual_metrics(canvas, valid_ratios(), metrics)
        self.assertIn("visual_score_final", out)
        self.assertIn("visual_validation_passed", out)

    def test_military_visual_discipline_score(self) -> None:
        out = mut.military_visual_discipline_score(valid_metrics())
        self.assertIn("visual_military_score", out)
        self.assertIn("visual_military_passed", out)


# ============================================================
# TESTS ADAPTATION / LOGS / DIAGNOSTICS
# ============================================================

class TestAdaptiveRejectionCorrection(GlobalStateMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_register_and_consume_profile_override(self) -> None:
        mut._register_profile_override(123, {"prefer_oblique": 1.0})
        self.assertIn(123, mut._ADAPTIVE_PROFILE_OVERRIDES)
        hint = mut._consume_profile_override(123)
        self.assertEqual(hint["prefer_oblique"], 1.0)
        self.assertIsNone(mut._consume_profile_override(123))

    def test_get_log_module_from_sys_modules(self) -> None:
        module = types.SimpleNamespace(log_event=lambda *a, **k: None)
        with patch.dict(sys.modules, {"log": module}, clear=False):
            got = mut._get_log_module()
        self.assertIs(got, module)

    def test_runtime_log_calls_log_event(self) -> None:
        event = Mock()
        module = types.SimpleNamespace(log_event=event)
        with patch.object(mut, "_get_log_module", return_value=module):
            mut._runtime_log("INFO", "source", "message", x=1)
        event.assert_called_once()

    def test_failure_payload_and_append_helpers(self) -> None:
        payload = mut._failure_payload("rule", 1.0, 0.2, min_value=0.0, max_value=2.0, target=1.2)
        self.assertEqual(payload["rule"], "rule")
        failures: List[Dict[str, Any]] = []
        mut._append_failure_range(failures, "r1", 0.1, 0.2, 0.8)
        mut._append_failure_min(failures, "r2", 0.1, 0.2)
        mut._append_failure_max(failures, "r3", 0.9, 0.8)
        mut._append_failure_abs_target(failures, "r4", 0.5, 0.2, 0.1)
        self.assertEqual([f["rule"] for f in failures], ["r1", "r2", "r3", "r4"])

    def test_fallback_analyze_candidate_failures_returns_rules(self) -> None:
        candidate = make_candidate(ratios=invalid_ratios_far())
        failures = mut._fallback_analyze_candidate_failures(candidate)
        self.assertTrue(any(item["rule"] == "ratio_coyote" for item in failures))

    def test_extract_rejection_failures_prefers_log_module_analyze_candidate(self) -> None:
        fake_failure = types.SimpleNamespace(rule="ratio_olive", actual=0.1, target=0.28, min_value=0.24, max_value=0.33, delta=0.14)
        diagnostic = types.SimpleNamespace(failures=[fake_failure])
        module = types.SimpleNamespace(analyze_candidate=Mock(return_value=diagnostic))
        candidate = make_candidate(ratios=invalid_ratios_far())
        with patch.object(mut, "_get_log_module", return_value=module):
            out = mut.extract_rejection_failures(candidate, 1, 1)
        self.assertEqual(out[0]["rule"], "ratio_olive")

    def test_build_adaptive_hint_wrapper(self) -> None:
        state = mut.AdaptiveGenerationState(olive_pressure=2.0)
        hint = mut.build_adaptive_hint(state)
        self.assertIn("olive_macro_target_scale", hint)

    def test_update_adaptive_state_after_attempt_success_and_failure(self) -> None:
        state = mut.AdaptiveGenerationState()
        accepted_candidate = make_candidate()
        rejected_candidate = make_candidate(ratios=invalid_ratios_far())

        out1 = mut.update_adaptive_state_after_attempt(state, accepted_candidate, True, 1, 1)
        self.assertEqual(out1, [])
        self.assertEqual(state.consecutive_rejections, 0)

        with patch.object(mut, "extract_rejection_failures", return_value=[{"rule": "ratio_olive", "actual": 0.1, "delta": 0.2, "min_value": 0.24, "max_value": 0.33, "target": 0.28}]):
            out2 = mut.update_adaptive_state_after_attempt(state, rejected_candidate, False, 1, 2)
        self.assertEqual(out2[0]["rule"], "ratio_olive")
        self.assertGreater(state.consecutive_rejections, 0)

    def test_generate_candidate_from_seed_with_hint_and_validate_wrappers(self) -> None:
        candidate = make_candidate(seed=999)
        with patch.object(mut, "generate_one_variant", return_value=(candidate.image, candidate.ratios, candidate.metrics)):
            cand = mut.generate_candidate_from_seed_with_hint(999, {"prefer_oblique": 1.0})
        self.assertEqual(cand.seed, 999)

        with patch.object(mut, "generate_candidate_from_seed_with_hint", return_value=candidate), patch.object(mut, "validate_candidate_result", return_value=True):
            out_candidate, accepted = mut.generate_and_validate_from_seed_with_hint(999, {"prefer_oblique": 1.0})
        self.assertTrue(accepted)
        self.assertEqual(out_candidate.seed, 999)

    def test_generate_and_validate_from_seed_wrapper(self) -> None:
        candidate = make_candidate(seed=123)
        with patch.object(mut, "generate_candidate_from_seed", return_value=candidate), patch.object(mut, "validate_candidate_result", return_value=False):
            out_candidate, accepted = mut.generate_and_validate_from_seed(123)
        self.assertFalse(accepted)
        self.assertEqual(out_candidate.seed, 123)

    def test_batch_attempt_seeds(self) -> None:
        batch = mut._batch_attempt_seeds(2, 3, 4, 100)
        self.assertEqual(len(batch), 4)
        self.assertEqual(batch[0][0], 3)
        self.assertEqual(batch[-1][0], 6)


# ============================================================
# TESTS GÉNÉRATION CANDIDATS
# ============================================================

class TestCandidateGeneration(TestAssertionsMixin, unittest.TestCase):
    def test_generate_one_variant_structure(self) -> None:
        profile = mut.make_profile(mut.DEFAULT_BASE_SEED)
        fake_image = Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0))
        fake_ratios = valid_ratios()
        fake_metrics = valid_metrics()
        dummy_mask = np.zeros((10, 10), dtype=bool)
        fake_macros = [mut.MacroRecord(mut.IDX_OLIVE, [], 0, (10, 10), dummy_mask, 2)]

        with ExitStack() as stack:
            stack.enter_context(patch.object(mut, "add_macros", return_value=fake_macros))
            stack.enter_context(patch.object(mut, "enforce_macro_angle_discipline"))
            stack.enter_context(patch.object(mut, "add_transitions"))
            stack.enter_context(patch.object(mut, "add_micro_clusters"))
            stack.enter_context(patch.object(mut, "repair_spatial_discipline"))
            stack.enter_context(patch.object(mut, "nudge_proportions"))
            stack.enter_context(patch.object(mut, "enforce_secondary_macro_budgets"))
            stack.enter_context(patch.object(mut, "add_forced_structural_macros"))
            stack.enter_context(patch.object(mut, "render_canvas", return_value=fake_image))
            stack.enter_context(patch.object(mut, "compute_ratios", return_value=fake_ratios))
            stack.enter_context(patch.object(mut, "orientation_score", return_value={
                "oblique_share": fake_metrics["oblique_share"],
                "vertical_share": fake_metrics["vertical_share"],
                "dominance_ratio": fake_metrics["angle_dominance_ratio"],
            }))
            stack.enter_context(patch.object(mut, "visible_origin_shares", return_value={
                "coyote_brown_macro_share": fake_metrics["coyote_brown_macro_share"],
                "coyote_brown_transition_share": fake_metrics["coyote_brown_transition_share"],
                "coyote_brown_micro_share": fake_metrics["coyote_brown_micro_share"],
                "vert_olive_macro_share": fake_metrics["vert_olive_macro_share"],
                "vert_olive_transition_share": fake_metrics["vert_olive_transition_share"],
                "vert_olive_micro_share": fake_metrics["vert_olive_micro_share"],
                "terre_de_france_macro_share": fake_metrics["terre_de_france_macro_share"],
                "terre_de_france_transition_share": fake_metrics["terre_de_france_transition_share"],
                "terre_de_france_micro_share": fake_metrics["terre_de_france_micro_share"],
                "vert_de_gris_macro_share": fake_metrics["vert_de_gris_macro_share"],
                "vert_de_gris_transition_share": fake_metrics["vert_de_gris_transition_share"],
                "vert_de_gris_micro_share": fake_metrics["vert_de_gris_micro_share"],
            }))
            stack.enter_context(patch.object(mut, "multiscale_metrics", return_value={
                "boundary_density_small": fake_metrics["boundary_density_small"],
                "boundary_density_tiny": fake_metrics["boundary_density_tiny"],
                "center_empty_ratio_small": fake_metrics["center_empty_ratio_small"],
                "largest_olive_component_ratio_small": fake_metrics["largest_olive_component_ratio_small"],
            }))
            stack.enter_context(patch.object(mut, "absolute_origin_color_ratios", return_value={
                "macro_olive_visible_ratio": fake_metrics["macro_olive_visible_ratio"],
                "macro_terre_visible_ratio": fake_metrics["macro_terre_visible_ratio"],
                "macro_gris_visible_ratio": fake_metrics["macro_gris_visible_ratio"],
                "transition_terre_visible_ratio": fake_metrics["transition_terre_visible_ratio"],
                "micro_gris_visible_ratio": fake_metrics["micro_gris_visible_ratio"],
            }))
            stack.enter_context(patch.object(mut, "spatial_discipline_metrics", return_value={
                "periphery_boundary_density": fake_metrics["periphery_boundary_density"],
                "center_boundary_density": fake_metrics["center_boundary_density"],
                "periphery_boundary_density_ratio": fake_metrics["periphery_boundary_density_ratio"],
                "periphery_non_coyote_density": fake_metrics["periphery_non_coyote_density"],
                "center_non_coyote_density": fake_metrics["center_non_coyote_density"],
                "periphery_non_coyote_ratio": fake_metrics["periphery_non_coyote_ratio"],
            }))
            stack.enter_context(patch.object(mut, "evaluate_visual_metrics", return_value={
                "visual_score_final": fake_metrics["visual_score_final"],
                "visual_score_ratio": fake_metrics["visual_score_ratio"],
                "visual_score_silhouette": fake_metrics["visual_score_silhouette"],
                "visual_score_contour": fake_metrics["visual_score_contour"],
                "visual_score_main": fake_metrics["visual_score_main"],
                "visual_silhouette_color_diversity": fake_metrics["visual_silhouette_color_diversity"],
                "visual_contour_break_score": fake_metrics["visual_contour_break_score"],
                "visual_outline_band_diversity": fake_metrics["visual_outline_band_diversity"],
                "visual_small_scale_structural_score": fake_metrics["visual_small_scale_structural_score"],
                "visual_validation_passed": fake_metrics["visual_validation_passed"],
            }))
            stack.enter_context(patch.object(mut, "military_visual_discipline_score", return_value={
                "visual_military_score": fake_metrics["visual_military_score"],
                "visual_military_passed": fake_metrics["visual_military_passed"],
            }))
            stack.enter_context(patch.object(mut, "largest_component_ratio", return_value=fake_metrics["largest_olive_component_ratio"]))
            stack.enter_context(patch.object(mut, "center_empty_ratio", return_value=fake_metrics["center_empty_ratio"]))
            stack.enter_context(patch.object(mut, "boundary_density", return_value=fake_metrics["boundary_density"]))
            stack.enter_context(patch.object(mut, "mirror_similarity_score", return_value=fake_metrics["mirror_similarity"]))
            stack.enter_context(patch.object(mut, "central_brown_continuity", return_value=fake_metrics["central_brown_continuity"]))
            image, ratios, metrics = mut.generate_one_variant(profile)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(ratios.shape, (4,))
        self.assertIsInstance(metrics, dict)
        self.assertAlmostEqual(float(np.sum(ratios)), 1.0, places=5)
        self.assertEqual(metrics["largest_olive_component_ratio_small"], fake_metrics["largest_olive_component_ratio_small"])
        self.assertEqual(metrics["central_brown_continuity"], fake_metrics["central_brown_continuity"])

    def test_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        seed = mut.DEFAULT_BASE_SEED
        profile = mut.make_profile(seed)
        fake_image = Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0))
        fake_ratios = valid_ratios()
        fake_metrics = valid_metrics()
        with patch.object(mut, "generate_one_variant", return_value=(fake_image, fake_ratios, fake_metrics)) as mock_generate:
            cand = mut.generate_candidate_from_seed(seed)
        self.assertCandidateLooksConsistent(cand)
        self.assertEqual(cand.seed, seed)
        self.assertEqual(cand.profile.allowed_angles, profile.allowed_angles)
        mock_generate.assert_called_once()


class TestCandidateGenerationAsync(TestAssertionsMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        fake_candidate = make_candidate(seed=mut.DEFAULT_BASE_SEED + 1)
        with patch.object(mut, "generate_candidate_from_seed", return_value=fake_candidate) as mock_generate:
            cand = await mut.async_generate_candidate_from_seed(mut.DEFAULT_BASE_SEED + 1)
        self.assertCandidateLooksConsistent(cand)
        mock_generate.assert_called_once()

    async def test_async_validate_candidate_result(self) -> None:
        self.assertTrue(await mut.async_validate_candidate_result(make_candidate()))


# ============================================================
# TESTS VALIDATION / EXPORTS
# ============================================================

class TestValidation(unittest.TestCase):
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
        self.assertIn("coyote_brown_pct", row)
        self.assertIn("visual_military_score", row)


class TestExports(TempDirMixin, unittest.TestCase):
    def test_save_candidate_image(self) -> None:
        candidate = make_candidate()
        out = mut.save_candidate_image(candidate, self.tmpdir / "x" / "img.png")
        self.assertTrue(out.exists())

    def test_write_report(self) -> None:
        row = mut.candidate_row(1, 1, 1, make_candidate())
        out = mut.write_report([row], self.tmpdir)
        self.assertTrue(out.exists())
        with out.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

    def test_write_report_empty(self) -> None:
        out = mut.write_report([], self.tmpdir)
        self.assertTrue(out.exists())
        self.assertEqual(out.read_text(encoding="utf-8"), "")


# ============================================================
# TESTS ORCHESTRATEURS SYNCHRONES
# ============================================================

class TestGenerateAllSync(TempDirMixin, unittest.TestCase):
    def test_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=101)
        progress_calls: List[tuple] = []

        def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted: bool) -> None:
            progress_calls.append((target_index, local_attempt, total_attempts, target_count, cand.seed, accepted))

        with patch.object(mut, "generate_candidate_from_seed", return_value=candidate) as mock_gen, \
             patch.object(mut, "validate_candidate_result", return_value=True) as mock_val:
            rows = mut.generate_all(target_count=1, output_dir=self.tmpdir, base_seed=999, progress_callback=progress_cb, parallel_attempts=False)

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.call_count, 1)
        self.assertEqual(mock_val.call_count, 1)
        self.assertEqual(len(progress_calls), 1)
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    def test_generate_all_retries_until_accept(self) -> None:
        rejected = make_candidate(seed=201)
        accepted = make_candidate(seed=202)
        with patch.object(mut, "generate_candidate_from_seed", side_effect=[rejected, accepted]) as mock_gen, \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]) as mock_val:
            rows = mut.generate_all(target_count=1, output_dir=self.tmpdir, parallel_attempts=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.call_count, 2)
        self.assertEqual(mock_val.call_count, 2)
        self.assertEqual(rows[0]["seed"], 202)

    def test_generate_all_stop_requested_writes_partial_report(self) -> None:
        with patch.object(mut, "generate_candidate_from_seed") as mock_gen:
            rows = mut.generate_all(target_count=3, output_dir=self.tmpdir, stop_requested=lambda: True, parallel_attempts=False)
        self.assertEqual(rows, [])
        self.assertEqual(mock_gen.call_count, 0)
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    def test_generate_all_parallel_path_with_adaptive_enabled(self) -> None:
        def fake_eval(seed: int, adaptive_hint: Dict[str, Any] | None = None):
            cand = make_candidate(seed=seed)
            accepted = (seed % 2 == 0)
            return cand, accepted

        with ThreadPoolExecutor(max_workers=2) as pool, \
             patch.object(mut, "get_process_pool", return_value=pool), \
             patch.object(mut, "generate_and_validate_from_seed_with_hint", side_effect=fake_eval):
            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=0,
                max_workers=2,
                attempt_batch_size=2,
                parallel_attempts=True,
                adaptive_rejection_correction=True,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())


# ============================================================
# TESTS ORCHESTRATEURS ASYNCHRONES
# ============================================================

class TestGenerateAllAsync(AsyncTempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=301)
        progress = AsyncMock()

        with patch.object(mut, "generate_candidate_from_seed", return_value=candidate), \
             patch.object(mut, "validate_candidate_result", return_value=True):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                progress_callback=progress,
                parallel_attempts=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(progress.await_count, 1)
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())

    async def test_async_generate_all_stop_requested(self) -> None:
        stop_cb = AsyncMock(return_value=True)
        rows = await mut.async_generate_all(
            target_count=3,
            output_dir=self.tmpdir,
            stop_requested=stop_cb,
            parallel_attempts=False,
        )
        self.assertEqual(rows, [])
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    async def test_async_generate_all_parallel_path(self) -> None:
        def fake_eval(seed: int, adaptive_hint: Dict[str, Any] | None = None):
            cand = make_candidate(seed=seed)
            accepted = (seed % 2 == 0)
            return cand, accepted

        with ThreadPoolExecutor(max_workers=2) as pool, \
             patch.object(mut, "get_process_pool", return_value=pool), \
             patch.object(mut, "generate_and_validate_from_seed_with_hint", side_effect=fake_eval):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=0,
                max_workers=2,
                attempt_batch_size=2,
                parallel_attempts=True,
                adaptive_rejection_correction=True,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)


# ============================================================
# TESTS CONSTANTES / SANITY CHECKS
# ============================================================

class TestConstants(unittest.TestCase):
    def test_target_is_probability_distribution(self) -> None:
        self.assertEqual(mut.TARGET.shape, (4,))
        self.assertAlmostEqual(float(np.sum(mut.TARGET)), 1.0, places=8)

    def test_max_abs_error_per_color_has_four_entries(self) -> None:
        self.assertEqual(mut.MAX_ABS_ERROR_PER_COLOR.shape, (4,))

    def test_color_names_alignment(self) -> None:
        self.assertEqual(len(mut.COLOR_NAMES), 4)

    def test_index_constants_alignment(self) -> None:
        self.assertEqual((mut.IDX_COYOTE, mut.IDX_OLIVE, mut.IDX_TERRE, mut.IDX_GRIS), (0, 1, 2, 3))

    def test_origin_constants_alignment(self) -> None:
        self.assertEqual((mut.ORIGIN_BACKGROUND, mut.ORIGIN_MACRO, mut.ORIGIN_TRANSITION, mut.ORIGIN_MICRO), (0, 1, 2, 3))


# ============================================================
# RESULT CLASS AVEC LOGS PRÉCIS
# ============================================================

class LoggedTextTestResult(unittest.TextTestResult):
    def startTestRun(self) -> None:
        LOGGER.info("========== START TEST RUN ==========")
        super().startTestRun()

    def stopTestRun(self) -> None:
        LOGGER.info(
            "========== STOP TEST RUN | testsRun=%s | failures=%s | errors=%s | skipped=%s | expectedFailures=%s | unexpectedSuccesses=%s ==========",
            self.testsRun,
            len(self.failures),
            len(self.errors),
            len(getattr(self, "skipped", [])),
            len(getattr(self, "expectedFailures", [])),
            len(getattr(self, "unexpectedSuccesses", [])),
        )
        super().stopTestRun()

    def startTest(self, test: unittest.case.TestCase) -> None:
        LOGGER.info("START | %s", test.id())
        super().startTest(test)

    def stopTest(self, test: unittest.case.TestCase) -> None:
        LOGGER.info("STOP  | %s", test.id())
        super().stopTest(test)

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        LOGGER.info("OK    | %s", test.id())
        super().addSuccess(test)

    def addFailure(self, test: unittest.case.TestCase, err) -> None:
        details = self._exc_info_to_string(err, test)
        LOGGER.error("FAIL  | %s\n%s", test.id(), details)
        super().addFailure(test, err)

    def addError(self, test: unittest.case.TestCase, err) -> None:
        details = self._exc_info_to_string(err, test)
        LOGGER.error("ERROR | %s\n%s", test.id(), details)
        super().addError(test, err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        LOGGER.warning("SKIP  | %s | reason=%s", test.id(), reason)
        super().addSkip(test, reason)

    def addExpectedFailure(self, test: unittest.case.TestCase, err) -> None:
        details = self._exc_info_to_string(err, test)
        LOGGER.warning("XFAIL | %s\n%s", test.id(), details)
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:
        LOGGER.warning("XPASS | %s", test.id())
        super().addUnexpectedSuccess(test)


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_main.py ==========")
    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=LoggedTextTestResult,
    )
    unittest.main(testRunner=runner, verbosity=2)