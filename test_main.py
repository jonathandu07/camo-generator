# -*- coding: utf-8 -*-
"""
test_main_corrected.py
Suite de tests unitaires corrigée pour la version corrigée de main.py.

Exécution :
    python -m unittest -v test_main_corrected.py
"""

from __future__ import annotations

import asyncio
import csv
import logging
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, Iterable, List
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image

try:
    import main_corrected as mut
except ImportError:
    import main as mut


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = Path(__file__).resolve().parent / "logs_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_main_corrected.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_main_corrected")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
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
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_corrected_")
        self.tmpdir = Path(self._tmpdir_obj.name)

    def tearDown(self) -> None:
        self._tmpdir_obj.cleanup()


class AsyncTempDirMixin:
    async def asyncSetUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_corrected_async_")
        self.tmpdir = Path(self._tmpdir_obj.name)

    async def asyncTearDown(self) -> None:
        self._tmpdir_obj.cleanup()


class TestAssertionsMixin:
    def assertFloatClose(self, a: float, b: float, places: int = 8, msg: str | None = None) -> None:
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
        self.assertIsInstance(candidate.metrics, dict)
        self.assertGreaterEqual(candidate.seed, 0)
        self.assertAlmostEqual(float(np.sum(candidate.ratios)), 1.0, places=5)


async def gather_concurrent(*aws):
    return await asyncio.gather(*aws)


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
        "olive_multizone_share": 0.60,
        "center_empty_ratio": 0.31,
        "center_empty_ratio_small": 0.34,
        "boundary_density": 0.145,
        "boundary_density_small": 0.110,
        "boundary_density_tiny": 0.12,
        "mirror_similarity": 0.44,
        "oblique_share": 0.72,
        "vertical_share": 0.15,
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
        "macro_terre_visible_ratio": 0.07,
        "macro_gris_visible_ratio": 0.008,
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


def summarize_candidate(candidate: mut.CandidateResult) -> str:
    rs = candidate.ratios
    return (
        f"seed={candidate.seed} | "
        f"ratios=({rs[0]:.4f}, {rs[1]:.4f}, {rs[2]:.4f}, {rs[3]:.4f}) | "
        f"boundary={candidate.metrics.get('boundary_density', float('nan')):.4f} | "
        f"mirror={candidate.metrics.get('mirror_similarity', float('nan')):.4f}"
    )


def make_canvas_quadrants() -> np.ndarray:
    canvas = np.zeros((8, 8), dtype=np.uint8)
    canvas[:4, 4:] = 1
    canvas[4:, :4] = 2
    canvas[4:, 4:] = 3
    return canvas


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
    yield "macro_terre_visible_ratio", mut.MAX_MACRO_TERRE_VISIBLE_RATIO + 0.001
    yield "macro_gris_visible_ratio", mut.MAX_MACRO_GRIS_VISIBLE_RATIO + 0.001
    yield "visual_military_score", mut.VISUAL_MIN_MILITARY_SCORE - 0.001


# ============================================================
# TESTS UTILITAIRES PURS
# ============================================================

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
        self.assertGreater(profile.angle_pool.count(0), 1)
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
        import random
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
        import random
        profile = mut.make_profile(111)
        angle = mut.pick_macro_angle([], profile, random.Random(42))
        self.assertIn(angle, profile.allowed_angles)

    def test_semantic_color_helpers(self) -> None:
        deficits = np.array([0.00, 0.04, -0.01, 0.02], dtype=float)
        chosen = mut.choose_semantic_color_for_origin(deficits, mut.ORIGIN_MICRO, mut.IDX_TERRE)
        self.assertEqual(chosen, mut.IDX_GRIS)
        self.assertTrue(mut.semantic_color_allowed(mut.ORIGIN_MACRO, mut.IDX_OLIVE))
        self.assertFalse(mut.semantic_color_allowed(mut.ORIGIN_MACRO, mut.IDX_GRIS))


# ============================================================
# TESTS ZONES / MORPHOLOGIE
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
            {"left_shoulder", "right_shoulder", "left_flank", "right_flank", "left_thigh", "right_thigh", "center_torso"},
        )

    def test_macro_zone_count(self) -> None:
        mask = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=bool)
        mask |= mut.ANATOMY_ZONES["left_shoulder"]
        mask |= mut.ANATOMY_ZONES["left_flank"]
        self.assertGreaterEqual(mut.macro_zone_count(mask), 2)

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

    def test_visible_origin_shares(self) -> None:
        canvas = np.array([[0, 1, 1, 2], [0, 1, 3, 3]], dtype=np.uint8)
        origin_map = np.array([[0, 1, 3, 2], [0, 1, 3, 3]], dtype=np.uint8)
        out = mut.visible_origin_shares(canvas, origin_map)
        self.assertFloatClose(out["vert_olive_macro_share"], 2 / 3)
        self.assertFloatClose(out["vert_olive_micro_share"], 1 / 3)

    def test_multiscale_metrics_keys(self) -> None:
        out = mut.multiscale_metrics(make_canvas_quadrants())
        self.assertEqual(
            set(out.keys()),
            {"boundary_density_small", "boundary_density_tiny", "center_empty_ratio_small", "largest_olive_component_ratio_small"},
        )


# ============================================================
# TESTS FORMES / GÉOMÉTRIE / CONTRÔLES
# ============================================================

class TestGeometryAndControls(unittest.TestCase):
    def test_jagged_spine_poly_returns_points(self) -> None:
        import random
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
        import random
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
        self.assertTrue(mut.local_parallel_conflict(macros, center=(130, 105), angle_deg=18, dist_threshold_px=100, angle_threshold_deg=8))

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
        self.assertFalse(mut.creates_new_mass(canvas, new_mask, mut.IDX_OLIVE, local_radius=10, max_local_area_ratio=0.50))


# ============================================================
# TESTS COUCHES / NUDGE
# ============================================================

class TestLayerBehaviors(unittest.TestCase):
    def test_apply_mask_changes_canvas_and_origin(self) -> None:
        canvas = np.zeros((6, 6), dtype=np.uint8)
        origin_map = np.zeros((6, 6), dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=bool)
        mask[2:4, 2:4] = True
        mut.apply_mask(canvas, origin_map, mask, mut.IDX_TERRE, mut.ORIGIN_TRANSITION)
        self.assertTrue(np.all(canvas[mask] == mut.IDX_TERRE))
        self.assertTrue(np.all(origin_map[mask] == mut.ORIGIN_TRANSITION))

    def test_nudge_proportions_no_boundary_no_crash(self) -> None:
        import random
        canvas = np.zeros((20, 20), dtype=np.uint8)
        origin_map = np.zeros((20, 20), dtype=np.uint8)
        before_canvas = canvas.copy()
        before_origin = origin_map.copy()
        mut.nudge_proportions(canvas, origin_map, random.Random(123))
        np.testing.assert_array_equal(canvas, before_canvas)
        np.testing.assert_array_equal(origin_map, before_origin)


# ============================================================
# TESTS VALIDATION / EXPORT
# ============================================================

class TestValidationAndExport(TempDirMixin, TestAssertionsMixin, unittest.TestCase):
    def test_variant_is_valid_accepts_valid_candidate(self) -> None:
        self.assertTrue(mut.variant_is_valid(valid_ratios(), valid_metrics()))

    def test_variant_is_valid_rejects_bad_ratios(self) -> None:
        self.assertFalse(mut.variant_is_valid(invalid_ratios_far(), valid_metrics()))

    def test_variant_is_valid_rejects_each_metric_threshold(self) -> None:
        rs = valid_ratios()
        for metric_name, bad_value in iter_metric_failure_cases():
            with self.subTest(metric=metric_name, bad_value=bad_value):
                metrics = valid_metrics()
                metrics[metric_name] = bad_value
                self.assertFalse(mut.variant_is_valid(rs, metrics))

    def test_validate_candidate_result_delegates_to_variant_is_valid(self) -> None:
        self.assertTrue(mut.validate_candidate_result(make_candidate()))

    def test_save_candidate_image(self) -> None:
        path = self.tmpdir / "img" / "camouflage_test.png"
        out = mut.save_candidate_image(make_candidate(), path)
        self.assertTrue(path.exists())
        self.assertEqual(out, path)

    def test_candidate_row_contains_expected_fields(self) -> None:
        row = mut.candidate_row(1, 2, 3, make_candidate(seed=999))
        required = {
            "index", "seed", "attempts_for_this_image", "global_attempt",
            "coyote_brown_pct", "vert_olive_pct", "terre_de_france_pct", "vert_de_gris_pct",
            "largest_olive_component_ratio", "largest_olive_component_ratio_small",
            "olive_multizone_share", "center_empty_ratio", "center_empty_ratio_small",
            "boundary_density", "boundary_density_small", "boundary_density_tiny",
            "mirror_similarity", "oblique_share", "vertical_share", "angle_dominance_ratio",
            "olive_macro_share", "terre_transition_share", "gris_micro_share", "gris_macro_share", "angles",
            "visual_score_final", "periphery_boundary_density_ratio", "visual_military_score",
        }
        self.assertTrue(required.issubset(row.keys()))
        self.assertEqual(row["seed"], 999)

    def test_write_report_with_rows(self) -> None:
        rows = [mut.candidate_row(1, 1, 1, make_candidate(seed=111)), mut.candidate_row(2, 1, 2, make_candidate(seed=222))]
        csv_path = mut.write_report(rows, self.tmpdir, filename="rapport.csv")
        self.assertTrue(csv_path.exists())
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 2)
        self.assertEqual(reader[0]["seed"], "111")
        self.assertEqual(reader[1]["seed"], "222")

    def test_write_report_with_empty_rows_creates_empty_file(self) -> None:
        csv_path = mut.write_report([], self.tmpdir, filename="empty.csv")
        self.assertTrue(csv_path.exists())
        self.assertEqual(csv_path.read_text(encoding="utf-8"), "")


# ============================================================
# TESTS ASYNC I/O
# ============================================================

class TestValidationAndExportAsync(AsyncTempDirMixin, TestAssertionsMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_save_candidate_image(self) -> None:
        path = self.tmpdir / "img" / "camouflage_async.png"
        out = await mut.async_save_candidate_image(make_candidate(), path)
        self.assertTrue(out.exists())

    async def test_async_write_report(self) -> None:
        rows = [mut.candidate_row(1, 1, 1, make_candidate(seed=1)), mut.candidate_row(2, 1, 2, make_candidate(seed=2))]
        path = await mut.async_write_report(rows, self.tmpdir, filename="async_report.csv")
        self.assertTrue(path.exists())
        with path.open("r", encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))
        self.assertEqual(len(data), 2)

    async def test_async_io_batch_is_parallelizable(self) -> None:
        async def fake_save(candidate: mut.CandidateResult, path: Path) -> Path:
            await asyncio.sleep(0.10)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok", encoding="utf-8")
            return path

        candidates = [make_candidate(seed=i) for i in range(4)]
        paths = [self.tmpdir / "parallel" / f"{i}.png" for i in range(4)]
        started = time.perf_counter()
        with patch.object(mut, "async_save_candidate_image", side_effect=fake_save):
            await gather_concurrent(*(mut.async_save_candidate_image(c, p) for c, p in zip(candidates, paths)))
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.30)


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

        with patch.object(mut, "add_macros", return_value=fake_macros), \
             patch.object(mut, "enforce_macro_angle_discipline"), \
             patch.object(mut, "add_transitions"), \
             patch.object(mut, "add_micro_clusters"), \
             patch.object(mut, "repair_spatial_discipline"), \
             patch.object(mut, "nudge_proportions"), \
             patch.object(mut, "render_canvas", return_value=fake_image), \
             patch.object(mut, "compute_ratios", return_value=fake_ratios), \
             patch.object(mut, "orientation_score", return_value={
                 "oblique_share": fake_metrics["oblique_share"],
                 "vertical_share": fake_metrics["vertical_share"],
                 "dominance_ratio": fake_metrics["angle_dominance_ratio"],
             }), \
             patch.object(mut, "visible_origin_shares", return_value={
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
             }), \
             patch.object(mut, "multiscale_metrics", return_value={
                 "boundary_density_small": fake_metrics["boundary_density_small"],
                 "boundary_density_tiny": fake_metrics["boundary_density_tiny"],
                 "center_empty_ratio_small": fake_metrics["center_empty_ratio_small"],
                 "largest_olive_component_ratio_small": fake_metrics["largest_olive_component_ratio_small"],
             }), \
             patch.object(mut, "absolute_origin_color_ratios", return_value={
                 "macro_olive_visible_ratio": fake_metrics["macro_olive_visible_ratio"],
                 "macro_terre_visible_ratio": fake_metrics["macro_terre_visible_ratio"],
                 "macro_gris_visible_ratio": fake_metrics["macro_gris_visible_ratio"],
                 "transition_terre_visible_ratio": fake_metrics["transition_terre_visible_ratio"],
                 "micro_gris_visible_ratio": fake_metrics["micro_gris_visible_ratio"],
             }), \
             patch.object(mut, "spatial_discipline_metrics", return_value={
                 "periphery_boundary_density": fake_metrics["periphery_boundary_density"],
                 "center_boundary_density": fake_metrics["center_boundary_density"],
                 "periphery_boundary_density_ratio": fake_metrics["periphery_boundary_density_ratio"],
                 "periphery_non_coyote_density": fake_metrics["periphery_non_coyote_density"],
                 "center_non_coyote_density": fake_metrics["center_non_coyote_density"],
                 "periphery_non_coyote_ratio": fake_metrics["periphery_non_coyote_ratio"],
             }), \
             patch.object(mut, "evaluate_visual_metrics", return_value={
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
             }), \
             patch.object(mut, "military_visual_discipline_score", return_value={
                 "visual_military_score": fake_metrics["visual_military_score"],
                 "visual_military_passed": fake_metrics["visual_military_passed"],
             }), \
             patch.object(mut, "largest_component_ratio", return_value=fake_metrics["largest_olive_component_ratio"]), \
             patch.object(mut, "center_empty_ratio", return_value=fake_metrics["center_empty_ratio"]), \
             patch.object(mut, "boundary_density", return_value=fake_metrics["boundary_density"]), \
             patch.object(mut, "mirror_similarity_score", return_value=fake_metrics["mirror_similarity"]):
            image, ratios, metrics = mut.generate_one_variant(profile)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(ratios.shape, (4,))
        self.assertIsInstance(metrics, dict)
        self.assertAlmostEqual(float(np.sum(ratios)), 1.0, places=5)
        self.assertEqual(metrics["largest_olive_component_ratio_small"], fake_metrics["largest_olive_component_ratio_small"])

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
# TESTS ORCHESTRATEURS
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
            rows = mut.generate_all(target_count=3, output_dir=self.tmpdir, stop_requested=lambda: True)
        self.assertEqual(rows, [])
        self.assertEqual(mock_gen.call_count, 0)
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())


class TestGenerateAllAsync(AsyncTempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=301)
        progress_calls: List[tuple] = []

        async def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted: bool) -> None:
            progress_calls.append((target_index, local_attempt, total_attempts, target_count, cand.seed, accepted))

        with patch.object(mut, "async_generate_candidate_from_seed", AsyncMock(return_value=candidate)) as mock_gen, \
             patch.object(mut, "async_validate_candidate_result", AsyncMock(return_value=True)) as mock_val:
            rows = await mut.async_generate_all(target_count=1, output_dir=self.tmpdir, progress_callback=progress_cb, parallel_attempts=False)

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.await_count, 1)
        self.assertEqual(mock_val.await_count, 1)
        self.assertEqual(len(progress_calls), 1)
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    async def test_async_generate_all_stop_requested(self) -> None:
        async def stop_requested() -> bool:
            return True

        with patch.object(mut, "async_generate_candidate_from_seed", AsyncMock()) as mock_gen:
            rows = await mut.async_generate_all(target_count=5, output_dir=self.tmpdir, stop_requested=stop_requested)
        self.assertEqual(rows, [])
        self.assertEqual(mock_gen.await_count, 0)
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())


# ============================================================
# TESTS ADAPTATION REJETS
# ============================================================

class TestAdaptiveRejectionCorrection(TempDirMixin, unittest.TestCase):
    def test_extract_rejection_failures_returns_expected_rules(self) -> None:
        metrics = valid_metrics()
        metrics["oblique_share"] = mut.MIN_OBLIQUE_SHARE - 0.05
        metrics["center_empty_ratio"] = mut.MAX_COYOTE_CENTER_EMPTY_RATIO + 0.04
        candidate = make_candidate(seed=777, ratios=invalid_ratios_far(), metrics=metrics)
        failures = mut.extract_rejection_failures(candidate, target_index=1, local_attempt=1)
        rules = {item["rule"] for item in failures}
        self.assertIn("ratio_coyote", rules)
        self.assertIn("ratio_olive", rules)
        self.assertIn("ratio_gris", rules)
        self.assertIn("oblique_share", rules)
        self.assertIn("center_empty_ratio", rules)
        self.assertIn("mean_abs_error", rules)

    def test_adaptive_state_builds_and_softens_hint(self) -> None:
        state = mut.AdaptiveGenerationState()
        failures = [
            {"rule": "ratio_olive", "actual": 0.20, "target": None, "min_value": 0.24, "max_value": 0.33, "delta": 0.04},
            {"rule": "oblique_share", "actual": 0.40, "target": None, "min_value": mut.MIN_OBLIQUE_SHARE, "max_value": None, "delta": mut.MIN_OBLIQUE_SHARE - 0.40},
            {"rule": "center_empty_ratio", "actual": 0.61, "target": None, "min_value": None, "max_value": mut.MAX_COYOTE_CENTER_EMPTY_RATIO, "delta": 0.10},
        ]
        state.register_failures(failures)
        hint_before = state.to_hint()
        self.assertGreater(state.consecutive_rejections, 0)
        self.assertGreater(hint_before["olive_macro_target_scale"], 1.0)
        self.assertGreater(hint_before["prefer_oblique"], 0.0)
        self.assertLess(hint_before["center_torso_overlap_scale"], 1.0)
        state.register_success()
        hint_after = state.to_hint()
        self.assertEqual(state.consecutive_rejections, 0)
        self.assertLess(hint_after["olive_macro_target_scale"], hint_before["olive_macro_target_scale"])
        self.assertLessEqual(hint_after["prefer_oblique"], hint_before["prefer_oblique"])


# ============================================================
# TESTS CONSTANTES
# ============================================================

class TestConstantsAndShapes(unittest.TestCase):
    def test_rgb_has_four_colors(self) -> None:
        self.assertEqual(mut.RGB.shape, (4, 3))

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


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_main_corrected.py ==========")
    unittest.main(verbosity=2)
