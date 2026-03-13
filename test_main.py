# -*- coding: utf-8 -*-
"""
test_main.py
Suite de tests unitaires avancée et renforcée pour main.py

Objectifs :
- couvrir les fonctions utilitaires pures ;
- couvrir les fonctions géométriques / morphologiques ;
- couvrir la validation métier des variantes ;
- couvrir les exports PNG / CSV ;
- couvrir les orchestrateurs sync et async ;
- fournir des logs lisibles, précis et exploitables ;
- rendre les diagnostics d'échec plus rapides à comprendre ;
- accélérer les tests async en exécutant en parallèle
  les opérations indépendantes (I/O, wrappers async, batchs).

Exécution :
    python -m unittest -v test_main.py
"""

from __future__ import annotations

import asyncio
import csv
import logging
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image

import main as mut


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = Path(__file__).resolve().parent / "logs_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_main.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_main")
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
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_")
        self.tmpdir = Path(self._tmpdir_obj.name)
        LOGGER.info("Répertoire temporaire créé : %s", self.tmpdir)

    def tearDown(self) -> None:
        LOGGER.info("Nettoyage du répertoire temporaire : %s", self.tmpdir)
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


class AsyncTempDirMixin:
    async def asyncSetUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_async_")
        self.tmpdir = Path(self._tmpdir_obj.name)
        LOGGER.info("Répertoire temporaire async créé : %s", self.tmpdir)

    async def asyncTearDown(self) -> None:
        LOGGER.info("Nettoyage du répertoire temporaire async : %s", self.tmpdir)
        self._tmpdir_obj.cleanup()


def valid_metrics() -> Dict[str, float]:
    return {
        "largest_olive_component_ratio": max(mut.MIN_OLIVE_CONNECTED_COMPONENT_RATIO + 0.05, 0.25),
        "largest_olive_component_ratio_small": 0.20,
        "olive_multizone_share": max(mut.MIN_OLIVE_MULTIZONE_SHARE + 0.10, 0.60),
        "center_empty_ratio": min(mut.MAX_COYOTE_CENTER_EMPTY_RATIO - 0.10, 0.35),
        "center_empty_ratio_small": min(mut.MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL - 0.10, 0.35),
        "boundary_density": (mut.MIN_BOUNDARY_DENSITY + mut.MAX_BOUNDARY_DENSITY) / 2.0,
        "boundary_density_small": (mut.MIN_BOUNDARY_DENSITY_SMALL + mut.MAX_BOUNDARY_DENSITY_SMALL) / 2.0,
        "boundary_density_tiny": 0.12,
        "mirror_similarity": min(mut.MAX_MIRROR_SIMILARITY - 0.10, 0.50),
        "oblique_share": max(mut.MIN_OBLIQUE_SHARE + 0.10, 0.70),
        "vertical_share": min(max(mut.MIN_VERTICAL_SHARE + 0.05, 0.12), mut.MAX_VERTICAL_SHARE - 0.05),
        "angle_dominance_ratio": min(mut.MAX_ANGLE_DOMINANCE_RATIO - 0.05, 0.20),

        "coyote_brown_macro_share": 0.00,
        "coyote_brown_transition_share": 0.05,
        "coyote_brown_micro_share": 0.00,

        "vert_olive_macro_share": max(mut.MIN_VISIBLE_OLIVE_MACRO_SHARE + 0.10, 0.72),
        "vert_olive_transition_share": 0.10,
        "vert_olive_micro_share": 0.05,

        "terre_de_france_macro_share": 0.20,
        "terre_de_france_transition_share": max(mut.MIN_VISIBLE_TERRE_TRANS_SHARE + 0.10, 0.50),
        "terre_de_france_micro_share": 0.05,

        "vert_de_gris_macro_share": min(mut.MAX_VISIBLE_GRIS_MACRO_SHARE - 0.05, 0.08),
        "vert_de_gris_transition_share": 0.05,
        "vert_de_gris_micro_share": max(mut.MIN_VISIBLE_GRIS_MICRO_SHARE + 0.10, 0.82),
    }


def valid_metrics_fixed() -> Dict[str, float]:
    """
    Valeurs métier fixes, non dérivées des constantes.
    Sert à éviter qu'un changement involontaire de seuil dans main.py
    fasse passer les tests "par suivisme".
    """
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
        "vert_de_gris_macro_share": 0.07,
        "vert_de_gris_transition_share": 0.05,
        "vert_de_gris_micro_share": 0.83,
    }


def valid_ratios() -> np.ndarray:
    return np.array([0.32, 0.28, 0.22, 0.18], dtype=float)


def invalid_ratios_far() -> np.ndarray:
    return np.array([0.50, 0.20, 0.20, 0.10], dtype=float)


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


async def gather_concurrent(*aws):
    return await asyncio.gather(*aws)


# ============================================================
# TESTS UTILITAIRES PURS
# ============================================================

class TestPureUtilities(TempDirMixin, TestAssertionsMixin, unittest.TestCase):
    def test_ensure_output_dir_creates_directory(self) -> None:
        path = self.tmpdir / "a" / "b" / "c"
        out = mut.ensure_output_dir(path)
        LOGGER.info("ensure_output_dir -> %s", out)
        self.assertEqual(out, path)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())

    def test_build_seed_is_deterministic(self) -> None:
        s1 = mut.build_seed(target_index=3, local_attempt=7, base_seed=1000)
        s2 = mut.build_seed(target_index=3, local_attempt=7, base_seed=1000)
        s3 = mut.build_seed(target_index=4, local_attempt=7, base_seed=1000)
        LOGGER.info("Seeds calculés : s1=%s s2=%s s3=%s", s1, s2, s3)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_build_seed_progression(self) -> None:
        base = 1000
        seeds = [mut.build_seed(1, i, base) for i in range(1, 5)]
        self.assertEqual(
            seeds,
            [
                mut.build_seed(1, 1, base),
                mut.build_seed(1, 2, base),
                mut.build_seed(1, 3, base),
                mut.build_seed(1, 4, base),
            ],
        )

    def test_make_profile_is_deterministic_for_same_seed(self) -> None:
        p1 = mut.make_profile(424242)
        p2 = mut.make_profile(424242)
        LOGGER.info("Profil seed=%s -> angles=%s", p1.seed, p1.allowed_angles)
        self.assertEqual(p1.allowed_angles, p2.allowed_angles)
        self.assertEqual(p1.micro_cluster_max, p2.micro_cluster_max)
        self.assertEqual(p1.seed, p2.seed)
        self.assertIn(0, p1.allowed_angles)
        self.assertEqual(sorted(set(p1.allowed_angles)), p1.allowed_angles)

    def test_make_profile_ranges(self) -> None:
        p = mut.make_profile(111)
        self.assertGreaterEqual(p.micro_cluster_min, 2)
        self.assertTrue(4 <= p.micro_cluster_max <= 5)
        self.assertTrue(0.22 <= p.macro_width_variation <= 0.30)
        self.assertTrue(0.14 <= p.macro_lateral_jitter <= 0.21)
        self.assertTrue(0.34 <= p.macro_tip_taper <= 0.43)
        self.assertTrue(0.10 <= p.macro_edge_break <= 0.15)
        self.assertTrue(0.18 <= p.micro_width_variation <= 0.25)
        self.assertTrue(0.12 <= p.micro_lateral_jitter <= 0.18)
        self.assertTrue(0.42 <= p.micro_tip_taper <= 0.52)
        self.assertTrue(0.12 <= p.micro_edge_break <= 0.18)

    def test_cm_to_px_returns_at_least_one(self) -> None:
        self.assertEqual(mut.cm_to_px(0.0), 1)
        self.assertGreaterEqual(mut.cm_to_px(0.01), 1)
        self.assertEqual(mut.cm_to_px(10.0), int(round(10.0 * mut.PX_PER_CM)))

    def test_compute_ratios_sums_to_one(self) -> None:
        canvas = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        ratios = mut.compute_ratios(canvas)
        LOGGER.info("Ratios calculés : %s", ratios)
        self.assertFloatClose(float(np.sum(ratios)), 1.0)
        self.assertArrayClose(ratios, np.array([0.25, 0.25, 0.25, 0.25]))

    def test_compute_ratios_on_uniform_canvas(self) -> None:
        canvas = np.full((10, 10), mut.IDX_TERRE, dtype=np.uint8)
        ratios = mut.compute_ratios(canvas)
        self.assertArrayClose(ratios, np.array([0.0, 0.0, 1.0, 0.0]))

    def test_render_canvas_returns_pil_image(self) -> None:
        canvas = np.zeros((5, 7), dtype=np.uint8)
        img = mut.render_canvas(canvas)
        self.assertImageSize(img, (7, 5))

    def test_render_canvas_uses_rgb_palette(self) -> None:
        canvas = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        img = mut.render_canvas(canvas)
        arr = np.array(img)
        self.assertArrayClose(arr[0], mut.RGB)

    def test_rotate_90_deg(self) -> None:
        x, y = mut.rotate(1.0, 0.0, 90.0)
        LOGGER.info("Rotation 90° -> (%.6f, %.6f)", x, y)
        self.assertFloatClose(x, 0.0, places=6)
        self.assertFloatClose(y, 1.0, places=6)

    def test_rotate_180_deg(self) -> None:
        x, y = mut.rotate(2.0, 3.0, 180.0)
        self.assertFloatClose(x, -2.0, places=6)
        self.assertFloatClose(y, -3.0, places=6)

    def test_choose_biased_center_is_in_bounds(self) -> None:
        import random
        rng = random.Random(123)
        for _ in range(50):
            x, y = mut.choose_biased_center(rng)
            self.assertGreaterEqual(x, 60)
            self.assertLessEqual(x, mut.WIDTH - 60)
            self.assertGreaterEqual(y, 60)
            self.assertLessEqual(y, mut.HEIGHT - 60)

    def test_polygon_mask_non_empty(self) -> None:
        poly = [(100, 100), (120, 100), (120, 130), (100, 130)]
        mask = mut.polygon_mask(poly)
        LOGGER.info("polygon_mask area=%s", int(mask.sum()))
        self.assertEqual(mask.shape, (mut.HEIGHT, mut.WIDTH))
        self.assertGreater(int(mask.sum()), 0)

    def test_compute_boundary_mask_detects_changes(self) -> None:
        canvas = np.zeros((6, 6), dtype=np.uint8)
        canvas[:, 3:] = 1
        boundary = mut.compute_boundary_mask(canvas)
        self.assertTrue(boundary.any())

    def test_compute_boundary_mask_uniform_is_false_everywhere(self) -> None:
        canvas = np.zeros((6, 6), dtype=np.uint8)
        boundary = mut.compute_boundary_mask(canvas)
        self.assertFalse(boundary.any())

    def test_dilate_mask_expands_true_area(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        dilated = mut.dilate_mask(mask, radius=1)
        self.assertGreater(int(dilated.sum()), int(mask.sum()))
        self.assertTrue(dilated[3, 3])

    def test_local_color_variety_counts_unique_values(self) -> None:
        canvas = np.array(
            [
                [0, 0, 1],
                [0, 2, 1],
                [3, 2, 1],
            ],
            dtype=np.uint8,
        )
        v = mut.local_color_variety(canvas, 1, 1, radius=1)
        self.assertEqual(v, 4)

    def test_downsample_nearest(self) -> None:
        canvas = np.arange(16, dtype=np.uint8).reshape(4, 4)
        ds = mut.downsample_nearest(canvas, factor=2)
        self.assertArrayClose(ds, np.array([[0, 2], [8, 10]], dtype=np.uint8))

    def test_boundary_density(self) -> None:
        canvas = np.zeros((10, 10), dtype=np.uint8)
        d0 = mut.boundary_density(canvas)
        canvas[:, 5:] = 1
        d1 = mut.boundary_density(canvas)
        LOGGER.info("Boundary density uniforme=%.4f | séparée=%.4f", d0, d1)
        self.assertEqual(d0, 0.0)
        self.assertGreater(d1, 0.0)

    def test_mirror_similarity_score(self) -> None:
        canvas = np.array(
            [
                [0, 1, 1, 0],
                [2, 3, 3, 2],
            ],
            dtype=np.uint8,
        )
        score = mut.mirror_similarity_score(canvas)
        self.assertFloatClose(score, 1.0)

    def test_mirror_similarity_score_non_symmetric(self) -> None:
        canvas = np.array(
            [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.uint8,
        )
        score = mut.mirror_similarity_score(canvas)
        self.assertLess(score, 1.0)

    def test_infer_origin_from_neighbors(self) -> None:
        canvas = np.array(
            [
                [0, 1, 1],
                [0, 1, 2],
                [3, 1, 2],
            ],
            dtype=np.uint8,
        )
        origin_map = np.array(
            [
                [0, 1, 1],
                [0, 1, 2],
                [3, 1, 2],
            ],
            dtype=np.uint8,
        )
        origin = mut.infer_origin_from_neighbors(canvas, origin_map, 1, 1, chosen_color=1, fallback_origin=0)
        self.assertEqual(origin, 1)

    def test_infer_origin_from_neighbors_fallback(self) -> None:
        canvas = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        origin_map = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        origin = mut.infer_origin_from_neighbors(canvas, origin_map, 0, 0, chosen_color=3, fallback_origin=2)
        self.assertEqual(origin, 2)


# ============================================================
# TESTS ZONES / MORPHOLOGIE / ORIGINES
# ============================================================

class TestMorphologyAndZones(TestAssertionsMixin, unittest.TestCase):
    def test_rect_mask_shape_and_population(self) -> None:
        mask = mut.rect_mask(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(mask.shape, (mut.HEIGHT, mut.WIDTH))
        self.assertTrue(mask.any())

    def test_anatomy_zone_masks_contains_expected_keys(self) -> None:
        zones = mut.anatomy_zone_masks()
        expected = {
            "left_shoulder",
            "right_shoulder",
            "left_flank",
            "right_flank",
            "left_thigh",
            "right_thigh",
            "center_torso",
        }
        self.assertEqual(set(zones.keys()), expected)

    def test_macro_zone_count(self) -> None:
        mask = np.zeros((mut.HEIGHT, mut.WIDTH), dtype=bool)
        mask |= mut.ANATOMY_ZONES["left_shoulder"]
        mask |= mut.ANATOMY_ZONES["left_flank"]
        count = mut.macro_zone_count(mask)
        LOGGER.info("macro_zone_count=%s", count)
        self.assertGreaterEqual(count, 2)

    def test_center_empty_ratio(self) -> None:
        canvas = np.full((mut.HEIGHT, mut.WIDTH), mut.IDX_COYOTE, dtype=np.uint8)
        ratio = mut.center_empty_ratio(canvas)
        self.assertEqual(ratio, 1.0)

    def test_largest_component_ratio(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[3:6, 3:6] = True
        ratio = mut.largest_component_ratio(mask)
        self.assertFloatClose(ratio, 9 / 13)

    def test_largest_component_ratio_empty(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        ratio = mut.largest_component_ratio(mask)
        self.assertEqual(ratio, 0.0)

    def test_orientation_score_empty(self) -> None:
        out = mut.orientation_score([])
        self.assertEqual(out["oblique_share"], 0.0)
        self.assertEqual(out["vertical_share"], 0.0)
        self.assertEqual(out["dominance_ratio"], 1.0)

    def test_orientation_score_non_empty(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], -20, (10, 10), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 0, (20, 20), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_TERRE, [], 25, (30, 30), dummy_mask, 2),
        ]
        out = mut.orientation_score(macros)
        LOGGER.info("orientation_score=%s", out)
        self.assertFloatClose(out["oblique_share"], 2 / 3)
        self.assertFloatClose(out["vertical_share"], 1 / 3)
        self.assertFloatClose(out["dominance_ratio"], 1 / 3)

    def test_visible_origin_shares(self) -> None:
        canvas = np.array(
            [
                [0, 1, 1, 2],
                [0, 1, 3, 3],
            ],
            dtype=np.uint8,
        )
        origin_map = np.array(
            [
                [0, 1, 3, 2],
                [0, 1, 3, 3],
            ],
            dtype=np.uint8,
        )
        out = mut.visible_origin_shares(canvas, origin_map)
        LOGGER.info("visible_origin_shares=%s", out)

        self.assertIn("vert_olive_macro_share", out)
        self.assertIn("terre_de_france_transition_share", out)
        self.assertIn("vert_de_gris_micro_share", out)

        self.assertFloatClose(out["vert_olive_macro_share"], 2 / 3)
        self.assertFloatClose(out["vert_olive_micro_share"], 1 / 3)

    def test_visible_origin_shares_empty_color(self) -> None:
        canvas = np.full((5, 5), mut.IDX_COYOTE, dtype=np.uint8)
        origin_map = np.zeros((5, 5), dtype=np.uint8)
        out = mut.visible_origin_shares(canvas, origin_map)
        self.assertEqual(out["vert_olive_macro_share"], 0.0)
        self.assertEqual(out["terre_de_france_transition_share"], 0.0)
        self.assertEqual(out["vert_de_gris_micro_share"], 0.0)

    def test_center_empty_ratio_upscaled_proxy(self) -> None:
        small = np.full((20, 20), mut.IDX_COYOTE, dtype=np.uint8)
        ratio = mut.center_empty_ratio_upscaled_proxy(small)
        self.assertEqual(ratio, 1.0)

    def test_multiscale_metrics_keys(self) -> None:
        canvas = make_canvas_quadrants()
        out = mut.multiscale_metrics(canvas)
        expected = {
            "boundary_density_small",
            "boundary_density_tiny",
            "center_empty_ratio_small",
            "largest_olive_component_ratio_small",
        }
        self.assertEqual(set(out.keys()), expected)


# ============================================================
# TESTS FORMES / GÉOMÉTRIE
# ============================================================

class TestGeometryAndShapes(unittest.TestCase):
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
        self.assertIsInstance(poly, list)
        self.assertEqual(len(poly), 16)

    def test_attached_transition_returns_polygon(self) -> None:
        import random
        rng = random.Random(42)
        parent = [(10, 10), (50, 10), (50, 50), (10, 50)]
        poly = mut.attached_transition(rng, parent, length_px=20, width_px=10)
        self.assertEqual(len(poly), 5)
        self.assertTrue(all(len(p) == 2 for p in poly))


# ============================================================
# TESTS CONTRÔLES STRUCTURELS
# ============================================================

class TestStructuralControls(unittest.TestCase):
    def test_angle_distance_deg(self) -> None:
        self.assertEqual(mut.angle_distance_deg(10, 20), 10)
        self.assertEqual(mut.angle_distance_deg(-35, 35), 70)

    def test_local_parallel_conflict(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (100, 100), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 22, (150, 110), dummy_mask, 2),
        ]
        conflict = mut.local_parallel_conflict(
            macros,
            center=(130, 105),
            angle_deg=18,
            dist_threshold_px=100,
            angle_threshold_deg=8,
        )
        self.assertTrue(conflict)

    def test_local_parallel_conflict_false_when_far(self) -> None:
        dummy_mask = np.zeros((10, 10), dtype=bool)
        macros = [
            mut.MacroRecord(mut.IDX_OLIVE, [], 20, (100, 100), dummy_mask, 2),
            mut.MacroRecord(mut.IDX_OLIVE, [], 22, (600, 600), dummy_mask, 2),
        ]
        conflict = mut.local_parallel_conflict(
            macros,
            center=(130, 105),
            angle_deg=18,
            dist_threshold_px=100,
            angle_threshold_deg=8,
        )
        self.assertFalse(conflict)

    def test_transition_is_attached(self) -> None:
        parent = np.zeros((20, 20), dtype=bool)
        parent[5:10, 5:10] = True
        transition = np.zeros((20, 20), dtype=bool)
        transition[9:13, 8:12] = True
        self.assertTrue(mut.transition_is_attached(parent, transition, min_touch_pixels=1))

    def test_transition_is_not_attached(self) -> None:
        parent = np.zeros((20, 20), dtype=bool)
        parent[2:5, 2:5] = True
        transition = np.zeros((20, 20), dtype=bool)
        transition[15:18, 15:18] = True
        self.assertFalse(mut.transition_is_attached(parent, transition, min_touch_pixels=1))

    def test_micro_is_on_boundary(self) -> None:
        boundary = np.zeros((10, 10), dtype=bool)
        boundary[3:7, 3:7] = True
        micro = np.zeros((10, 10), dtype=bool)
        micro[4:6, 4:6] = True
        self.assertTrue(mut.micro_is_on_boundary(boundary, micro, min_boundary_coverage=0.5))

    def test_micro_is_not_on_boundary_when_empty(self) -> None:
        boundary = np.zeros((10, 10), dtype=bool)
        micro = np.zeros((10, 10), dtype=bool)
        self.assertFalse(mut.micro_is_on_boundary(boundary, micro))

    def test_creates_new_mass(self) -> None:
        canvas = np.zeros((100, 100), dtype=np.uint8)
        canvas[40:60, 40:60] = mut.IDX_OLIVE
        new_mask = np.zeros((100, 100), dtype=bool)
        new_mask[30:70, 30:70] = True
        result = mut.creates_new_mass(
            canvas=canvas,
            new_mask=new_mask,
            color_idx=mut.IDX_OLIVE,
            local_radius=20,
            max_local_area_ratio=0.20,
        )
        self.assertTrue(result)

    def test_creates_new_mass_false_when_small_and_sparse(self) -> None:
        canvas = np.zeros((100, 100), dtype=np.uint8)
        new_mask = np.zeros((100, 100), dtype=bool)
        new_mask[10:12, 10:12] = True
        result = mut.creates_new_mass(
            canvas=canvas,
            new_mask=new_mask,
            color_idx=mut.IDX_OLIVE,
            local_radius=10,
            max_local_area_ratio=0.50,
        )
        self.assertFalse(result)


# ============================================================
# TESTS COUCHES / APPLICATION DE MASQUES
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
        rng = random.Random(123)

        before_canvas = canvas.copy()
        before_origin = origin_map.copy()

        mut.nudge_proportions(canvas, origin_map, rng)

        np.testing.assert_array_equal(canvas, before_canvas)
        np.testing.assert_array_equal(origin_map, before_origin)


# ============================================================
# TESTS VALIDATION / EXPORT
# ============================================================

class TestValidationAndExport(TempDirMixin, TestAssertionsMixin, unittest.TestCase):
    def test_variant_is_valid_accepts_valid_candidate(self) -> None:
        rs = valid_ratios()
        metrics = valid_metrics()
        ok = mut.variant_is_valid(rs, metrics)
        LOGGER.info("variant_is_valid(valid) -> %s", ok)
        self.assertTrue(ok)

    def test_variant_is_valid_accepts_fixed_business_case(self) -> None:
        ok = mut.variant_is_valid(valid_ratios(), valid_metrics_fixed())
        LOGGER.info("variant_is_valid(fixed business case) -> %s", ok)
        self.assertTrue(ok)

    def test_variant_is_valid_rejects_bad_ratios(self) -> None:
        ok = mut.variant_is_valid(invalid_ratios_far(), valid_metrics())
        self.assertFalse(ok)

    def test_variant_is_valid_rejects_each_metric_threshold(self) -> None:
        rs = valid_ratios()

        for metric_name, bad_value in iter_metric_failure_cases():
            with self.subTest(metric=metric_name, bad_value=bad_value):
                metrics = valid_metrics()
                metrics[metric_name] = bad_value
                ok = mut.variant_is_valid(rs, metrics)
                LOGGER.info("Échec attendu | metric=%s | value=%s | ok=%s", metric_name, bad_value, ok)
                self.assertFalse(ok)

    def test_variant_is_valid_rejects_mean_error(self) -> None:
        rs = valid_ratios() + np.array([0.03, -0.03, 0.03, -0.03], dtype=float)
        ok = mut.variant_is_valid(rs, valid_metrics())
        self.assertFalse(ok)

    def test_validate_candidate_result_delegates_to_variant_is_valid(self) -> None:
        cand = make_candidate()
        self.assertTrue(mut.validate_candidate_result(cand))

    def test_save_candidate_image(self) -> None:
        cand = make_candidate()
        path = self.tmpdir / "img" / "camouflage_test.png"
        out = mut.save_candidate_image(cand, path)
        LOGGER.info("Image sauvegardée : %s", out)
        self.assertTrue(path.exists())
        self.assertEqual(out, path)

    def test_candidate_row_contains_expected_fields(self) -> None:
        cand = make_candidate(seed=999)
        row = mut.candidate_row(
            target_index=1,
            local_attempt=2,
            global_attempt=3,
            candidate=cand,
        )
        LOGGER.info("candidate_row=%s", row)

        required_keys = {
            "index",
            "seed",
            "attempts_for_this_image",
            "global_attempt",
            "coyote_brown_pct",
            "vert_olive_pct",
            "terre_de_france_pct",
            "vert_de_gris_pct",
            "largest_olive_component_ratio",
            "largest_olive_component_ratio_small",
            "olive_multizone_share",
            "center_empty_ratio",
            "center_empty_ratio_small",
            "boundary_density",
            "boundary_density_small",
            "boundary_density_tiny",
            "mirror_similarity",
            "oblique_share",
            "vertical_share",
            "angle_dominance_ratio",
            "olive_macro_share",
            "terre_transition_share",
            "gris_micro_share",
            "gris_macro_share",
            "angles",
        }
        self.assertTrue(required_keys.issubset(row.keys()))
        self.assertEqual(row["seed"], 999)
        self.assertEqual(row["index"], 1)

    def test_candidate_row_rounding(self) -> None:
        cand = make_candidate(seed=42)
        row = mut.candidate_row(1, 1, 1, cand)
        self.assertIsInstance(row["largest_olive_component_ratio"], float)
        self.assertIsInstance(row["coyote_brown_pct"], float)

    def test_write_report_with_rows(self) -> None:
        rows = [
            mut.candidate_row(1, 1, 1, make_candidate(seed=111)),
            mut.candidate_row(2, 1, 2, make_candidate(seed=222)),
        ]
        csv_path = mut.write_report(rows, self.tmpdir, filename="rapport.csv")
        LOGGER.info("CSV écrit : %s", csv_path)

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
# TESTS VALIDATION / EXPORT ASYNC CONCURRENTS
# ============================================================

class TestValidationAndExportAsync(AsyncTempDirMixin, TestAssertionsMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_save_candidate_image(self) -> None:
        path = self.tmpdir / "img" / "camouflage_async.png"
        cand = make_candidate()
        out = await mut.async_save_candidate_image(cand, path)
        self.assertTrue(out.exists())

    async def test_async_write_report(self) -> None:
        rows = [
            mut.candidate_row(1, 1, 1, make_candidate(seed=1)),
            mut.candidate_row(2, 1, 2, make_candidate(seed=2)),
        ]
        path = await mut.async_write_report(rows, self.tmpdir, filename="async_report.csv")
        self.assertTrue(path.exists())

        with path.open("r", encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["seed"], "1")
        self.assertEqual(data[1]["seed"], "2")

    async def test_async_save_multiple_candidate_images_concurrently(self) -> None:
        candidates = [make_candidate(seed=i) for i in range(10, 14)]
        paths = [self.tmpdir / "batch" / f"camouflage_{i}.png" for i in range(10, 14)]

        results = await gather_concurrent(
            *(mut.async_save_candidate_image(c, p) for c, p in zip(candidates, paths))
        )

        self.assertEqual(len(results), 4)
        for p in paths:
            self.assertTrue(p.exists(), f"Fichier absent: {p}")

    async def test_async_write_multiple_reports_concurrently(self) -> None:
        rows1 = [mut.candidate_row(1, 1, 1, make_candidate(seed=101))]
        rows2 = [mut.candidate_row(2, 1, 2, make_candidate(seed=202))]
        rows3 = [mut.candidate_row(3, 1, 3, make_candidate(seed=303))]

        p1 = self.tmpdir / "r1"
        p2 = self.tmpdir / "r2"
        p3 = self.tmpdir / "r3"

        out1, out2, out3 = await gather_concurrent(
            mut.async_write_report(rows1, p1, filename="a.csv"),
            mut.async_write_report(rows2, p2, filename="b.csv"),
            mut.async_write_report(rows3, p3, filename="c.csv"),
        )

        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())
        self.assertTrue(out3.exists())

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

        LOGGER.info("Temps batch async simulé = %.4fs", elapsed)
        self.assertLess(elapsed, 0.30)


# ============================================================
# TESTS GÉNÉRATION D'UNE VARIANTE / CANDIDAT
# ============================================================

class TestCandidateGeneration(TestAssertionsMixin, unittest.TestCase):
    def test_generate_one_variant_structure(self) -> None:
        profile = mut.make_profile(mut.DEFAULT_BASE_SEED)
        fake_image = Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0))
        fake_ratios = valid_ratios()
        fake_metrics = valid_metrics()

        with patch.object(mut, "add_macros"), \
             patch.object(mut, "add_transitions"), \
             patch.object(mut, "add_micro_clusters"), \
             patch.object(mut, "nudge_proportions"), \
             patch.object(mut, "render_canvas", return_value=fake_image), \
             patch.object(mut, "compute_ratios", return_value=fake_ratios), \
             patch.object(mut, "largest_component_ratio", return_value=fake_metrics["largest_olive_component_ratio"]), \
             patch.object(mut, "center_empty_ratio", return_value=fake_metrics["center_empty_ratio"]), \
             patch.object(mut, "boundary_density", side_effect=[fake_metrics["boundary_density"], fake_metrics["boundary_density_small"], fake_metrics["boundary_density_tiny"]]), \
             patch.object(mut, "mirror_similarity_score", return_value=fake_metrics["mirror_similarity"]), \
             patch.object(mut, "macro_zone_count", return_value=3), \
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
             }):
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

        LOGGER.info("Candidate généré : %s", summarize_candidate(cand))
        self.assertCandidateLooksConsistent(cand)
        self.assertEqual(cand.seed, seed)
        self.assertEqual(cand.profile.allowed_angles, profile.allowed_angles)
        mock_generate.assert_called_once()

    def test_validate_candidate_result_false_candidate(self) -> None:
        cand = make_candidate(ratios=invalid_ratios_far())
        self.assertFalse(mut.validate_candidate_result(cand))


class TestCandidateGenerationAsync(TestAssertionsMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        fake_candidate = make_candidate(seed=mut.DEFAULT_BASE_SEED + 1)
        with patch.object(mut, "generate_candidate_from_seed", return_value=fake_candidate) as mock_generate:
            cand = await mut.async_generate_candidate_from_seed(mut.DEFAULT_BASE_SEED + 1)
        LOGGER.info("Async candidate généré : %s", summarize_candidate(cand))
        self.assertCandidateLooksConsistent(cand)
        mock_generate.assert_called_once()

    async def test_async_validate_candidate_result(self) -> None:
        cand = make_candidate()
        ok = await mut.async_validate_candidate_result(cand)
        self.assertTrue(ok)

    async def test_async_generate_multiple_candidates_concurrently(self) -> None:
        async def fake_generate(seed: int) -> mut.CandidateResult:
            await asyncio.sleep(0.01)
            return make_candidate(seed=seed)

        seeds = [mut.DEFAULT_BASE_SEED + i for i in range(4)]
        with patch.object(mut, "async_generate_candidate_from_seed", side_effect=fake_generate):
            candidates = await gather_concurrent(*(mut.async_generate_candidate_from_seed(seed) for seed in seeds))

        self.assertEqual(len(candidates), 4)
        for seed, cand in zip(seeds, candidates):
            self.assertCandidateLooksConsistent(cand)
            self.assertEqual(cand.seed, seed)

    async def test_async_generate_multiple_candidates_parallel_mocked(self) -> None:
        async def fake_generate(seed: int) -> mut.CandidateResult:
            await asyncio.sleep(0.10)
            return make_candidate(seed=seed)

        seeds = [1001, 1002, 1003, 1004]
        started = time.perf_counter()

        with patch.object(mut, "async_generate_candidate_from_seed", side_effect=fake_generate):
            candidates = await gather_concurrent(*(mut.async_generate_candidate_from_seed(seed) for seed in seeds))

        elapsed = time.perf_counter() - started
        LOGGER.info("Temps génération async simulée = %.4fs", elapsed)

        self.assertEqual([c.seed for c in candidates], seeds)
        self.assertLess(elapsed, 0.30)


# ============================================================
# TESTS ORCHESTRATEUR SYNCHRONE
# ============================================================

class TestGenerateAllSync(TempDirMixin, unittest.TestCase):
    def test_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=101)
        progress_calls: List[tuple] = []

        def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted: bool) -> None:
            LOGGER.info(
                "Progress sync | idx=%s local=%s total=%s accepted=%s seed=%s",
                target_index, local_attempt, total_attempts, accepted, cand.seed
            )
            progress_calls.append((target_index, local_attempt, total_attempts, target_count, cand.seed, accepted))

        with patch.object(mut, "generate_candidate_from_seed", return_value=candidate) as mock_gen, \
             patch.object(mut, "validate_candidate_result", return_value=True) as mock_val:

            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=999,
                progress_callback=progress_cb,
                parallel_attempts=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.call_count, 1)
        self.assertEqual(mock_val.call_count, 1)
        self.assertEqual(len(progress_calls), 1)
        self.assertTrue(progress_calls[0][-1])
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    def test_generate_all_multiple_targets(self) -> None:
        candidates = [make_candidate(seed=11), make_candidate(seed=22)]

        with patch.object(mut, "generate_candidate_from_seed", side_effect=candidates), \
             patch.object(mut, "validate_candidate_result", return_value=True):

            rows = mut.generate_all(target_count=2, output_dir=self.tmpdir, base_seed=555, parallel_attempts=False)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["seed"], 11)
        self.assertEqual(rows[1]["seed"], 22)
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "camouflage_002.png").exists())

    def test_generate_all_retries_until_accept(self) -> None:
        rejected = make_candidate(seed=201)
        accepted = make_candidate(seed=202)
        progress_calls: List[tuple] = []

        def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted_flag: bool) -> None:
            progress_calls.append((target_index, local_attempt, total_attempts, accepted_flag, cand.seed))

        with patch.object(mut, "generate_candidate_from_seed", side_effect=[rejected, accepted]) as mock_gen, \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]) as mock_val:

            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=123,
                progress_callback=progress_cb,
                parallel_attempts=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.call_count, 2)
        self.assertEqual(mock_val.call_count, 2)
        self.assertEqual(len(progress_calls), 2)
        self.assertFalse(progress_calls[0][3])
        self.assertTrue(progress_calls[1][3])
        self.assertEqual(rows[0]["seed"], 202)

    def test_generate_all_stop_requested_writes_partial_report(self) -> None:
        def stop_requested() -> bool:
            return True

        with patch.object(mut, "generate_candidate_from_seed") as mock_gen:
            rows = mut.generate_all(
                target_count=3,
                output_dir=self.tmpdir,
                stop_requested=stop_requested,
            )

        self.assertEqual(rows, [])
        self.assertEqual(mock_gen.call_count, 0)
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    def test_generate_all_builds_expected_seed_sequence(self) -> None:
        seen_seeds: List[int] = []

        def fake_generate(seed: int) -> mut.CandidateResult:
            seen_seeds.append(seed)
            return make_candidate(seed=seed)

        with patch.object(mut, "generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]):

            rows = mut.generate_all(target_count=1, output_dir=self.tmpdir, base_seed=1000, parallel_attempts=False)

        self.assertEqual(len(rows), 1)
        self.assertEqual(seen_seeds, [mut.build_seed(1, 1, 1000), mut.build_seed(1, 2, 1000)])

    def test_generate_all_progress_callback_receives_target_count(self) -> None:
        observed: List[int] = []

        def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted: bool) -> None:
            observed.append(target_count)

        with patch.object(mut, "generate_candidate_from_seed", return_value=make_candidate(seed=9)), \
             patch.object(mut, "validate_candidate_result", return_value=True):

            mut.generate_all(target_count=3, output_dir=self.tmpdir, progress_callback=progress_cb, parallel_attempts=False)

        self.assertEqual(observed, [3, 3, 3])


# ============================================================
# TESTS ORCHESTRATEUR ASYNCHRONE
# ============================================================

class TestGenerateAllAsync(AsyncTempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=301)
        progress_calls: List[tuple] = []

        async def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted: bool) -> None:
            progress_calls.append((target_index, local_attempt, total_attempts, target_count, cand.seed, accepted))

        with patch.object(mut, "async_generate_candidate_from_seed", AsyncMock(return_value=candidate)) as mock_gen, \
             patch.object(mut, "async_validate_candidate_result", AsyncMock(return_value=True)) as mock_val:

            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=5000,
                progress_callback=progress_cb,
                parallel_attempts=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.await_count, 1)
        self.assertEqual(mock_val.await_count, 1)
        self.assertEqual(len(progress_calls), 1)
        self.assertTrue(progress_calls[0][-1])
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    async def test_async_generate_all_retries_until_accept(self) -> None:
        rejected = make_candidate(seed=401)
        accepted = make_candidate(seed=402)
        order: List[str] = []

        async def fake_generate(seed: int) -> mut.CandidateResult:
            order.append(f"gen:{seed}")
            return rejected if len(order) == 1 else accepted

        async def fake_validate(candidate: mut.CandidateResult) -> bool:
            order.append(f"val:{candidate.seed}")
            return candidate.seed == 402

        async def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted_flag: bool) -> None:
            order.append(f"cb:{cand.seed}:{accepted_flag}")

        with patch.object(mut, "async_generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(mut, "async_validate_candidate_result", side_effect=fake_validate):

            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                progress_callback=progress_cb,
                parallel_attempts=False,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 402)
        self.assertEqual(
            order,
            [
                f"gen:{mut.build_seed(1, 1, mut.DEFAULT_BASE_SEED)}",
                "val:401",
                "cb:401:False",
                f"gen:{mut.build_seed(1, 2, mut.DEFAULT_BASE_SEED)}",
                "val:402",
                "cb:402:True",
            ],
        )

    async def test_async_generate_all_stop_requested(self) -> None:
        async def stop_requested() -> bool:
            return True

        with patch.object(mut, "async_generate_candidate_from_seed", AsyncMock()) as mock_gen:
            rows = await mut.async_generate_all(
                target_count=5,
                output_dir=self.tmpdir,
                stop_requested=stop_requested,
            )

        self.assertEqual(rows, [])
        self.assertEqual(mock_gen.await_count, 0)
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    async def test_async_generate_all_is_strictly_sequential(self) -> None:
        history: List[str] = []

        async def fake_generate(seed: int) -> mut.CandidateResult:
            history.append(f"generate:{seed}")
            if seed == mut.build_seed(1, 1, mut.DEFAULT_BASE_SEED):
                return make_candidate(seed=1001)
            if seed == mut.build_seed(1, 2, mut.DEFAULT_BASE_SEED):
                return make_candidate(seed=1002)
            if seed == mut.build_seed(2, 1, mut.DEFAULT_BASE_SEED):
                return make_candidate(seed=2001)
            raise AssertionError(f"Seed inattendu: {seed}")

        async def fake_validate(candidate: mut.CandidateResult) -> bool:
            history.append(f"validate:{candidate.seed}")
            return candidate.seed in {1002, 2001}

        with patch.object(mut, "async_generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(mut, "async_validate_candidate_result", side_effect=fake_validate):

            rows = await mut.async_generate_all(
                target_count=2,
                output_dir=self.tmpdir,
                parallel_attempts=False,
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            history,
            [
                f"generate:{mut.build_seed(1, 1, mut.DEFAULT_BASE_SEED)}",
                "validate:1001",
                f"generate:{mut.build_seed(1, 2, mut.DEFAULT_BASE_SEED)}",
                "validate:1002",
                f"generate:{mut.build_seed(2, 1, mut.DEFAULT_BASE_SEED)}",
                "validate:2001",
            ],
        )

    async def test_async_generate_all_multiple_targets_first_try(self) -> None:
        candidates = [make_candidate(seed=501), make_candidate(seed=502)]

        with patch.object(mut, "async_generate_candidate_from_seed", AsyncMock(side_effect=candidates)), \
             patch.object(mut, "async_validate_candidate_result", AsyncMock(return_value=True)):

            rows = await mut.async_generate_all(target_count=2, output_dir=self.tmpdir, parallel_attempts=False)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["seed"], 501)
        self.assertEqual(rows[1]["seed"], 502)


# ============================================================
# TESTS ROBUSTESSE CONSTANTES / STRUCTURES
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
        self.assertEqual(mut.IDX_COYOTE, 0)
        self.assertEqual(mut.IDX_OLIVE, 1)
        self.assertEqual(mut.IDX_TERRE, 2)
        self.assertEqual(mut.IDX_GRIS, 3)

    def test_origin_constants_alignment(self) -> None:
        self.assertEqual(mut.ORIGIN_BACKGROUND, 0)
        self.assertEqual(mut.ORIGIN_MACRO, 1)
        self.assertEqual(mut.ORIGIN_TRANSITION, 2)
        self.assertEqual(mut.ORIGIN_MICRO, 3)


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_main.py ==========")
    unittest.main(verbosity=2)
