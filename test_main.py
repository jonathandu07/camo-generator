# -*- coding: utf-8 -*-
"""
test_main.py
Suite de tests unitaires avancée pour main.py

Objectifs :
- couvrir les fonctions utilitaires pures ;
- couvrir la validation métier des variantes ;
- couvrir les exports PNG / CSV ;
- couvrir les orchestrateurs sync et async ;
- fournir des logs lisibles, précis et exploitables.

Exécution :
    python -m unittest -v test_main.py

Pré-requis :
- ce fichier doit pouvoir importer le module cible via : import main as mut
- adapter éventuellement cette ligne si ton fichier main.py est ailleurs.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

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

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = configure_logger()


# ============================================================
# AIDES
# ============================================================

def valid_metrics() -> Dict[str, float]:
    """
    Jeu de métriques entièrement valides vis-à-vis de variant_is_valid().
    """
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
        "olive_multizone_share": max(mut.MIN_OLIVE_MULTIZONE_SHARE + 0.10, 0.60),

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


def valid_ratios() -> np.ndarray:
    """
    Ratios parfaitement valides et proches de TARGET.
    """
    return np.array([0.32, 0.28, 0.22, 0.18], dtype=float)


def make_candidate(seed: int = 123456, ratios: np.ndarray | None = None, metrics: Dict[str, float] | None = None) -> mut.CandidateResult:
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
        f"boundary={candidate.metrics.get('boundary_density', 'n/a'):.4f} | "
        f"mirror={candidate.metrics.get('mirror_similarity', 'n/a'):.4f}"
    )


class TempDirMixin:
    def setUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_")
        self.tmpdir = Path(self._tmpdir_obj.name)
        LOGGER.info("Répertoire temporaire créé : %s", self.tmpdir)

    def tearDown(self) -> None:
        LOGGER.info("Nettoyage du répertoire temporaire : %s", self.tmpdir)
        self._tmpdir_obj.cleanup()


# ============================================================
# TESTS UTILITAIRES
# ============================================================

class TestPureUtilities(TempDirMixin, unittest.TestCase):
    def test_ensure_output_dir_creates_directory(self) -> None:
        path = self.tmpdir / "a" / "b" / "c"
        out = mut.ensure_output_dir(path)
        LOGGER.info("ensure_output_dir -> %s", out)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertEqual(out, path)

    def test_build_seed_is_deterministic(self) -> None:
        s1 = mut.build_seed(target_index=3, local_attempt=7, base_seed=1000)
        s2 = mut.build_seed(target_index=3, local_attempt=7, base_seed=1000)
        s3 = mut.build_seed(target_index=4, local_attempt=7, base_seed=1000)
        LOGGER.info("Seeds calculés : s1=%s s2=%s s3=%s", s1, s2, s3)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_make_profile_is_deterministic_for_same_seed(self) -> None:
        p1 = mut.make_profile(424242)
        p2 = mut.make_profile(424242)
        LOGGER.info("Profil seed=%s -> angles=%s", p1.seed, p1.allowed_angles)
        self.assertEqual(p1.allowed_angles, p2.allowed_angles)
        self.assertEqual(p1.micro_cluster_max, p2.micro_cluster_max)
        self.assertEqual(p1.seed, p2.seed)
        self.assertIn(0, p1.allowed_angles)
        self.assertTrue(8 <= len(p1.allowed_angles) <= len(set(mut.BASE_ANGLES + [0])))

    def test_cm_to_px_returns_at_least_one(self) -> None:
        self.assertEqual(mut.cm_to_px(0.0), 1)
        self.assertGreaterEqual(mut.cm_to_px(0.01), 1)
        self.assertEqual(mut.cm_to_px(10.0), int(round(10.0 * mut.PX_PER_CM)))

    def test_compute_ratios_sums_to_one(self) -> None:
        canvas = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        ratios = mut.compute_ratios(canvas)
        LOGGER.info("Ratios calculés : %s", ratios)
        self.assertAlmostEqual(float(np.sum(ratios)), 1.0, places=8)
        np.testing.assert_allclose(ratios, np.array([0.25, 0.25, 0.25, 0.25]))

    def test_render_canvas_returns_pil_image(self) -> None:
        canvas = np.zeros((5, 7), dtype=np.uint8)
        img = mut.render_canvas(canvas)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (7, 5))

    def test_rotate_90_deg(self) -> None:
        x, y = mut.rotate(1.0, 0.0, 90.0)
        LOGGER.info("Rotation 90° -> (%.6f, %.6f)", x, y)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 1.0, places=6)

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
        LOGGER.info("Boundary density locale=%.4f", float(np.mean(boundary)))
        self.assertTrue(boundary.any())

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
        np.testing.assert_array_equal(ds, np.array([[0, 2], [8, 10]], dtype=np.uint8))

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
        self.assertAlmostEqual(score, 1.0, places=8)

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


# ============================================================
# TESTS ZONES / MORPHOLOGIE / ORIGINES
# ============================================================

class TestMorphologyAndZones(unittest.TestCase):
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
        mask[0:2, 0:2] = True   # 4
        mask[3:6, 3:6] = True   # 9
        ratio = mut.largest_component_ratio(mask)
        self.assertAlmostEqual(ratio, 9 / 13, places=8)

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
        self.assertAlmostEqual(out["oblique_share"], 2 / 3, places=8)
        self.assertAlmostEqual(out["vertical_share"], 1 / 3, places=8)
        self.assertAlmostEqual(out["dominance_ratio"], 1 / 3, places=8)

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

        self.assertAlmostEqual(out["vert_olive_macro_share"], 2 / 3, places=8)
        self.assertAlmostEqual(out["vert_olive_micro_share"], 1 / 3, places=8)


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
        conflict = mut.local_parallel_conflict(macros, center=(130, 105), angle_deg=18, dist_threshold_px=100, angle_threshold_deg=8)
        self.assertTrue(conflict)

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

    def test_creates_new_mass(self) -> None:
        canvas = np.zeros((100, 100), dtype=np.uint8)
        canvas[40:60, 40:60] = mut.IDX_OLIVE

        new_mask = np.zeros((100, 100), dtype=bool)
        new_mask[35:65, 35:65] = True

        result = mut.creates_new_mass(
            canvas=canvas,
            new_mask=new_mask,
            color_idx=mut.IDX_OLIVE,
            local_radius=20,
            max_local_area_ratio=0.20,
        )
        self.assertTrue(result)


# ============================================================
# TESTS GÉNÉRATION / VALIDATION / EXPORT
# ============================================================

class TestValidationAndExport(TempDirMixin, unittest.TestCase):
    def test_variant_is_valid_accepts_valid_candidate(self) -> None:
        rs = valid_ratios()
        metrics = valid_metrics()
        ok = mut.variant_is_valid(rs, metrics)
        LOGGER.info("variant_is_valid(valid) -> %s", ok)
        self.assertTrue(ok)

    def test_variant_is_valid_rejects_bad_ratios(self) -> None:
        rs = np.array([0.50, 0.20, 0.20, 0.10], dtype=float)
        metrics = valid_metrics()
        ok = mut.variant_is_valid(rs, metrics)
        LOGGER.info("variant_is_valid(bad_ratios) -> %s", ok)
        self.assertFalse(ok)

    def test_variant_is_valid_rejects_bad_metrics(self) -> None:
        rs = valid_ratios()
        metrics = valid_metrics()
        metrics["mirror_similarity"] = mut.MAX_MIRROR_SIMILARITY + 0.05
        ok = mut.variant_is_valid(rs, metrics)
        LOGGER.info("variant_is_valid(bad_metrics) -> %s", ok)
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
            "olive_multizone_share",
            "boundary_density",
            "mirror_similarity",
            "angles",
        }
        self.assertTrue(required_keys.issubset(row.keys()))
        self.assertEqual(row["seed"], 999)
        self.assertEqual(row["index"], 1)

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
# TESTS GÉNÉRATION D'UNE VARIANTE / CANDIDAT
# ============================================================

class TestCandidateGeneration(unittest.TestCase):
    def test_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        """
        Test d'intégration léger : on vérifie la structure du retour.
        Il peut être plus coûteux qu'un test purement mocké.
        """
        seed = mut.DEFAULT_BASE_SEED
        cand = mut.generate_candidate_from_seed(seed)
        LOGGER.info("Candidate généré : %s", summarize_candidate(cand))

        self.assertIsInstance(cand, mut.CandidateResult)
        self.assertEqual(cand.seed, seed)
        self.assertIsInstance(cand.profile, mut.VariantProfile)
        self.assertIsInstance(cand.image, Image.Image)
        self.assertEqual(cand.ratios.shape, (4,))
        self.assertIsInstance(cand.metrics, dict)

    def test_async_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        async def runner() -> None:
            cand = await mut.async_generate_candidate_from_seed(mut.DEFAULT_BASE_SEED + 1)
            LOGGER.info("Async candidate généré : %s", summarize_candidate(cand))
            self.assertIsInstance(cand, mut.CandidateResult)
            self.assertEqual(cand.ratios.shape, (4,))

        asyncio.run(runner())

    def test_async_validate_candidate_result(self) -> None:
        async def runner() -> None:
            cand = make_candidate()
            ok = await mut.async_validate_candidate_result(cand)
            self.assertTrue(ok)

        asyncio.run(runner())

    def test_async_save_candidate_image(self) -> None:
        async def runner() -> None:
            with tempfile.TemporaryDirectory(prefix="test_async_save_") as td:
                path = Path(td) / "async_saved.png"
                cand = make_candidate()
                out = await mut.async_save_candidate_image(cand, path)
                self.assertTrue(out.exists())

        asyncio.run(runner())


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
            progress_calls.append((target_index, local_attempt, total_attempts, accepted, cand.seed))

        with patch.object(mut, "generate_candidate_from_seed", return_value=candidate) as mock_gen, \
             patch.object(mut, "validate_candidate_result", return_value=True) as mock_val:

            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=999,
                progress_callback=progress_cb,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.call_count, 1)
        self.assertEqual(mock_val.call_count, 1)
        self.assertEqual(len(progress_calls), 1)
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    def test_generate_all_retries_until_accept(self) -> None:
        rejected = make_candidate(seed=201)
        accepted = make_candidate(seed=202)

        progress_calls: List[tuple] = []

        def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted_flag: bool) -> None:
            progress_calls.append((target_index, local_attempt, total_attempts, accepted_flag, cand.seed))
            LOGGER.info(
                "Retry sync | idx=%s local=%s total=%s accepted=%s seed=%s",
                target_index, local_attempt, total_attempts, accepted_flag, cand.seed
            )

        with patch.object(mut, "generate_candidate_from_seed", side_effect=[rejected, accepted]) as mock_gen, \
             patch.object(mut, "validate_candidate_result", side_effect=[False, True]) as mock_val:

            rows = mut.generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=123,
                progress_callback=progress_cb,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.call_count, 2)
        self.assertEqual(mock_val.call_count, 2)
        self.assertEqual(len(progress_calls), 2)
        self.assertFalse(progress_calls[0][3])
        self.assertTrue(progress_calls[1][3])
        self.assertEqual(rows[0]["seed"], 202)

    def test_generate_all_stop_requested_writes_partial_report(self) -> None:
        stop_calls = {"count": 0}

        def stop_requested() -> bool:
            stop_calls["count"] += 1
            # stop avant toute génération réelle
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

            rows = mut.generate_all(target_count=1, output_dir=self.tmpdir, base_seed=1000)

        self.assertEqual(len(rows), 1)
        self.assertEqual(seen_seeds, [mut.build_seed(1, 1, 1000), mut.build_seed(1, 2, 1000)])


# ============================================================
# TESTS ORCHESTRATEUR ASYNCHRONE
# ============================================================

class TestGenerateAllAsync(TempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=301)
        progress_calls: List[tuple] = []

        async def progress_cb(target_index: int, local_attempt: int, total_attempts: int, target_count: int, cand: mut.CandidateResult, accepted: bool) -> None:
            progress_calls.append((target_index, local_attempt, total_attempts, accepted, cand.seed))
            LOGGER.info(
                "Progress async | idx=%s local=%s total=%s accepted=%s seed=%s",
                target_index, local_attempt, total_attempts, accepted, cand.seed
            )

        with patch.object(mut, "async_generate_candidate_from_seed", AsyncMock(return_value=candidate)) as mock_gen, \
             patch.object(mut, "async_validate_candidate_result", AsyncMock(return_value=True)) as mock_val:

            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=5000,
                progress_callback=progress_cb,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_gen.await_count, 1)
        self.assertEqual(mock_val.await_count, 1)
        self.assertEqual(len(progress_calls), 1)
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
            )

        LOGGER.info("Ordre async observé : %s", order)

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
        """
        Vérifie le comportement séquentiel strict :
        - tentative 1 de l'image 1 refusée ;
        - tentative 2 de l'image 1 acceptée ;
        - seulement ensuite l'image 2 démarre.
        """
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
            )

        LOGGER.info("Historique séquentiel async : %s", history)

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


# ============================================================
# TESTS RAPPORTS ASYNC
# ============================================================

class TestAsyncReport(TempDirMixin, unittest.IsolatedAsyncioTestCase):
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


# ============================================================
# TESTS ROBUSTESSE SUR LES CONSTANTES
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


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_main.py ==========")
    unittest.main(verbosity=2)