# -*- coding: utf-8 -*-
"""
Suite de tests alignée sur le main.py générique multi-classes :
- validation stricte + best-of obligatoire ;
- nettoyage d'orphelins / fragmentation ;
- exports JSONL pour logs / ML / DL ;
- génération async avec lots, stop, mode parallèle et export rapport.

Exécution :
    MUT_MODULE=main python -m unittest -v test_main.py
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import io
import json
import logging
import os
import sys
import tempfile
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from PIL import Image


TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))


def _import_mut():
    candidates = []
    env_name = os.getenv("MUT_MODULE", "").strip()
    if env_name:
        candidates.append(env_name)
    candidates.append("main")
    last_exc = None
    for name in dict.fromkeys(candidates):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    raise RuntimeError(f"Impossible d'importer le module cible parmi {candidates!r}") from last_exc


mut = _import_mut()

LOG_DIR = Path(os.getenv("LOG_OUTPUT_DIR", TEST_DIR / "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_main.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_main_generic")
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
LOGGER.info("Module cible importé pour les tests: %s", getattr(mut, "__name__", type(mut).__name__))
LOGGER.info("Fichier de log des tests: %s", LOG_FILE)


class LoggedTestMixin:
    def setUp(self) -> None:
        super().setUp()
        self._test_started_at = time.perf_counter()
        LOGGER.info("▶ START %s", self.id())

    def tearDown(self) -> None:
        elapsed = time.perf_counter() - getattr(self, "_test_started_at", time.perf_counter())
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
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_main_generic_")
        self.tmpdir = Path(self._tmpdir_obj.name)
        LOGGER.info("tmpdir prêt: %s", self.tmpdir)

    def tearDown(self) -> None:
        LOGGER.info("tmpdir cleanup: %s", getattr(self, "tmpdir", "<absent>"))
        self._tmpdir_obj.cleanup()
        super().tearDown()


class GlobalStateMixin:
    def setUp(self) -> None:
        super().setUp()
        self._orig_pool = getattr(mut, "_PROCESS_POOL", None)
        self._orig_pool_workers = getattr(mut, "_PROCESS_POOL_WORKERS", None)
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
        if hasattr(mut, "_PROCESS_POOL"):
            mut._PROCESS_POOL = self._orig_pool
        if hasattr(mut, "_PROCESS_POOL_WORKERS"):
            mut._PROCESS_POOL_WORKERS = self._orig_pool_workers
        super().tearDown()


class GeometryMixin:
    def setUp(self) -> None:
        super().setUp()
        self._orig_geometry = (
            mut.WIDTH,
            mut.HEIGHT,
            mut.PHYSICAL_WIDTH_CM,
            mut.PHYSICAL_HEIGHT_CM,
            getattr(mut, "MOTIF_SCALE", None),
        )

    def tearDown(self) -> None:
        width, height, physical_width_cm, physical_height_cm, motif_scale = self._orig_geometry
        mut.set_canvas_geometry(width, height, physical_width_cm, physical_height_cm, motif_scale)
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
        self.assertIsInstance(candidate.label_map, np.ndarray)
        self.assertEqual(candidate.label_map.shape, (mut.HEIGHT, mut.WIDTH))
        self.assertEqual(candidate.image.size, (mut.WIDTH, mut.HEIGHT))
        self.assertEqual(candidate.ratios.shape, (mut.N_CLASSES,))
        self.assertAlmostEqual(float(np.sum(candidate.ratios)), 1.0, places=5)
        self.assertIsInstance(candidate.metrics, dict)
        LOGGER.info(
            "candidate ok | seed=%s | size=%sx%s | ratio_sum=%.6f | metrics=%s",
            getattr(candidate, "seed", "?"),
            getattr(candidate.image, "size", ("?", "?"))[0],
            getattr(candidate.image, "size", ("?", "?"))[1],
            float(np.sum(candidate.ratios)),
            sorted(candidate.metrics.keys()),
        )

    def assertOutcomeLooksConsistent(self, outcome: Any) -> None:
        self.assertIsInstance(outcome, mut.ValidationOutcome)
        self.assertIsInstance(outcome.reasons, list)
        self.assertIsInstance(outcome.fragmentation, dict)
        self.assertIsInstance(outcome.subscores, dict)
        LOGGER.info(
            "outcome ok | accepted=%s | strict=%s | bestof_ok=%s | score=%.6f | reasons=%s",
            bool(outcome.accepted),
            bool(outcome.passed_strict),
            bool(outcome.bestof_ok),
            float(outcome.bestof_score),
            list(outcome.reasons),
        )


REQUIRED_METRIC_KEYS = {
    "largest_component_ratio_class_1",
    "boundary_density",
    "boundary_density_small",
    "boundary_density_tiny",
    "mirror_similarity",
    "edge_contact_ratio",
    "overscan",
    "shift_strength",
    "width",
    "height",
    "physical_width_cm",
    "physical_height_cm",
    "px_per_cm",
    "motif_scale",
    "orphan_pixels_fixed",
    "orphan_cleanup_passes",
}

REQUIRED_SUBSCORES = {
    "ratio_score",
    "per_class_score",
    "fragmentation_score",
    "symmetry_score",
    "edge_score",
}


def make_balanced_label_map(h: int = 80, w: int = 80) -> np.ndarray:
    label_map = np.zeros((h, w), dtype=np.uint8)
    label_map[: h // 2, w // 2 :] = getattr(mut, "IDX_1", 1)
    label_map[h // 2 :, : w // 2] = getattr(mut, "IDX_2", 2)
    label_map[h // 2 :, w // 2 :] = getattr(mut, "IDX_3", 3)
    return label_map


def valid_ratios() -> np.ndarray:
    return np.asarray(mut.TARGET, dtype=float).copy()


def valid_metrics() -> Dict[str, float]:
    return {
        "largest_component_ratio_class_1": 1.0,
        "boundary_density": 0.05,
        "boundary_density_small": 0.05,
        "boundary_density_tiny": 0.05,
        "mirror_similarity": 0.10,
        "edge_contact_ratio": 0.10,
        "overscan": 1.10,
        "shift_strength": 0.80,
        "width": 128.0,
        "height": 72.0,
        "physical_width_cm": 40.0,
        "physical_height_cm": 22.5,
        "px_per_cm": 3.2,
        "motif_scale": getattr(mut, "DEFAULT_MOTIF_SCALE", 0.55),
        "orphan_pixels_fixed": 0.0,
        "orphan_cleanup_passes": 1.0,
    }


def invalid_ratios_far() -> np.ndarray:
    return np.array([0.50, 0.20, 0.20, 0.10], dtype=float)


def valid_fragmentation() -> Dict[str, Any]:
    return {
        "pixel_count": 6400,
        "megapixels": 0.0064,
        "orphan_pixels": 0,
        "orphan_ratio": 0.0,
        "weak_pixels": 0,
        "weak_ratio": 0.0,
        "micro_components_total": 0,
        "micro_components_per_mp": 0.0,
        "component_count_total": 4,
        "by_class": {str(i): {"components": 1, "micro_components": 0} for i in range(mut.N_CLASSES)},
    }


def make_validation_outcome(
    *,
    accepted: bool = True,
    passed_strict: bool = True,
    bestof_ok: bool = True,
    bestof_score: float = 0.99,
    reasons: List[str] | None = None,
    fragmentation: Dict[str, Any] | None = None,
    subscores: Dict[str, float] | None = None,
):
    return mut.ValidationOutcome(
        accepted=accepted,
        passed_strict=passed_strict,
        bestof_ok=bestof_ok,
        bestof_score=bestof_score,
        reasons=list(reasons or []),
        fragmentation=dict(fragmentation or valid_fragmentation()),
        subscores=dict(subscores or {k: 1.0 for k in REQUIRED_SUBSCORES}),
    )


def make_candidate(
    seed: int = 123456,
    label_map: np.ndarray | None = None,
    ratios: np.ndarray | None = None,
    metrics: Dict[str, float] | None = None,
):
    label_map = make_balanced_label_map(mut.HEIGHT, mut.WIDTH) if label_map is None else np.asarray(label_map, dtype=np.uint8)
    ratios = valid_ratios() if ratios is None else np.asarray(ratios, dtype=float)
    metrics = valid_metrics() if metrics is None else dict(metrics)
    return mut.CandidateResult(
        seed=seed,
        profile=mut.make_profile(seed),
        image=Image.new("RGB", (mut.WIDTH, mut.HEIGHT), (0, 0, 0)),
        label_map=label_map,
        ratios=ratios,
        metrics=metrics,
    )


def fake_snapshot(*, machine_intensity: float = 0.90, available_mb: float = 8192.0, disk_free_mb: float = 4096.0):
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


def log_csv_preview(path: Path, max_lines: int = 3) -> None:
    if not path.exists():
        LOGGER.info("csv absent: %s", path)
        return
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        LOGGER.info("csv illisible: %s | %s", path, exc)
        return
    preview = "\n".join(text.splitlines()[:max_lines])
    LOGGER.info("csv %s | preview=\n%s", path.name, preview)


def iter_metric_failure_cases() -> Iterable[tuple[str, float]]:
    yield "boundary_density", mut.MIN_BOUNDARY_DENSITY - 0.001
    yield "boundary_density", mut.MAX_BOUNDARY_DENSITY + 0.001
    yield "boundary_density_small", mut.MIN_BOUNDARY_DENSITY_SMALL - 0.001
    yield "boundary_density_small", mut.MAX_BOUNDARY_DENSITY_SMALL + 0.001
    yield "boundary_density_tiny", mut.MIN_BOUNDARY_DENSITY_TINY - 0.001
    yield "boundary_density_tiny", mut.MAX_BOUNDARY_DENSITY_TINY + 0.001
    yield "mirror_similarity", mut.MAX_MIRROR_SIMILARITY + 0.001
    yield "largest_component_ratio_class_1", mut.MIN_LARGEST_COMPONENT_RATIO_CLASS_1 - 0.001
    yield "edge_contact_ratio", mut.MAX_EDGE_CONTACT_RATIO + 0.001


class TestConstantsAndDataclasses(LoggedTestMixin, AssertionsMixin, unittest.TestCase):
    def test_global_constants_are_coherent(self) -> None:
        self.assertEqual(mut.N_CLASSES, 4)
        self.assertEqual(tuple(mut.RGB.shape), (4, 3))
        self.assertEqual(tuple(mut.TARGET.shape), (4,))
        self.assertEqual(float(np.sum(mut.TARGET)), 1.0)
        self.assertGreater(mut.WIDTH, 0)
        self.assertGreater(mut.HEIGHT, 0)
        self.assertGreater(mut.PX_PER_CM, 0)
        self.assertGreaterEqual(mut.MIN_MOTIF_SCALE, 0.0)
        self.assertGreater(mut.MAX_MOTIF_SCALE, mut.MIN_MOTIF_SCALE)

    def test_validation_outcome_to_dict(self) -> None:
        out = make_validation_outcome(accepted=False, passed_strict=False, bestof_ok=False, bestof_score=0.5, reasons=["x"]).to_dict()
        self.assertEqual(out["accepted"], False)
        self.assertEqual(out["passed_strict"], False)
        self.assertEqual(out["bestof_ok"], False)
        self.assertEqual(out["bestof_score"], 0.5)
        self.assertEqual(out["reasons"], ["x"])

    def test_resource_snapshot_to_dict(self) -> None:
        snap = fake_snapshot()
        out = snap.to_dict()
        self.assertEqual(
            set(out.keys()),
            {
                "ts",
                "cpu_count",
                "process_cpu_percent",
                "system_cpu_percent",
                "process_rss_mb",
                "system_available_mb",
                "system_total_mb",
                "disk_free_mb",
                "machine_intensity",
            },
        )

    def test_runtime_tuning_normalized(self) -> None:
        rt = mut.RuntimeTuning(0, 0, True, 5.0).normalized()
        self.assertEqual(rt.max_workers, 1)
        self.assertEqual(rt.attempt_batch_size, 1)
        self.assertFalse(rt.parallel_attempts)
        self.assertEqual(rt.machine_intensity, 1.0)

    def test_live_counters_line_contains_useful_fields(self) -> None:
        counters = mut.LiveCounters(target_count=5, accepted=2, passed_validation=4, rejected=3, attempts=7, in_flight=1, start_ts=time.time() - 2.0)
        line = counters.line(current_target=3, workers=4)
        self.assertIn("fait=2/5", line)
        self.assertIn("valides=4", line)
        self.assertIn("workers=4", line)

    def test_make_profile_is_deterministic(self) -> None:
        p1 = mut.make_profile(424242)
        p2 = mut.make_profile(424242)
        self.assertEqual(p1, p2)
        self.assertTrue(1.08 <= p1.overscan <= 1.16)
        self.assertTrue(0.50 <= p1.shift_strength <= 1.10)
        self.assertEqual(len(p1.palette_bias), 4)


class TestSystemHelpers(LoggedTestMixin, GlobalStateMixin, TempDirMixin, GeometryMixin, AssertionsMixin, unittest.TestCase):
    def test_worker_initializer_can_limit_numeric_threads(self) -> None:
        with patch.dict(os.environ, {"TEXTURE_LIMIT_NUMERIC_THREADS": "1"}, clear=True):
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

    def test_ensure_output_dir_creates_path(self) -> None:
        out = mut.ensure_output_dir(self.tmpdir / "nested" / "output")
        self.assertTrue(out.exists())
        self.assertTrue(out.is_dir())

    def test_clip_float(self) -> None:
        self.assertEqual(mut._clip_float(5.0, 0.1, 1.0), 1.0)
        self.assertEqual(mut._clip_float(-1.0, 0.1, 1.0), 0.1)
        self.assertEqual(mut._clip_float(0.5, 0.1, 1.0), 0.5)

    def test_set_canvas_geometry_updates_globals(self) -> None:
        mut.set_canvas_geometry(320, 180, 100.0, 56.25, 0.65)
        self.assertEqual(mut.WIDTH, 320)
        self.assertEqual(mut.HEIGHT, 180)
        self.assertFloatClose(mut.PX_PER_CM, 3.2)
        self.assertFloatClose(mut.MOTIF_SCALE, 0.65)
        mut.set_canvas_geometry(320, 180, 100.0, 56.25, 0.001)
        self.assertFloatClose(mut.MOTIF_SCALE, mut.MIN_MOTIF_SCALE)
        with self.assertRaises(ValueError):
            mut.set_canvas_geometry(0, 180, 100.0, 56.25)
        with self.assertRaises(ValueError):
            mut.set_canvas_geometry(320, 180, 100.0, 56.25, 0.0)

    def test_sample_process_resources_returns_snapshot(self) -> None:
        snap = mut.sample_process_resources(machine_intensity=0.7, output_dir=self.tmpdir)
        self.assertIsInstance(snap, mut.ResourceSnapshot)
        self.assertEqual(snap.cpu_count, mut.CPU_COUNT)
        self.assertGreaterEqual(snap.disk_free_mb, 0.0)

    def test_compute_runtime_tuning_uses_memory_thresholds(self) -> None:
        low_mem = fake_snapshot(available_mb=2000.0)
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

    def test_validate_generation_request_accepts_normal_case_and_writes_snapshot(self) -> None:
        with patch.object(mut, "sample_process_resources", return_value=fake_snapshot(disk_free_mb=4096.0)):
            mut.validate_generation_request(
                target_count=1,
                output_dir=self.tmpdir,
                base_seed=1,
                machine_intensity=0.5,
                max_workers=1,
                attempt_batch_size=1,
            )
        self.assertFalse((self.tmpdir / ".write_probe.tmp").exists())
        self.assertTrue((self.tmpdir / "preflight_snapshot.json").exists())

    def test_validate_generation_request_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            mut.validate_generation_request(target_count=0, output_dir=self.tmpdir, base_seed=1, machine_intensity=0.5, max_workers=1, attempt_batch_size=1)
        with self.assertRaises(ValueError):
            mut.validate_generation_request(target_count=1, output_dir=self.tmpdir, base_seed=-1, machine_intensity=0.5, max_workers=1, attempt_batch_size=1)
        with self.assertRaises(ValueError):
            mut.validate_generation_request(target_count=1, output_dir=self.tmpdir, base_seed=1, machine_intensity=0.5, max_workers=0, attempt_batch_size=1)
        with self.assertRaises(ValueError):
            mut.validate_generation_request(target_count=1, output_dir=self.tmpdir, base_seed=1, machine_intensity=0.5, max_workers=1, attempt_batch_size=0)


class TestPureUtilities(LoggedTestMixin, TempDirMixin, AssertionsMixin, GeometryMixin, unittest.TestCase):
    def test_build_seed_and_build_batch_are_deterministic(self) -> None:
        self.assertEqual(mut.build_seed(3, 7, 1000), mut.build_seed(3, 7, 1000))
        self.assertNotEqual(mut.build_seed(3, 7, 1000), mut.build_seed(4, 7, 1000))
        self.assertEqual(mut.build_batch(2, 3, 4, 1000), [(3, 201003), (4, 201004), (5, 201005), (6, 201006)])

    def test_compute_ratios_and_render_label_map(self) -> None:
        label_map = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        ratios = mut.compute_ratios(label_map)
        self.assertArrayClose(ratios, np.array([0.25, 0.25, 0.25, 0.25]))
        img = mut.render_label_map(label_map)
        self.assertEqual(img.size, (2, 2))
        self.assertIsInstance(img, Image.Image)

    def test_boundary_and_component_helpers(self) -> None:
        label_map = np.zeros((6, 6), dtype=np.uint8)
        label_map[:, 3:] = 1
        boundary = mut.boundary_mask(label_map)
        self.assertTrue(boundary.any())
        self.assertGreater(mut.boundary_density(label_map), 0.0)
        self.assertLessEqual(mut.mirror_similarity_score(label_map), 1.0)
        self.assertGreater(mut.largest_component_ratio(label_map == 1), 0.0)
        self.assertGreaterEqual(mut.edge_contact_ratio(label_map), 0.0)
        self.assertLessEqual(mut.edge_contact_ratio(label_map), 1.0)

    def test_scaled_patch_size(self) -> None:
        sx, sy = mut.scaled_patch_size(10.0, 5.0, 0.5)
        self.assertFloatClose(sx, 5.0)
        self.assertFloatClose(sy, 2.5)
        sx2, sy2 = mut.scaled_patch_size(10.0, 5.0, 0.0)
        self.assertFloatClose(sx2, 10.0 * mut.MIN_MOTIF_SCALE)
        self.assertFloatClose(sy2, 5.0 * mut.MIN_MOTIF_SCALE)

    def test_downsample_center_crop_shift_reflect_and_cells(self) -> None:
        arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
        ds = mut.downsample_nearest(arr, factor=2)
        self.assertArrayClose(ds, np.array([[0, 2], [8, 10]], dtype=np.uint8))
        crop2d = mut.center_crop(np.arange(25, dtype=np.uint8).reshape(5, 5), 3, 3)
        self.assertEqual(crop2d.shape, (3, 3))
        crop3d = mut.center_crop(np.arange(2 * 5 * 5, dtype=np.uint8).reshape(2, 5, 5), 3, 3)
        self.assertEqual(crop3d.shape, (2, 3, 3))
        shifted = mut.shift_reflect(np.arange(9, dtype=np.uint8).reshape(3, 3), 1, -1)
        self.assertEqual(shifted.shape, (3, 3))
        cells_x, cells_y = mut.cells_for_patch_size(10.0, 5.0, 320, 180)
        self.assertGreaterEqual(cells_x, 3)
        self.assertGreaterEqual(cells_y, 3)


class TestCleanupAndFragmentation(LoggedTestMixin, AssertionsMixin, unittest.TestCase):
    def test_neighbors_same_and_dominant_class(self) -> None:
        label_map = np.zeros((3, 3), dtype=np.uint8)
        label_map[1, 1] = 1
        same = mut.same_neighbor_count(label_map)
        winner, winner_count = mut.dominant_neighbor_class(label_map, class_count=4)
        self.assertEqual(same[1, 1], 0)
        self.assertEqual(int(winner[1, 1]), 0)
        self.assertGreaterEqual(int(winner_count[1, 1]), 4)

    def test_cleanup_orphan_pixels_changes_isolated_pixel(self) -> None:
        label_map = np.zeros((5, 5), dtype=np.uint8)
        label_map[2, 2] = 1
        cleaned, info = mut.cleanup_orphan_pixels(label_map, class_count=4)
        self.assertEqual(int(cleaned[2, 2]), 0)
        self.assertGreaterEqual(info["orphan_pixels_fixed"], 1)
        self.assertGreaterEqual(info["orphan_cleanup_passes"], 1)

    def test_connected_component_areas_python_fallback(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        mask[4:6, 4:6] = True
        with patch.object(mut, "cv2", None):
            areas = mut.connected_component_areas(mask)
        self.assertEqual(sorted(areas), [1, 4])

    def test_fragmentation_report_detects_micro_components(self) -> None:
        label_map = np.zeros((10, 10), dtype=np.uint8)
        label_map[1, 1] = 1
        label_map[8, 8] = 2
        report = mut.fragmentation_report(label_map, class_count=4, min_component_pixels=(12, 12, 12, 12))
        self.assertIn("orphan_ratio", report)
        self.assertIn("micro_components_per_mp", report)
        self.assertGreaterEqual(report["micro_components_total"], 1)


class TestBestOfAndValidation(LoggedTestMixin, TempDirMixin, AssertionsMixin, GeometryMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        mut.set_canvas_geometry(80, 80, 40.0, 40.0, 0.55)

    def test_compute_bestof_score_returns_high_score_for_clean_candidate(self) -> None:
        score, subscores = mut.compute_bestof_score(
            ratios=valid_ratios(),
            target=mut.TARGET,
            metrics=valid_metrics(),
            fragmentation=valid_fragmentation(),
        )
        self.assertGreaterEqual(score, mut.BESTOF_MIN_SCORE)
        self.assertEqual(set(subscores.keys()), REQUIRED_SUBSCORES)

    def test_validate_with_reasons_accepts_valid_candidate(self) -> None:
        candidate = make_candidate(seed=111, label_map=make_balanced_label_map(80, 80))
        outcome = mut.validate_with_reasons(candidate)
        self.assertOutcomeLooksConsistent(outcome)
        self.assertTrue(outcome.accepted)
        self.assertTrue(outcome.passed_strict)
        self.assertTrue(outcome.bestof_ok)
        self.assertEqual(outcome.reasons, [])

    def test_validate_with_reasons_rejects_metric_failures_and_bestof(self) -> None:
        metrics = valid_metrics()
        metrics["mirror_similarity"] = 0.99
        candidate = make_candidate(seed=112, label_map=make_balanced_label_map(80, 80), metrics=metrics)
        outcome = mut.validate_with_reasons(candidate)
        self.assertFalse(outcome.accepted)
        self.assertIn("mirror_similarity", outcome.reasons)
        self.assertIn("not_bestof", outcome.reasons)

    def test_validate_with_reasons_rejects_ratio_failures(self) -> None:
        candidate = make_candidate(seed=113, label_map=make_balanced_label_map(80, 80), ratios=invalid_ratios_far())
        outcome = mut.validate_with_reasons(candidate)
        self.assertFalse(outcome.accepted)
        self.assertIn("mean_abs_error", outcome.reasons)

    def test_emit_validation_payload_writes_jsonl_files(self) -> None:
        candidate = make_candidate(seed=114, label_map=make_balanced_label_map(80, 80))
        outcome_ok = make_validation_outcome()
        outcome_ko = make_validation_outcome(accepted=False, passed_strict=False, bestof_ok=False, bestof_score=0.2, reasons=["mirror_similarity", "not_bestof"])

        payload_ok = mut.emit_validation_payload(
            output_dir=self.tmpdir,
            target_index=1,
            local_attempt=1,
            global_attempt=1,
            candidate=candidate,
            outcome=outcome_ok,
        )
        payload_ko = mut.emit_validation_payload(
            output_dir=self.tmpdir,
            target_index=1,
            local_attempt=2,
            global_attempt=2,
            candidate=candidate,
            outcome=outcome_ko,
        )
        self.assertTrue((self.tmpdir / mut.EVENTS_JSONL).exists())
        self.assertTrue((self.tmpdir / mut.ACCEPTS_JSONL).exists())
        self.assertTrue((self.tmpdir / mut.REJECTIONS_JSONL).exists())
        self.assertTrue((self.tmpdir / mut.FULL_DATASET_JSONL).exists())
        self.assertEqual(payload_ok["accepted"], True)
        self.assertEqual(payload_ko["accepted"], False)


class TestGeneratorInternals(LoggedTestMixin, AssertionsMixin, GeometryMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        mut.set_canvas_geometry(128, 72, 40.0, 22.5, 0.55)

    def test_random_blob_layer_shape_and_range(self) -> None:
        rng = np.random.default_rng(123)
        out = mut.random_blob_layer(128, 72, rng, 5, 4, 20.0)
        self.assertEqual(out.shape, (72, 128))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_build_field_shape_and_range(self) -> None:
        rng = np.random.default_rng(123)
        plan = [(4, 3, 0.0, 1.0), (6, 4, 15.0, 0.5)]
        out = mut.build_field(128, 72, rng, plan, shift_strength=1.0)
        self.assertEqual(out.shape, (72, 128))
        self.assertEqual(out.dtype, np.float16)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_build_all_fields_shape_and_custom_scale(self) -> None:
        profile = mut.make_profile(123)
        fields_default = mut.build_all_fields(160, 96, profile, crop_height=72, crop_width=128, motif_scale=mut.DEFAULT_MOTIF_SCALE)
        fields_small = mut.build_all_fields(160, 96, profile, crop_height=72, crop_width=128, motif_scale=0.20)
        self.assertEqual(fields_default.shape, (4, 72, 128))
        self.assertEqual(fields_default.dtype, np.float16)
        self.assertFalse(np.array_equal(fields_default, fields_small))

    def test_sequential_assign_respects_target_counts(self) -> None:
        fields = np.zeros((4, 2, 4), dtype=np.float16)
        fields[1] = np.array([[0.9, 0.8, 0.1, 0.1], [0.7, 0.6, 0.2, 0.2]], dtype=np.float16)
        fields[2] = np.array([[0.1, 0.1, 0.9, 0.8], [0.2, 0.2, 0.7, 0.6]], dtype=np.float16)
        fields[3] = np.array([[0.05, 0.05, 0.05, 0.05], [0.95, 0.95, 0.05, 0.05]], dtype=np.float16)
        target_counts = np.array([2, 2, 2, 2], dtype=int)
        labels = mut.sequential_assign(fields, target_counts)
        counts = np.bincount(labels.ravel(), minlength=4)
        self.assertEqual(tuple(counts), (2, 2, 2, 2))

    def test_exactify_and_force_exact_target_counts_hit_target(self) -> None:
        labels = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [2, 2, 3, 3], [0, 0, 0, 0]], dtype=np.uint8)
        fields = np.zeros((4, 4, 4), dtype=np.float16)
        fields[1, :, :] = 0.2
        fields[2, :, :] = 0.2
        fields[3, :, :] = 0.2
        target_counts = np.array([10, 2, 2, 2], dtype=int)
        out = mut.exactify_proportions(labels, fields, target_counts)
        counts = np.bincount(out.ravel(), minlength=4)
        self.assertEqual(tuple(counts), tuple(target_counts))
        forced = mut.force_exact_target_counts(np.zeros((4, 4), dtype=np.uint8), fields, np.array([8, 3, 3, 2], dtype=int))
        counts2 = np.bincount(forced.ravel(), minlength=4)
        self.assertEqual(tuple(counts2), (8, 3, 3, 2))

    def test_generate_one_variant_returns_expected_structure(self) -> None:
        profile = mut.make_profile(777)
        candidate = mut.generate_one_variant(profile)
        self.assertCandidateLooksConsistent(candidate)
        self.assertTrue(set(REQUIRED_METRIC_KEYS).issubset(candidate.metrics.keys()))
        self.assertAlmostEqual(float(candidate.ratios.sum()), 1.0, places=6)
        self.assertFloatClose(candidate.metrics["motif_scale"], mut.MOTIF_SCALE)

    def test_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        candidate = mut.generate_candidate_from_seed(12345)
        self.assertCandidateLooksConsistent(candidate)
        self.assertEqual(candidate.seed, 12345)

    def test_generate_and_validate_from_seed_returns_tuple(self) -> None:
        candidate = make_candidate(seed=42)
        outcome = make_validation_outcome()
        with patch.object(mut, "generate_candidate_from_seed", return_value=candidate), patch.object(mut, "validate_with_reasons", return_value=outcome):
            got_candidate, got_outcome = mut.generate_and_validate_from_seed(42)
        self.assertCandidateLooksConsistent(got_candidate)
        self.assertOutcomeLooksConsistent(got_outcome)
        self.assertTrue(got_outcome.accepted)


class TestExports(TempDirMixin, AssertionsMixin, GeometryMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        mut.set_canvas_geometry(128, 72, 40.0, 22.5, 0.55)

    def test_unique_pattern_helpers_and_deduped_save(self) -> None:
        candidate = make_candidate(seed=888)
        path1 = mut.build_unique_pattern_path(self.tmpdir, 1, candidate.seed, 2, 3)
        self.assertTrue(path1.name.startswith("pattern_001_"))
        saved1 = mut.save_candidate_image(candidate, path1)
        saved2 = mut.save_candidate_image(candidate, path1)
        self.assertTrue(saved1.exists())
        self.assertTrue(saved2.exists())
        self.assertNotEqual(saved1, saved2)

    def test_candidate_row_and_write_report(self) -> None:
        candidate = make_candidate(seed=777)
        outcome = make_validation_outcome(accepted=True, bestof_score=0.99)
        row = mut.candidate_row(1, 2, 3, candidate, outcome, image_name="x.png", image_path="/tmp/x.png")
        self.assertEqual(row["index"], 1)
        self.assertEqual(row["seed"], 777)
        self.assertEqual(row["attempts_for_this_image"], 2)
        self.assertIn("bestof_score", row)
        self.assertIn("image_name", row)
        csv1 = mut.write_report([row], self.tmpdir)
        csv2 = mut.write_report([], self.tmpdir, filename="empty.csv")
        self.assertTrue(csv1.exists())
        self.assertTrue(csv2.exists())
        with csv1.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["index"], "1")
        self.assertEqual(csv2.read_text(encoding="utf-8"), "")


class TestAsyncHelpersAndOrchestrator(LoggedTestMixin, GlobalStateMixin, TempDirMixin, GeometryMixin, AssertionsMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        mut.set_canvas_geometry(128, 72, 40.0, 22.5, 0.55)

    async def test_await_attempt(self) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        fut.set_result((make_candidate(seed=42), make_validation_outcome()))
        out = await mut._await_attempt(fut, 3, 42)
        self.assertEqual(out[0], 3)
        self.assertEqual(out[1], 42)
        self.assertIsInstance(out[2], mut.CandidateResult)
        self.assertIsInstance(out[3], mut.ValidationOutcome)

    async def test_console_progress_writes_to_stdout(self) -> None:
        counters = mut.LiveCounters(target_count=2, accepted=1, passed_validation=1, rejected=0, attempts=1, in_flight=0, start_ts=time.time() - 1.0)
        buf = io.StringIO()
        with patch.object(sys, "stdout", buf):
            mut.console_progress(counters, current_target=1, workers=2)
        self.assertIn("fait=1/2", buf.getvalue())

    async def test_async_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=211)
        outcome = make_validation_outcome()
        progress = AsyncMock()
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "emit_validation_payload", return_value={"accepted": True}) as emit_mock, \
             patch.object(mut, "generate_and_validate_from_seed", return_value=(candidate, outcome)):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                progress_callback=progress,
                live_console=False,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 211)
        progress.assert_awaited_once()
        emit_mock.assert_called_once()
        saved_images = sorted(self.tmpdir.glob("pattern_*.png"))
        self.assertEqual(len(saved_images), 1)
        self.assertEqual(rows[0]["image_name"], saved_images[0].name)
        self.assertEqual(Path(rows[0]["image_path"]), saved_images[0])
        self.assertTrue((self.tmpdir / "rapport_textures.csv").exists())
        self.assertTrue((self.tmpdir / "run_summary.json").exists())
        summary = json.loads((self.tmpdir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["passed_validation"], 1)
        self.assertTrue(summary["bestof_required"])

    async def test_async_generate_all_retries_until_accept(self) -> None:
        side_effects = [
            (make_candidate(seed=301), make_validation_outcome(accepted=False, passed_strict=False, bestof_ok=False, bestof_score=0.2, reasons=["not_bestof"])),
            (make_candidate(seed=302), make_validation_outcome()),
        ]
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "emit_validation_payload", return_value={}) as emit_mock, \
             patch.object(mut, "generate_and_validate_from_seed", side_effect=side_effects):
            rows = await mut.async_generate_all(target_count=1, output_dir=self.tmpdir, live_console=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 302)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)
        self.assertEqual(rows[0]["global_attempt"], 2)
        self.assertEqual(emit_mock.call_count, 2)
        summary = json.loads((self.tmpdir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["passed_validation"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["attempts"], 2)

    async def test_async_generate_all_parallel_branch(self) -> None:
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            with patch.object(mut, "validate_generation_request", return_value=None), \
                 patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(2, 2, True, 0.9, "test")), \
                 patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
                 patch.object(mut, "get_process_pool", return_value=pool), \
                 patch.object(mut, "emit_validation_payload", return_value={}), \
                 patch.object(mut, "generate_and_validate_from_seed", side_effect=lambda seed: (make_candidate(seed=seed), make_validation_outcome(accepted=(seed % 2 == 0), passed_strict=(seed % 2 == 0), bestof_ok=(seed % 2 == 0), bestof_score=0.99 if seed % 2 == 0 else 0.1, reasons=[] if seed % 2 == 0 else ["not_bestof"]))):
                rows = await mut.async_generate_all(target_count=1, output_dir=self.tmpdir, live_console=False)
        finally:
            pool.shutdown(wait=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)
        self.assertEqual(rows[0]["seed"] % 2, 0)

    async def test_async_generate_all_stop_requested(self) -> None:
        stop = AsyncMock(side_effect=[True])
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()):
            rows = await mut.async_generate_all(target_count=2, output_dir=self.tmpdir, stop_requested=stop, live_console=False)
        self.assertEqual(rows, [])
        self.assertTrue((self.tmpdir / "rapport_textures.csv").exists())

    async def test_async_generate_all_memory_pressure_disables_parallel(self) -> None:
        candidate = make_candidate(seed=501)
        outcome = make_validation_outcome()
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(2, 2, True, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", side_effect=[fake_snapshot(available_mb=16000.0), fake_snapshot(available_mb=3000.0)]), \
             patch.object(mut, "get_process_pool") as get_pool_mock, \
             patch.object(mut, "emit_validation_payload", return_value={}), \
             patch.object(mut, "generate_and_validate_from_seed", return_value=(candidate, outcome)):
            rows = await mut.async_generate_all(target_count=1, output_dir=self.tmpdir, live_console=False)
        self.assertEqual(len(rows), 1)
        get_pool_mock.assert_not_called()


class TestCliEntrypoints(LoggedTestMixin, GlobalStateMixin, TempDirMixin, GeometryMixin, AssertionsMixin, unittest.TestCase):
    def test_parse_cli_args_defaults(self) -> None:
        with patch.object(sys, "argv", ["prog"]):
            args = mut.parse_cli_args()
        self.assertEqual(args.target_count, mut.N_VARIANTS_REQUIRED)
        self.assertEqual(args.width, mut.DEFAULT_WIDTH)
        self.assertEqual(args.height, mut.DEFAULT_HEIGHT)
        self.assertFalse(args.disable_parallel_attempts)
        self.assertFloatClose(args.motif_scale, mut.DEFAULT_MOTIF_SCALE)

    def test_parse_cli_args_custom_flags(self) -> None:
        with patch.object(sys, "argv", ["prog", "--disable-parallel-attempts", "--random-seed", "77", "--machine-intensity", "0.5"]):
            args = mut.parse_cli_args()
        self.assertTrue(args.disable_parallel_attempts)
        self.assertEqual(args.random_seed, 77)
        self.assertEqual(args.machine_intensity, 0.5)

    def test_main_calls_async_generate_all_and_shutdown_pool(self) -> None:
        async_mock = AsyncMock(return_value=[{"index": 1}])
        with patch.object(sys, "argv", [
            "prog",
            "--target-count", "1",
            "--output-dir", str(self.tmpdir),
            "--width", "128",
            "--height", "72",
            "--physical-width-cm", "40",
            "--physical-height-cm", "22.5",
            "--motif-scale", "0.61",
            "--no-live-console",
        ]), \
             patch.object(mut, "async_generate_all", async_mock), \
             patch.object(mut, "shutdown_process_pool") as shutdown_mock, \
             patch.object(mut, "set_canvas_geometry") as geometry_mock:
            mut.main()
        geometry_mock.assert_called_once_with(width=128, height=72, physical_width_cm=40.0, physical_height_cm=22.5, motif_scale=0.61)
        async_mock.assert_awaited_once()
        shutdown_mock.assert_called_once()


class TestLoggingArtifacts(LoggedTestMixin, TempDirMixin, unittest.TestCase):
    def test_logger_file_exists_and_helpers_preview_files(self) -> None:
        self.assertTrue(LOG_FILE.exists() or LOG_FILE.parent.exists())
        sample_json = self.tmpdir / "sample.json"
        sample_csv = self.tmpdir / "sample.csv"
        sample_json.write_text(json.dumps({"ok": True, "items": [1, 2, 3]}, ensure_ascii=False), encoding="utf-8")
        sample_csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        log_json_preview(sample_json)
        log_csv_preview(sample_csv)
        self.assertTrue(sample_json.exists())
        self.assertTrue(sample_csv.exists())


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_main.py ==========")
    unittest.main(verbosity=2)
