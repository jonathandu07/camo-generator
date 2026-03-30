# -*- coding: utf-8 -*-
"""
Suite de tests alignée sur l'API réelle de main.py.

Objectifs :
- couvrir les helpers purs, les dataclasses, le logging et le préflight ;
- couvrir le moteur organique sur géométrie réduite pour rester rapide ;
- couvrir la validation, les exports et le rapport CSV ;
- couvrir l'orchestrateur async sur les branches essentielles ;
- rester déterministe, rapide et compatible avec `main`.

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
from typing import Any, Dict, Iterable
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
    candidates.extend(["main"])
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
        if motif_scale is None:
            mut.set_canvas_geometry(width, height, physical_width_cm, physical_height_cm)
        else:
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
        self.assertEqual(candidate.image.size, (mut.WIDTH, mut.HEIGHT))
        self.assertEqual(candidate.ratios.shape, (4,))
        self.assertAlmostEqual(float(np.sum(candidate.ratios)), 1.0, places=5)
        self.assertIsInstance(candidate.metrics, dict)


REQUIRED_METRIC_KEYS = {
    "largest_olive_component_ratio",
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
}

VALID_METRICS = {
    "largest_olive_component_ratio": 0.20,
    "boundary_density": 0.05,
    "boundary_density_small": 0.05,
    "boundary_density_tiny": 0.05,
    "mirror_similarity": 0.40,
    "edge_contact_ratio": 0.40,
    "overscan": 1.10,
    "shift_strength": 0.80,
    "width": 128.0,
    "height": 72.0,
    "physical_width_cm": 40.0,
    "physical_height_cm": 22.5,
    "px_per_cm": 3.2,
    "motif_scale": mut.DEFAULT_MOTIF_SCALE,
}

INVALID_RATIOS_FAR = np.array([0.50, 0.20, 0.20, 0.10], dtype=float)


def valid_ratios() -> np.ndarray:
    return mut.TARGET.copy()


def valid_metrics() -> Dict[str, float]:
    return dict(VALID_METRICS)


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
    yield "boundary_density", mut.MIN_BOUNDARY_DENSITY - 0.001
    yield "boundary_density", mut.MAX_BOUNDARY_DENSITY + 0.001
    yield "boundary_density_small", mut.MIN_BOUNDARY_DENSITY_SMALL - 0.001
    yield "boundary_density_small", mut.MAX_BOUNDARY_DENSITY_SMALL + 0.001
    yield "boundary_density_tiny", mut.MIN_BOUNDARY_DENSITY_TINY - 0.001
    yield "boundary_density_tiny", mut.MAX_BOUNDARY_DENSITY_TINY + 0.001
    yield "mirror_similarity", mut.MAX_MIRROR_SIMILARITY + 0.001
    yield "largest_olive_component_ratio", mut.MIN_LARGEST_OLIVE_COMPONENT_RATIO - 0.001
    yield "edge_contact_ratio", mut.MAX_EDGE_CONTACT_RATIO + 0.001


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
        self.assertEqual(out["cpu_count"], float(mut.CPU_COUNT))

    def test_runtime_tuning_normalized(self) -> None:
        rt = mut.RuntimeTuning(
            max_workers=0,
            attempt_batch_size=0,
            parallel_attempts=True,
            machine_intensity=2.0,
        ).normalized()
        self.assertEqual(rt.max_workers, 1)
        self.assertEqual(rt.attempt_batch_size, 1)
        self.assertFalse(rt.parallel_attempts)
        self.assertEqual(rt.machine_intensity, 1.0)

    def test_live_counters_line_contains_useful_fields(self) -> None:
        counters = mut.LiveCounters(
            target_count=5,
            accepted=2,
            passed_validation=4,
            rejected=3,
            attempts=7,
            in_flight=1,
            start_ts=time.time() - 2.0,
        )
        line = counters.line(current_target=3, workers=4)
        self.assertIn("fait=2/5", line)
        self.assertIn("valides=4", line)
        self.assertIn("rejets=3", line)
        self.assertIn("workers=4", line)

    def test_make_profile_is_deterministic(self) -> None:
        p1 = mut.make_profile(424242)
        p2 = mut.make_profile(424242)
        self.assertEqual(p1, p2)
        self.assertTrue(1.08 <= p1.overscan <= 1.16)
        self.assertTrue(0.50 <= p1.shift_strength <= 1.10)
        self.assertEqual(len(p1.palette_bias), 4)


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

    def test_supervisor_feedback_returns_dict(self) -> None:
        fake = types.SimpleNamespace(feedback_runtime_event=lambda **kwargs: {"max_workers": kwargs.get("workers", 1)})
        with patch.dict(sys.modules, {"log": fake}, clear=False):
            mut._LOG_MODULE_CACHE = None
            mut._LOG_MODULE_ATTEMPTED = False
            out = mut._supervisor_feedback("evt", workers=3)
        self.assertEqual(out, {"max_workers": 3})

    def test_merge_supervisor_tuning_normalizes_values(self) -> None:
        tuning = mut.RuntimeTuning(1, 1, False, 0.5, "base")
        advice = {
            "max_workers": 0,
            "attempt_batch_size": 0,
            "parallel_attempts": True,
            "machine_intensity": 5.0,
            "reason": "advice",
        }
        out = mut._merge_supervisor_tuning(tuning, advice)
        self.assertEqual(out.max_workers, 1)
        self.assertEqual(out.attempt_batch_size, 1)
        self.assertFalse(out.parallel_attempts)
        self.assertEqual(out.machine_intensity, 1.0)
        self.assertEqual(out.reason, "advice")


class TestSystemHelpers(GlobalStateMixin, TempDirMixin, AssertionsMixin, GeometryMixin, unittest.TestCase):
    def test_worker_initializer_can_limit_numeric_threads(self) -> None:
        with patch.dict(os.environ, {"CAMO_LIMIT_NUMERIC_THREADS": "1"}, clear=True):
            mut._worker_initializer()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "1")

    def test_ensure_output_dir_creates_path(self) -> None:
        out = mut.ensure_output_dir(self.tmpdir / "nested" / "output")
        self.assertTrue(out.exists())
        self.assertTrue(out.is_dir())

    def test_set_canvas_geometry_updates_globals(self) -> None:
        mut.set_canvas_geometry(320, 180, 100.0, 56.25, 0.65)
        self.assertEqual(mut.WIDTH, 320)
        self.assertEqual(mut.HEIGHT, 180)
        self.assertFloatClose(mut.PX_PER_CM, 3.2)
        self.assertFloatClose(mut.MOTIF_SCALE, 0.65)
        with self.assertRaises(ValueError):
            mut.set_canvas_geometry(0, 180, 100.0, 56.25)
        with self.assertRaises(ValueError):
            mut.set_canvas_geometry(320, 180, 100.0, 56.25, 0.0)

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

    def test_sample_process_resources_returns_snapshot(self) -> None:
        snap = mut.sample_process_resources(machine_intensity=0.7, output_dir=self.tmpdir)
        self.assertIsInstance(snap, mut.ResourceSnapshot)
        self.assertEqual(snap.cpu_count, mut.CPU_COUNT)
        self.assertGreaterEqual(snap.disk_free_mb, 0.0)

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

    def test_validate_generation_request_accepts_normal_case(self) -> None:
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


class TestPureUtilities(TempDirMixin, AssertionsMixin, GeometryMixin, unittest.TestCase):
    def test_build_seed_and_build_batch_are_deterministic(self) -> None:
        self.assertEqual(mut.build_seed(3, 7, 1000), mut.build_seed(3, 7, 1000))
        self.assertNotEqual(mut.build_seed(3, 7, 1000), mut.build_seed(4, 7, 1000))
        self.assertEqual(mut.build_batch(2, 3, 4, 1000), [(3, 201003), (4, 201004), (5, 201005), (6, 201006)])

    def test_clip_float(self) -> None:
        self.assertEqual(mut._clip_float(5.0, 0.1, 1.0), 1.0)
        self.assertEqual(mut._clip_float(-1.0, 0.1, 1.0), 0.1)
        self.assertEqual(mut._clip_float(0.5, 0.1, 1.0), 0.5)

    def test_compute_ratios_and_render_canvas(self) -> None:
        canvas = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        ratios = mut.compute_ratios(canvas)
        self.assertArrayClose(ratios, np.array([0.25, 0.25, 0.25, 0.25]))
        img = mut.render_canvas(canvas)
        self.assertEqual(img.size, (2, 2))
        self.assertIsInstance(img, Image.Image)

    def test_boundary_and_component_helpers(self) -> None:
        canvas = np.zeros((6, 6), dtype=np.uint8)
        canvas[:, 3:] = 1
        boundary = mut.boundary_mask(canvas)
        self.assertTrue(boundary.any())
        self.assertGreater(mut.boundary_density(canvas), 0.0)
        self.assertLessEqual(mut.mirror_similarity_score(canvas), 1.0)
        self.assertGreater(mut.largest_component_ratio(canvas == 1), 0.0)
        self.assertGreaterEqual(mut.edge_contact_ratio(canvas), 0.0)
        self.assertLessEqual(mut.edge_contact_ratio(canvas), 1.0)

    def test_scaled_patch_size(self) -> None:
        sx, sy = mut.scaled_patch_size(10.0, 5.0, 0.5)
        self.assertFloatClose(sx, 5.0)
        self.assertFloatClose(sy, 2.5)
        sx2, sy2 = mut.scaled_patch_size(10.0, 5.0, 0.0)
        self.assertFloatClose(sx2, 2.5)
        self.assertFloatClose(sy2, 1.25)

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


class TestGeneratorInternals(AssertionsMixin, GeometryMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        mut.set_canvas_geometry(128, 72, 40.0, 22.5)

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

    def test_build_all_fields_shape(self) -> None:
        profile = mut.make_profile(123)
        fields = mut.build_all_fields(160, 96, profile, crop_height=72, crop_width=128)
        self.assertEqual(fields.shape, (4, 72, 128))
        self.assertEqual(fields.dtype, np.float16)

    def test_build_all_fields_accepts_custom_motif_scale(self) -> None:
        profile = mut.make_profile(124)
        fields_default = mut.build_all_fields(
            160,
            96,
            profile,
            crop_height=72,
            crop_width=128,
            motif_scale=mut.DEFAULT_MOTIF_SCALE,
        )
        fields_small = mut.build_all_fields(
            160,
            96,
            profile,
            crop_height=72,
            crop_width=128,
            motif_scale=0.55,
        )
        self.assertEqual(fields_default.shape, fields_small.shape)
        self.assertFalse(np.array_equal(fields_default, fields_small))

    def test_sequential_assign_respects_target_counts(self) -> None:
        fields = np.zeros((4, 2, 4), dtype=np.float16)
        fields[mut.IDX_OLIVE] = np.array([[0.9, 0.8, 0.1, 0.1], [0.7, 0.6, 0.2, 0.2]], dtype=np.float16)
        fields[mut.IDX_TERRE] = np.array([[0.1, 0.1, 0.9, 0.8], [0.2, 0.2, 0.7, 0.6]], dtype=np.float16)
        fields[mut.IDX_GRIS] = np.array([[0.05, 0.05, 0.05, 0.05], [0.95, 0.95, 0.05, 0.05]], dtype=np.float16)
        target_counts = np.array([2, 2, 2, 2], dtype=int)
        labels = mut.sequential_assign(fields, target_counts)
        counts = np.bincount(labels.ravel(), minlength=4)
        self.assertEqual(tuple(counts), (2, 2, 2, 2))

    def test_exactify_proportions_preserves_target_when_already_exact(self) -> None:
        labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [2, 2, 3, 3],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        fields = np.zeros((4, 4, 4), dtype=np.float16)
        fields[1, :, :] = 0.2
        fields[2, :, :] = 0.2
        fields[3, :, :] = 0.2
        target_counts = np.array([10, 2, 2, 2], dtype=int)
        out = mut.exactify_proportions(labels, fields, target_counts)
        counts = np.bincount(out.ravel(), minlength=4)
        self.assertEqual(tuple(counts), tuple(target_counts))

    def test_force_exact_target_counts_hits_target_counts(self) -> None:
        labels = np.zeros((4, 4), dtype=np.uint8)
        fields = np.zeros((4, 4, 4), dtype=np.float16)

        positions_1 = [(0, 0), (0, 1), (1, 0)]
        positions_2 = [(2, 2), (2, 3), (3, 2)]
        positions_3 = [(1, 3), (3, 0)]

        for y, x in positions_1:
            fields[1, y, x] = 1.0
        for y, x in positions_2:
            fields[2, y, x] = 1.0
        for y, x in positions_3:
            fields[3, y, x] = 1.0

        target_counts = np.array([8, 3, 3, 2], dtype=int)
        out = mut.force_exact_target_counts(labels, fields, target_counts)
        counts = np.bincount(out.ravel(), minlength=4)
        self.assertEqual(tuple(counts), tuple(target_counts))

    def test_generate_one_variant_returns_expected_keys(self) -> None:
        profile = mut.make_profile(777)
        image, ratios, metrics = mut.generate_one_variant(profile)
        self.assertEqual(image.size, (128, 72))
        self.assertEqual(ratios.shape, (4,))
        self.assertTrue(set(REQUIRED_METRIC_KEYS).issubset(set(metrics.keys())))
        self.assertAlmostEqual(float(ratios.sum()), 1.0, places=6)
        self.assertFloatClose(metrics["motif_scale"], mut.MOTIF_SCALE)

    def test_generate_candidate_from_seed_returns_candidate_result(self) -> None:
        candidate = mut.generate_candidate_from_seed(12345)
        self.assertCandidateLooksConsistent(candidate)
        self.assertEqual(candidate.seed, 12345)

    def test_generate_and_validate_from_seed_returns_tuple(self) -> None:
        with patch.object(mut, "generate_candidate_from_seed", return_value=make_candidate(seed=42)), \
             patch.object(mut, "validate_candidate_result", return_value=True):
            candidate, accepted = mut.generate_and_validate_from_seed(42)
        self.assertCandidateLooksConsistent(candidate)
        self.assertTrue(accepted)


class TestValidationAndExports(TempDirMixin, AssertionsMixin, GeometryMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        mut.set_canvas_geometry(128, 72, 40.0, 22.5)

    def test_variant_is_valid_accepts_and_rejects(self) -> None:
        self.assertTrue(mut.variant_is_valid(valid_ratios(), valid_metrics()))
        self.assertFalse(mut.variant_is_valid(INVALID_RATIOS_FAR, valid_metrics()))
        for key, value in iter_metric_failure_cases():
            metrics = valid_metrics()
            metrics[key] = value
            with self.subTest(metric=key, value=value):
                self.assertFalse(mut.variant_is_valid(valid_ratios(), metrics))

    def test_validate_candidate_result_delegates(self) -> None:
        candidate = make_candidate()
        self.assertTrue(mut.validate_candidate_result(candidate))

    def test_save_candidate_image_and_candidate_row(self) -> None:
        candidate = make_candidate(seed=777)
        out = mut.save_candidate_image(candidate, self.tmpdir / "x" / "img.png")
        self.assertTrue(out.exists())
        row = mut.candidate_row(1, 2, 3, candidate)
        self.assertEqual(row["index"], 1)
        self.assertEqual(row["seed"], 777)
        self.assertEqual(row["attempts_for_this_image"], 2)
        self.assertIn("largest_olive_component_ratio", row)
        self.assertIn("physical_width_cm", row)
        self.assertIn("motif_scale", row)

    def test_write_report_with_rows_and_empty_rows(self) -> None:
        row = mut.candidate_row(1, 1, 1, make_candidate())
        csv1 = mut.write_report([row], self.tmpdir)
        csv2 = mut.write_report([], self.tmpdir, filename="empty.csv")
        self.assertTrue(csv1.exists())
        self.assertTrue(csv2.exists())
        with csv1.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["index"], "1")
        self.assertEqual(csv2.read_text(encoding="utf-8"), "")


class TestAsyncHelpersAndOrchestrator(
    GlobalStateMixin,
    TempDirMixin,
    GeometryMixin,
    AssertionsMixin,
    unittest.IsolatedAsyncioTestCase,
):
    def setUp(self) -> None:
        super().setUp()
        mut.set_canvas_geometry(128, 72, 40.0, 22.5)

    async def test_await_attempt(self) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        fut.set_result((make_candidate(seed=42), True))
        out = await mut._await_attempt(fut, 3, 42)
        self.assertEqual(out[0], 3)
        self.assertEqual(out[1], 42)
        self.assertTrue(out[3])

    async def test_console_progress_writes_to_stdout(self) -> None:
        counters = mut.LiveCounters(
            target_count=2,
            accepted=1,
            passed_validation=1,
            rejected=0,
            attempts=1,
            in_flight=0,
            start_ts=time.time() - 1.0,
        )
        buf = io.StringIO()
        with patch.object(sys, "stdout", buf):
            mut.console_progress(counters, current_target=1, workers=2)
        self.assertIn("fait=1/2", buf.getvalue())

    async def test_async_generate_all_accepts_first_attempt(self) -> None:
        candidate = make_candidate(seed=211)
        progress = AsyncMock()
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "generate_and_validate_from_seed", return_value=(candidate, True)):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                progress_callback=progress,
                enable_live_supervisor=False,
                strict_preflight=False,
                live_console=False,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 211)
        progress.assert_awaited_once()
        self.assertTrue((self.tmpdir / "camouflage_001.png").exists())
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())
        self.assertTrue((self.tmpdir / "run_summary.json").exists())

        summary = json.loads((self.tmpdir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["passed_validation"], 1)

    async def test_async_generate_all_retries_until_accept(self) -> None:
        side_effects = [
            (make_candidate(seed=301), False),
            (make_candidate(seed=302), True),
        ]
        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(1, 1, False, 0.9, "test")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "generate_and_validate_from_seed", side_effect=side_effects):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                enable_live_supervisor=False,
                strict_preflight=False,
                live_console=False,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 302)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)
        self.assertEqual(rows[0]["global_attempt"], 2)

        summary = json.loads((self.tmpdir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["passed_validation"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["attempts"], 2)

    async def test_async_generate_all_parallel_branch(self) -> None:
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            with patch.object(mut, "validate_generation_request", return_value=None), \
                 patch.object(mut, "_run_log_preflight", return_value=None), \
                 patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(2, 2, True, 0.9, "test")), \
                 patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
                 patch.object(mut, "get_process_pool", return_value=pool), \
                 patch.object(mut, "generate_and_validate_from_seed", side_effect=lambda seed: (make_candidate(seed=seed), seed % 2 == 0)):
                rows = await mut.async_generate_all(
                    target_count=1,
                    output_dir=self.tmpdir,
                    enable_live_supervisor=False,
                    strict_preflight=False,
                    live_console=False,
                )
        finally:
            pool.shutdown(wait=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts_for_this_image"], 2)
        self.assertEqual(rows[0]["seed"] % 2, 0)

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
                live_console=False,
            )
        self.assertEqual(rows, [])
        self.assertTrue((self.tmpdir / "rapport_camouflages.csv").exists())

    async def test_async_generate_all_supervisor_can_adjust_tuning(self) -> None:
        candidate = make_candidate(seed=401)

        def supervisor_side_effect(event_type: str, **payload: Any):
            if event_type == "generation_started":
                return {
                    "max_workers": 1,
                    "attempt_batch_size": 1,
                    "parallel_attempts": False,
                    "machine_intensity": 0.6,
                    "reason": "supervised",
                }
            return None

        with patch.object(mut, "validate_generation_request", return_value=None), \
             patch.object(mut, "_run_log_preflight", return_value=None), \
             patch.object(mut, "compute_runtime_tuning", return_value=mut.RuntimeTuning(4, 4, True, 0.9, "base")), \
             patch.object(mut, "sample_process_resources", return_value=fake_snapshot()), \
             patch.object(mut, "_supervisor_feedback", side_effect=supervisor_side_effect) as supervisor_mock, \
             patch.object(mut, "generate_and_validate_from_seed", return_value=(candidate, True)):
            rows = await mut.async_generate_all(
                target_count=1,
                output_dir=self.tmpdir,
                enable_live_supervisor=True,
                strict_preflight=False,
                live_console=False,
            )
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(supervisor_mock.call_count, 2)


class TestCliEntrypoints(GlobalStateMixin, TempDirMixin, GeometryMixin, AssertionsMixin, unittest.TestCase):
    def test_parse_cli_args_defaults(self) -> None:
        with patch.object(sys, "argv", ["prog"]):
            args = mut.parse_cli_args()
        self.assertEqual(args.target_count, mut.N_VARIANTS_REQUIRED)
        self.assertEqual(args.width, mut.DEFAULT_WIDTH)
        self.assertEqual(args.height, mut.DEFAULT_HEIGHT)
        self.assertFalse(args.disable_parallel_attempts)
        self.assertFloatClose(args.motif_scale, mut.DEFAULT_MOTIF_SCALE)

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

        geometry_mock.assert_called_once_with(
            width=128,
            height=72,
            physical_width_cm=40.0,
            physical_height_cm=22.5,
            motif_scale=0.61,
        )
        async_mock.assert_awaited_once()
        shutdown_mock.assert_called_once()


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_main.py ==========")
    unittest.main(verbosity=2)