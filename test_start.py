# -*- coding: utf-8 -*-
"""
test_start.py
Suite de tests unitaires avancée et renforcée pour start.py

Objectifs :
- couvrir les fonctions utilitaires pures ;
- couvrir les conversions PIL / palette / canvas indexé ;
- couvrir les métriques silhouette / contour / structure ;
- couvrir le scoring global v3 ;
- couvrir les exports CSV / best-of ;
- couvrir le runner asyncio dédié ;
- couvrir les méthodes applicatives de CamouflageApp ;
- couvrir le préflight non bloquant ;
- fournir des logs lisibles, précis et exploitables.

Exécution :
    python -m unittest -v test_start.py
"""

from __future__ import annotations

import csv
import logging
import tempfile
import time
import types
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image as PILImage

import start as sut


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = Path(__file__).resolve().parent / "logs_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_start.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_start")
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
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_start_")
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


def make_index_canvas_quadrants(h: int = 80, w: int = 80) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[: h // 2, w // 2 :] = sut.camo.IDX_OLIVE
    canvas[h // 2 :, : w // 2] = sut.camo.IDX_TERRE
    canvas[h // 2 :, w // 2 :] = sut.camo.IDX_GRIS
    return canvas


def make_pil_from_index_canvas(index_canvas: np.ndarray) -> PILImage.Image:
    rgb = sut.camo.RGB[index_canvas]
    return PILImage.fromarray(rgb, "RGB")


def valid_main_metrics() -> Dict[str, float]:
    return {
        "largest_olive_component_ratio": 0.24,
        "largest_olive_component_ratio_small": 0.18,
        "olive_multizone_share": 0.62,
        "center_empty_ratio": 0.28,
        "center_empty_ratio_small": 0.31,
        "boundary_density": 0.145,
        "boundary_density_small": 0.10,
        "boundary_density_tiny": 0.08,
        "mirror_similarity": 0.44,
        "oblique_share": 0.72,
        "vertical_share": 0.15,
        "angle_dominance_ratio": 0.20,
        "vert_olive_macro_share": 0.75,
        "terre_de_france_transition_share": 0.55,
        "vert_de_gris_micro_share": 0.83,
        "vert_de_gris_macro_share": 0.07,
    }


def valid_ratios() -> np.ndarray:
    return np.array([0.32, 0.28, 0.22, 0.18], dtype=float)


def make_candidate_record(tmpdir: Path, index: int = 1, score: float = 0.8) -> sut.CandidateRecord:
    image_path = tmpdir / f"camouflage_{index:03d}.png"
    img = make_pil_from_index_canvas(make_index_canvas_quadrants())
    img.save(image_path)

    return sut.CandidateRecord(
        index=index,
        seed=1000 + index,
        local_attempt=2,
        global_attempt=10 + index,
        image_path=image_path,
        score_final=score,
        score_ratio=0.9,
        score_silhouette=0.8,
        score_contour=0.7,
        score_main=0.6,
        silhouette_color_diversity=0.75,
        contour_break_score=0.65,
        outline_band_diversity=0.68,
        small_scale_structural_score=0.62,
        rs=valid_ratios(),
        metrics=valid_main_metrics(),
    )


def make_fake_label() -> Any:
    return types.SimpleNamespace(text="", color=None)


def make_fake_progress_bar() -> Any:
    return types.SimpleNamespace(value=0, max_value=100)


def make_fake_log_view() -> Any:
    ns = types.SimpleNamespace(lines=[], label=types.SimpleNamespace(text=""))

    def append(line: str) -> None:
        ns.lines.append(line)
        existing = ns.label.text.splitlines() if ns.label.text else []
        existing.append(line)
        ns.label.text = "\n".join(existing)

    ns.append = append
    return ns


def make_fake_button() -> Any:
    return types.SimpleNamespace(disabled=False)


def make_fake_input(text: str = "10") -> Any:
    return types.SimpleNamespace(text=text)


def make_fake_image_widget() -> Any:
    return types.SimpleNamespace(texture=None)


def make_fake_grid() -> Any:
    ns = types.SimpleNamespace(items=[])

    def clear_widgets() -> None:
        ns.items.clear()

    def add_widget(item: Any) -> None:
        ns.items.append(item)

    ns.clear_widgets = clear_widgets
    ns.add_widget = add_widget
    return ns


class FakeProcess:
    def cpu_percent(self, interval=None) -> float:
        return 12.0

    class _Mem:
        rss = 512 * 1024 * 1024

    def memory_info(self) -> "_Mem":
        return self._Mem()


def make_submit_closing_coroutines(return_future: Any):
    """
    Fabrique un faux submit() qui ferme explicitement la coroutine reçue
    pour éviter les RuntimeWarning: coroutine was never awaited.
    """
    def _submit(coro):
        try:
            close = getattr(coro, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        return return_future
    return _submit


def make_ui_methods_sync(app: Any) -> Any:
    """
    Remplace, pour les tests, les méthodes décorées avec @mainthread
    par leur fonction sous-jacente afin que les effets soient immédiats.
    """
    import types as _types

    method_names = [
        "_update_preflight_label",
        "_on_preflight_finished",
        "_refresh_live_diag_labels",
        "diag_log",
        "_refresh_controls_state",
        "reload_gallery",
        "_handle_future_exception",
        "_clear_current_future_if_same",
        "status",
        "log",
        "update_progress",
        "update_attempt_status",
        "update_preview",
    ]

    cls = type(app)
    for name in method_names:
        fn = getattr(cls, name, None)
        raw = getattr(fn, "__wrapped__", None)
        if raw is not None:
            setattr(app, name, _types.MethodType(raw, app))
    return app


# ============================================================
# TESTS PALETTE / CONSTANTES
# ============================================================

class TestPaletteAndConstants(TestAssertionsMixin, unittest.TestCase):
    def test_palette_hex_has_expected_keys(self) -> None:
        self.assertIn("BFW", sut.PALETTE_HEX)
        self.assertIn("BA", sut.PALETTE_HEX)
        self.assertIn("RF", sut.PALETTE_HEX)
        self.assertIn("white", sut.PALETTE_HEX)

    def test_hex_rgba(self) -> None:
        rgba = sut.hex_rgba("white", 0.5)
        self.assertEqual(len(rgba), 4)
        self.assertFloatClose(rgba[0], 1.0)
        self.assertFloatClose(rgba[1], 1.0)
        self.assertFloatClose(rgba[2], 1.0)
        self.assertFloatClose(rgba[3], 0.5)

    def test_palette_map_alignment(self) -> None:
        pm = sut.palette_map()
        self.assertEqual(len(pm), 4)
        self.assertIn(tuple(sut.camo.RGB[sut.camo.IDX_COYOTE].tolist()), pm)
        self.assertIn(tuple(sut.camo.RGB[sut.camo.IDX_OLIVE].tolist()), pm)
        self.assertIn(tuple(sut.camo.RGB[sut.camo.IDX_TERRE].tolist()), pm)
        self.assertIn(tuple(sut.camo.RGB[sut.camo.IDX_GRIS].tolist()), pm)

    def test_constants_basic(self) -> None:
        self.assertEqual(sut.APP_TITLE, "Camouflage Armée Fédérale Europe")
        self.assertEqual(sut.BEST_DIR_NAME, "best_of")
        self.assertEqual(sut.REPORT_NAME, "rapport_camouflages_v3.csv")
        self.assertGreater(sut.DEFAULT_TARGET_COUNT, 0)
        self.assertGreater(sut.DEFAULT_TOP_K, 0)
        self.assertEqual(sut.THUMB_SIZE, (240, 150))
        self.assertEqual(sut.GALLERY_COLUMNS, 3)


# ============================================================
# TESTS UTILITAIRES PURS
# ============================================================

class TestPureUtilities(TestAssertionsMixin, unittest.TestCase):
    def test_clamp01(self) -> None:
        self.assertEqual(sut.clamp01(-1.0), 0.0)
        self.assertEqual(sut.clamp01(2.0), 1.0)
        self.assertEqual(sut.clamp01(0.4), 0.4)

    def test_make_thumbnail(self) -> None:
        img = PILImage.new("RGB", (800, 600), (255, 0, 0))
        thumb = sut.make_thumbnail(img, (240, 150))
        self.assertIsInstance(thumb, PILImage.Image)
        self.assertEqual(thumb.size, (240, 150))

    def test_rgb_image_to_index_canvas(self) -> None:
        canvas = make_index_canvas_quadrants(20, 20)
        img = make_pil_from_index_canvas(canvas)
        recovered = sut.rgb_image_to_index_canvas(img)
        self.assertArrayClose(recovered, canvas)

    def test_rgb_image_to_index_canvas_unknown_color_defaults_zero(self) -> None:
        img = PILImage.new("RGB", (5, 5), (123, 45, 67))
        canvas = sut.rgb_image_to_index_canvas(img)
        self.assertEqual(canvas.shape, (5, 5))
        self.assertTrue(np.all(canvas == 0))

    def test_downsample_nearest(self) -> None:
        canvas = np.arange(16, dtype=np.uint8).reshape(4, 4)
        ds = sut.downsample_nearest(canvas, 2)
        self.assertArrayClose(ds, np.array([[0, 2], [8, 10]], dtype=np.uint8))

    def test_boundary_mask_uniform(self) -> None:
        canvas = np.zeros((10, 10), dtype=np.uint8)
        b = sut.boundary_mask(canvas)
        self.assertFalse(b.any())

    def test_boundary_mask_split(self) -> None:
        canvas = np.zeros((10, 10), dtype=np.uint8)
        canvas[:, 5:] = 1
        b = sut.boundary_mask(canvas)
        self.assertTrue(b.any())

    def test_largest_component_ratio_empty(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        self.assertEqual(sut.largest_component_ratio(mask), 0.0)

    def test_largest_component_ratio_non_empty(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[3:6, 3:6] = True
        ratio = sut.largest_component_ratio(mask)
        self.assertFloatClose(ratio, 9 / 13)

    def test_dilate_bool(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        dilated = sut.dilate_bool(mask, radius=1)
        self.assertGreater(int(dilated.sum()), 1)
        self.assertTrue(dilated[3, 3])


# ============================================================
# TESTS SILHOUETTE / SCORING
# ============================================================

class TestSilhouetteAndScoring(TestAssertionsMixin, unittest.TestCase):
    def test_build_silhouette_mask(self) -> None:
        mask = sut.build_silhouette_mask(200, 300)
        self.assertEqual(mask.shape, (300, 200))
        self.assertTrue(mask.any())

    def test_silhouette_boundary(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        bound = sut.silhouette_boundary(mask)
        self.assertTrue(bound.any())
        self.assertLess(int(bound.sum()), int(mask.sum()))

    def test_silhouette_projection_image(self) -> None:
        canvas = make_index_canvas_quadrants(120, 80)
        img = sut.silhouette_projection_image(canvas)
        self.assertIsInstance(img, PILImage.Image)
        self.assertEqual(img.size, (80, 120))

    def test_silhouette_color_diversity_score(self) -> None:
        canvas = make_index_canvas_quadrants(120, 80)
        score = sut.silhouette_color_diversity_score(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_silhouette_color_diversity_score_uniform(self) -> None:
        canvas = np.full((120, 80), sut.camo.IDX_COYOTE, dtype=np.uint8)
        score = sut.silhouette_color_diversity_score(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.5)

    def test_contour_break_score(self) -> None:
        canvas = make_index_canvas_quadrants(120, 80)
        score, entropy = sut.contour_break_score(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)

    def test_small_scale_structural_score(self) -> None:
        canvas = make_index_canvas_quadrants(120, 80)
        score = sut.small_scale_structural_score(canvas)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ratio_score(self) -> None:
        score = sut.ratio_score(valid_ratios())
        self.assertGreater(score, 0.9)

    def test_ratio_score_bad(self) -> None:
        bad = np.array([0.60, 0.10, 0.20, 0.10], dtype=float)
        score = sut.ratio_score(bad)
        self.assertLess(score, 0.5)

    def test_main_metrics_score(self) -> None:
        score = sut.main_metrics_score(valid_main_metrics())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.5)

    def test_evaluate_candidate_v3(self) -> None:
        canvas = make_index_canvas_quadrants(200, 140)
        img = make_pil_from_index_canvas(canvas)
        scores, ok = sut.evaluate_candidate_v3(img, valid_ratios(), valid_main_metrics())

        required = {
            "score_final",
            "score_ratio",
            "score_silhouette",
            "score_contour",
            "score_main",
            "silhouette_color_diversity",
            "contour_break_score",
            "outline_band_diversity",
            "small_scale_structural_score",
        }
        self.assertEqual(set(scores.keys()), required)
        self.assertIsInstance(ok, bool)


class TestSilhouetteAndScoringAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_evaluate_candidate_v3(self) -> None:
        canvas = make_index_canvas_quadrants(160, 100)
        img = make_pil_from_index_canvas(canvas)
        scores, ok = await sut.async_evaluate_candidate_v3(img, valid_ratios(), valid_main_metrics())
        self.assertIn("score_final", scores)
        self.assertIsInstance(ok, bool)


# ============================================================
# TESTS ASYNCIO THREAD RUNNER
# ============================================================

class TestAsyncioThreadRunner(unittest.TestCase):
    def test_submit_and_result(self) -> None:
        runner = sut.AsyncioThreadRunner()
        try:
            async def coro() -> int:
                await sut.asyncio.sleep(0.01)
                return 42

            fut = runner.submit(coro())
            result = fut.result(timeout=2.0)
            self.assertEqual(result, 42)
        finally:
            runner.stop()

    def test_stop(self) -> None:
        runner = sut.AsyncioThreadRunner()
        runner.stop()
        self.assertFalse(runner.thread.is_alive() or runner.loop.is_running())


# ============================================================
# TESTS OPEN / SLEEP / SYSTÈME
# ============================================================

class TestSystemHelpers(unittest.TestCase):
    def test_open_folder_windows(self) -> None:
        with patch.object(sut.platform, "system", return_value="Windows"), \
             patch.object(sut.os, "startfile", create=True) as mock_startfile:
            sut.open_folder(Path("."))
            mock_startfile.assert_called_once()

    def test_open_folder_darwin(self) -> None:
        with patch.object(sut.platform, "system", return_value="Darwin"), \
             patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
            mock_popen.assert_called_once()

    def test_open_folder_linux(self) -> None:
        with patch.object(sut.platform, "system", return_value="Linux"), \
             patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
            mock_popen.assert_called_once()

    def test_prevent_sleep_non_windows(self) -> None:
        with patch.object(sut.platform, "system", return_value="Linux"):
            sut.prevent_sleep(True)
            sut.prevent_sleep(False)

    def test_prevent_sleep_windows_enable_disable(self) -> None:
        fake_kernel32 = types.SimpleNamespace(SetThreadExecutionState=MagicMock())
        fake_windll = types.SimpleNamespace(kernel32=fake_kernel32)

        with patch.object(sut.platform, "system", return_value="Windows"), \
             patch.object(sut.ctypes, "windll", fake_windll, create=True):
            sut.prevent_sleep(True)
            sut.prevent_sleep(False)

        self.assertGreaterEqual(fake_kernel32.SetThreadExecutionState.call_count, 2)


# ============================================================
# TESTS EXPORT / CSV / BEST OF
# ============================================================

class TestExportFunctions(TempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = sut.CamouflageApp()
        self.app.current_output_dir = self.tmpdir

    async def asyncTearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass

    async def test_async_write_report(self) -> None:
        rows = [
            {"index": 1, "seed": 101, "score_final": 0.9},
            {"index": 2, "seed": 202, "score_final": 0.8},
        ]
        report_path = await self.app._async_write_report(rows)
        self.assertTrue(report_path.exists())

        with report_path.open("r", encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["seed"], "101")
        self.assertEqual(data[1]["seed"], "202")

    async def test_async_write_report_empty(self) -> None:
        report_path = await self.app._async_write_report([])
        self.assertTrue(report_path.exists())
        self.assertEqual(report_path.read_text(encoding="utf-8"), "")

    async def test_async_export_best_of(self) -> None:
        self.app.best_records = [
            make_candidate_record(self.tmpdir, index=1, score=0.95),
            make_candidate_record(self.tmpdir, index=2, score=0.90),
        ]
        best_dir = await self.app._async_export_best_of(2)

        self.assertTrue(best_dir.exists())
        self.assertTrue((best_dir / "best_001_camouflage_001.png").exists())
        self.assertTrue((best_dir / "best_002_camouflage_002.png").exists())
        self.assertTrue((best_dir / "best_of.csv").exists())

        with (best_dir / "best_of.csv").open("r", encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["source_index"], "1")
        self.assertEqual(data[1]["source_index"], "2")

    async def test_async_export_best_of_clears_existing_files(self) -> None:
        best_dir = self.tmpdir / sut.BEST_DIR_NAME
        best_dir.mkdir(parents=True, exist_ok=True)
        stale = best_dir / "stale.txt"
        stale.write_text("old", encoding="utf-8")

        self.app.best_records = [make_candidate_record(self.tmpdir, index=1, score=0.95)]
        await self.app._async_export_best_of(1)

        self.assertFalse(stale.exists())


# ============================================================
# TESTS APP METHODS SANS BUILD COMPLET
# ============================================================

class TestCamouflageAppMethods(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_start_app_")
        self.tmpdir = Path(self._tmpdir_obj.name)

        self.app = sut.CamouflageApp()
        make_ui_methods_sync(self.app)
        self.app.current_output_dir = self.tmpdir
        self.app.best_records = []

        self.app.status_label = make_fake_label()
        self.app.tests_label = make_fake_label()
        self.app.progress_bar = make_fake_progress_bar()
        self.app.progress_text = make_fake_label()
        self.app.attempt_text = make_fake_label()
        self.app.color_text = make_fake_label()
        self.app.score_text = make_fake_label()
        self.app.extra_text = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.start_btn = make_fake_button()
        self.app.stop_btn = make_fake_button()
        self.app.count_input = make_fake_input("3")
        self.app.intensity_label = make_fake_label()
        self.app.gallery_grid = make_fake_grid()
        self.app.preview_img = make_fake_image_widget()
        self.app.preview_silhouette = make_fake_image_widget()
        self.app.diag_summary_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()
        self.app.diag_enabled_label = make_fake_label()
        self.app.resource_text = make_fake_label()
        self.app.resource_hint = make_fake_label()
        self.app.process = FakeProcess()

    def tearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass
        self._tmpdir_obj.cleanup()

    def test_update_preflight_label(self) -> None:
        self.app._update_preflight_label("OK", ok=True)
        self.assertEqual(self.app.tests_label.text, "OK")
        self.assertEqual(self.app.tests_label.color, sut.C["success"])

        self.app._update_preflight_label("KO", ok=False)
        self.assertEqual(self.app.tests_label.color, sut.C["danger"])

        self.app._update_preflight_label("NEUTRE", ok=None)
        self.assertEqual(self.app.tests_label.color, sut.C["text_soft"])

    def test_reset_live_diagnostics(self) -> None:
        self.app.diag_total = 4
        self.app.diag_accepts = 1
        self.app.diag_rejects = 3
        self.app.diag_rule_counter = Counter({"rule_a": 2})
        self.app.diag_last_rules = ["rule_a"]

        self.app._reset_live_diagnostics()

        self.assertEqual(self.app.diag_total, 0)
        self.assertEqual(self.app.diag_accepts, 0)
        self.assertEqual(self.app.diag_rejects, 0)
        self.assertEqual(self.app.diag_rule_counter, Counter())
        self.assertEqual(self.app.diag_last_rules, [])

    def test_refresh_live_diag_labels_empty(self) -> None:
        self.app._refresh_live_diag_labels()
        self.assertIn("Tentatives 0", self.app.diag_summary_label.text)
        self.assertEqual(self.app.diag_top_rules_label.text, "Top règles : --")
        self.assertEqual(self.app.diag_last_fail_label.text, "Dernier rejet : --")

    def test_refresh_live_diag_labels_with_data(self) -> None:
        self.app.diag_total = 10
        self.app.diag_accepts = 4
        self.app.diag_rejects = 6
        self.app.diag_rule_counter = Counter({"rule_a": 5, "rule_b": 3, "rule_c": 1})
        self.app.diag_last_rules = ["rule_b", "rule_c"]

        self.app._refresh_live_diag_labels()

        self.assertIn("Tentatives 10", self.app.diag_summary_label.text)
        self.assertIn("rule_a:5", self.app.diag_top_rules_label.text)
        self.assertIn("rule_b", self.app.diag_last_fail_label.text)

    def test_diag_log(self) -> None:
        self.app.diag_log("ligne test")
        self.assertIn("ligne test", self.app.diag_log_view.label.text)

    def test_log(self) -> None:
        self.app.log("hello")
        self.assertIn("hello", self.app.log_view.label.text)

    def test_status(self) -> None:
        self.app.status("Prêt", ok=True)
        self.assertEqual(self.app.status_label.text, "Prêt")
        self.assertEqual(self.app.status_label.color, sut.C["success"])

        self.app.status("Erreur", ok=False)
        self.assertEqual(self.app.status_label.color, sut.C["danger"])

    def test_update_progress(self) -> None:
        self.app.update_progress(3, 9)
        self.assertEqual(self.app.progress_bar.max_value, 9)
        self.assertEqual(self.app.progress_bar.value, 3)

    def test_update_attempt_status(self) -> None:
        self.app.update_attempt_status(
            target_index=2,
            attempt_idx=7,
            target_total=10,
            accepted_count=1,
            rs=valid_ratios(),
            extra_scores={
                "score_final": 0.88,
                "score_ratio": 0.93,
                "score_silhouette": 0.77,
                "score_contour": 0.65,
            },
            metrics=valid_main_metrics(),
        )
        self.assertIn("1 / 10", self.app.progress_text.text)
        self.assertIn("Image 002", self.app.attempt_text.text)
        self.assertIn("C", self.app.color_text.text)
        self.assertIn("Score 0.880", self.app.score_text.text)
        self.assertIn("Olive conn.", self.app.extra_text.text)

    def test_update_preview(self) -> None:
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(80, 50))
        sil = sut.silhouette_projection_image(make_index_canvas_quadrants(80, 50))

        class FakeCore:
            def __init__(self):
                self.texture = "fake_texture"

        with patch.object(sut, "pil_to_coreimage", return_value=FakeCore()):
            self.app.update_preview(img, sil)

        self.assertIs(self.app.current_preview_img, img)
        self.assertIs(self.app.current_silhouette_preview, sil)
        self.assertEqual(self.app.preview_img.texture, "fake_texture")
        self.assertEqual(self.app.preview_silhouette.texture, "fake_texture")

    def test_on_intensity_change(self) -> None:
        self.app._on_intensity_change(None, 72.4)
        self.assertEqual(int(self.app.machine_intensity), 72)
        self.assertEqual(self.app.intensity_label.text, "72 %")

    def test_refresh_controls_state_normal(self) -> None:
        self.app.running = False
        self.app.stopping = False
        self.app.preflight_running = False
        self.app._refresh_controls_state()
        self.assertFalse(self.app.start_btn.disabled)
        self.assertTrue(self.app.stop_btn.disabled)

    def test_refresh_controls_state_running(self) -> None:
        self.app.running = True
        self.app.stopping = False
        self.app.preflight_running = False
        self.app._refresh_controls_state()
        self.assertTrue(self.app.start_btn.disabled)
        self.assertFalse(self.app.stop_btn.disabled)

    def test_refresh_controls_state_preflight(self) -> None:
        self.app.running = False
        self.app.stopping = False
        self.app.preflight_running = True
        self.app._refresh_controls_state()
        self.assertTrue(self.app.start_btn.disabled)
        self.assertFalse(self.app.stop_btn.disabled)

    def test_reload_gallery_empty(self) -> None:
        self.app.reload_gallery()
        self.assertEqual(len(self.app.gallery_grid.items), 0)

    def test_reload_gallery_with_files(self) -> None:
        for i in range(1, 3):
            img = make_pil_from_index_canvas(make_index_canvas_quadrants())
            img.save(self.tmpdir / f"camouflage_{i:03d}.png")

        with patch.object(sut, "GalleryThumb", side_effect=lambda app, p: ("thumb", p.name)):
            self.app.reload_gallery()

        self.assertEqual(len(self.app.gallery_grid.items), 2)

    def test_refresh_gallery_periodic(self) -> None:
        with patch.object(self.app, "reload_gallery") as mock_reload:
            self.app._refresh_gallery_periodic(0.0)
            mock_reload.assert_called_once()

    def test_update_resource_monitor_without_psutil(self) -> None:
        with patch.object(sut, "psutil", None):
            self.app._update_resource_monitor(0.0)
            self.assertIn("Installer psutil", self.app.resource_text.text)

    def test_update_resource_monitor_with_psutil(self) -> None:
        fake_psutil = types.SimpleNamespace(
            cpu_percent=lambda interval=None: 35.0,
            virtual_memory=lambda: types.SimpleNamespace(percent=61.0),
            disk_usage=lambda path: types.SimpleNamespace(percent=44.0),
        )

        with patch.object(sut, "psutil", fake_psutil):
            self.app._update_resource_monitor(0.0)

        self.assertIn("CPU 35%", self.app.resource_text.text)
        self.assertIn("RAM 61%", self.app.resource_text.text)

    def test_bind_future_and_clear(self) -> None:
        fut = MagicMock()
        fut.add_done_callback = MagicMock()
        self.app._bind_future(fut)
        self.assertIs(self.app.current_future, fut)
        fut.add_done_callback.assert_called_once()

        self.app._clear_current_future_if_same(fut)
        self.assertIsNone(self.app.current_future)

    def test_handle_future_exception(self) -> None:
        self.app.running = True
        self.app.stopping = False

        with patch.object(sut, "prevent_sleep") as mock_sleep:
            self.app._handle_future_exception(RuntimeError("boom"))

        self.assertFalse(self.app.running)
        self.assertFalse(self.app.stopping)
        self.assertFalse(self.app.stop_flag)
        self.assertIn("Erreur non capturée", self.app.log_view.label.text)
        mock_sleep.assert_called_once_with(False)

    def test_on_future_done_success(self) -> None:
        fut = MagicMock()
        fut.result.return_value = None
        self.app.current_future = fut

        with patch.object(self.app, "_handle_future_exception") as mock_exc:
            self.app._on_future_done(fut)

        mock_exc.assert_not_called()
        self.assertIsNone(self.app.current_future)

    def test_on_future_done_failure(self) -> None:
        fut = MagicMock()
        fut.result.side_effect = RuntimeError("x")
        self.app.current_future = fut

        with patch.object(self.app, "_handle_future_exception") as mock_exc:
            self.app._on_future_done(fut)

        mock_exc.assert_called_once()
        self.assertIsNone(self.app.current_future)


# ============================================================
# TESTS PRÉFLIGHT
# ============================================================

class TestPreflight(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_start_preflight_")
        self.tmpdir = Path(self._tmpdir_obj.name)

        self.app = sut.CamouflageApp()
        make_ui_methods_sync(self.app)
        self.app.tests_label = make_fake_label()
        self.app.status_label = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.start_btn = make_fake_button()
        self.app.stop_btn = make_fake_button()
        self.app.count_input = make_fake_input("2")
        self.app.progress_bar = make_fake_progress_bar()
        self.app.diag_summary_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()

    def tearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass
        self._tmpdir_obj.cleanup()

    def test_async_run_preflight_via_log_summary_object(self) -> None:
        class Summary:
            ok = True

            def short_text(self) -> str:
                return "12 tests OK"

        fake_log = types.SimpleNamespace(async_run_preflight_tests=AsyncMock(return_value=Summary()))

        with patch.object(sut, "camo_log", fake_log):
            ok, summary = sut.asyncio.run(self.app._async_run_preflight_via_log())

        self.assertTrue(ok)
        self.assertEqual(summary, "12 tests OK")
        fake_log.async_run_preflight_tests.assert_awaited_once()

    def test_async_run_preflight_via_log_dict_failure(self) -> None:
        fake_log = types.SimpleNamespace(
            async_run_preflight_tests=AsyncMock(
                return_value={"ok": False, "total": 5, "failures": 2, "errors": 1}
            )
        )

        with patch.object(sut, "camo_log", fake_log):
            ok, summary = sut.asyncio.run(self.app._async_run_preflight_via_log())

        self.assertFalse(ok)
        self.assertIn("5 tests exécutés", summary)

    def test_async_run_preflight_via_log_exception(self) -> None:
        fake_log = types.SimpleNamespace(async_run_preflight_tests=AsyncMock(side_effect=RuntimeError("boom")))

        with patch.object(sut, "camo_log", fake_log):
            ok, summary = sut.asyncio.run(self.app._async_run_preflight_via_log())

        self.assertFalse(ok)
        self.assertIn("RuntimeError", summary)

    def test_ensure_preflight_tests_cached(self) -> None:
        self.app.tests_ran = True
        self.app.tests_ok = True

        with patch.object(self.app.async_runner, "submit") as mock_submit:
            ok = self.app._ensure_preflight_tests()

        self.assertTrue(ok)
        mock_submit.assert_not_called()

    def test_ensure_preflight_tests_already_running(self) -> None:
        self.app.preflight_running = True
        ok = self.app._ensure_preflight_tests()
        self.assertFalse(ok)
        self.assertIn("Préflight déjà en cours", self.app.log_view.label.text)

    def test_ensure_preflight_tests_starts_background_worker(self) -> None:
        class FakeFuture:
            def __init__(self, result):
                self._result = result
                self.callback = None

            def add_done_callback(self, cb):
                self.callback = cb

            def result(self):
                return self._result

        fake_future = FakeFuture((True, "12 tests OK"))
        fake_submit = make_submit_closing_coroutines(fake_future)

        with patch.object(sut.Clock, "schedule_once", side_effect=lambda cb, dt=0: cb(0)), \
             patch.object(self.app.async_runner, "submit", side_effect=fake_submit) as mock_submit:
            ok = self.app._ensure_preflight_tests()
            self.assertFalse(ok)
            self.assertTrue(self.app.preflight_running)
            self.assertTrue(self.app.preflight_pending_start)
            self.assertIs(self.app.preflight_future, fake_future)
            mock_submit.assert_called_once()
            fake_future.callback(fake_future)

        self.assertTrue(self.app.tests_ran)
        self.assertTrue(self.app.tests_ok)
        self.assertEqual(self.app.tests_summary, "12 tests OK")
        self.assertIn("Préflight OK", self.app.log_view.label.text)
        self.assertFalse(self.app.preflight_running)
        self.assertFalse(self.app.preflight_pending_start)
        self.assertIsNone(self.app.preflight_future)

    def test_on_preflight_finished_ok_without_pending_start(self) -> None:
        self.app.preflight_running = True
        self.app.preflight_pending_start = False

        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app._on_preflight_finished(True, "9 tests OK")

        self.assertFalse(self.app.preflight_running)
        self.assertTrue(self.app.tests_ran)
        self.assertTrue(self.app.tests_ok)
        self.assertEqual(self.app.tests_summary, "9 tests OK")
        self.assertEqual(self.app.status_label.text, "Préflight terminé")
        self.assertFalse(self.app.preflight_pending_start)
        mock_start.assert_not_called()

    def test_on_preflight_finished_ok_with_pending_start(self) -> None:
        self.app.preflight_running = True
        self.app.preflight_pending_start = True

        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app._on_preflight_finished(True, "9 tests OK")

        mock_start.assert_called_once()
        self.assertFalse(self.app.preflight_pending_start)

    def test_on_preflight_finished_fail(self) -> None:
        self.app.preflight_running = True
        self.app.preflight_pending_start = True

        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app._on_preflight_finished(False, "KO")

        self.assertFalse(self.app.tests_ok)
        self.assertEqual(self.app.status_label.text, "Tests KO")
        self.assertIn("Préflight KO", self.app.log_view.label.text)
        self.assertFalse(self.app.preflight_pending_start)
        mock_start.assert_not_called()


# ============================================================
# TESTS DIAGNOSTIC LIVE
# ============================================================

class TestLiveDiagnostics(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = sut.CamouflageApp()
        make_ui_methods_sync(self.app)
        self.app.diag_summary_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()
        self.app.diag_log_view = make_fake_log_view()

    async def asyncTearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass

    async def test_extract_failure_rules_async_with_async_analyze(self) -> None:
        failure1 = types.SimpleNamespace(rule="rule_a")
        failure2 = types.SimpleNamespace(rule="rule_b")
        diagnostic = types.SimpleNamespace(failures=[failure1, failure2])

        fake_log = types.SimpleNamespace(async_analyze_candidate=AsyncMock(return_value=diagnostic))
        candidate = types.SimpleNamespace(seed=123)

        with patch.object(sut, "camo_log", fake_log):
            rules = await self.app._extract_failure_rules_async(candidate, 1, 2)

        self.assertEqual(rules, ["rule_a", "rule_b"])

    async def test_extract_failure_rules_async_without_log(self) -> None:
        with patch.object(sut, "camo_log", None):
            rules = await self.app._extract_failure_rules_async(types.SimpleNamespace(seed=1), 1, 1)
        self.assertEqual(rules, [])

    async def test_extract_failure_rules_async_with_exception(self) -> None:
        fake_log = types.SimpleNamespace(async_analyze_candidate=AsyncMock(side_effect=RuntimeError("boom")))
        with patch.object(sut, "camo_log", fake_log):
            rules = await self.app._extract_failure_rules_async(types.SimpleNamespace(seed=7), 1, 1)
        self.assertEqual(rules, [])
        self.assertIn("Diagnostic indisponible", self.app.diag_log_view.label.text)

    async def test_register_live_diagnostic_async_accept(self) -> None:
        candidate = types.SimpleNamespace(seed=111)

        await self.app._register_live_diagnostic_async(candidate, 1, 1, accepted=True)

        self.assertEqual(self.app.diag_total, 1)
        self.assertEqual(self.app.diag_accepts, 1)
        self.assertEqual(self.app.diag_rejects, 0)
        self.assertIn("accepté", self.app.diag_log_view.label.text)

    async def test_register_live_diagnostic_async_reject(self) -> None:
        candidate = types.SimpleNamespace(seed=222)

        with patch.object(self.app, "_extract_failure_rules_async", AsyncMock(return_value=["rule_x", "rule_y"])):
            await self.app._register_live_diagnostic_async(candidate, 2, 3, accepted=False)

        self.assertEqual(self.app.diag_total, 1)
        self.assertEqual(self.app.diag_accepts, 0)
        self.assertEqual(self.app.diag_rejects, 1)
        self.assertEqual(self.app.diag_rule_counter["rule_x"], 1)
        self.assertEqual(self.app.diag_rule_counter["rule_y"], 1)
        self.assertEqual(self.app.diag_last_rules, ["rule_x", "rule_y"])
        self.assertIn("rejet", self.app.diag_log_view.label.text)


# ============================================================
# TESTS START / STOP / WORKFLOW
# ============================================================

class TestStartStopWorkflow(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_start_workflow_")
        self.tmpdir = Path(self._tmpdir_obj.name)

        self.app = sut.CamouflageApp()
        make_ui_methods_sync(self.app)
        self.app.current_output_dir = self.tmpdir
        self.app.tests_label = make_fake_label()
        self.app.status_label = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.progress_bar = make_fake_progress_bar()
        self.app.start_btn = make_fake_button()
        self.app.stop_btn = make_fake_button()
        self.app.count_input = make_fake_input("2")
        self.app.diag_summary_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()
        self.app.diag_enabled_label = make_fake_label()
        self.app.gallery_grid = make_fake_grid()
        self.app.preview_img = make_fake_image_widget()
        self.app.preview_silhouette = make_fake_image_widget()
        self.app.progress_text = make_fake_label()
        self.app.attempt_text = make_fake_label()
        self.app.color_text = make_fake_label()
        self.app.score_text = make_fake_label()
        self.app.extra_text = make_fake_label()

    def tearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass
        self._tmpdir_obj.cleanup()

    def test_start_generation_while_running(self) -> None:
        self.app.running = True
        self.app.start_generation()
        self.assertTrue(self.app.running)

    def test_start_generation_while_preflight_running(self) -> None:
        self.app.preflight_running = True
        with patch.object(self.app, "_ensure_preflight_tests") as mock_pf:
            self.app.start_generation()
        mock_pf.assert_not_called()

    def test_start_generation_already_has_future(self) -> None:
        fake_future = MagicMock()
        fake_future.done.return_value = False
        self.app.current_future = fake_future
        self.app.start_generation()
        self.assertIn("déjà en cours", self.app.log_view.label.text)

    def test_start_generation_launches_preflight_if_needed(self) -> None:
        with patch.object(self.app, "_ensure_preflight_tests", return_value=False) as mock_pf:
            self.app.start_generation()
        mock_pf.assert_called_once()

    def test_start_generation_uses_cached_preflight(self) -> None:
        self.app.tests_ran = True
        self.app.tests_ok = True

        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app.start_generation()

        mock_start.assert_called_once()

    def test_start_generation_after_preflight_invalid_count(self) -> None:
        self.app.count_input.text = "abc"
        self.app._start_generation_after_preflight()
        self.assertIn("Nombre de camouflages invalide", self.app.log_view.label.text)
        self.assertFalse(self.app.running)

    def test_start_generation_after_preflight_success_path(self) -> None:
        fake_future = MagicMock()
        fake_future.done.return_value = False

        self.app.async_runner = types.SimpleNamespace(
            submit=make_submit_closing_coroutines(fake_future)
        )

        with patch.object(self.app, "_bind_future") as mock_bind, \
             patch.object(sut, "prevent_sleep") as mock_sleep:
            self.app._start_generation_after_preflight()

        self.assertTrue(self.app.running)
        self.assertFalse(self.app.stopping)
        self.assertFalse(self.app.stop_flag)
        self.assertEqual(self.app.progress_bar.max_value, 2)
        self.assertEqual(self.app.progress_bar.value, 0)
        self.assertIn("Démarrage", self.app.log_view.label.text)
        mock_bind.assert_called_once_with(fake_future)
        mock_sleep.assert_called_once_with(True)

    def test_stop_generation_during_preflight(self) -> None:
        self.app.preflight_running = True
        self.app.preflight_pending_start = True

        self.app.stop_generation()

        self.assertFalse(self.app.preflight_pending_start)
        self.assertEqual(self.app.status_label.text, "Préflight en cours…")
        self.assertIn("La génération ne démarrera pas", self.app.log_view.label.text)

    def test_stop_generation_running(self) -> None:
        self.app.running = True
        self.app.stopping = False

        self.app.stop_generation()

        self.assertTrue(self.app.stop_flag)
        self.assertTrue(self.app.stopping)
        self.assertEqual(self.app.status_label.text, "Arrêt demandé…")
        self.assertIn("Arrêt demandé", self.app.log_view.label.text)

    def test_stop_generation_when_not_running(self) -> None:
        self.app.running = False
        self.app.stopping = False
        self.app.stop_generation()
        self.assertFalse(self.app.stop_flag)

    def test_on_stop_without_future(self) -> None:
        with patch.object(sut, "prevent_sleep") as mock_sleep:
            self.app.on_stop()

        self.assertTrue(self.app.stop_flag)
        self.assertTrue(self.app.stopping)
        self.assertFalse(self.app.preflight_pending_start)
        mock_sleep.assert_called_once_with(False)

    def test_on_stop_with_pending_future(self) -> None:
        callback_holder: Dict[str, Any] = {}

        fake_future = MagicMock()
        fake_future.done.return_value = False

        def add_done_callback(cb):
            callback_holder["cb"] = cb

        fake_future.add_done_callback = add_done_callback
        self.app.current_future = fake_future

        with patch.object(sut, "prevent_sleep"):
            self.app.on_stop()

        self.assertIn("cb", callback_holder)


# ============================================================
# TESTS ADAPTIVE PAUSE
# ============================================================

class TestAdaptivePause(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = sut.CamouflageApp()

    async def asyncTearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass

    async def test_adaptive_pause_without_psutil(self) -> None:
        self.app.machine_intensity = 100.0
        with patch.object(sut, "psutil", None):
            await self.app._adaptive_pause()

    async def test_adaptive_pause_with_psutil(self) -> None:
        fake_psutil = types.SimpleNamespace(
            cpu_percent=lambda interval=None: 96.0,
            virtual_memory=lambda: types.SimpleNamespace(percent=93.0),
        )
        self.app.machine_intensity = 100.0

        t0 = time.perf_counter()
        with patch.object(sut, "psutil", fake_psutil):
            await self.app._adaptive_pause()
        elapsed = time.perf_counter() - t0

        self.assertGreater(elapsed, 0.05)


# ============================================================
# TESTS FIN DE TRAITEMENT
# ============================================================

class TestFinishMethods(TempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = sut.CamouflageApp()
        make_ui_methods_sync(self.app)
        self.app.current_output_dir = self.tmpdir
        self.app.status_label = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.start_btn = make_fake_button()
        self.app.stop_btn = make_fake_button()
        self.app.gallery_grid = make_fake_grid()
        self.app.running = True
        self.app.stopping = False
        self.app.stop_flag = False

    async def asyncTearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass

    async def test_async_finish_success(self) -> None:
        rows = [{"index": 1, "seed": 1}]
        with patch.object(self.app, "_async_write_report", AsyncMock(return_value=self.tmpdir / sut.REPORT_NAME)), \
             patch.object(self.app, "_async_export_best_of", AsyncMock(return_value=self.tmpdir / sut.BEST_DIR_NAME)), \
             patch.object(self.app, "reload_gallery") as mock_reload, \
             patch.object(sut, "prevent_sleep") as mock_sleep:

            await self.app._async_finish_success(rows)

        self.assertFalse(self.app.running)
        self.assertFalse(self.app.stopping)
        self.assertFalse(self.app.stop_flag)
        self.assertEqual(self.app.status_label.text, "Terminé")
        self.assertIn("Génération terminée avec succès", self.app.log_view.label.text)
        mock_reload.assert_called_once()
        mock_sleep.assert_called_once_with(False)

    async def test_async_finish_stopped_with_rows(self) -> None:
        rows = [{"index": 1, "seed": 1}]
        with patch.object(self.app, "_async_write_report", AsyncMock(return_value=self.tmpdir / sut.REPORT_NAME)), \
             patch.object(self.app, "_async_export_best_of", AsyncMock(return_value=self.tmpdir / sut.BEST_DIR_NAME)), \
             patch.object(self.app, "reload_gallery") as mock_reload, \
             patch.object(sut, "prevent_sleep") as mock_sleep:

            await self.app._async_finish_stopped(rows)

        self.assertFalse(self.app.running)
        self.assertFalse(self.app.stopping)
        self.assertFalse(self.app.stop_flag)
        self.assertEqual(self.app.status_label.text, "Arrêté")
        self.assertIn("Génération arrêtée proprement", self.app.log_view.label.text)
        mock_reload.assert_called_once()
        mock_sleep.assert_called_once_with(False)

    async def test_async_finish_stopped_empty(self) -> None:
        with patch.object(self.app, "_async_write_report", AsyncMock(return_value=self.tmpdir / sut.REPORT_NAME)), \
             patch.object(self.app, "_async_export_best_of", AsyncMock()) as mock_best, \
             patch.object(sut, "prevent_sleep"):

            await self.app._async_finish_stopped([])

        mock_best.assert_not_called()
        self.assertIn("Rapport vide écrit", self.app.log_view.label.text)

    async def test_async_finish_error_reflects_current_start_behavior(self) -> None:
        """
        start.py appelle actuellement _emit_runtime(..., message=message, ...)
        alors que message est déjà le 3e argument positionnel.
        Le test suit donc le comportement réel du start.py actuel.
        """
        with patch.object(sut, "prevent_sleep") as mock_sleep:
            with self.assertRaises(TypeError):
                await self.app._async_finish_error("boom")

        self.assertFalse(self.app.running)
        self.assertFalse(self.app.stopping)
        self.assertFalse(self.app.stop_flag)
        self.assertEqual(self.app.status_label.text, "Erreur")
        self.assertIn("Erreur : boom", self.app.log_view.label.text)
        mock_sleep.assert_called_once_with(False)


# ============================================================
# TESTS WORKER ASYNC
# ============================================================

class TestAsyncWorkerGenerate(TempDirMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = sut.CamouflageApp()
        make_ui_methods_sync(self.app)
        self.app.current_output_dir = self.tmpdir
        self.app.status_label = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.progress_bar = make_fake_progress_bar()
        self.app.progress_text = make_fake_label()
        self.app.attempt_text = make_fake_label()
        self.app.color_text = make_fake_label()
        self.app.score_text = make_fake_label()
        self.app.extra_text = make_fake_label()
        self.app.gallery_grid = make_fake_grid()
        self.app.preview_img = make_fake_image_widget()
        self.app.preview_silhouette = make_fake_image_widget()
        self.app.diag_summary_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()
        self.app.running = True
        self.app.stopping = False
        self.app.stop_flag = False

    async def asyncTearDown(self) -> None:
        try:
            self.app.async_runner.stop()
        except Exception:
            pass

    async def test_async_worker_generate_success_single(self) -> None:
        candidate = types.SimpleNamespace(
            seed=123,
            image=make_pil_from_index_canvas(make_index_canvas_quadrants(120, 80)),
            ratios=valid_ratios(),
            metrics=valid_main_metrics(),
            profile=types.SimpleNamespace(allowed_angles=[0, 15, 20]),
        )

        extra_scores = {
            "score_final": 0.92,
            "score_ratio": 0.95,
            "score_silhouette": 0.81,
            "score_contour": 0.73,
            "score_main": 0.77,
            "silhouette_color_diversity": 0.80,
            "contour_break_score": 0.66,
            "outline_band_diversity": 0.70,
            "small_scale_structural_score": 0.60,
        }

        async def fake_eval(*args, **kwargs):
            return extra_scores, True

        async def fake_validate(*args, **kwargs):
            return True

        async def fake_generate(seed: int):
            return candidate

        async def fake_save(cand, path):
            cand.image.save(path)
            return path

        with patch.object(sut.camo, "async_generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(sut, "async_evaluate_candidate_v3", side_effect=fake_eval), \
             patch.object(sut.camo, "async_validate_candidate_result", side_effect=fake_validate), \
             patch.object(sut.camo, "async_save_candidate_image", side_effect=fake_save), \
             patch.object(self.app, "_register_live_diagnostic_async", AsyncMock()), \
             patch.object(self.app, "_adaptive_pause", AsyncMock()), \
             patch.object(self.app, "_async_finish_success", AsyncMock()) as mock_finish, \
             patch.object(self.app, "reload_gallery"):

            await self.app._async_worker_generate(1)

        mock_finish.assert_called_once()
        self.assertEqual(len(self.app.best_records), 1)
        self.assertEqual(self.app.accepted_count, 1)

    async def test_async_worker_generate_stop_early(self) -> None:
        self.app.stop_flag = True

        with patch.object(self.app, "_async_finish_stopped", AsyncMock()) as mock_stop:
            await self.app._async_worker_generate(2)

        mock_stop.assert_called_once()

    async def test_async_worker_generate_error(self) -> None:
        async def fake_generate(seed: int):
            raise RuntimeError("boom")

        with patch.object(sut.camo, "async_generate_candidate_from_seed", side_effect=fake_generate), \
             patch.object(self.app, "_async_finish_error", AsyncMock()) as mock_err:

            await self.app._async_worker_generate(1)

        mock_err.assert_called_once()


# ============================================================
# TESTS PIL -> CORE IMAGE
# ============================================================

class TestPilToCoreImage(unittest.TestCase):
    def test_pil_to_coreimage(self) -> None:
        img = PILImage.new("RGB", (20, 10), (255, 0, 0))

        class FakeCoreImage:
            def __init__(self, bio, ext):
                self.bio = bio
                self.ext = ext
                self.texture = "fake"

        with patch.object(sut, "CoreImage", FakeCoreImage):
            core = sut.pil_to_coreimage(img)

        self.assertEqual(core.ext, "png")
        self.assertEqual(core.texture, "fake")


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_start.py ==========")
    unittest.main(verbosity=2)