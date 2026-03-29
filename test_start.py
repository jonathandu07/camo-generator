# -*- coding: utf-8 -*-
"""
test_start.py
Suite de tests exhaustive pour le start.py actuellement chargé.

Buts :
- couvrir les fonctions utilitaires, conversions et scoring ;
- couvrir les exports et le runner asyncio ;
- couvrir les branches importantes de CamouflageApp ;
- produire des logs de test précis et exploitables.
"""

from __future__ import annotations

import csv
import functools
import logging
import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image as PILImage


# ============================================================
# FAUX KIVY
# ============================================================

def _install_fake_kivy() -> None:
    if "kivy" in sys.modules:
        return

    class _CanvasCtx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def clear(self):
            return None

    class _Canvas:
        def __init__(self):
            self.before = _CanvasCtx()
            self.after = _CanvasCtx()
        def clear(self):
            return None

    class _DummyWidget:
        def __init__(self, **kwargs):
            self.children = []
            self.canvas = _Canvas()
            self.size = kwargs.get("size", (0, 0))
            self.pos = kwargs.get("pos", (0, 0))
            self.width = kwargs.get("width", 0)
            self.height = kwargs.get("height", 0)
            self.x = kwargs.get("x", 0)
            self.y = kwargs.get("y", 0)
            self.text = kwargs.get("text", "")
            self.texture = kwargs.get("texture", None)
            self.texture_size = kwargs.get("texture_size", (0, 0))
            self.size_hint = kwargs.get("size_hint", (1, 1))
            self.size_hint_x = kwargs.get("size_hint_x", 1)
            self.size_hint_y = kwargs.get("size_hint_y", 1)
            self.disabled = kwargs.get("disabled", False)
            self.state = kwargs.get("state", "normal")
            self.color = kwargs.get("color", None)
            self.value = kwargs.get("value", 0)
            self.max_value = kwargs.get("max_value", 100)
            self.minimum_height = kwargs.get("minimum_height", 0)
            self.text_size = kwargs.get("text_size", None)
            self.bar_width = kwargs.get("bar_width", 0)
            self.scroll_y = kwargs.get("scroll_y", 1)
            self.role = kwargs.get("role", "neutral")
        def bind(self, **kwargs):
            return None
        def add_widget(self, widget):
            self.children.append(widget)
        def clear_widgets(self):
            self.children.clear()
        def setter(self, name):
            def _set(_instance, value):
                setattr(self, name, value)
            return _set
        def _redraw(self, *_):
            return None

    def _prop(default=None):
        return default

    def _identity_decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return _wrapped

    mods = {}
    for name in [
        "kivy", "kivy.config", "kivy.app", "kivy.clock", "kivy.core.image", "kivy.core.window",
        "kivy.graphics", "kivy.metrics", "kivy.properties", "kivy.uix.boxlayout", "kivy.uix.button",
        "kivy.uix.gridlayout", "kivy.uix.image", "kivy.uix.label", "kivy.uix.scrollview",
        "kivy.uix.slider", "kivy.uix.textinput", "kivy.uix.widget",
    ]:
        mods[name] = types.ModuleType(name)

    class _Config:
        @staticmethod
        def set(*args, **kwargs):
            return None

    class _App:
        def __init__(self, **kwargs):
            super().__init__()
        def run(self):
            return None

    class _Clock:
        @staticmethod
        def schedule_once(*args, **kwargs):
            return None
        @staticmethod
        def schedule_interval(*args, **kwargs):
            return None

    class _CoreImage:
        def __init__(self, *args, **kwargs):
            self.texture = None

    class _Window:
        clearcolor = None
        @staticmethod
        def maximize():
            return None

    class _Graphic:
        def __init__(self, *args, **kwargs):
            return None

    mods["kivy.config"].Config = _Config
    mods["kivy.app"].App = _App
    mods["kivy.clock"].Clock = _Clock
    mods["kivy.clock"].mainthread = _identity_decorator
    mods["kivy.core.image"].Image = _CoreImage
    mods["kivy.core.window"].Window = _Window
    mods["kivy.graphics"].Color = _Graphic
    mods["kivy.graphics"].Line = _Graphic
    mods["kivy.graphics"].RoundedRectangle = _Graphic
    mods["kivy.metrics"].dp = lambda x: x
    mods["kivy.metrics"].sp = lambda x: x
    mods["kivy.properties"].NumericProperty = _prop
    mods["kivy.properties"].StringProperty = _prop
    mods["kivy.uix.boxlayout"].BoxLayout = _DummyWidget
    mods["kivy.uix.button"].Button = _DummyWidget
    mods["kivy.uix.gridlayout"].GridLayout = _DummyWidget
    mods["kivy.uix.image"].Image = _DummyWidget
    mods["kivy.uix.label"].Label = _DummyWidget
    mods["kivy.uix.scrollview"].ScrollView = _DummyWidget
    mods["kivy.uix.slider"].Slider = _DummyWidget
    mods["kivy.uix.textinput"].TextInput = _DummyWidget
    mods["kivy.uix.widget"].Widget = _DummyWidget

    for name, mod in mods.items():
        sys.modules[name] = mod


try:
    import kivy  # type: ignore # noqa
except Exception:
    _install_fake_kivy()

import start as sut


# ============================================================
# LOGS
# ============================================================

LOG_DIR = Path(os.getenv("LOG_OUTPUT_DIR", Path(__file__).resolve().parent / "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_start_exhaustive.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_start_exhaustive")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


LOGGER = configure_logger()


class LoggedTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        LOGGER.info("START %s", self.id())
    def tearDown(self):
        LOGGER.info("END   %s", self.id())
        super().tearDown()


class LoggedAsyncCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        LOGGER.info("START %s", self.id())
    def tearDown(self):
        LOGGER.info("END   %s", self.id())
        super().tearDown()


# ============================================================
# HELPERS
# ============================================================

def make_index_canvas_quadrants(h: int = 80, w: int = 80) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[: h // 2, w // 2 :] = sut.camo.IDX_OLIVE
    canvas[h // 2 :, : w // 2] = sut.camo.IDX_TERRE
    canvas[h // 2 :, w // 2 :] = sut.camo.IDX_GRIS
    return canvas


def make_pil_from_index_canvas(index_canvas: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(sut.camo.RGB[index_canvas], "RGB")


def valid_ratios() -> np.ndarray:
    return np.array([0.32, 0.28, 0.22, 0.18], dtype=float)


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
        "vert_de_gris_macro_share": 0.05,
        "visual_score_final": 0.78,
        "visual_score_ratio": 0.92,
        "visual_score_silhouette": 0.80,
        "visual_score_contour": 0.74,
        "visual_score_main": 0.70,
        "visual_silhouette_color_diversity": 0.80,
        "visual_contour_break_score": 0.66,
        "visual_outline_band_diversity": 0.70,
        "visual_small_scale_structural_score": 0.60,
        "visual_military_score": 0.72,
        "macro_olive_visible_ratio": 0.14,
        "macro_terre_visible_ratio": 0.10,
        "macro_gris_visible_ratio": 0.08,
        "transition_terre_visible_ratio": 0.12,
        "micro_gris_visible_ratio": 0.14,
        "periphery_boundary_density": 0.18,
        "center_boundary_density": 0.12,
        "periphery_boundary_density_ratio": 1.50,
        "periphery_non_coyote_density": 0.72,
        "center_non_coyote_density": 0.58,
        "periphery_non_coyote_ratio": 1.24,
    }


def make_candidate_record(tmpdir: Path, index: int = 1, score: float = 0.8) -> sut.CandidateRecord:
    image_path = tmpdir / f"camouflage_{index:03d}.png"
    make_pil_from_index_canvas(make_index_canvas_quadrants()).save(image_path)
    return sut.CandidateRecord(
        index=index,
        seed=1000 + index,
        local_attempt=2,
        global_attempt=10 + index,
        image_path=image_path,
        score_final=score,
        score_ratio=0.91,
        score_silhouette=0.81,
        score_contour=0.72,
        score_main=0.66,
        silhouette_color_diversity=0.78,
        contour_break_score=0.64,
        outline_band_diversity=0.68,
        small_scale_structural_score=0.61,
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
    return types.SimpleNamespace(disabled=False, text="", role="neutral", _redraw=lambda *a, **k: None)


def make_fake_input(text: str = "10") -> Any:
    return types.SimpleNamespace(text=text)


def make_fake_image_widget() -> Any:
    return types.SimpleNamespace(texture=None)


def make_fake_grid() -> Any:
    ns = types.SimpleNamespace(items=[])
    ns.clear_widgets = lambda: ns.items.clear()
    ns.add_widget = lambda item: ns.items.append(item)
    return ns


class FakeProcess:
    def cpu_percent(self, interval=None) -> float:
        return 12.0
    class _Mem:
        rss = 512 * 1024 * 1024
    def memory_info(self):
        return self._Mem()


def make_submit_closing_coroutines(return_future: Any):
    def _submit(coro):
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        return return_future
    return _submit


def make_ui_methods_sync(app: Any) -> Any:
    import types as _types
    method_names = [
        "_update_preflight_label",
        "_on_preflight_finished",
        "_refresh_live_diag_labels",
        "diag_log",
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
# TESTS FONCTIONS PURES
# ============================================================

class TestHelpers(LoggedTestCase):
    def test_hex_rgba(self):
        rgba = sut.hex_rgba("BL", 0.5)
        self.assertEqual(len(rgba), 4)
        self.assertAlmostEqual(rgba[3], 0.5)

    def test_constants(self):
        self.assertEqual(sut.APP_TITLE, "Camouflage Armée Fédérale Europe")
        self.assertEqual(sut.BEST_DIR_NAME, "best_of")
        self.assertEqual(sut.REPORT_NAME, "rapport_camouflages_v3.csv")
        self.assertEqual(sut.GALLERY_COLUMNS, 3)

    def test_make_thumbnail(self):
        img = PILImage.new("RGB", (800, 600), (255, 0, 0))
        thumb = sut.make_thumbnail(img, (240, 150))
        self.assertEqual(thumb.size, (240, 150))

    def test_palette_map(self):
        pm = sut.palette_map()
        self.assertEqual(len(pm), 4)
        self.assertIn(tuple(sut.camo.RGB[sut.camo.IDX_COYOTE].tolist()), pm)

    def test_rgb_image_to_index_canvas(self):
        canvas = make_index_canvas_quadrants(20, 20)
        img = make_pil_from_index_canvas(canvas)
        recovered = sut.rgb_image_to_index_canvas(img)
        np.testing.assert_array_equal(recovered, canvas)

    def test_downsample_nearest(self):
        canvas = np.arange(16, dtype=np.uint8).reshape(4, 4)
        ds = sut.downsample_nearest(canvas, 2)
        np.testing.assert_array_equal(ds, np.array([[0, 2], [8, 10]], dtype=np.uint8))

    def test_boundary_mask_uniform(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        self.assertFalse(sut.boundary_mask(canvas).any())

    def test_boundary_mask_split(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        canvas[:, 5:] = 1
        self.assertTrue(sut.boundary_mask(canvas).any())

    def test_largest_component_ratio(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[3:6, 3:6] = True
        self.assertAlmostEqual(sut.largest_component_ratio(mask), 9 / 13)

    def test_dilate_bool(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        dilated = sut.dilate_bool(mask, radius=1)
        self.assertTrue(dilated[3, 3])
        self.assertGreater(int(dilated.sum()), 1)

    def test_build_silhouette_mask(self):
        mask = sut.build_silhouette_mask(200, 300)
        self.assertEqual(mask.shape, (300, 200))
        self.assertTrue(mask.any())

    def test_silhouette_boundary(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        bound = sut.silhouette_boundary(mask)
        self.assertTrue(bound.any())
        self.assertLess(int(bound.sum()), int(mask.sum()))

    def test_silhouette_projection_image(self):
        canvas = make_index_canvas_quadrants(120, 80)
        img = sut.silhouette_projection_image(canvas)
        self.assertEqual(img.size, (80, 120))

    def test_silhouette_color_diversity_score(self):
        score = sut.silhouette_color_diversity_score(make_index_canvas_quadrants(120, 80))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_contour_break_score(self):
        score, entropy = sut.contour_break_score(make_index_canvas_quadrants(120, 80))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)

    def test_small_scale_structural_score(self):
        score = sut.small_scale_structural_score(make_index_canvas_quadrants(120, 80))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ratio_score(self):
        self.assertGreater(sut.ratio_score(valid_ratios()), 0.9)
        self.assertLess(sut.ratio_score(np.array([0.60, 0.10, 0.20, 0.10], dtype=float)), 0.5)

    def test_main_metrics_score(self):
        score = sut.main_metrics_score(valid_main_metrics())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_candidate_v3(self):
        canvas = make_index_canvas_quadrants(200, 140)
        img = make_pil_from_index_canvas(canvas)
        scores, ok = sut.evaluate_candidate_v3(img, valid_ratios(), valid_main_metrics())
        self.assertIn("score_final", scores)
        self.assertIn("small_scale_structural_score", scores)
        self.assertIsInstance(ok, bool)


# ============================================================
# TESTS SYSTÈME
# ============================================================

class TestSystemHelpers(LoggedTestCase):
    def test_pil_to_coreimage(self):
        img = PILImage.new("RGB", (20, 10), (255, 0, 0))
        class FakeCoreImage:
            def __init__(self, bio, ext):
                self.texture = "fake"
                self.ext = ext
        with patch.object(sut, "CoreImage", FakeCoreImage):
            core = sut.pil_to_coreimage(img)
        self.assertEqual(core.texture, "fake")
        self.assertEqual(core.ext, "png")

    def test_asyncio_thread_runner(self):
        runner = sut.AsyncioThreadRunner()
        try:
            async def coro():
                await sut.asyncio.sleep(0.01)
                return 42
            fut = runner.submit(coro())
            self.assertEqual(fut.result(timeout=2.0), 42)
        finally:
            runner.stop()

    def test_open_folder_windows(self):
        with patch.object(sut.platform, "system", return_value="Windows"), patch.object(sut.os, "startfile", create=True) as mock_start:
            sut.open_folder(Path("."))
        mock_start.assert_called_once()

    def test_open_folder_darwin(self):
        with patch.object(sut.platform, "system", return_value="Darwin"), patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
        mock_popen.assert_called_once()

    def test_open_folder_linux(self):
        with patch.object(sut.platform, "system", return_value="Linux"), patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
        mock_popen.assert_called_once()

    def test_prevent_sleep_non_windows(self):
        with patch.object(sut.platform, "system", return_value="Linux"):
            sut.prevent_sleep(True)
            sut.prevent_sleep(False)

    def test_prevent_sleep_windows(self):
        fake_kernel32 = types.SimpleNamespace(SetThreadExecutionState=MagicMock())
        fake_windll = types.SimpleNamespace(kernel32=fake_kernel32)
        with patch.object(sut.platform, "system", return_value="Windows"), patch.object(sut.ctypes, "windll", fake_windll, create=True):
            sut.prevent_sleep(True)
            sut.prevent_sleep(False)
        self.assertGreaterEqual(fake_kernel32.SetThreadExecutionState.call_count, 2)


# ============================================================
# APP SYNC
# ============================================================

class TestCamouflageAppMethods(LoggedTestCase):
    def setUp(self):
        super().setUp()
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_start_app_")
        self.tmpdir = Path(self.tmpdir_obj.name)
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
        self.app.struct_text = make_fake_label()
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
        self.app.backend_stats_label = make_fake_label()
        self.app.backend_state_label = make_fake_label()
        self.app.resource_text = make_fake_label()
        self.app.resource_hint = make_fake_label()
        self.app.run_mode_label = make_fake_label()
        self.app.mode_blocking_btn = make_fake_button()
        self.app.mode_non_blocking_btn = make_fake_button()
        self.app.mode_skip_tests_btn = make_fake_button()
        self.app.runtime_last_label = make_fake_label()
        self.app.process = FakeProcess()

    def tearDown(self):
        try:
            self.app.async_runner.stop()
        except Exception:
            pass
        self.tmpdir_obj.cleanup()
        super().tearDown()

    def test_update_preflight_label(self):
        self.app._update_preflight_label("OK", ok=True)
        self.assertEqual(self.app.tests_label.color, sut.C["success"])
        self.app._update_preflight_label("KO", ok=False)
        self.assertEqual(self.app.tests_label.color, sut.C["danger"])

    def test_reset_live_diagnostics(self):
        self.app.diag_total = 4
        self.app.diag_accepts = 1
        self.app.diag_rejects = 3
        self.app.diag_rule_counter = Counter({"rule_a": 2})
        self.app.diag_last_rules = ["rule_a"]
        self.app._reset_live_diagnostics()
        self.assertEqual(self.app.diag_total, 0)
        self.assertEqual(self.app.diag_rule_counter, Counter())
        self.assertEqual(self.app.diag_log_view.label.text, "")

    def test_refresh_live_diag_labels(self):
        self.app._refresh_live_diag_labels()
        self.assertIn("Tentatives 0", self.app.diag_summary_label.text)
        self.app.diag_total = 10
        self.app.diag_accepts = 4
        self.app.diag_rejects = 6
        self.app.diag_rule_counter = Counter({"rule_a": 5, "rule_b": 3})
        self.app.diag_last_rules = ["rule_b", "rule_c"]
        self.app.running = True
        self.app._refresh_live_diag_labels()
        self.assertIn("rule_a:5", self.app.diag_top_rules_label.text)
        self.assertIn("rule_b", self.app.diag_last_fail_label.text)
        self.assertIn("génération active", self.app.backend_state_label.text)

    def test_status_log_diag(self):
        self.app.status("Prêt", ok=True)
        self.assertEqual(self.app.status_label.text, "Prêt")
        self.app.log("hello")
        self.app.diag_log("diag")
        self.assertIn("hello", self.app.log_view.label.text)
        self.assertIn("diag", self.app.diag_log_view.label.text)

    def test_update_progress(self):
        self.app.update_progress(3, 9)
        self.assertEqual(self.app.progress_bar.max_value, 9)
        self.assertEqual(self.app.progress_bar.value, 3)

    def test_update_attempt_status(self):
        self.app.update_attempt_status(
            target_index=2,
            attempt_idx=7,
            global_attempt=11,
            seed=123,
            target_total=10,
            accepted_count=1,
            rejected_count=5,
            accepted=False,
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
        self.assertIn("Score 0.880", self.app.score_text.text)
        self.assertEqual(self.app.struct_text.text, "")

    def test_update_preview(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(80, 50))
        sil = sut.silhouette_projection_image(make_index_canvas_quadrants(80, 50))
        class FakeCore:
            def __init__(self):
                self.texture = "fake_texture"
        with patch.object(sut, "pil_to_coreimage", return_value=FakeCore()):
            self.app.update_preview(img, sil)
        self.assertEqual(self.app.preview_img.texture, "fake_texture")
        self.assertEqual(self.app.preview_silhouette.texture, "fake_texture")

    def test_on_intensity_change(self):
        self.app._on_intensity_change(None, 72.4)
        self.assertEqual(int(self.app.machine_intensity), 72)
        self.assertEqual(self.app.intensity_label.text, "72 %")

    def test_refresh_controls_state(self):
        self.app.running = False
        self.app.preflight_running = False
        self.app.stopping = False
        self.app._refresh_controls_state()
        self.assertFalse(self.app.start_btn.disabled)
        self.assertTrue(self.app.stop_btn.disabled)
        self.app.running = True
        self.app._refresh_controls_state()
        self.assertTrue(self.app.start_btn.disabled)

    def test_run_mode_helpers(self):
        self.assertEqual(self.app._run_mode_text(sut.RUN_MODE_SKIP_TESTS), "sans tests")
        self.app._set_run_mode(sut.RUN_MODE_NON_BLOCKING)
        self.assertEqual(self.app.run_mode, sut.RUN_MODE_NON_BLOCKING)
        self.assertIn("non bloquants", self.app.log_view.label.text)

    def test_start_generation_skip_tests(self):
        self.app.run_mode = sut.RUN_MODE_SKIP_TESTS
        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app.start_generation()
        mock_start.assert_called_once()

    def test_start_generation_non_blocking(self):
        self.app.run_mode = sut.RUN_MODE_NON_BLOCKING
        with patch.object(self.app, "_ensure_preflight_tests", return_value=False) as mock_pf, patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app.start_generation()
        mock_pf.assert_called_once_with(pending_start=False)
        mock_start.assert_called_once_with(allow_during_preflight=True)

    def test_start_generation_blocking(self):
        with patch.object(self.app, "_ensure_preflight_tests", return_value=False) as mock_pf:
            self.app.start_generation()
        mock_pf.assert_called_once_with(pending_start=True)

    def test_start_generation_after_preflight_invalid(self):
        self.app.count_input.text = "abc"
        self.app._start_generation_after_preflight()
        self.assertIn("Nombre de camouflages invalide", self.app.log_view.label.text)

    def test_start_generation_after_preflight_success(self):
        fake_future = MagicMock()
        fake_future.done.return_value = False
        fake_future.add_done_callback = MagicMock()
        self.app.async_runner = types.SimpleNamespace(submit=make_submit_closing_coroutines(fake_future))
        with patch.object(sut, "prevent_sleep") as mock_sleep:
            self.app._start_generation_after_preflight()
        self.assertTrue(self.app.running)
        self.assertIs(self.app.current_future, fake_future)
        mock_sleep.assert_called_once_with(True)
        fake_future.add_done_callback.assert_called_once()

    def test_stop_generation(self):
        self.app.running = True
        self.app.stop_generation()
        self.assertTrue(self.app.stop_flag)
        self.assertTrue(self.app.stopping)
        self.app.running = False
        self.app.preflight_running = True
        self.app.stop_generation()
        self.assertIn("préflight", self.app.log_view.label.text.lower())

    def test_on_preflight_finished(self):
        self.app.preflight_pending_start = True
        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app._on_preflight_finished(True, "OK")
        self.assertTrue(self.app.tests_ok)
        mock_start.assert_called_once()
        self.app.preflight_pending_start = True
        self.app._on_preflight_finished(False, "KO")
        self.assertFalse(self.app.tests_ok)
        self.assertIn("KO", self.app.tests_label.text)

    def test_update_resource_monitor(self):
        with patch.object(sut.psutil, "cpu_percent", return_value=10.0), patch.object(sut.psutil, "virtual_memory", return_value=types.SimpleNamespace(percent=20.0)), patch.object(sut.psutil, "disk_usage", return_value=types.SimpleNamespace(percent=30.0)):
            self.app._update_resource_monitor(None)
        self.assertIn("CPU 10%", self.app.resource_text.text)

    def test_runtime_format_and_subscription(self):
        event = types.SimpleNamespace(ts=0, level="info", source="src", message="msg", payload={"a": 1})
        line = self.app._format_runtime_event_line(event)
        self.assertIn("src", line)
        manager_calls = []
        manager = types.SimpleNamespace(subscribe=lambda cb: manager_calls.append(("sub", cb)), unsubscribe=lambda cb: manager_calls.append(("unsub", cb)))
        with patch.object(sut, "camo_log", types.SimpleNamespace(LOG_MANAGER=manager, log_event=lambda *a, **k: None, get_recent_runtime_lines=lambda n: ["L1", "L2"])):
            self.app._subscribe_runtime_feed()
            self.app._bootstrap_runtime_feed()
            self.app._unsubscribe_runtime_feed()
        self.assertEqual(manager_calls[0][0], "sub")
        self.assertEqual(manager_calls[1][0], "unsub")
        self.assertIn("L1", self.app.log_view.label.text)

    def test_on_start_and_on_stop(self):
        fut = MagicMock()
        fut.done.return_value = True
        self.app.current_future = fut
        stopped = {"n": 0}
        self.app.async_runner = types.SimpleNamespace(stop=lambda: stopped.__setitem__("n", stopped["n"] + 1))
        with patch.object(sut.Window, "maximize"), patch.object(self.app, "_subscribe_runtime_feed") as mock_sub, patch.object(self.app, "_unsubscribe_runtime_feed") as mock_unsub, patch.object(self.app, "_emit_runtime") as mock_emit, patch.object(sut, "prevent_sleep"):
            self.app.on_start()
            self.app.on_stop()
        mock_sub.assert_called_once()
        mock_unsub.assert_called_once()
        self.assertGreaterEqual(mock_emit.call_count, 2)
        self.assertEqual(stopped["n"], 1)


# ============================================================
# EXPORTS ET ASYNC APP
# ============================================================

class TestCamouflageAppAsync(LoggedAsyncCase):
    async def asyncSetUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_start_async_")
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.app = sut.CamouflageApp()
        make_ui_methods_sync(self.app)
        self.app.current_output_dir = self.tmpdir
        self.app.status_label = make_fake_label()
        self.app.tests_label = make_fake_label()
        self.app.progress_bar = make_fake_progress_bar()
        self.app.progress_text = make_fake_label()
        self.app.attempt_text = make_fake_label()
        self.app.color_text = make_fake_label()
        self.app.score_text = make_fake_label()
        self.app.extra_text = make_fake_label()
        self.app.struct_text = make_fake_label()
        self.app.runtime_last_label = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.start_btn = make_fake_button()
        self.app.stop_btn = make_fake_button()
        self.app.open_btn = make_fake_button()
        self.app.mode_blocking_btn = make_fake_button()
        self.app.mode_non_blocking_btn = make_fake_button()
        self.app.mode_skip_tests_btn = make_fake_button()
        self.app.run_mode_label = make_fake_label()
        self.app.count_input = make_fake_input("2")
        self.app.intensity_label = make_fake_label()
        self.app.gallery_grid = make_fake_grid()
        self.app.preview_img = make_fake_image_widget()
        self.app.preview_silhouette = make_fake_image_widget()
        self.app.diag_summary_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()
        self.app.backend_stats_label = make_fake_label()
        self.app.backend_state_label = make_fake_label()
        self.app.resource_text = make_fake_label()
        self.app.resource_hint = make_fake_label()
        self.app.process = FakeProcess()

    async def asyncTearDown(self):
        try:
            self.app.async_runner.stop()
        except Exception:
            pass
        self.tmpdir_obj.cleanup()

    async def test_async_run_preflight_via_log_none(self):
        with patch.object(sut, "camo_log", None):
            ok, summary = await self.app._async_run_preflight_via_log()
        self.assertFalse(ok)
        self.assertIn("impossible", summary.lower())

    async def test_async_run_preflight_via_log_dict(self):
        fake_log = types.SimpleNamespace(async_run_preflight_tests=AsyncMock(return_value={"ok": True, "total": 5, "failures": 0, "errors": 0}))
        with patch.object(sut, "camo_log", fake_log):
            ok, summary = await self.app._async_run_preflight_via_log()
        self.assertTrue(ok)
        self.assertIn("5 tests", summary)

    async def test_extract_failure_rules_async(self):
        failure_a = types.SimpleNamespace(rule="rule_a")
        failure_b = types.SimpleNamespace(rule="rule_b")
        diagnostic = types.SimpleNamespace(failures=[failure_a, failure_b])
        fake_log = types.SimpleNamespace(async_analyze_candidate=AsyncMock(return_value=diagnostic))
        candidate = types.SimpleNamespace(seed=1)
        with patch.object(sut, "camo_log", fake_log):
            rules = await self.app._extract_failure_rules_async(candidate, 1, 1)
        self.assertEqual(rules, ["rule_a", "rule_b"])

    async def test_register_live_diagnostic_async(self):
        candidate = types.SimpleNamespace(seed=1)
        await self.app._register_live_diagnostic_async(candidate, 1, 1, True)
        self.assertEqual(self.app.diag_accepts, 1)
        with patch.object(self.app, "_extract_failure_rules_async", AsyncMock(return_value=["rule_a", "rule_b"])):
            await self.app._register_live_diagnostic_async(candidate, 1, 2, False)
        self.assertEqual(self.app.diag_rejects, 1)
        self.assertEqual(self.app.diag_rule_counter["rule_a"], 1)

    async def test_adaptive_pause(self):
        with patch.object(sut.asyncio, "sleep", AsyncMock()) as mock_sleep:
            await self.app._adaptive_pause()
        self.assertGreaterEqual(mock_sleep.await_count, 0)

    async def test_async_write_report(self):
        rows = [{"index": 1, "seed": 101, "score_final": 0.9}, {"index": 2, "seed": 202, "score_final": 0.8}]
        report_path = await self.app._async_write_report(rows)
        self.assertTrue(report_path.exists())
        with report_path.open("r", encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))
        self.assertEqual(len(data), 2)

    async def test_async_export_best_of(self):
        self.app.best_records = [make_candidate_record(self.tmpdir, 1, 0.95), make_candidate_record(self.tmpdir, 2, 0.90)]
        best_dir = await self.app._async_export_best_of(2)
        self.assertTrue((best_dir / "best_001_camouflage_001.png").exists())
        self.assertTrue((best_dir / "best_of.csv").exists())

    async def test_async_worker_generate_success(self):
        canvas = make_index_canvas_quadrants(120, 80)
        candidate = types.SimpleNamespace(seed=123, image=make_pil_from_index_canvas(canvas), ratios=valid_ratios(), metrics=valid_main_metrics(), profile=types.SimpleNamespace(allowed_angles=[-35,-20,0,20,35]))
        async def fake_generate(seed: int):
            return candidate
        async def fake_eval(img, rs, metrics):
            return ({"score_final": 0.9, "score_ratio": 0.9, "score_silhouette": 0.8, "score_contour": 0.7, "score_main": 0.6, "silhouette_color_diversity": 0.75, "contour_break_score": 0.65, "outline_band_diversity": 0.68, "small_scale_structural_score": 0.62}, True)
        async def fake_validate(_candidate):
            return True
        async def fake_save(_candidate, path):
            candidate.image.save(path)
            return path
        with patch.object(sut.camo, "async_generate_candidate_from_seed", side_effect=fake_generate), patch.object(sut, "async_evaluate_candidate_v3", side_effect=fake_eval), patch.object(sut.camo, "async_validate_candidate_result", side_effect=fake_validate), patch.object(sut.camo, "async_save_candidate_image", side_effect=fake_save), patch.object(self.app, "_register_live_diagnostic_async", AsyncMock()), patch.object(self.app, "_adaptive_pause", AsyncMock()), patch.object(self.app, "_async_finish_success", AsyncMock()) as mock_finish, patch.object(self.app, "reload_gallery"):
            await self.app._async_worker_generate(1)
        mock_finish.assert_called_once()
        self.assertEqual(len(self.app.best_records), 1)
        self.assertEqual(self.app.accepted_count, 1)

    async def test_async_worker_generate_reject_then_accept(self):
        canvas = make_index_canvas_quadrants(120, 80)
        candidate = types.SimpleNamespace(seed=123, image=make_pil_from_index_canvas(canvas), ratios=valid_ratios(), metrics=valid_main_metrics(), profile=types.SimpleNamespace(allowed_angles=[-35,-20,0,20,35]))
        seq = [False, True]
        async def fake_generate(seed: int):
            return candidate
        async def fake_eval(img, rs, metrics):
            return ({"score_final": 0.9, "score_ratio": 0.9, "score_silhouette": 0.8, "score_contour": 0.7, "score_main": 0.6, "silhouette_color_diversity": 0.75, "contour_break_score": 0.65, "outline_band_diversity": 0.68, "small_scale_structural_score": 0.62}, True)
        async def fake_validate(_candidate):
            return seq.pop(0)
        async def fake_save(_candidate, path):
            candidate.image.save(path)
            return path
        with patch.object(sut.camo, "async_generate_candidate_from_seed", side_effect=fake_generate), patch.object(sut, "async_evaluate_candidate_v3", side_effect=fake_eval), patch.object(sut.camo, "async_validate_candidate_result", side_effect=fake_validate), patch.object(sut.camo, "async_save_candidate_image", side_effect=fake_save), patch.object(self.app, "_register_live_diagnostic_async", AsyncMock()), patch.object(self.app, "_adaptive_pause", AsyncMock()), patch.object(self.app, "_async_finish_success", AsyncMock()) as mock_finish, patch.object(self.app, "reload_gallery"):
            await self.app._async_worker_generate(1)
        mock_finish.assert_called_once()
        self.assertEqual(self.app.total_attempts, 2)

    async def test_async_worker_generate_stop_early(self):
        self.app.stop_flag = True
        with patch.object(self.app, "_async_finish_stopped", AsyncMock()) as mock_stop:
            await self.app._async_worker_generate(2)
        mock_stop.assert_called_once()

    async def test_async_worker_generate_error(self):
        async def fake_generate(seed: int):
            raise RuntimeError("boom")
        with patch.object(sut.camo, "async_generate_candidate_from_seed", side_effect=fake_generate), patch.object(self.app, "_async_finish_error", AsyncMock()) as mock_err:
            await self.app._async_worker_generate(1)
        mock_err.assert_called_once()

    async def test_async_finish_success_stopped_error(self):
        with patch.object(self.app, "_async_write_report", AsyncMock(return_value=self.tmpdir / "r.csv")), patch.object(self.app, "_async_export_best_of", AsyncMock(return_value=self.tmpdir / "best")), patch.object(sut, "prevent_sleep") as mock_sleep, patch.object(self.app, "reload_gallery"):
            await self.app._async_finish_success([])
            await self.app._async_finish_stopped([{"a": 1}])
        self.assertGreaterEqual(mock_sleep.call_count, 2)
        with patch.object(sut, "prevent_sleep") as mock_sleep_err:
            await self.app._async_finish_error("boom")
        mock_sleep_err.assert_called_once_with(False)
        self.assertEqual(self.app.status_label.text, "Erreur")


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
    LOGGER.info("========== DÉBUT DES TESTS test_start.py ==========")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=LoggedTextTestResult)
    unittest.main(testRunner=runner, verbosity=2)
