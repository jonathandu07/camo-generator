# -*- coding: utf-8 -*-
"""
Suite de tests alignée sur l'API réelle de start.py.

Objectifs :
- couvrir les utilitaires front et les conversions d'image ;
- couvrir l'orchestrateur UI/async de CamouflageApp ;
- rester rapide, déterministe et sans dépendre d'un vrai Kivy.

Exécution :
    python -m unittest -v test_start.py
"""

from __future__ import annotations

import asyncio
import csv
import functools
import logging
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
from PIL import Image as PILImage


# ============================================================
# FAUX KIVY
# ============================================================

def _install_fake_kivy() -> None:
    # Les tests doivent rester déterministes, même si Kivy est installé
    # sur la machine. On retire donc tout module Kivy déjà chargé puis
    # on injecte une implémentation minimale contrôlée.
    for name in list(sys.modules):
        if name == "kivy" or name.startswith("kivy."):
            sys.modules.pop(name, None)

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
            self.orientation = kwargs.get("orientation", "vertical")
            self.padding = kwargs.get("padding", 0)
            self.spacing = kwargs.get("spacing", 0)

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
        "kivy",
        "kivy.config",
        "kivy.app",
        "kivy.clock",
        "kivy.core.image",
        "kivy.core.window",
        "kivy.graphics",
        "kivy.metrics",
        "kivy.properties",
        "kivy.uix.boxlayout",
        "kivy.uix.button",
        "kivy.uix.gridlayout",
        "kivy.uix.image",
        "kivy.uix.label",
        "kivy.uix.scrollview",
        "kivy.uix.slider",
        "kivy.uix.textinput",
        "kivy.uix.widget",
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


USE_REAL_KIVY = os.getenv("CAMO_TEST_USE_REAL_KIVY", "0").strip().lower() in {"1", "true", "yes", "on"}

if not USE_REAL_KIVY:
    _install_fake_kivy()
else:
    try:
        import kivy  # type: ignore # noqa: F401
    except Exception:
        _install_fake_kivy()

# start.py peut laisser des ResourceWarning liés à des boucles asyncio non
# fermées selon la plateforme et la politique warnings locale. Les tests
# nettoient eux-mêmes les runners quand c'est possible ; ce filtre évite
# qu'un environnement plus strict rende la suite non déterministe.
warnings.filterwarnings("ignore", category=ResourceWarning, message=r"unclosed event loop.*")

import start as sut


# ============================================================
# LOGS
# ============================================================

LOG_DIR = Path(os.getenv("LOG_OUTPUT_DIR", Path(__file__).resolve().parent / "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_start.log"


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("test_start")
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
    logger.propagate = False
    return logger


LOGGER = configure_logger()


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


def valid_metrics() -> Dict[str, float]:
    return {
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
        "largest_olive_component_ratio": 0.24,
        "center_empty_ratio": 0.28,
        "boundary_density": 0.145,
        "mirror_similarity": 0.44,
        "macro_terre_visible_ratio": 0.18,
        "macro_gris_visible_ratio": 0.14,
        "central_brown_continuity": 0.20,
    }


def make_candidate(seed: int = 1234) -> Any:
    return sut.camo.CandidateResult(
        seed=seed,
        profile=sut.camo.make_profile(seed),
        image=make_pil_from_index_canvas(make_index_canvas_quadrants(60, 60)),
        ratios=valid_ratios(),
        metrics=valid_metrics(),
    )


def make_candidate_record(tmpdir: Path, index: int = 1, score: float = 0.8) -> sut.CandidateRecord:
    image_path = tmpdir / f"camouflage_{index:03d}.png"
    make_pil_from_index_canvas(make_index_canvas_quadrants()).save(image_path)
    metrics = valid_metrics()
    metrics["visual_score_final"] = score
    return sut.CandidateRecord(
        index=index,
        seed=1000 + index,
        local_attempt=2,
        global_attempt=10 + index,
        image_path=image_path,
        metrics=metrics,
        ratios=valid_ratios(),
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


def make_submit_closing_coroutines(fake_future: Any):
    def _submit(coro):
        try:
            coro.close()
        except Exception:
            pass
        return fake_future

    return _submit




def stop_async_runner_safely(runner: Any) -> None:
    if runner is None:
        return
    try:
        stop = getattr(runner, "stop", None)
        if callable(stop):
            stop()
    except Exception:
        pass

    loop = getattr(runner, "loop", None)
    if loop is not None:
        try:
            is_closed = getattr(loop, "is_closed", None)
            if callable(is_closed):
                if not is_closed():
                    loop.close()
            else:
                loop.close()
        except Exception:
            pass

class FakeProcess:
    def cpu_percent(self, interval=None) -> float:
        return 12.0

    class _Mem:
        rss = 512 * 1024 * 1024

    def memory_info(self):
        return self._Mem()


class TempDirMixin:
    def setUp(self) -> None:
        super().setUp()
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_start_")
        self.tmpdir = Path(self._tmpdir_obj.name)

    def tearDown(self) -> None:
        self._tmpdir_obj.cleanup()
        super().tearDown()


class AppMixin(TempDirMixin):
    def setUp(self) -> None:
        super().setUp()
        self.app = sut.CamouflageApp()
        stop_async_runner_safely(getattr(self.app, "async_runner", None))
        self.app.async_runner = types.SimpleNamespace(submit=make_submit_closing_coroutines(MagicMock()), stop=Mock())
        self.app.current_output_dir = self.tmpdir
        self.app.process = FakeProcess()

        self.app.status_label = make_fake_label()
        self.app.attempt_text = make_fake_label()
        self.app.progress_text = make_fake_label()
        self.app.color_text = make_fake_label()
        self.app.score_text = make_fake_label()
        self.app.extra_text = make_fake_label()
        self.app.struct_text = make_fake_label()
        self.app.resource_text = make_fake_label()
        self.app.tests_label = make_fake_label()
        self.app.run_mode_label = make_fake_label()
        self.app.diag_summary_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()
        self.app.runtime_last_label = make_fake_label()
        self.app.live_stage_label = make_fake_label()
        self.app.live_counts_label = make_fake_label()
        self.app.live_meta_label = make_fake_label()
        self.app.preview_img = make_fake_image_widget()
        self.app.preview_silhouette = make_fake_image_widget()
        self.app.live_preview_img = make_fake_image_widget()
        self.app.progress_bar = make_fake_progress_bar()
        self.app.count_input = make_fake_input("2")
        self.app.start_btn = make_fake_button()
        self.app.stop_btn = make_fake_button()
        self.app.open_btn = make_fake_button()
        self.app.mode_blocking_btn = make_fake_button()
        self.app.mode_non_blocking_btn = make_fake_button()
        self.app.mode_skip_tests_btn = make_fake_button()
        self.app.intensity_label = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.gallery_grid = make_fake_grid()

    def tearDown(self) -> None:
        stop_async_runner_safely(getattr(self.app, "async_runner", None))
        super().tearDown()


# ============================================================
# TESTS UTILITAIRES
# ============================================================

class TestUtilities(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(sut.REPORT_NAME, "rapport_camouflages_front.csv")
        self.assertEqual(sut.BEST_DIR_NAME, "best_of")
        self.assertEqual(sut.DEFAULT_TOP_K, 20)

    def test_hex_rgba(self):
        rgba = sut.hex_rgba("BL", 0.5)
        self.assertEqual(len(rgba), 4)
        self.assertAlmostEqual(rgba[3], 0.5)

    def test_open_folder_windows(self):
        with patch.object(sut.platform, "system", return_value="Windows"), \
             patch.object(sut.os, "startfile", create=True) as mock_startfile:
            sut.open_folder(Path("."))
        mock_startfile.assert_called_once()

    def test_open_folder_darwin(self):
        with patch.object(sut.platform, "system", return_value="Darwin"), \
             patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
        mock_popen.assert_called_once()

    def test_open_folder_linux(self):
        with patch.object(sut.platform, "system", return_value="Linux"), \
             patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
        mock_popen.assert_called_once()

    def test_prevent_sleep_windows(self):
        fake_kernel = types.SimpleNamespace(SetThreadExecutionState=Mock())
        fake_windll = types.SimpleNamespace(kernel32=fake_kernel)
        with patch.object(sut.platform, "system", return_value="Windows"), \
             patch.object(sut.ctypes, "windll", fake_windll, create=True):
            sut.prevent_sleep(True)
            sut.prevent_sleep(False)
        self.assertEqual(fake_kernel.SetThreadExecutionState.call_count, 2)

    def test_pil_to_coreimage(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(20, 20))
        core = sut.pil_to_coreimage(img)
        self.assertTrue(hasattr(core, "texture"))

    def test_make_thumbnail(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(120, 90))
        thumb = sut.make_thumbnail(img, (60, 40))
        self.assertEqual(thumb.size, (60, 40))

    def test_palette_map_and_rgb_image_to_index_canvas(self):
        palette = sut.palette_map()
        self.assertEqual(len(palette), 4)
        idx = make_index_canvas_quadrants(40, 40)
        img = make_pil_from_index_canvas(idx)
        out = sut.rgb_image_to_index_canvas(img)
        np.testing.assert_array_equal(out, idx)

    def test_build_silhouette_mask_and_boundary(self):
        mask = sut.build_silhouette_mask(80, 120)
        boundary = sut.silhouette_boundary(mask)
        self.assertEqual(mask.shape, (120, 80))
        self.assertTrue(mask.any())
        self.assertTrue(boundary.any())
        self.assertTrue(np.all(boundary <= mask))

    def test_silhouette_projection_image(self):
        idx = make_index_canvas_quadrants(80, 50)
        img = sut.silhouette_projection_image(idx)
        self.assertIsInstance(img, PILImage.Image)
        self.assertEqual(img.size, (50, 80))

    def test_extract_backend_scores(self):
        scores = sut.extract_backend_scores(valid_metrics())
        self.assertEqual(scores["score_final"], valid_metrics()["visual_score_final"])
        self.assertEqual(scores["military_score"], valid_metrics()["visual_military_score"])

    def test_asyncio_thread_runner(self):
        runner = sut.AsyncioThreadRunner()
        try:
            fut = runner.submit(asyncio.sleep(0, result=42))
            self.assertEqual(fut.result(timeout=2), 42)
        finally:
            stop_async_runner_safely(runner)


# ============================================================
# TESTS CAMOUFLAGE APP — SYNCHRONE
# ============================================================

class TestCamouflageAppMethods(AppMixin, unittest.TestCase):
    def test_status_log_diag(self):
        self.app.status("Prêt", ok=True)
        self.app.log("hello")
        self.app.diag_log("diag")
        self.assertEqual(self.app.status_label.text, "Prêt")
        self.assertIn("hello", self.app.log_view.label.text)
        self.assertIn("diag", self.app.diag_log_view.label.text)

    def test_update_progress(self):
        self.app.update_progress(3, 9)
        self.assertEqual(self.app.progress_bar.max_value, 9)
        self.assertEqual(self.app.progress_bar.value, 3)

    def test_update_preview(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(80, 50))
        sil = sut.silhouette_projection_image(make_index_canvas_quadrants(80, 50))

        class FakeCore:
            def __init__(self, texture: str):
                self.texture = texture

        with patch.object(sut, "pil_to_coreimage", side_effect=[FakeCore("img"), FakeCore("sil")]):
            self.app.update_preview(img, sil)
        self.assertEqual(self.app.preview_img.texture, "img")
        self.assertEqual(self.app.preview_silhouette.texture, "sil")

    def test_update_live_stage(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(60, 60))

        class FakeCore:
            texture = "live_tex"

        with patch.object(sut, "pil_to_coreimage", return_value=FakeCore()):
            self.app.update_live_stage("validation", 2, 7, 123, 8, 0, 0, pil_img=img)
        self.assertIn("validation", self.app.live_stage_label.text)
        self.assertIn("Macros 8", self.app.live_counts_label.text)
        self.assertIn("Image 002", self.app.live_meta_label.text)
        self.assertEqual(self.app.live_preview_img.texture, "live_tex")

    def test_maybe_handle_live_runtime_payload(self):
        with patch.object(self.app, "update_live_stage") as mock_stage:
            evt = types.SimpleNamespace(payload={
                "stage": "build",
                "target_index": 1,
                "local_attempt": 2,
                "seed": 3,
                "macro_count": 4,
                "transition_count": 5,
                "micro_count": 6,
                "preview_path": "frame.png",
            })
            self.app._maybe_handle_live_runtime_payload(evt)
        mock_stage.assert_called_once()

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
        self.assertFalse(self.app.stop_btn.disabled)

    def test_on_intensity_change(self):
        self.app._on_intensity_change(None, 72.4)
        self.assertEqual(int(self.app.machine_intensity), 72)
        self.assertEqual(self.app.intensity_label.text, "72 %")

    def test_run_mode_helpers(self):
        self.assertEqual(self.app._run_mode_text(sut.RUN_MODE_SKIP_TESTS), "sans tests")
        self.app._set_run_mode(sut.RUN_MODE_NON_BLOCKING)
        self.assertEqual(self.app.run_mode, sut.RUN_MODE_NON_BLOCKING)
        self.assertIn("non bloquants", self.app.log_view.label.text)

    def test_refresh_run_mode_buttons(self):
        self.app.run_mode = sut.RUN_MODE_NON_BLOCKING
        self.app._refresh_run_mode_buttons()
        self.assertIn("●", self.app.mode_non_blocking_btn.text)
        self.assertIn("○", self.app.mode_blocking_btn.text)
        self.assertIn("Mode actuel", self.app.run_mode_label.text)

    def test_refresh_diag_labels(self):
        self.app.diag_total = 6
        self.app.diag_accepts = 2
        self.app.diag_rejects = 4
        self.app.diag_rule_counter.update({"rule_a": 5, "rule_b": 3})
        self.app.diag_last_rules = ["rule_b", "rule_c"]
        self.app._refresh_diag_labels()
        self.assertIn("Tentatives 6", self.app.diag_summary_label.text)
        self.assertIn("rule_a:5", self.app.diag_top_rules_label.text)
        self.assertIn("rule_b", self.app.diag_last_fail_label.text)

    def test_reload_gallery(self):
        for i in range(2):
            make_pil_from_index_canvas(make_index_canvas_quadrants()).save(self.tmpdir / f"camouflage_{i+1:03d}.png")
        with patch.object(sut, "GalleryThumb", side_effect=lambda app, path: path.name):
            self.app.reload_gallery()
        self.assertEqual(len(self.app.gallery_grid.items), 2)

    def test_update_preflight_label(self):
        self.app._update_preflight_label("OK", ok=True)
        self.assertEqual(self.app.tests_label.text, "OK")
        self.app._update_preflight_label("KO", ok=False)
        self.assertEqual(self.app.tests_label.text, "KO")

    def test_format_runtime_event_and_append(self):
        event = types.SimpleNamespace(ts=0.0, level="info", source="src", message="hello", payload={"a": 1})
        line = self.app._format_runtime_event(event)
        self.assertIn("src", line)
        self.assertIn("hello", line)
        self.app._append_runtime_line(line)
        self.assertIn("hello", self.app.log_view.label.text)
        self.assertIn("Dernier runtime", self.app.runtime_last_label.text)

    def test_on_runtime_event(self):
        event = types.SimpleNamespace(ts=0.0, level="info", source="src", message="hello", payload={})
        with patch.object(self.app, "_append_runtime_line") as mock_append, \
             patch.object(self.app, "_maybe_handle_live_runtime_payload") as mock_payload:
            self.app._on_runtime_event(event)
        mock_append.assert_called_once()
        mock_payload.assert_called_once_with(event)

    def test_emit_runtime(self):
        fake_log = types.SimpleNamespace(log_event=Mock())
        with patch.object(sut, "camo_log", fake_log):
            self.app._emit_runtime("INFO", "start", "msg", value=1)
        fake_log.log_event.assert_called_once()

    def test_subscribe_and_unsubscribe_runtime_feed(self):
        manager = types.SimpleNamespace(subscribe=Mock(), unsubscribe=Mock())
        fake_log = types.SimpleNamespace(LOG_MANAGER=manager)
        with patch.object(sut, "camo_log", fake_log):
            self.app._subscribe_runtime_feed()
            self.assertTrue(self.app._runtime_subscription_active)
            self.app._unsubscribe_runtime_feed()
        manager.subscribe.assert_called_once()
        manager.unsubscribe.assert_called_once()
        self.assertFalse(self.app._runtime_subscription_active)

    def test_ensure_preflight(self):
        fake_future = MagicMock()
        fake_future.add_done_callback = Mock()
        self.app.async_runner = types.SimpleNamespace(submit=make_submit_closing_coroutines(fake_future), stop=Mock())
        with patch.object(self.app, "status") as mock_status, \
             patch.object(self.app, "_refresh_controls_state") as mock_refresh:
            out = self.app._ensure_preflight(pending_start=True)
        self.assertFalse(out)
        self.assertTrue(self.app.preflight_running)
        self.assertTrue(self.app.preflight_pending_start)
        mock_status.assert_called_once()
        mock_refresh.assert_called_once()
        fake_future.add_done_callback.assert_called_once()

    def test_ensure_preflight_returns_true_when_already_ok(self):
        self.app.tests_ran = True
        self.app.tests_ok = True
        self.assertTrue(self.app._ensure_preflight())

    def test_finish_preflight(self):
        self.app.preflight_running = True
        self.app.preflight_pending_start = True
        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app._finish_preflight(True, "OK")
        self.assertTrue(self.app.tests_ok)
        self.assertEqual(self.app.tests_label.text, "OK")
        mock_start.assert_called_once()
        self.app.preflight_running = True
        self.app.preflight_pending_start = True
        self.app._finish_preflight(False, "KO")
        self.assertFalse(self.app.tests_ok)
        self.assertEqual(self.app.status_label.text, "Tests KO")

    def test_start_generation_skip_tests(self):
        self.app.run_mode = sut.RUN_MODE_SKIP_TESTS
        with patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app.start_generation()
        mock_start.assert_called_once()
        self.assertIn("Tests ignorés", self.app.tests_label.text)

    def test_start_generation_non_blocking(self):
        self.app.run_mode = sut.RUN_MODE_NON_BLOCKING
        with patch.object(self.app, "_ensure_preflight", return_value=False) as mock_pf, \
             patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app.start_generation()
        mock_pf.assert_called_once_with(pending_start=False)
        mock_start.assert_called_once_with(allow_during_preflight=True)

    def test_start_generation_blocking(self):
        with patch.object(self.app, "_ensure_preflight", return_value=False) as mock_pf:
            self.app.start_generation()
        mock_pf.assert_called_once_with(pending_start=True)

    def test_start_generation_ignores_when_already_running(self):
        self.app.running = True
        with patch.object(self.app, "_ensure_preflight") as mock_pf:
            self.app.start_generation()
        mock_pf.assert_not_called()

    def test_start_generation_after_preflight_invalid(self):
        self.app.count_input.text = "abc"
        self.app._start_generation_after_preflight()
        self.assertIn("Nombre de camouflages invalide", self.app.log_view.label.text)

    def test_start_generation_after_preflight_success(self):
        fake_future = MagicMock()
        fake_future.done.return_value = False
        fake_future.add_done_callback = MagicMock()
        self.app.async_runner = types.SimpleNamespace(submit=make_submit_closing_coroutines(fake_future), stop=Mock())
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

    def test_on_generation_done_success(self):
        fut = Mock()
        fut.result.return_value = None
        with patch.object(sut.Clock, "schedule_once", side_effect=lambda cb, dt=0: cb(dt)), \
             patch.object(self.app, "_clear_future") as mock_clear, \
             patch.object(self.app, "_finish_error") as mock_error:
            self.app._on_generation_done(fut)
        mock_clear.assert_called_once_with(fut)
        mock_error.assert_not_called()

    def test_on_generation_done_error(self):
        fut = Mock()
        fut.result.side_effect = RuntimeError("boom")
        with patch.object(sut.Clock, "schedule_once", side_effect=lambda cb, dt=0: cb(dt)), \
             patch.object(self.app, "_clear_future") as mock_clear, \
             patch.object(self.app, "_finish_error") as mock_error:
            self.app._on_generation_done(fut)
        mock_error.assert_called_once()
        mock_clear.assert_called_once_with(fut)

    def test_clear_future(self):
        fut = Mock()
        self.app.current_future = fut
        self.app._clear_future(fut)
        self.assertIsNone(self.app.current_future)

    def test_finish_error(self):
        self.app.running = True
        self.app._finish_error("boom")
        self.assertFalse(self.app.running)
        self.assertEqual(self.app.status_label.text, "Erreur")
        self.assertIn("boom", self.app.log_view.label.text)

    def test_update_resource_monitor_without_psutil(self):
        with patch.object(sut, "psutil", None):
            self.app._update_resource_monitor(None)
        self.assertIn("Installer psutil", self.app.resource_text.text)

    def test_update_resource_monitor_with_psutil(self):
        fake_psutil = types.SimpleNamespace(
            cpu_percent=lambda interval=None: 10.0,
            virtual_memory=lambda: types.SimpleNamespace(percent=20.0),
            disk_usage=lambda anchor: types.SimpleNamespace(percent=30.0),
        )
        with patch.object(sut, "psutil", fake_psutil):
            self.app._update_resource_monitor(None)
        self.assertIn("CPU 10%", self.app.resource_text.text)

    def test_on_stop_with_pending_future(self):
        pending = Mock()
        pending.done.return_value = False
        pending.add_done_callback = Mock()
        self.app.current_future = pending
        self.app.async_runner = types.SimpleNamespace(stop=Mock())
        with patch.object(sut, "prevent_sleep") as mock_sleep, \
             patch.object(self.app, "_emit_runtime") as mock_emit, \
             patch.object(self.app, "_unsubscribe_runtime_feed") as mock_unsub:
            self.app.on_stop()
        mock_sleep.assert_called_once_with(False)
        mock_emit.assert_called_once()
        mock_unsub.assert_called_once()
        pending.add_done_callback.assert_called_once()

    def test_on_stop_without_pending_future(self):
        done_fut = Mock()
        done_fut.done.return_value = True
        self.app.current_future = done_fut
        self.app.async_runner = types.SimpleNamespace(stop=Mock())
        with patch.object(sut, "prevent_sleep"):
            self.app.on_stop()
        self.app.async_runner.stop.assert_called_once()


# ============================================================
# TESTS CAMOUFLAGE APP — ASYNCHRONE
# ============================================================

class TestCamouflageAppAsync(AppMixin, unittest.IsolatedAsyncioTestCase):
    async def test_async_run_preflight_without_log(self):
        with patch.object(sut, "camo_log", None):
            ok, summary = await self.app._async_run_preflight()
        self.assertFalse(ok)
        self.assertIn("log.py indisponible", summary)

    async def test_async_run_preflight_with_dict_summary(self):
        fake_log = types.SimpleNamespace(
            async_run_preflight_tests=AsyncMock(return_value={"ok": True, "total": 10, "failures": 0, "errors": 0})
        )
        with patch.object(sut, "camo_log", fake_log):
            ok, summary = await self.app._async_run_preflight()
        self.assertTrue(ok)
        self.assertIn("10 tests", summary)

    async def test_extract_failure_rules(self):
        candidate = make_candidate()
        with patch.object(sut.camo, "extract_rejection_failures", return_value=[{"rule": "r1"}, {"rule": "r2"}], create=True):
            rules = await self.app._extract_failure_rules(candidate, 1, 2)
        self.assertEqual(rules, ["r1", "r2"])

    async def test_register_live_diag_accept_and_reject(self):
        candidate = make_candidate()
        await self.app._register_live_diag(candidate, 1, 1, True)
        self.assertEqual(self.app.diag_accepts, 1)
        with patch.object(self.app, "_extract_failure_rules", AsyncMock(return_value=["rule_a", "rule_b"])):
            await self.app._register_live_diag(candidate, 1, 2, False)
        self.assertEqual(self.app.diag_rejects, 1)
        self.assertIn("rule_a", self.app.diag_top_rules_label.text)

    async def test_adaptive_pause(self):
        fake_psutil = types.SimpleNamespace(
            cpu_percent=lambda interval=None: 99.0,
            virtual_memory=lambda: types.SimpleNamespace(percent=96.0),
        )
        self.app.machine_intensity = 50.0
        with patch.object(sut, "psutil", fake_psutil), \
             patch.object(asyncio, "sleep", AsyncMock()) as mock_sleep:
            await self.app._adaptive_pause()
        mock_sleep.assert_awaited()

    async def test_async_write_report(self):
        with patch.object(sut.camo, "async_write_report", AsyncMock(return_value=self.tmpdir / "r.csv")) as mock_write:
            out = await self.app._async_write_report([{"a": 1}])
        self.assertEqual(out, self.tmpdir / "r.csv")
        mock_write.assert_awaited_once_with([{"a": 1}], self.app.current_output_dir, filename=sut.REPORT_NAME)

    async def test_async_export_best_of(self):
        self.app.best_records = [
            make_candidate_record(self.tmpdir, index=1, score=0.9),
            make_candidate_record(self.tmpdir, index=2, score=0.8),
        ]
        out = await self.app._async_export_best_of(2)
        self.assertTrue(out.exists())
        self.assertTrue((out / "best_001_camouflage_001.png").exists())
        self.assertTrue((out / "best_of.csv").exists())

    async def test_write_csv_sync(self):
        path = self.tmpdir / "x.csv"
        self.app._write_csv_sync(path, [{"a": 1, "b": 2}])
        self.assertTrue(path.exists())
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        path_empty = self.tmpdir / "y.csv"
        self.app._write_csv_sync(path_empty, [])
        self.assertEqual(path_empty.read_text(encoding="utf-8"), "")

    async def test_async_finish_success_stopped_error(self):
        with patch.object(self.app, "_async_write_report", AsyncMock(return_value=self.tmpdir / "rapport.csv")), \
             patch.object(self.app, "_async_export_best_of", AsyncMock(return_value=self.tmpdir / "best")), \
             patch.object(sut, "prevent_sleep") as mock_sleep, \
             patch.object(self.app, "reload_gallery"):
            await self.app._async_finish_success([])
            await self.app._async_finish_stopped([{"a": 1}])
        self.assertGreaterEqual(mock_sleep.call_count, 2)
        with patch.object(sut, "prevent_sleep") as mock_sleep_err:
            await self.app._async_finish_error("boom")
        mock_sleep_err.assert_called_once_with(False)
        self.assertEqual(self.app.status_label.text, "Erreur")

    async def test_async_worker_generate_success(self):
        candidate = make_candidate(seed=999)
        self.app.current_output_dir = self.tmpdir
        with patch.object(self.app, "_async_should_stop", AsyncMock(return_value=False)), \
             patch.object(sut.camo, "async_generate_candidate_from_seed", AsyncMock(return_value=candidate)), \
             patch.object(sut.camo, "async_validate_candidate_result", AsyncMock(return_value=True)), \
             patch.object(sut.camo, "async_save_candidate_image", AsyncMock()) as mock_save, \
             patch.object(sut.camo, "candidate_row", return_value={"index": 1}), \
             patch.object(self.app, "_register_live_diag", AsyncMock()) as mock_diag, \
             patch.object(self.app, "_async_finish_success", AsyncMock()) as mock_finish, \
             patch.object(self.app, "reload_gallery"):
            await self.app._async_worker_generate(1)
        mock_diag.assert_awaited()
        mock_save.assert_awaited()
        mock_finish.assert_awaited_once()
        self.assertEqual(self.app.accepted_count, 1)
        self.assertEqual(len(self.app.best_records), 1)

    async def test_async_worker_generate_stop(self):
        with patch.object(self.app, "_async_should_stop", AsyncMock(return_value=True)), \
             patch.object(self.app, "_async_finish_stopped", AsyncMock()) as mock_stop:
            await self.app._async_worker_generate(1)
        mock_stop.assert_awaited_once()

    async def test_async_worker_generate_error(self):
        with patch.object(self.app, "_async_should_stop", AsyncMock(return_value=False)), \
             patch.object(sut.camo, "async_generate_candidate_from_seed", AsyncMock(side_effect=RuntimeError("boom"))), \
             patch.object(self.app, "_async_finish_error", AsyncMock()) as mock_err:
            await self.app._async_worker_generate(1)
        mock_err.assert_awaited_once()

    async def test_async_should_stop(self):
        self.app.stop_flag = True
        self.assertTrue(await self.app._async_should_stop())

    def test_update_attempt_status(self):
        self.app._update_attempt_status(
            target_index=2,
            attempt_idx=7,
            global_attempt=11,
            seed=123,
            target_total=10,
            accepted_count=1,
            rejected_count=5,
            accepted=False,
            rs=valid_ratios(),
            scores=sut.extract_backend_scores(valid_metrics()),
            metrics=valid_metrics(),
        )
        self.assertIn("1 / 10", self.app.progress_text.text)
        self.assertIn("Image 002", self.app.attempt_text.text)
        self.assertIn("Score 0.780", self.app.score_text.text)
        self.assertIn("militaire", self.app.struct_text.text)


if __name__ == "__main__":
    LOGGER.info("========== DÉBUT DES TESTS test_start.py ==========")
    unittest.main(verbosity=2)
