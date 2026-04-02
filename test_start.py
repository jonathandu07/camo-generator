# -*- coding: utf-8 -*-
"""
test_start.py

Suite de tests approfondie pour start.py avec faux Kivy, logs détaillés
et exécution déterministe.

Objectifs :
- couvrir les utilitaires front, conversions et helpers backend/front ;
- couvrir CamouflageApp sans dépendre d'un vrai runtime Kivy ;
- tester les chemins sync + async les plus importants ;
- produire des logs précis et lisibles dans logs/test_start.log.

Exécution :
    python -m unittest -v test_start.py
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
            self.orientation = kwargs.get("orientation", "vertical")
            self.padding = kwargs.get("padding", 0)
            self.spacing = kwargs.get("spacing", 0)
            self.bold = kwargs.get("bold", False)

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
        def schedule_once(cb, *_args, **_kwargs):
            if callable(cb):
                return cb(0)
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


try:
    import kivy  # type: ignore # noqa: F401
except Exception:
    _install_fake_kivy()

import start as sut  # type: ignore


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


class LoggedTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        LOGGER.info("START %s", self.id())

    def tearDown(self) -> None:
        LOGGER.info("END   %s", self.id())
        super().tearDown()


# ============================================================
# HELPERS
# ============================================================

def make_index_canvas_quadrants(h: int = 80, w: int = 80) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[: h // 2, w // 2:] = sut.BACKEND_IDX_1
    canvas[h // 2:, : w // 2] = sut.BACKEND_IDX_2
    canvas[h // 2:, w // 2:] = sut.BACKEND_IDX_3
    return canvas


def make_pil_from_index_canvas(index_canvas: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(sut.camo.RGB[index_canvas], "RGB")


def valid_ratios() -> np.ndarray:
    return np.array([0.32, 0.28, 0.22, 0.18], dtype=float)


def valid_metrics() -> Dict[str, float]:
    return {
        "largest_component_ratio_class_1": 0.24,
        "largest_olive_component_ratio": 0.24,
        "boundary_density": 0.05,
        "boundary_density_small": 0.06,
        "boundary_density_tiny": 0.07,
        "mirror_similarity": 0.44,
        "edge_contact_ratio": 0.30,
        "overscan": 1.11,
        "shift_strength": 0.92,
        "width": 256.0,
        "height": 144.0,
        "physical_width_cm": 240.0,
        "physical_height_cm": 135.0,
        "px_per_cm": 1.066,
        "motif_scale": 0.55,
        "bestof_score": 0.97,
    }


def make_candidate(seed: int = 1234) -> Any:
    idx = make_index_canvas_quadrants(60, 60)
    return sut.camo.CandidateResult(
        seed=seed,
        profile=sut.camo.make_profile(seed),
        image=make_pil_from_index_canvas(idx),
        label_map=idx,
        ratios=valid_ratios(),
        metrics=valid_metrics(),
    )


def make_invalid_candidate(seed: int = 1234) -> Any:
    idx = make_index_canvas_quadrants(60, 60)
    bad = valid_metrics()
    bad["mirror_similarity"] = 0.99
    return sut.camo.CandidateResult(
        seed=seed,
        profile=sut.camo.make_profile(seed),
        image=make_pil_from_index_canvas(idx),
        label_map=idx,
        ratios=valid_ratios(),
        metrics=bad,
    )


def make_candidate_record(tmpdir: Path, index: int = 1, mirror: float = 0.4) -> sut.CandidateRecord:
    image_path = tmpdir / f"pattern_{index:03d}.png"
    make_pil_from_index_canvas(make_index_canvas_quadrants()).save(image_path)
    metrics = valid_metrics()
    metrics["mirror_similarity"] = mirror
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
        try:
            self.app.async_runner.stop()
            if hasattr(self.app.async_runner, "loop") and self.app.async_runner.loop is not None:
                self.app.async_runner.loop.close()
        except Exception:
            pass
        self.app.async_runner = types.SimpleNamespace(submit=make_submit_closing_coroutines(MagicMock()), stop=Mock(), loop=None)
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
        self.app.diag_summary_mini_label = make_fake_label()
        self.app.diag_top_rules_label = make_fake_label()
        self.app.diag_top_rules_mini_label = make_fake_label()
        self.app.diag_last_fail_label = make_fake_label()
        self.app.runtime_last_label = make_fake_label()
        self.app.live_stage_label = make_fake_label()
        self.app.live_counts_label = make_fake_label()
        self.app.live_meta_label = make_fake_label()
        self.app.manual_review_label = make_fake_label()
        self.app.manual_review_mini_label = make_fake_label()
        self.app.preview_img = make_fake_image_widget()
        self.app.preview_silhouette = make_fake_image_widget()
        self.app.live_preview_img = make_fake_image_widget()
        self.app.progress_bar = make_fake_progress_bar()
        self.app.count_input = make_fake_input("2")
        self.app.start_btn = make_fake_button()
        self.app.stop_btn = make_fake_button()
        self.app.open_btn = make_fake_button()
        self.app.manual_accept_btn = make_fake_button()
        self.app.manual_skip_btn = make_fake_button()
        self.app.mode_blocking_btn = make_fake_button()
        self.app.mode_non_blocking_btn = make_fake_button()
        self.app.mode_skip_tests_btn = make_fake_button()
        self.app.motif_scale_label = make_fake_label()
        self.app.projection_scale_label = make_fake_label()
        self.app.log_view = make_fake_log_view()
        self.app.diag_log_view = make_fake_log_view()
        self.app.gallery_grid = make_fake_grid()
        self.app.best_records = []
        self.app.generated_rows = []

    def tearDown(self) -> None:
        try:
            self.app.async_runner.stop()
            if hasattr(self.app.async_runner, "loop") and self.app.async_runner.loop is not None:
                self.app.async_runner.loop.close()
        except Exception:
            pass
        super().tearDown()


# ============================================================
# TESTS UTILITAIRES
# ============================================================

class TestUtilities(TempDirMixin, LoggedTestCase):
    def test_constants_and_defaults(self):
        self.assertEqual(sut.APP_TITLE, "Camouflage Armée Fédérale Europe")
        self.assertEqual(sut.BEST_DIR_NAME, "best_of")
        self.assertEqual(sut.MANNEQUIN_DIR_NAME, "mannequin_previews")
        self.assertEqual(sut.REPORT_NAME, "rapport_textures.csv")
        self.assertTrue(sut.DEFAULT_TARGET_COUNT > 0)

    def test_hex_rgba(self):
        rgba = sut.hex_rgba("BL", 0.5)
        self.assertEqual(len(rgba), 4)
        self.assertAlmostEqual(rgba[3], 0.5)

    def test_open_folder_windows(self):
        with patch.object(sut.platform, "system", return_value="Windows"),              patch.object(sut.os, "startfile", create=True) as mock_startfile:
            sut.open_folder(Path("."))
        mock_startfile.assert_called_once()

    def test_open_folder_darwin(self):
        with patch.object(sut.platform, "system", return_value="Darwin"),              patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
        mock_popen.assert_called_once()

    def test_open_folder_linux(self):
        with patch.object(sut.platform, "system", return_value="Linux"),              patch.object(sut.subprocess, "Popen") as mock_popen:
            sut.open_folder(Path("."))
        mock_popen.assert_called_once()

    def test_prevent_sleep_windows(self):
        fake_kernel = types.SimpleNamespace(SetThreadExecutionState=Mock())
        fake_windll = types.SimpleNamespace(kernel32=fake_kernel)
        with patch.object(sut.platform, "system", return_value="Windows"),              patch.object(sut.ctypes, "windll", fake_windll, create=True):
            sut.prevent_sleep(True)
            sut.prevent_sleep(False)
        self.assertEqual(fake_kernel.SetThreadExecutionState.call_count, 2)

    def test_make_thumbnail(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(120, 90))
        thumb = sut.make_thumbnail(img, (60, 40))
        self.assertEqual(thumb.size, (60, 40))

    def test_image_conversions(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(20, 20))
        bgr = sut.pil_rgb_to_bgr(img)
        self.assertEqual(bgr.shape, (20, 20, 3))
        back = sut.bgr_to_pil_rgb(bgr)
        self.assertIsInstance(back, PILImage.Image)
        self.assertEqual(back.size, img.size)

    def test_read_helpers(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(20, 20))
        path = self.tmpdir / "img.png"
        img.save(path)
        bgr = sut.read_bgr(path)
        self.assertEqual(bgr.shape[:2], (20, 20))
        pil = sut.read_pil_rgb(path)
        self.assertEqual(pil.size, (20, 20))

    def test_palette_map_and_rgb_image_to_index_canvas(self):
        palette = sut.palette_map()
        self.assertEqual(len(palette), 4)
        idx = make_index_canvas_quadrants(40, 40)
        img = make_pil_from_index_canvas(idx)
        out = sut.rgb_image_to_index_canvas(img)
        np.testing.assert_array_equal(out, idx)

    def test_backend_machine_intensity_and_safe_metric(self):
        self.assertEqual(sut.backend_machine_intensity(100), 1.0)
        self.assertEqual(sut.backend_machine_intensity(5), 0.10)
        self.assertEqual(sut.backend_machine_intensity(50), 0.5)
        self.assertEqual(sut.safe_metric({"x": 1.2}, "x"), 1.2)
        self.assertEqual(sut.safe_metric({}, "x", 3.4), 3.4)

    def test_extract_backend_scores(self):
        scores = sut.extract_backend_scores(valid_ratios(), valid_metrics())
        self.assertIn("ratio_mae", scores)
        self.assertIn("primary_component_ratio", scores)
        self.assertIn("bestof_score", scores)

    def test_rejection_rules_from_outcome(self):
        candidate = make_invalid_candidate()
        outcome = types.SimpleNamespace(accepted=False, reasons=["mirror_similarity"])
        rules = sut.rejection_rules_for_candidate(candidate, outcome)
        self.assertEqual(rules, ["mirror_similarity"])

    def test_rejection_rules_from_metrics(self):
        candidate = make_invalid_candidate()
        rules = sut.rejection_rules_for_candidate(candidate, None)
        self.assertIn("mirror_similarity", rules)

    def test_candidate_rank_key(self):
        rec1 = make_candidate_record(self.tmpdir, index=1, mirror=0.40)
        rec2 = make_candidate_record(self.tmpdir, index=2, mirror=0.80)
        self.assertLess(sut.candidate_rank_key(rec1), sut.candidate_rank_key(rec2))

    def test_build_backend_compatible_output_path(self):
        candidate = make_candidate()
        path = sut.build_backend_compatible_output_path(self.tmpdir, 1, 2, 3, candidate)
        self.assertEqual(path.suffix.lower(), ".png")
        self.assertTrue(path.parent.exists())

    def test_build_candidate_row_compatible(self):
        candidate = make_candidate()
        outcome = types.SimpleNamespace(accepted=True, bestof_score=0.95, reasons=["ok"])
        saved_path = self.tmpdir / "pattern_001.png"
        saved_path.write_bytes(b"x")
        row = sut.build_candidate_row_compatible(1, 2, 3, candidate, outcome, saved_path)
        self.assertEqual(row["image_name"], saved_path.name)
        self.assertEqual(row["image_path"], str(saved_path))

    def test_projection_preview_image(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(40, 40))
        analysis = types.SimpleNamespace(subject_bgr=np.zeros((40, 40, 3), dtype=np.uint8))
        projected = np.full((40, 40, 3), 127, dtype=np.uint8)
        with patch.object(sut, "get_projection_subject_analysis", return_value=analysis),              patch.object(sut, "apply_camo_to_reference", return_value=(projected, np.ones((40, 40), dtype=np.uint8) * 255)):
            out = sut.projection_preview_image(img)
        self.assertIsInstance(out, PILImage.Image)
        self.assertEqual(out.size, (40, 40))

    def test_async_helpers(self):
        candidate = make_candidate()

        async def _go():
            with patch.object(sut.camo, "generate_candidate_from_seed", return_value=candidate) as mock_gen,                  patch.object(sut.camo, "validate_with_reasons", return_value=types.SimpleNamespace(accepted=True)) as mock_val,                  patch.object(sut.camo, "save_candidate_image", return_value=Path("x.png")) as mock_save,                  patch.object(sut.camo, "write_report", return_value=Path("r.csv")) as mock_rep:
                out1 = await sut.async_generate_candidate_from_seed(1)
                out2 = await sut.async_validate_candidate_result(candidate)
                out3 = await sut.async_save_candidate_image(candidate, Path("x.png"))
                out4 = await sut.async_write_report([{"a": 1}], Path("."))
            self.assertIs(out1, candidate)
            self.assertTrue(getattr(out2, "accepted", False))
            self.assertEqual(out3, Path("x.png"))
            self.assertEqual(out4, Path("r.csv"))
            mock_gen.assert_called_once()
            mock_val.assert_called_once()
            mock_save.assert_called_once()
            mock_rep.assert_called_once()

        asyncio.run(_go())

    def test_asyncio_thread_runner(self):
        runner = sut.AsyncioThreadRunner()
        try:
            fut = runner.submit(asyncio.sleep(0, result=42))
            self.assertEqual(fut.result(timeout=2), 42)
        finally:
            runner.stop()
            try:
                runner.loop.close()
            except Exception:
                pass


# ============================================================
# TESTS CAMOUFLAGE APP — INIT / MÉTHODES SYNC
# ============================================================

class TestCamouflageAppInit(AppMixin, LoggedTestCase):
    def test_init_defaults(self):
        self.assertEqual(self.app.generated_rows, [])
        self.assertEqual(self.app.run_mode, sut.RUN_MODE_BLOCKING)
        self.assertIsInstance(self.app.gallery_projection_cache, dict)
        self.assertIsNone(self.app.pending_manual_review)


class TestCamouflageAppMethods(AppMixin, LoggedTestCase):
    def test_status_log_diag_and_progress(self):
        self.app.status("Prêt", ok=True)
        self.app.log("hello")
        self.app.diag_log("diag")
        self.app.update_progress(3, 9)
        self.assertEqual(self.app.status_label.text, "Prêt")
        self.assertIn("hello", self.app.log_view.label.text)
        self.assertIn("diag", self.app.diag_log_view.label.text)
        self.assertEqual(self.app.progress_bar.max_value, 9)
        self.assertEqual(self.app.progress_bar.value, 3)

    def test_update_preview(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(80, 50))
        proj = make_pil_from_index_canvas(make_index_canvas_quadrants(80, 50))

        class FakeCore:
            def __init__(self, texture: str):
                self.texture = texture

        with patch.object(sut, "pil_to_coreimage", side_effect=[FakeCore("img"), FakeCore("proj")]):
            self.app.update_preview(img, proj)
        self.assertEqual(self.app.preview_img.texture, "img")
        self.assertEqual(self.app.preview_silhouette.texture, "proj")

    def test_update_live_stage_and_payload(self):
        img = make_pil_from_index_canvas(make_index_canvas_quadrants(60, 60))

        class FakeCore:
            texture = "live_tex"

        with patch.object(sut, "pil_to_coreimage", return_value=FakeCore()):
            self.app.update_live_stage("validation", 2, 7, 123, "bd 0.05 | miroir 0.44", pil_img=img)
        self.assertIn("validation", self.app.live_stage_label.text)
        self.assertIn("bd 0.05", self.app.live_counts_label.text)
        self.assertIn("Image 002", self.app.live_meta_label.text)
        self.assertEqual(self.app.live_preview_img.texture, "live_tex")

        with patch.object(self.app, "update_live_stage") as mock_stage:
            evt = types.SimpleNamespace(payload={
                "stage": "build",
                "target_index": 1,
                "local_attempt": 2,
                "seed": 3,
                "preview_path": "frame.png",
                "metrics": {"boundary_density": 0.05, "mirror_similarity": 0.4, "edge_contact_ratio": 0.3},
            })
            self.app._maybe_handle_live_runtime_payload(evt)
        mock_stage.assert_called_once()

    def test_refresh_controls_state(self):
        self.app.running = False
        self.app.preflight_running = False
        self.app.stopping = False
        self.app.pending_manual_review = None
        self.app._refresh_controls_state()
        self.assertFalse(self.app.start_btn.disabled)
        self.assertTrue(self.app.stop_btn.disabled)

        self.app.running = True
        self.app.pending_manual_review = types.SimpleNamespace(manually_saved=False)
        self.app._refresh_controls_state()
        self.assertTrue(self.app.start_btn.disabled)
        self.assertFalse(self.app.stop_btn.disabled)
        self.assertFalse(self.app.manual_accept_btn.disabled)

    def test_apply_backend_motif_scale_and_slider(self):
        with patch.object(sut.camo, "set_canvas_geometry") as mock_set:
            out = self.app._apply_backend_motif_scale()
        self.assertAlmostEqual(out, self.app.motif_scale, places=6)
        mock_set.assert_called_once()

        with patch.object(self.app, "_apply_backend_motif_scale", return_value=0.72) as mock_apply:
            self.app._on_motif_scale_change(None, 0.72)
        self.assertAlmostEqual(self.app.motif_scale, 0.72, places=6)
        self.assertEqual(self.app.motif_scale_label.text, "0.72")
        mock_apply.assert_called_once()

    def test_on_projection_scale_change(self):
        raw = make_pil_from_index_canvas(make_index_canvas_quadrants(60, 60))
        self.app._current_preview_raw_img = raw
        fake_future = types.SimpleNamespace(add_done_callback=lambda cb: None)
        self.app.async_runner = types.SimpleNamespace(submit=Mock(side_effect=make_submit_closing_coroutines(fake_future)), stop=Mock(), loop=None)
        with patch.object(sut.Clock, "schedule_once") as mock_sched:
            self.app._on_projection_scale_change(None, 0.55)
        self.assertEqual(self.app.projection_scale_label.text, "0.55")
        self.app.async_runner.submit.assert_called_once()
        mock_sched.assert_called_once()

    def test_manual_review_helpers(self):
        review = sut.PendingManualReview(
            target_index=1,
            local_attempt=2,
            global_attempt=3,
            candidate=make_invalid_candidate(),
            outcome=types.SimpleNamespace(accepted=False, reasons=["mirror_similarity"]),
            projection_img=make_pil_from_index_canvas(make_index_canvas_quadrants()),
            metrics_text="x",
        )
        self.app._arm_manual_review(review)
        self.assertIs(self.app.pending_manual_review, review)
        self.assertIn("image 001", self.app.manual_review_label.text)

        self.app.manual_skip_current_reject()
        self.assertIsNone(self.app.pending_manual_review)
        self.assertIn("Aucun rejet", self.app.manual_review_label.text)

    def test_run_mode_helpers_and_buttons(self):
        self.assertEqual(self.app._run_mode_text(sut.RUN_MODE_SKIP_TESTS), "sans tests")
        self.app._set_run_mode(sut.RUN_MODE_NON_BLOCKING)
        self.assertEqual(self.app.run_mode, sut.RUN_MODE_NON_BLOCKING)
        self.assertIn("non bloquants", self.app.log_view.label.text)

        self.app._refresh_run_mode_buttons()
        self.assertIn("●", self.app.mode_non_blocking_btn.text)
        self.assertIn("○", self.app.mode_blocking_btn.text)
        self.assertIn("Mode actuel", self.app.run_mode_label.text)

    def test_refresh_diag_labels(self):
        self.app.diag_total = 5
        self.app.diag_accepts = 2
        self.app.diag_rejects = 3
        self.app.diag_rule_counter.update({"mirror_similarity": 2, "edge_contact_ratio": 1})
        self.app.diag_last_rules = ["mirror_similarity"]
        self.app._refresh_diag_labels()
        self.assertIn("Tentatives 5", self.app.diag_summary_label.text)
        self.assertIn("Top règles", self.app.diag_top_rules_label.text)
        self.assertIn("mirror_similarity", self.app.diag_last_fail_label.text)

    def test_reload_gallery(self):
        for i in range(3):
            (self.tmpdir / f"pattern_{i:03d}.png").write_bytes(b"x")
        with patch.object(sut, "GalleryThumb", side_effect=lambda app, path: ("thumb", path.name)):
            self.app.reload_gallery()
        self.assertEqual(len(self.app.gallery_grid.items), 3)

    def test_update_attempt_status(self):
        rs = valid_ratios()
        scores = sut.extract_backend_scores(rs, valid_metrics())
        self.app._update_attempt_status(
            target_index=1,
            attempt_idx=2,
            global_attempt=3,
            seed=123,
            target_total=5,
            accepted_count=1,
            rejected_count=2,
            accepted=True,
            rs=rs,
            scores=scores,
            metrics=valid_metrics(),
        )
        self.assertIn("1 / 5", self.app.progress_text.text)
        self.assertIn("Image 001", self.app.attempt_text.text)
        self.assertIn("class_0", self.app.color_text.text)
        self.assertIn("MAE ratio", self.app.score_text.text)
        self.assertIn("overscan", self.app.struct_text.text)

    def test_finish_manual_accept(self):
        review = sut.PendingManualReview(
            target_index=1,
            local_attempt=2,
            global_attempt=3,
            candidate=make_invalid_candidate(),
            outcome=types.SimpleNamespace(accepted=False, reasons=["mirror_similarity"]),
            projection_img=make_pil_from_index_canvas(make_index_canvas_quadrants()),
            metrics_text="x",
            manually_saved=True,
        )
        self.app.generated_rows = [{"x": 1}]
        with patch.object(self.app, "reload_gallery") as mock_reload:
            self.app._finish_manual_accept(Path("a.png"), Path("b.png"), review)
        self.assertEqual(self.app.accepted_count, 1)
        self.assertIn("Rejet validé manuellement", self.app.log_view.label.text)
        mock_reload.assert_called_once()


# ============================================================
# TESTS CAMOUFLAGE APP — ASYNC
# ============================================================

class TestCamouflageAppAsync(AppMixin, LoggedTestCase):
    def test_async_run_preflight_ok_and_ko(self):
        async def _go():
            class Snap:
                cpu_count = 8
                system_available_mb = 4096.0
                disk_free_mb = 8192.0

            with patch.object(self.app, "_apply_backend_motif_scale"),                  patch.object(sut.camo, "validate_generation_request", return_value=None),                  patch.object(sut.camo, "sample_process_resources", return_value=Snap()):
                ok, summary = await self.app._async_run_preflight()
            self.assertTrue(ok)
            self.assertIn("Préflight OK", summary)

            with patch.object(self.app, "_apply_backend_motif_scale"),                  patch.object(sut.camo, "validate_generation_request", side_effect=RuntimeError("boom")):
                ok2, summary2 = await self.app._async_run_preflight()
            self.assertFalse(ok2)
            self.assertIn("Préflight impossible", summary2)

        asyncio.run(_go())

    def test_register_live_diag_accept_and_reject(self):
        candidate = make_candidate()

        async def _go():
            await self.app._register_live_diag(candidate, 1, 1, types.SimpleNamespace(accepted=True))
            self.assertEqual(self.app.diag_accepts, 1)
            bad = make_invalid_candidate()
            await self.app._register_live_diag(bad, 1, 2, types.SimpleNamespace(accepted=False, reasons=["mirror_similarity"]))
            self.assertEqual(self.app.diag_rejects, 1)
            self.assertIn("rejet", self.app.diag_log_view.label.text)

        asyncio.run(_go())

    def test_start_generation_modes(self):
        with patch.object(self.app, "_ensure_preflight") as mock_preflight,              patch.object(self.app, "_start_generation_after_preflight") as mock_start:
            self.app.run_mode = sut.RUN_MODE_BLOCKING
            self.app.tests_ran = False
            self.app.tests_ok = False
            self.app.start_generation()
        mock_preflight.assert_called_once_with(pending_start=True)
        mock_start.assert_not_called()

        with patch.object(self.app, "_update_preflight_label") as mock_label,              patch.object(self.app, "_start_generation_after_preflight") as mock_start2:
            self.app.run_mode = sut.RUN_MODE_SKIP_TESTS
            self.app.start_generation()
        mock_label.assert_called_once()
        mock_start2.assert_called_once()

        with patch.object(self.app, "_ensure_preflight") as mock_preflight2,              patch.object(self.app, "_start_generation_after_preflight") as mock_start3:
            self.app.run_mode = sut.RUN_MODE_NON_BLOCKING
            self.app.tests_ran = False
            self.app.tests_ok = False
            self.app.start_generation()
        mock_preflight2.assert_called_once_with(pending_start=False)
        mock_start3.assert_called_once_with(allow_during_preflight=True)

    def test_start_generation_after_preflight(self):
        with patch.object(self.app, "_apply_backend_motif_scale"),              patch.object(sut, "prevent_sleep"),              patch.object(self.app, "_refresh_diag_labels"),              patch.object(self.app, "_refresh_controls_state"):
            self.app.count_input.text = "2"
            fake_future = MagicMock()
            self.app.async_runner = types.SimpleNamespace(submit=Mock(side_effect=make_submit_closing_coroutines(fake_future)), stop=Mock(), loop=None)
            self.app._start_generation_after_preflight()
        self.assertTrue(self.app.running)
        self.assertEqual(self.app.accepted_count, 0)
        self.assertEqual(self.app.current_future, fake_future)
        self.assertEqual(self.app.generated_rows, [])

        self.app.running = False
        self.app.stopping = False
        self.app.preflight_running = False
        self.app.count_input.text = "abc"
        with patch.object(self.app, "log") as mock_log:
            self.app._start_generation_after_preflight()
        mock_log.assert_called_once()

    def test_async_save_candidate_bundle(self):
        candidate = make_candidate()
        outcome = types.SimpleNamespace(accepted=True, bestof_score=0.95, reasons=[])

        async def _go():
            with patch.object(sut, "async_save_candidate_image", new=AsyncMock(return_value=self.tmpdir / "pattern.png")),                  patch.object(sut, "async_save_mannequin_projection", new=AsyncMock(return_value=self.tmpdir / "pattern__mannequin.png")),                  patch.object(self.app, "reload_gallery") as mock_reload:
                rows: List[dict] = []
                saved, mannequin = await self.app._async_save_candidate_bundle(
                    rows,
                    target_index=1,
                    local_attempt=2,
                    global_attempt=3,
                    candidate=candidate,
                    outcome=outcome,
                    projection_img=make_pil_from_index_canvas(make_index_canvas_quadrants()),
                    manual_accept=False,
                )
            self.assertEqual(saved.name, "pattern.png")
            self.assertEqual(mannequin.name, "pattern__mannequin.png")
            self.assertEqual(len(rows), 1)
            self.assertEqual(self.app.accepted_count, 1)
            self.assertEqual(len(self.app.best_records), 1)
            mock_reload.assert_called_once()

        asyncio.run(_go())

    def test_async_export_best_of_and_csv(self):
        rec = make_candidate_record(self.tmpdir, index=1, mirror=0.4)
        self.app.best_records = [rec]

        async def _go():
            best_dir = await self.app._async_export_best_of(1)
            self.assertTrue(best_dir.exists())
            self.assertTrue((best_dir / "best_of.csv").exists())

        asyncio.run(_go())

    def test_async_finish_success_stopped_error(self):
        async def _go():
            with patch.object(sut, "async_write_report", new=AsyncMock(return_value=self.tmpdir / "report.csv")),                  patch.object(self.app, "_async_export_best_of", new=AsyncMock(return_value=self.tmpdir / "best_of")),                  patch.object(sut, "prevent_sleep"),                  patch.object(self.app, "reload_gallery"),                  patch.object(self.app, "_refresh_controls_state"):
                self.app.running = True
                await self.app._async_finish_success([{"x": 1}])
                self.assertFalse(self.app.running)
                self.assertIn("Rapport écrit", self.app.log_view.label.text)

            with patch.object(sut, "async_write_report", new=AsyncMock(return_value=self.tmpdir / "report.csv")),                  patch.object(self.app, "_async_export_best_of", new=AsyncMock(return_value=self.tmpdir / "best_of")),                  patch.object(sut, "prevent_sleep"),                  patch.object(self.app, "reload_gallery"),                  patch.object(self.app, "_refresh_controls_state"):
                self.app.running = True
                await self.app._async_finish_stopped([{"x": 1}])
                self.assertFalse(self.app.running)
                self.assertIn("Rapport partiel", self.app.log_view.label.text)

            with patch.object(sut, "prevent_sleep"),                  patch.object(self.app, "_refresh_controls_state"):
                self.app.running = True
                await self.app._async_finish_error("boom")
                self.assertFalse(self.app.running)
                self.assertIn("Erreur : boom", self.app.log_view.label.text)
                self.assertIn("Erreur diagnostic", self.app.diag_log_view.label.text)

        asyncio.run(_go())

    def test_finish_error_sync(self):
        with patch.object(sut, "prevent_sleep"),              patch.object(self.app, "_refresh_controls_state"):
            self.app.running = True
            self.app._finish_error("oops")
        self.assertFalse(self.app.running)
        self.assertEqual(self.app.status_label.text, "Erreur")
        self.assertIn("Erreur : oops", self.app.log_view.label.text)

    def test_async_manual_accept_review(self):
        review = sut.PendingManualReview(
            target_index=1,
            local_attempt=2,
            global_attempt=3,
            candidate=make_invalid_candidate(),
            outcome=types.SimpleNamespace(accepted=False, reasons=["mirror_similarity"]),
            projection_img=make_pil_from_index_canvas(make_index_canvas_quadrants()),
            metrics_text="x",
        )

        async def _go():
            with patch.object(self.app, "_async_save_candidate_bundle", new=AsyncMock(return_value=(Path("a.png"), Path("b.png")))):
                saved, mannequin, out_review = await self.app._async_manual_accept_review(review)
            self.assertEqual(saved, Path("a.png"))
            self.assertEqual(mannequin, Path("b.png"))
            self.assertTrue(out_review.manually_saved)

        asyncio.run(_go())

    def test_update_resource_monitor(self):
        with patch.object(sut, "psutil") as fake_psutil:
            fake_psutil.cpu_percent.return_value = 11.0
            fake_psutil.virtual_memory.return_value = types.SimpleNamespace(percent=22.0)
            fake_psutil.disk_usage.return_value = types.SimpleNamespace(percent=33.0)
            self.app._update_resource_monitor(0)
        self.assertIn("CPU", self.app.resource_text.text)
        self.assertIn("RAM", self.app.resource_text.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
