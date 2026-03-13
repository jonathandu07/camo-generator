# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("KIVY_NO_ARGS", "1")

import start


class DummyLogView:
    def __init__(self):
        self.lines = []
        self.label = SimpleNamespace(text="")

    def append(self, line: str):
        self.lines.append(line)


class TestStartApp(unittest.TestCase):
    def make_app(self) -> start.CamouflageApp:
        app = start.CamouflageApp()

        app.status_label = SimpleNamespace(text="", color=None)
        app.log_view = DummyLogView()
        app.diag_log_view = DummyLogView()

        app.diag_summary_label = SimpleNamespace(text="")
        app.diag_top_rules_label = SimpleNamespace(text="")
        app.diag_last_fail_label = SimpleNamespace(text="")
        app.diag_enabled_label = SimpleNamespace(text="", color=None)

        return app

    def test_reset_live_diagnostics(self):
        app = self.make_app()
        app.diag_total = 12
        app.diag_accepts = 2
        app.diag_rejects = 10
        app.diag_rule_counter.update(["ratio_olive", "ratio_olive", "mirror_similarity"])
        app.diag_last_rules = ["ratio_olive"]

        app._reset_live_diagnostics()

        self.assertEqual(app.diag_total, 0)
        self.assertEqual(app.diag_accepts, 0)
        self.assertEqual(app.diag_rejects, 0)
        self.assertEqual(len(app.diag_rule_counter), 0)
        self.assertEqual(app.diag_last_rules, [])

    def test_refresh_live_diag_labels(self):
        app = self.make_app()
        app.diag_total = 10
        app.diag_accepts = 2
        app.diag_rejects = 8
        app.diag_rule_counter.update(["ratio_olive", "ratio_olive", "mirror_similarity"])
        app.diag_last_rules = ["ratio_olive", "mirror_similarity"]

        app._refresh_live_diag_labels()

        self.assertIn("Tentatives 10", app.diag_summary_label.text)
        self.assertIn("ratio_olive:2", app.diag_top_rules_label.text)
        self.assertIn("ratio_olive", app.diag_last_fail_label.text)

    def test_extract_failure_rules(self):
        app = self.make_app()
        fake_candidate = SimpleNamespace(seed=123)

        fake_failure_1 = SimpleNamespace(rule="ratio_olive")
        fake_failure_2 = SimpleNamespace(rule="mirror_similarity")
        fake_diag = SimpleNamespace(failures=[fake_failure_1, fake_failure_2])

        with patch.object(start, "camo_log", SimpleNamespace(analyze_candidate=lambda *args, **kwargs: fake_diag)):
            rules = app._extract_failure_rules(fake_candidate, 1, 2)

        self.assertEqual(rules, ["ratio_olive", "mirror_similarity"])

    def test_ensure_preflight_tests_blocks_on_failure(self):
        app = self.make_app()

        with patch.object(app, "_run_preflight_tests_once", return_value=(False, "2 tests failed")):
            ok = app._ensure_preflight_tests()

        self.assertFalse(ok)
        self.assertEqual(app.tests_ok, False)
        self.assertIn("2 tests failed", app.tests_summary)
        self.assertEqual(app.status_label.text, "Tests KO")

    def test_ensure_preflight_tests_allows_start_on_success(self):
        app = self.make_app()

        with patch.object(app, "_run_preflight_tests_once", return_value=(True, "4 tests OK")):
            ok = app._ensure_preflight_tests()

        self.assertTrue(ok)
        self.assertEqual(app.tests_ok, True)
        self.assertIn("4 tests OK", app.tests_summary)


if __name__ == "__main__":
    unittest.main()