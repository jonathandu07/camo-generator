# -*- coding: utf-8 -*-
"""
log.py
Analyseur de rejets, orchestrateur de tests et infrastructure de logs
continus / asynchrones pour le générateur de camouflage.

Rôles principaux :
- analyser précisément pourquoi un candidat est rejeté ;
- exposer une API sync/async stable pour start.py ;
- lancer les préflight tests unitaires (test_main.py, test_start.py) ;
- agréger des logs runtime continus dans des fichiers dédiés ;
- exporter des diagnostics exploitables pour main.py et start.py ;
- permettre un suivi console en direct des tests et des diagnostics.

Sorties principales :
- logs_generation/diagnostic_candidates.csv
- logs_generation/diagnostic_summary.json
- logs_generation/diagnostic_summary.txt
- logs_generation/runtime.log
- logs_generation/tests_summary.json
- logs_generation/runtime_snapshot.json
"""

from __future__ import annotations

import os
os.environ.setdefault("KIVY_NO_ARGS", "1")

import argparse
import asyncio
import csv
import importlib
import json
import logging
import statistics
import subprocess
import sys
import threading
import time
import unittest
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import numpy as np

import main as camo


# ============================================================
# CONFIG
# ============================================================

DEFAULT_ANALYSIS_COUNT = 20
DEFAULT_OUTPUT_DIR = Path("logs_generation")
DEFAULT_TEST_MODULES: tuple[str, ...] = ("test_main", "test_start")
DEFAULT_RUNTIME_LOG_FILE = "runtime.log"
DEFAULT_TEST_SUMMARY_FILE = "tests_summary.json"
DEFAULT_HISTORY_LIMIT = 5000
DEFAULT_MAX_CONCURRENCY = max(1, min(4, (os.cpu_count() or 4)))
DEFAULT_TEST_TIMEOUT_S: float | None = None
DEFAULT_LIVE_CONSOLE = True
DEFAULT_CONSOLE_LEVEL = "INFO"

LOGGER_NAME = "camo_log"


# ============================================================
# STRUCTURES
# ============================================================

@dataclass
class RuleFailure:
    rule: str
    actual: float
    target: float | None
    min_value: float | None
    max_value: float | None
    delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "actual": round(float(self.actual), 8),
            "target": None if self.target is None else round(float(self.target), 8),
            "min_value": None if self.min_value is None else round(float(self.min_value), 8),
            "max_value": None if self.max_value is None else round(float(self.max_value), 8),
            "delta": round(float(self.delta), 8),
        }


@dataclass
class CandidateDiagnostic:
    seed: int
    target_index: int
    local_attempt: int
    accepted: bool
    ratios: Dict[str, float]
    metrics: Dict[str, float]
    failures: List[RuleFailure]

    def to_csv_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "seed": self.seed,
            "target_index": self.target_index,
            "local_attempt": self.local_attempt,
            "accepted": int(self.accepted),
            "fail_count": len(self.failures),
            "fail_rules": " | ".join(f.rule for f in self.failures),
        }

        for k, v in self.ratios.items():
            row[k] = round(float(v), 8)

        for k, v in self.metrics.items():
            row[k] = round(float(v), 8)

        for i, failure in enumerate(self.failures[:12], start=1):
            row[f"fail_{i}_rule"] = failure.rule
            row[f"fail_{i}_actual"] = round(float(failure.actual), 8)
            row[f"fail_{i}_delta"] = round(float(failure.delta), 8)

        return row

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "target_index": int(self.target_index),
            "local_attempt": int(self.local_attempt),
            "accepted": bool(self.accepted),
            "ratios": {k: round(float(v), 8) for k, v in self.ratios.items()},
            "metrics": {k: round(float(v), 8) for k, v in self.metrics.items()},
            "failures": [f.to_dict() for f in self.failures],
        }


@dataclass
class RuntimeEvent:
    ts: float
    level: str
    source: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": float(self.ts),
            "level": str(self.level),
            "source": str(self.source),
            "message": str(self.message),
            "payload": dict(self.payload),
        }

    def format_line(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))
        if self.payload:
            return (
                f"{stamp} | {self.level:<8} | {self.source} | {self.message} | "
                f"{json.dumps(self.payload, ensure_ascii=False, sort_keys=True, default=str)}"
            )
        return f"{stamp} | {self.level:<8} | {self.source} | {self.message}"


@dataclass
class TestModuleSummary:
    module: str
    ok: bool
    returncode: int
    duration_s: float
    command: List[str]
    stdout: str
    stderr: str
    log_file: str | None = None
    completed: bool = True
    timed_out: bool = False
    timeout_s: float | None = None
    exception_message: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestModuleSummary":
        return cls(**data)


@dataclass
class TestRunSummary:
    ok: bool
    total: int
    failures: int
    errors: int
    skipped: int
    expected_failures: int
    unexpected_successes: int
    duration_s: float
    modules: List[str]
    log_files: Dict[str, str]
    details: Dict[str, List[Dict[str, str]]]
    per_module: List[TestModuleSummary] = field(default_factory=list)
    parallel: bool = False
    completed: bool = True
    timed_out: bool = False
    timed_out_modules: List[str] = field(default_factory=list)

    def short_text(self) -> str:
        if self.ok and self.completed and not self.timed_out:
            return f"{self.total} tests OK"
        if self.timed_out:
            return (
                f"{self.total} tests comptés | {self.failures} échec(s) | "
                f"{self.errors} erreur(s) | timeout module(s): {', '.join(self.timed_out_modules)}"
            )
        return f"{self.total} tests exécutés | {self.failures} échec(s) | {self.errors} erreur(s)"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["per_module"] = [m.to_dict() for m in self.per_module]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestRunSummary":
        copied = dict(data)
        copied["per_module"] = [
            item if isinstance(item, TestModuleSummary) else TestModuleSummary.from_dict(item)
            for item in copied.get("per_module", [])
        ]
        return cls(**copied)


# ============================================================
# OUTILS GÉNÉRIQUES
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if np.isnan(out) or np.isinf(out):
        return float(default)
    return out


def _metric(metrics: Dict[str, Any], name: str, default: float = 0.0) -> float:
    return _safe_float(metrics.get(name, default), default=default)


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _normalize_module_names(module_names: Sequence[str] | None) -> List[str]:
    if not module_names:
        return list(DEFAULT_TEST_MODULES)

    out: List[str] = []
    seen = set()
    for name in module_names:
        name = str(name).strip()
        if not name:
            continue
        if name.endswith(".py"):
            name = name[:-3]
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out or list(DEFAULT_TEST_MODULES)


def _should_block_on_tests(summary: TestRunSummary) -> bool:
    return (not summary.ok) or (not summary.completed) or summary.timed_out


def _subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["KIVY_NO_ARGS"] = "1"
    return env


def _coerce_test_run_summary(value: Any) -> TestRunSummary:
    if isinstance(value, TestRunSummary):
        return value
    if isinstance(value, dict):
        return TestRunSummary.from_dict(value)
    raise TypeError(f"Impossible de convertir en TestRunSummary: {type(value)!r}")


def _normalize_timeout(timeout_s: float | None) -> float | None:
    if timeout_s is None:
        return None
    try:
        value = float(timeout_s)
    except Exception:
        return None
    return None if value <= 0 else value


# ============================================================
# LOGGER CENTRAL + FLUX CONTINU
# ============================================================

class ContinuousLogManager:
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        runtime_filename: str = DEFAULT_RUNTIME_LOG_FILE,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self.output_dir = ensure_dir(Path(output_dir))
        self.runtime_path = self.output_dir / runtime_filename
        self.history: Deque[RuntimeEvent] = deque(maxlen=int(history_limit))
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[RuntimeEvent], None]] = []
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        runtime_path = self.runtime_path.resolve()
        already = False
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if Path(handler.baseFilename).resolve() == runtime_path:
                        already = True
                        break
                except Exception:
                    pass

        if not already:
            fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh = logging.FileHandler(runtime_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        return logger

    def subscribe(self, callback: Callable[[RuntimeEvent], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[RuntimeEvent], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def emit(self, level: str, source: str, message: str, **payload: Any) -> RuntimeEvent:
        event = RuntimeEvent(
            ts=time.time(),
            level=str(level).upper(),
            source=str(source),
            message=str(message),
            payload=dict(payload),
        )
        with self._lock:
            self.history.append(event)
            log_message = event.message
            if payload:
                log_message = (
                    f"{log_message} | payload="
                    f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"
                )
            self.logger.log(getattr(logging, event.level, logging.INFO), f"[{event.source}] {log_message}")
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                pass

        return event

    async def async_emit(self, level: str, source: str, message: str, **payload: Any) -> RuntimeEvent:
        return await asyncio.to_thread(self.emit, level, source, message, **payload)

    def last_lines(self, n: int = 50) -> List[str]:
        with self._lock:
            return [evt.format_line() for evt in list(self.history)[-max(1, int(n)):]]

    def snapshot(self, n: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            return [evt.to_dict() for evt in list(self.history)[-max(1, int(n)):]]


LOG_MANAGER = ContinuousLogManager(DEFAULT_OUTPUT_DIR)


def log_event(level: str, source: str, message: str, **payload: Any) -> RuntimeEvent:
    return LOG_MANAGER.emit(level, source, message, **payload)


async def async_log_event(level: str, source: str, message: str, **payload: Any) -> RuntimeEvent:
    return await LOG_MANAGER.async_emit(level, source, message, **payload)


def get_recent_runtime_lines(n: int = 50) -> List[str]:
    return LOG_MANAGER.last_lines(n)


def get_runtime_snapshot(n: int = 200) -> List[Dict[str, Any]]:
    return LOG_MANAGER.snapshot(n)


def attach_live_console_printer(min_level: str = DEFAULT_CONSOLE_LEVEL) -> Callable[[RuntimeEvent], None]:
    min_value = getattr(logging, str(min_level).upper(), logging.INFO)
    print_lock = threading.Lock()

    def _printer(event: RuntimeEvent) -> None:
        level_value = getattr(logging, str(event.level).upper(), logging.INFO)
        if level_value < min_value:
            return
        with print_lock:
            print(event.format_line(), flush=True)

    LOG_MANAGER.subscribe(_printer)
    return _printer


# ============================================================
# RÈGLES DE DIAGNOSTIC
# ============================================================

def rule_fail_min(name: str, actual: float, min_value: float) -> RuleFailure | None:
    if actual >= min_value:
        return None
    return RuleFailure(
        rule=name,
        actual=actual,
        target=None,
        min_value=min_value,
        max_value=None,
        delta=min_value - actual,
    )


def rule_fail_max(name: str, actual: float, max_value: float) -> RuleFailure | None:
    if actual <= max_value:
        return None
    return RuleFailure(
        rule=name,
        actual=actual,
        target=None,
        min_value=None,
        max_value=max_value,
        delta=actual - max_value,
    )


def rule_fail_range(name: str, actual: float, min_value: float, max_value: float) -> RuleFailure | None:
    if actual < min_value:
        return RuleFailure(
            rule=name,
            actual=actual,
            target=None,
            min_value=min_value,
            max_value=max_value,
            delta=min_value - actual,
        )
    if actual > max_value:
        return RuleFailure(
            rule=name,
            actual=actual,
            target=None,
            min_value=min_value,
            max_value=max_value,
            delta=actual - max_value,
        )
    return None


def rule_fail_target_abs(name: str, actual: float, target: float, max_abs_error: float) -> RuleFailure | None:
    delta = abs(actual - target)
    if delta <= max_abs_error:
        return None
    return RuleFailure(
        rule=name,
        actual=actual,
        target=target,
        min_value=None,
        max_value=max_abs_error,
        delta=delta - max_abs_error,
    )


# ============================================================
# ANALYSE D'UN CANDIDAT
# ============================================================

def analyze_candidate(
    candidate: camo.CandidateResult,
    target_index: int,
    local_attempt: int,
) -> CandidateDiagnostic:
    rs = np.asarray(candidate.ratios, dtype=float)
    m = dict(candidate.metrics)

    failures: List[RuleFailure] = []

    abs_err = np.abs(rs - camo.TARGET)
    mean_abs_err = float(np.mean(abs_err))

    per_color_rules = [
        (
            "abs_err_coyote",
            float(rs[camo.IDX_COYOTE]),
            float(camo.TARGET[camo.IDX_COYOTE]),
            float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_COYOTE]),
        ),
        (
            "abs_err_olive",
            float(rs[camo.IDX_OLIVE]),
            float(camo.TARGET[camo.IDX_OLIVE]),
            float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_OLIVE]),
        ),
        (
            "abs_err_terre",
            float(rs[camo.IDX_TERRE]),
            float(camo.TARGET[camo.IDX_TERRE]),
            float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_TERRE]),
        ),
        (
            "abs_err_gris",
            float(rs[camo.IDX_GRIS]),
            float(camo.TARGET[camo.IDX_GRIS]),
            float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_GRIS]),
        ),
    ]
    for name, actual, target, max_abs in per_color_rules:
        fail = rule_fail_target_abs(name, actual, target, max_abs)
        if fail is not None:
            failures.append(fail)

    fail = rule_fail_max("mean_abs_error", mean_abs_err, float(camo.MAX_MEAN_ABS_ERROR))
    if fail is not None:
        failures.append(fail)

    ratio_checks = [
        ("ratio_coyote", float(rs[camo.IDX_COYOTE]), 0.27, 0.37),
        ("ratio_olive", float(rs[camo.IDX_OLIVE]), 0.24, 0.33),
        ("ratio_terre", float(rs[camo.IDX_TERRE]), 0.19, 0.26),
        ("ratio_gris", float(rs[camo.IDX_GRIS]), 0.14, 0.21),
    ]
    for name, actual, min_v, max_v in ratio_checks:
        fail = rule_fail_range(name, actual, min_v, max_v)
        if fail is not None:
            failures.append(fail)

    metric_min_checks = [
        ("largest_olive_component_ratio", _metric(m, "largest_olive_component_ratio"), float(camo.MIN_OLIVE_CONNECTED_COMPONENT_RATIO)),
        ("largest_olive_component_ratio_small", _metric(m, "largest_olive_component_ratio_small"), 0.12),
        ("olive_multizone_share", _metric(m, "olive_multizone_share"), float(camo.MIN_OLIVE_MULTIZONE_SHARE)),
        ("boundary_density", _metric(m, "boundary_density"), float(camo.MIN_BOUNDARY_DENSITY)),
        ("boundary_density_small", _metric(m, "boundary_density_small"), float(camo.MIN_BOUNDARY_DENSITY_SMALL)),
        ("oblique_share", _metric(m, "oblique_share"), float(camo.MIN_OBLIQUE_SHARE)),
        ("vert_olive_macro_share", _metric(m, "vert_olive_macro_share"), float(camo.MIN_VISIBLE_OLIVE_MACRO_SHARE)),
        ("terre_de_france_transition_share", _metric(m, "terre_de_france_transition_share"), float(camo.MIN_VISIBLE_TERRE_TRANS_SHARE)),
        ("vert_de_gris_micro_share", _metric(m, "vert_de_gris_micro_share"), float(camo.MIN_VISIBLE_GRIS_MICRO_SHARE)),
    ]
    for name, actual, min_v in metric_min_checks:
        fail = rule_fail_min(name, float(actual), float(min_v))
        if fail is not None:
            failures.append(fail)

    metric_max_checks = [
        ("center_empty_ratio", _metric(m, "center_empty_ratio"), float(camo.MAX_COYOTE_CENTER_EMPTY_RATIO)),
        ("center_empty_ratio_small", _metric(m, "center_empty_ratio_small"), float(camo.MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL)),
        ("boundary_density", _metric(m, "boundary_density"), float(camo.MAX_BOUNDARY_DENSITY)),
        ("boundary_density_small", _metric(m, "boundary_density_small"), float(camo.MAX_BOUNDARY_DENSITY_SMALL)),
        ("mirror_similarity", _metric(m, "mirror_similarity"), float(camo.MAX_MIRROR_SIMILARITY)),
        ("angle_dominance_ratio", _metric(m, "angle_dominance_ratio"), float(camo.MAX_ANGLE_DOMINANCE_RATIO)),
        ("vert_de_gris_macro_share", _metric(m, "vert_de_gris_macro_share"), float(camo.MAX_VISIBLE_GRIS_MACRO_SHARE)),
    ]
    for name, actual, max_v in metric_max_checks:
        fail = rule_fail_max(name, float(actual), float(max_v))
        if fail is not None:
            failures.append(fail)

    vertical_fail = rule_fail_range(
        "vertical_share",
        _metric(m, "vertical_share"),
        float(camo.MIN_VERTICAL_SHARE),
        float(camo.MAX_VERTICAL_SHARE),
    )
    if vertical_fail is not None:
        failures.append(vertical_fail)

    accepted = len(failures) == 0

    ratios = {
        "ratio_coyote": float(rs[camo.IDX_COYOTE]),
        "ratio_olive": float(rs[camo.IDX_OLIVE]),
        "ratio_terre": float(rs[camo.IDX_TERRE]),
        "ratio_gris": float(rs[camo.IDX_GRIS]),
        "mean_abs_error": mean_abs_err,
        "abs_err_coyote": float(abs_err[camo.IDX_COYOTE]),
        "abs_err_olive": float(abs_err[camo.IDX_OLIVE]),
        "abs_err_terre": float(abs_err[camo.IDX_TERRE]),
        "abs_err_gris": float(abs_err[camo.IDX_GRIS]),
    }

    metrics = {k: _safe_float(v) for k, v in m.items()}

    diagnostic = CandidateDiagnostic(
        seed=int(candidate.seed),
        target_index=int(target_index),
        local_attempt=int(local_attempt),
        accepted=accepted,
        ratios=ratios,
        metrics=metrics,
        failures=failures,
    )

    log_event(
        "INFO" if diagnostic.accepted else "WARNING",
        "diagnostic",
        "Candidat analysé",
        seed=diagnostic.seed,
        target_index=diagnostic.target_index,
        local_attempt=diagnostic.local_attempt,
        accepted=diagnostic.accepted,
        fail_count=len(diagnostic.failures),
        fail_rules=[f.rule for f in diagnostic.failures],
    )
    return diagnostic


async def async_analyze_candidate(
    candidate: camo.CandidateResult,
    target_index: int,
    local_attempt: int,
) -> CandidateDiagnostic:
    return await asyncio.to_thread(analyze_candidate, candidate, target_index, local_attempt)


# ============================================================
# GÉNÉRATION DE DIAGNOSTICS
# ============================================================

def _generate_one_diagnostic(i: int, base_seed: int) -> CandidateDiagnostic:
    seed = camo.build_seed(target_index=1, local_attempt=i, base_seed=base_seed)
    t0 = time.perf_counter()
    log_event("INFO", "generate_diagnostics", "Début génération candidat", index=i, seed=seed)
    candidate = camo.generate_candidate_from_seed(seed)
    log_event(
        "INFO",
        "generate_diagnostics",
        "Fin génération candidat",
        index=i,
        seed=seed,
        duration_s=round(time.perf_counter() - t0, 6),
    )
    diagnostic = analyze_candidate(candidate, target_index=1, local_attempt=i)
    log_event(
        "INFO",
        "generate_diagnostics",
        "Diagnostic candidat terminé",
        index=i,
        seed=seed,
        accepted=diagnostic.accepted,
    )
    return diagnostic


def generate_diagnostics(
    count: int,
    base_seed: int = camo.DEFAULT_BASE_SEED,
    parallel: bool = False,
    max_workers: int = DEFAULT_MAX_CONCURRENCY,
) -> List[CandidateDiagnostic]:
    diagnostics: List[CandidateDiagnostic] = []
    count = max(0, int(count))
    max_workers = max(1, int(max_workers))

    log_event(
        "INFO",
        "generate_diagnostics",
        "Début diagnostic synchrone",
        count=count,
        base_seed=int(base_seed),
        parallel=bool(parallel),
        max_workers=max_workers,
    )

    if count == 0:
        log_event("INFO", "generate_diagnostics", "Fin diagnostic synchrone", count=0)
        return diagnostics

    if not parallel or count == 1:
        for i in range(1, count + 1):
            diagnostics.append(_generate_one_diagnostic(i, base_seed))
            log_event(
                "INFO",
                "generate_diagnostics",
                "Progression diagnostics",
                completed=len(diagnostics),
                total=count,
            )
    else:
        ordered: Dict[int, CandidateDiagnostic] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="diag") as executor:
            futures = {executor.submit(_generate_one_diagnostic, i, base_seed): i for i in range(1, count + 1)}
            for future in as_completed(futures):
                idx = futures[future]
                ordered[idx] = future.result()
                log_event(
                    "INFO",
                    "generate_diagnostics",
                    "Progression diagnostics parallèles",
                    completed=len(ordered),
                    total=count,
                    last_index=idx,
                )
        diagnostics = [ordered[i] for i in range(1, count + 1)]

    log_event(
        "INFO",
        "generate_diagnostics",
        "Fin diagnostic synchrone",
        count=len(diagnostics),
        parallel=bool(parallel),
    )
    return diagnostics


async def async_generate_diagnostics(
    count: int,
    base_seed: int = camo.DEFAULT_BASE_SEED,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    parallel: bool = True,
) -> List[CandidateDiagnostic]:
    count = max(0, int(count))
    max_concurrency = max(1, int(max_concurrency))

    await async_log_event(
        "INFO",
        "async_generate_diagnostics",
        "Début diagnostic asynchrone",
        count=count,
        base_seed=int(base_seed),
        parallel=bool(parallel),
        max_concurrency=max_concurrency,
    )

    if count == 0:
        await async_log_event("INFO", "async_generate_diagnostics", "Fin diagnostic asynchrone", count=0)
        return []

    async def _one_logged(i: int) -> CandidateDiagnostic:
        seed = camo.build_seed(target_index=1, local_attempt=i, base_seed=base_seed)
        t0 = time.perf_counter()
        await async_log_event("INFO", "async_generate_diagnostics", "Début génération candidat", index=i, seed=seed)
        candidate = await camo.async_generate_candidate_from_seed(seed)
        await async_log_event(
            "INFO",
            "async_generate_diagnostics",
            "Fin génération candidat",
            index=i,
            seed=seed,
            duration_s=round(time.perf_counter() - t0, 6),
        )
        diagnostic = await async_analyze_candidate(candidate, target_index=1, local_attempt=i)
        await async_log_event(
            "INFO",
            "async_generate_diagnostics",
            "Diagnostic candidat terminé",
            index=i,
            seed=seed,
            accepted=diagnostic.accepted,
        )
        return diagnostic

    if not parallel or count == 1:
        diagnostics: List[CandidateDiagnostic] = []
        for i in range(1, count + 1):
            diagnostics.append(await _one_logged(i))
            await async_log_event(
                "INFO",
                "async_generate_diagnostics",
                "Progression diagnostics",
                completed=len(diagnostics),
                total=count,
            )
        await async_log_event(
            "INFO",
            "async_generate_diagnostics",
            "Fin diagnostic asynchrone",
            count=len(diagnostics),
            parallel=False,
        )
        return diagnostics

    sem = asyncio.Semaphore(max_concurrency)

    async def one(i: int) -> tuple[int, CandidateDiagnostic]:
        async with sem:
            diagnostic = await _one_logged(i)
            await async_log_event(
                "INFO",
                "async_generate_diagnostics",
                "Progression diagnostics parallèles",
                completed=i,
                total=count,
                last_index=i,
            )
            return i, diagnostic

    results = await asyncio.gather(*(one(i) for i in range(1, count + 1)))
    results.sort(key=lambda x: x[0])
    diagnostics = [diag for _, diag in results]

    await async_log_event(
        "INFO",
        "async_generate_diagnostics",
        "Fin diagnostic asynchrone",
        count=len(diagnostics),
        parallel=True,
        max_concurrency=max_concurrency,
    )
    return diagnostics


# ============================================================
# SYNTHÈSE
# ============================================================

def build_summary(diagnostics: List[CandidateDiagnostic]) -> Dict[str, Any]:
    total = len(diagnostics)
    accepted = sum(1 for d in diagnostics if d.accepted)
    rejected = total - accepted

    fail_counter: Counter[str] = Counter()
    fail_deltas: Dict[str, List[float]] = defaultdict(list)
    fail_examples: Dict[str, List[int]] = defaultdict(list)

    metric_values: Dict[str, List[float]] = defaultdict(list)
    ratio_values: Dict[str, List[float]] = defaultdict(list)

    combo_counter: Counter[str] = Counter()

    for d in diagnostics:
        for k, v in d.ratios.items():
            ratio_values[k].append(float(v))
        for k, v in d.metrics.items():
            metric_values[k].append(float(v))

        if d.failures:
            combo_key = " | ".join(sorted(f.rule for f in d.failures))
            combo_counter[combo_key] += 1

        for f in d.failures:
            fail_counter[f.rule] += 1
            fail_deltas[f.rule].append(float(f.delta))
            if len(fail_examples[f.rule]) < 10:
                fail_examples[f.rule].append(int(d.seed))

    def describe_dist(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "min": float(min(values)),
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
        }

    fail_stats = []
    for rule, count in fail_counter.most_common():
        fail_stats.append(
            {
                "rule": rule,
                "count": int(count),
                "rate_over_all": float(count / total) if total else 0.0,
                "avg_delta": float(statistics.fmean(fail_deltas[rule])) if fail_deltas[rule] else 0.0,
                "max_delta": float(max(fail_deltas[rule])) if fail_deltas[rule] else 0.0,
                "example_seeds": fail_examples[rule],
            }
        )

    metric_summary = {k: describe_dist(v) for k, v in sorted(metric_values.items())}
    ratio_summary = {k: describe_dist(v) for k, v in sorted(ratio_values.items())}

    top_fail_combinations = [
        {"rules": combo, "count": int(count)}
        for combo, count in combo_counter.most_common(20)
    ]

    summary = {
        "total_candidates": int(total),
        "accepted": int(accepted),
        "rejected": int(rejected),
        "acceptance_rate": float(accepted / total) if total else 0.0,
        "top_failure_rules": fail_stats,
        "top_failure_combinations": top_fail_combinations,
        "ratios_distribution": ratio_summary,
        "metrics_distribution": metric_summary,
    }

    log_event(
        "INFO",
        "summary",
        "Synthèse diagnostics calculée",
        total_candidates=summary["total_candidates"],
        accepted=summary["accepted"],
        rejected=summary["rejected"],
    )
    return summary


# ============================================================
# EXPORTS
# ============================================================

def write_candidates_csv(diagnostics: List[CandidateDiagnostic], output_dir: Path) -> Path:
    output_dir = ensure_dir(Path(output_dir))
    path = output_dir / "diagnostic_candidates.csv"
    rows = [d.to_csv_row() for d in diagnostics]

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log_event("INFO", "export", "CSV diagnostics écrit", path=str(path.resolve()), rows=len(rows))
    return path


async def async_write_candidates_csv(diagnostics: List[CandidateDiagnostic], output_dir: Path) -> Path:
    return await asyncio.to_thread(write_candidates_csv, diagnostics, output_dir)


def write_summary_json(summary: Dict[str, Any], output_dir: Path) -> Path:
    output_dir = ensure_dir(Path(output_dir))
    path = output_dir / "diagnostic_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("INFO", "export", "JSON synthèse écrit", path=str(path.resolve()))
    return path


async def async_write_summary_json(summary: Dict[str, Any], output_dir: Path) -> Path:
    return await asyncio.to_thread(write_summary_json, summary, output_dir)


def write_summary_txt(summary: Dict[str, Any], output_dir: Path) -> Path:
    output_dir = ensure_dir(Path(output_dir))
    path = output_dir / "diagnostic_summary.txt"

    lines: List[str] = []
    lines.append("=== DIAGNOSTIC GENERATION CAMOUFLAGE ===")
    lines.append("")
    lines.append(f"Total candidats : {summary['total_candidates']}")
    lines.append(f"Acceptés        : {summary['accepted']}")
    lines.append(f"Rejetés         : {summary['rejected']}")
    lines.append(f"Taux acceptation: {summary['acceptance_rate']:.4%}")
    lines.append("")
    lines.append("=== REGLES LES PLUS BLOQUANTES ===")

    for item in summary["top_failure_rules"][:20]:
        lines.append(
            f"- {item['rule']}: "
            f"{item['count']} échec(s), "
            f"{item['rate_over_all']:.2%} des candidats, "
            f"delta moyen={item['avg_delta']:.6f}, "
            f"delta max={item['max_delta']:.6f}, "
            f"seeds={item['example_seeds']}"
        )

    lines.append("")
    lines.append("=== COMBINAISONS D'ECHECS LES PLUS FREQUENTES ===")
    for item in summary["top_failure_combinations"][:15]:
        lines.append(f"- {item['count']}x :: {item['rules']}")

    path.write_text("\n".join(lines), encoding="utf-8")
    log_event("INFO", "export", "TXT synthèse écrit", path=str(path.resolve()))
    return path


async def async_write_summary_txt(summary: Dict[str, Any], output_dir: Path) -> Path:
    return await asyncio.to_thread(write_summary_txt, summary, output_dir)


def export_runtime_snapshot(output_dir: Path, filename: str = "runtime_snapshot.json") -> Path:
    output_dir = ensure_dir(Path(output_dir))
    path = output_dir / filename
    path.write_text(json.dumps(get_runtime_snapshot(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ============================================================
# TESTS UNITAIRES — OUTILS
# ============================================================

def _extract_test_case_tuples(items: Sequence[tuple[Any, str]]) -> List[Dict[str, str]]:
    extracted: List[Dict[str, str]] = []
    for case, trace in items:
        extracted.append({"test": str(case), "trace": str(trace)})
    return extracted


def discover_test_log_files(module_names: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            log_file = getattr(module, "LOG_FILE", None)
            if log_file is not None:
                out[module_name] = str(Path(log_file).resolve())
        except Exception:
            continue
    return out


def _count_test_methods_in_module(module_name: str) -> int:
    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        return int(suite.countTestCases())
    except Exception:
        return 0


def _collect_parallel_test_counts(module_names: Sequence[str]) -> int:
    return sum(_count_test_methods_in_module(module_name) for module_name in module_names)


def _parse_unittest_output_for_failures(stdout: str, stderr: str) -> tuple[int, int]:
    text = f"{stdout}\n{stderr}"
    failures = 0
    errors = 0

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("FAILED"):
            continue
        if "(" not in line or ")" not in line:
            continue

        inside = line[line.find("(") + 1 : line.rfind(")")]
        parts = [p.strip() for p in inside.split(",")]
        for part in parts:
            if part.startswith("failures="):
                try:
                    failures = int(part.split("=", 1)[1])
                except Exception:
                    pass
            elif part.startswith("errors="):
                try:
                    errors = int(part.split("=", 1)[1])
                except Exception:
                    pass

    return failures, errors


def _build_test_command(module_name: str) -> List[str]:
    return [sys.executable, "-m", "unittest", "-v", module_name]


def _read_declared_test_log_file(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
        lf = getattr(module, "LOG_FILE", None)
        if lf is not None:
            return _safe_relpath(Path(lf))
    except Exception:
        pass
    return None


def _run_test_module_subprocess(module_name: str, timeout_s: float | None = None) -> TestModuleSummary:
    timeout_s = _normalize_timeout(timeout_s)
    cmd = _build_test_command(module_name)
    t0 = time.perf_counter()

    log_event(
        "INFO",
        "tests",
        "Lancement module de test",
        module=module_name,
        command=cmd,
        timeout_s=timeout_s,
        mode="sync-subprocess",
    )

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_subprocess_env(),
            timeout=timeout_s,
        )
        dt = time.perf_counter() - t0
        summary = TestModuleSummary(
            module=module_name,
            ok=(completed.returncode == 0),
            returncode=int(completed.returncode),
            duration_s=float(dt),
            command=cmd,
            stdout=completed.stdout,
            stderr=completed.stderr,
            log_file=_read_declared_test_log_file(module_name),
            completed=True,
            timed_out=False,
            timeout_s=timeout_s,
            exception_message=None,
        )
    except subprocess.TimeoutExpired as exc:
        dt = time.perf_counter() - t0
        stdout = ""
        stderr = ""
        if exc.stdout:
            stdout = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace")
        if exc.stderr:
            stderr = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="replace")

        summary = TestModuleSummary(
            module=module_name,
            ok=False,
            returncode=-9,
            duration_s=float(dt),
            command=cmd,
            stdout=stdout,
            stderr=stderr or f"Timeout expiré après {timeout_s} s",
            log_file=_read_declared_test_log_file(module_name),
            completed=False,
            timed_out=True,
            timeout_s=timeout_s,
            exception_message=f"TimeoutExpired({timeout_s})",
        )
    except Exception as exc:
        dt = time.perf_counter() - t0
        summary = TestModuleSummary(
            module=module_name,
            ok=False,
            returncode=-1,
            duration_s=float(dt),
            command=cmd,
            stdout="",
            stderr=f"Exception durant l'exécution du module {module_name}: {exc}",
            log_file=_read_declared_test_log_file(module_name),
            completed=False,
            timed_out=False,
            timeout_s=timeout_s,
            exception_message=str(exc),
        )

    log_event(
        "INFO" if summary.ok else "ERROR",
        "tests",
        "Fin module de test",
        module=module_name,
        ok=summary.ok,
        returncode=summary.returncode,
        duration_s=round(summary.duration_s, 6),
        completed=summary.completed,
        timed_out=summary.timed_out,
    )
    return summary


async def _async_run_test_module_subprocess(
    module_name: str,
    timeout_s: float | None = None,
) -> TestModuleSummary:
    timeout_s = _normalize_timeout(timeout_s)
    cmd = _build_test_command(module_name)
    t0 = time.perf_counter()

    await async_log_event(
        "INFO",
        "tests",
        "Lancement module de test",
        module=module_name,
        command=cmd,
        timeout_s=timeout_s,
        mode="async-subprocess",
    )

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_subprocess_env(),
        )

        if timeout_s is None:
            stdout_b, stderr_b = await proc.communicate()
        else:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

        dt = time.perf_counter() - t0
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        summary = TestModuleSummary(
            module=module_name,
            ok=(proc.returncode == 0),
            returncode=int(proc.returncode or 0),
            duration_s=float(dt),
            command=cmd,
            stdout=stdout,
            stderr=stderr,
            log_file=_read_declared_test_log_file(module_name),
            completed=True,
            timed_out=False,
            timeout_s=timeout_s,
            exception_message=None,
        )

    except asyncio.TimeoutError:
        dt = time.perf_counter() - t0
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                stdout_b, stderr_b = await proc.communicate()
            except Exception:
                stdout_b, stderr_b = b"", b""

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
        else:
            stdout = ""
            stderr = ""

        if not stderr:
            stderr = f"Timeout expiré après {timeout_s} s"

        summary = TestModuleSummary(
            module=module_name,
            ok=False,
            returncode=-9,
            duration_s=float(dt),
            command=cmd,
            stdout=stdout,
            stderr=stderr,
            log_file=_read_declared_test_log_file(module_name),
            completed=False,
            timed_out=True,
            timeout_s=timeout_s,
            exception_message=f"TimeoutError({timeout_s})",
        )

    except Exception as exc:
        dt = time.perf_counter() - t0
        summary = TestModuleSummary(
            module=module_name,
            ok=False,
            returncode=-1,
            duration_s=float(dt),
            command=cmd,
            stdout="",
            stderr=f"Exception durant l'exécution du module {module_name}: {exc}",
            log_file=_read_declared_test_log_file(module_name),
            completed=False,
            timed_out=False,
            timeout_s=timeout_s,
            exception_message=str(exc),
        )

    await async_log_event(
        "INFO" if summary.ok else "ERROR",
        "tests",
        "Fin module de test",
        module=module_name,
        ok=summary.ok,
        returncode=summary.returncode,
        duration_s=round(summary.duration_s, 6),
        completed=summary.completed,
        timed_out=summary.timed_out,
    )
    return summary


def _module_summary_to_details(module_summary: TestModuleSummary) -> Dict[str, List[Dict[str, str]]]:
    failures: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    if module_summary.ok and module_summary.completed and not module_summary.timed_out:
        return {"failures": failures, "errors": errors}

    trace_blob = module_summary.stdout.strip()
    if module_summary.stderr.strip():
        trace_blob = f"{trace_blob}\n\nSTDERR:\n{module_summary.stderr.strip()}".strip()

    failure_count, error_count = _parse_unittest_output_for_failures(module_summary.stdout, module_summary.stderr)

    if module_summary.timed_out:
        errors.append({"test": f"{module_summary.module}::timeout", "trace": trace_blob})
        return {"failures": failures, "errors": errors}

    if module_summary.exception_message and failure_count <= 0 and error_count <= 0:
        errors.append({"test": f"{module_summary.module}::exception", "trace": trace_blob})
        return {"failures": failures, "errors": errors}

    if failure_count <= 0 and error_count <= 0 and not module_summary.ok:
        errors.append({"test": module_summary.module, "trace": trace_blob})
        return {"failures": failures, "errors": errors}

    for i in range(failure_count):
        failures.append({"test": f"{module_summary.module}::failure_{i+1}", "trace": trace_blob})
    for i in range(error_count):
        errors.append({"test": f"{module_summary.module}::error_{i+1}", "trace": trace_blob})

    return {"failures": failures, "errors": errors}


def _merge_parallel_module_summaries(module_summaries: Sequence[TestModuleSummary]) -> TestRunSummary:
    modules = [m.module for m in module_summaries]
    log_files = {m.module: m.log_file for m in module_summaries if m.log_file}

    all_failures: List[Dict[str, str]] = []
    all_errors: List[Dict[str, str]] = []

    failures = 0
    errors = 0
    timed_out_modules: List[str] = []

    for module_summary in module_summaries:
        f_count, e_count = _parse_unittest_output_for_failures(module_summary.stdout, module_summary.stderr)
        failures += f_count
        errors += e_count

        details = _module_summary_to_details(module_summary)
        all_failures.extend(details["failures"])
        all_errors.extend(details["errors"])

        if module_summary.timed_out:
            timed_out_modules.append(module_summary.module)

    total = _collect_parallel_test_counts(modules)
    duration_s = max((m.duration_s for m in module_summaries), default=0.0)
    completed = all(m.completed for m in module_summaries)
    timed_out = any(m.timed_out for m in module_summaries)
    ok = all(m.ok for m in module_summaries) and completed and not timed_out

    if not ok and failures == 0 and errors == 0:
        errors = sum(1 for m in module_summaries if not m.ok or not m.completed or m.timed_out)

    return TestRunSummary(
        ok=ok,
        total=int(total),
        failures=int(failures),
        errors=int(errors),
        skipped=0,
        expected_failures=0,
        unexpected_successes=0,
        duration_s=float(duration_s),
        modules=list(modules),
        log_files=log_files,
        details={"failures": all_failures, "errors": all_errors},
        per_module=list(module_summaries),
        parallel=True,
        completed=completed,
        timed_out=timed_out,
        timed_out_modules=timed_out_modules,
    )


# ============================================================
# TESTS UNITAIRES — ORCHESTRATION
# ============================================================

def run_preflight_tests(
    module_names: Sequence[str] = DEFAULT_TEST_MODULES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_s: float | None = DEFAULT_TEST_TIMEOUT_S,
) -> TestRunSummary:
    timeout_s = _normalize_timeout(timeout_s)
    module_names = _normalize_module_names(module_names)
    ensure_dir(Path(output_dir))
    log_event(
        "INFO",
        "tests",
        "Début préflight synchrone",
        modules=list(module_names),
        parallel=False,
        timeout_s=timeout_s,
    )

    t0 = time.perf_counter()
    importlib.invalidate_caches()

    module_summaries = [
        _run_test_module_subprocess(module_name, timeout_s=timeout_s)
        for module_name in module_names
    ]

    summary = _merge_parallel_module_summaries(module_summaries)
    summary.parallel = False
    summary.duration_s = float(time.perf_counter() - t0)

    write_test_summary_json(summary, output_dir=output_dir)
    log_event(
        "INFO" if summary.ok else "ERROR",
        "tests",
        "Fin préflight synchrone",
        ok=summary.ok,
        total=summary.total,
        failures=summary.failures,
        errors=summary.errors,
        duration_s=round(summary.duration_s, 6),
        parallel=False,
        completed=summary.completed,
        timed_out=summary.timed_out,
        timed_out_modules=summary.timed_out_modules,
    )
    return summary


async def async_run_preflight_tests(
    module_names: Sequence[str] = DEFAULT_TEST_MODULES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_s: float | None = DEFAULT_TEST_TIMEOUT_S,
) -> TestRunSummary:
    timeout_s = _normalize_timeout(timeout_s)
    ensure_dir(Path(output_dir))
    module_names = tuple(_normalize_module_names(module_names))

    await async_log_event(
        "INFO",
        "tests",
        "Début préflight asynchrone",
        modules=list(module_names),
        parallel=True,
        timeout_s=timeout_s,
    )
    t0 = time.perf_counter()

    module_summaries = await asyncio.gather(
        *[_async_run_test_module_subprocess(module_name, timeout_s=timeout_s) for module_name in module_names]
    )

    summary = _merge_parallel_module_summaries(module_summaries)
    summary.duration_s = float(time.perf_counter() - t0)

    await asyncio.to_thread(write_test_summary_json, summary, output_dir)
    await async_log_event(
        "INFO" if summary.ok else "ERROR",
        "tests",
        "Fin préflight asynchrone",
        ok=summary.ok,
        total=summary.total,
        failures=summary.failures,
        errors=summary.errors,
        duration_s=round(summary.duration_s, 6),
        parallel=True,
        completed=summary.completed,
        timed_out=summary.timed_out,
        timed_out_modules=summary.timed_out_modules,
    )
    return summary


def write_test_summary_json(summary: TestRunSummary, output_dir: Path, filename: str = DEFAULT_TEST_SUMMARY_FILE) -> Path:
    output_dir = ensure_dir(Path(output_dir))
    path = output_dir / filename
    path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ============================================================
# ORCHESTRATION GLOBALE
# ============================================================

def run_full_diagnostic(
    count: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_seed: int = camo.DEFAULT_BASE_SEED,
    run_tests_first: bool = True,
    test_modules: Sequence[str] = DEFAULT_TEST_MODULES,
    parallel_diagnostics: bool = False,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    tests_non_blocking: bool = False,
    test_timeout_s: float | None = DEFAULT_TEST_TIMEOUT_S,
) -> Dict[str, Any]:
    output_dir = ensure_dir(Path(output_dir))

    tests_summary: Optional[TestRunSummary] = None
    if run_tests_first:
        tests_summary = run_preflight_tests(
            module_names=test_modules,
            output_dir=output_dir,
            timeout_s=test_timeout_s,
        )

        if _should_block_on_tests(tests_summary):
            if tests_non_blocking:
                log_event(
                    "WARNING",
                    "tests",
                    "Préflight non bloquant : poursuite du diagnostic malgré échec/timeout/incomplétude",
                    ok=tests_summary.ok,
                    completed=tests_summary.completed,
                    timed_out=tests_summary.timed_out,
                    timed_out_modules=tests_summary.timed_out_modules,
                )
            else:
                raise RuntimeError(
                    "Préflight bloquant: tests en échec, incomplets ou timeout. "
                    "Active tests_non_blocking=True pour continuer malgré cela."
                )

    diagnostics = generate_diagnostics(
        count=count,
        base_seed=base_seed,
        parallel=parallel_diagnostics,
        max_workers=max_concurrency,
    )
    summary = build_summary(diagnostics)

    csv_path = write_candidates_csv(diagnostics, output_dir)
    json_path = write_summary_json(summary, output_dir)
    txt_path = write_summary_txt(summary, output_dir)
    runtime_snapshot_path = export_runtime_snapshot(output_dir)

    return {
        "tests_summary": None if tests_summary is None else tests_summary.to_dict(),
        "summary": summary,
        "paths": {
            "csv": str(csv_path.resolve()),
            "json": str(json_path.resolve()),
            "txt": str(txt_path.resolve()),
            "runtime_log": str(LOG_MANAGER.runtime_path.resolve()),
            "runtime_snapshot": str(runtime_snapshot_path.resolve()),
        },
    }


async def async_run_full_diagnostic(
    count: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_seed: int = camo.DEFAULT_BASE_SEED,
    run_tests_first: bool = True,
    test_modules: Sequence[str] = DEFAULT_TEST_MODULES,
    parallel_tests: bool = True,
    parallel_diagnostics: bool = True,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    tests_non_blocking: bool = False,
    test_timeout_s: float | None = DEFAULT_TEST_TIMEOUT_S,
) -> Dict[str, Any]:
    output_dir = ensure_dir(Path(output_dir))

    tests_summary: Optional[TestRunSummary] = None
    if run_tests_first:
        if parallel_tests:
            tests_summary = await async_run_preflight_tests(
                module_names=test_modules,
                output_dir=output_dir,
                timeout_s=test_timeout_s,
            )
        else:
            tests_summary = await asyncio.to_thread(
                run_preflight_tests,
                tuple(test_modules),
                output_dir,
                test_timeout_s,
            )

        if _should_block_on_tests(tests_summary):
            if tests_non_blocking:
                await async_log_event(
                    "WARNING",
                    "tests",
                    "Préflight non bloquant : poursuite du diagnostic malgré échec/timeout/incomplétude",
                    ok=tests_summary.ok,
                    completed=tests_summary.completed,
                    timed_out=tests_summary.timed_out,
                    timed_out_modules=tests_summary.timed_out_modules,
                )
            else:
                raise RuntimeError(
                    "Préflight bloquant: tests en échec, incomplets ou timeout. "
                    "Active tests_non_blocking=True pour continuer malgré cela."
                )

    diagnostics = await async_generate_diagnostics(
        count=count,
        base_seed=base_seed,
        max_concurrency=max_concurrency,
        parallel=parallel_diagnostics,
    )
    summary = build_summary(diagnostics)

    csv_path = await async_write_candidates_csv(diagnostics, output_dir)
    json_path = await async_write_summary_json(summary, output_dir)
    txt_path = await async_write_summary_txt(summary, output_dir)
    runtime_snapshot_path = await asyncio.to_thread(export_runtime_snapshot, output_dir)

    return {
        "tests_summary": None if tests_summary is None else tests_summary.to_dict(),
        "summary": summary,
        "paths": {
            "csv": str(csv_path.resolve()),
            "json": str(json_path.resolve()),
            "txt": str(txt_path.resolve()),
            "runtime_log": str(LOG_MANAGER.runtime_path.resolve()),
            "runtime_snapshot": str(runtime_snapshot_path.resolve()),
        },
    }


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic des rejets, logs continus et orchestration de tests pour la génération camouflage."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_ANALYSIS_COUNT, help="Nombre de candidats à analyser.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Dossier de sortie des logs.")
    parser.add_argument("--base-seed", type=int, default=camo.DEFAULT_BASE_SEED, help="Seed de base.")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Utiliser la version asynchrone.")
    parser.add_argument("--skip-tests", action="store_true", help="Ne pas lancer les tests avant le diagnostic.")
    parser.add_argument("--tests-only", action="store_true", help="Lancer uniquement les tests unitaires.")
    parser.add_argument(
        "--test-modules",
        nargs="*",
        default=list(DEFAULT_TEST_MODULES),
        help="Liste des modules de tests à lancer, ex: test_main test_start",
    )
    parser.add_argument(
        "--parallel-tests",
        action="store_true",
        help="En mode async, lancer les modules de tests en parallèle dans des sous-processus.",
    )
    parser.add_argument(
        "--parallel-diagnostics",
        action="store_true",
        help="Paralléliser la génération/analyse des candidats.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Nombre maximal de workers/concurrence pour les diagnostics async/sync parallèles.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Mode rapide : count=10, test_modules=test_main, parallel_tests et parallel_diagnostics activés.",
    )
    parser.add_argument(
        "--tests-non-blocking",
        action="store_true",
        help="Continue le diagnostic même si les tests échouent, crashent ou timeout.",
    )
    parser.add_argument(
        "--test-timeout",
        type=float,
        default=DEFAULT_TEST_TIMEOUT_S,
        help="Timeout par module de tests en secondes. <= 0 désactive le timeout.",
    )
    parser.add_argument(
        "--live-console",
        dest="live_console",
        action="store_true",
        default=DEFAULT_LIVE_CONSOLE,
        help="Affiche les événements runtime en direct dans la console.",
    )
    parser.add_argument(
        "--no-live-console",
        dest="live_console",
        action="store_false",
        help="Désactive l'affichage runtime en direct dans la console.",
    )
    parser.add_argument(
        "--console-level",
        type=str,
        default=DEFAULT_CONSOLE_LEVEL,
        help="Niveau minimum affiché en direct dans la console (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--tail-runtime",
        type=int,
        default=0,
        help="Affiche les N dernières lignes du runtime log puis quitte.",
    )
    return parser.parse_args(argv)


def _print_test_summary(summary: TestRunSummary) -> None:
    print("\nPréflight tests terminé.")
    print(f"Modules            : {', '.join(summary.modules)}")
    print(f"Mode parallèle     : {'oui' if summary.parallel else 'non'}")
    print(f"Résultat           : {'OK' if summary.ok else 'KO'}")
    print(f"Terminé            : {'oui' if summary.completed else 'non'}")
    print(f"Timeout détecté    : {'oui' if summary.timed_out else 'non'}")
    if summary.timed_out_modules:
        print(f"Modules timeout    : {', '.join(summary.timed_out_modules)}")
    print(f"Tests exécutés     : {summary.total}")
    print(f"Échecs             : {summary.failures}")
    print(f"Erreurs            : {summary.errors}")
    print(f"Skips              : {summary.skipped}")
    print(f"Durée              : {summary.duration_s:.2f} s")
    if summary.log_files:
        print("Logs modules       :")
        for module_name, path in summary.log_files.items():
            print(f"  - {module_name}: {path}")
    if summary.per_module:
        print("Détail modules     :")
        for item in summary.per_module:
            print(
                f"  - {item.module}: "
                f"{'OK' if item.ok else 'KO'} | "
                f"rc={item.returncode} | "
                f"{item.duration_s:.2f} s | "
                f"completed={'oui' if item.completed else 'non'} | "
                f"timeout={'oui' if item.timed_out else 'non'}"
            )


def _apply_fast_mode(args: argparse.Namespace) -> None:
    if not args.fast:
        return
    args.count = 10
    args.test_modules = ["test_main"]
    args.parallel_tests = True
    args.parallel_diagnostics = True
    args.tests_non_blocking = True
    if int(args.max_concurrency) < 2:
        args.max_concurrency = 2


def _handle_tail_runtime(args: argparse.Namespace) -> bool:
    if int(args.tail_runtime) <= 0:
        return False
    for line in get_recent_runtime_lines(int(args.tail_runtime)):
        print(line)
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _apply_fast_mode(args)

    if _handle_tail_runtime(args):
        return

    subscriber = None
    if args.live_console:
        subscriber = attach_live_console_printer(args.console_level)

    try:
        output_dir = ensure_dir(Path(args.output))
        test_modules = _normalize_module_names(args.test_modules)
        timeout_s = _normalize_timeout(args.test_timeout)

        if args.tests_only:
            summary = run_preflight_tests(
                module_names=test_modules,
                output_dir=output_dir,
                timeout_s=timeout_s,
            )
            _print_test_summary(summary)
            return

        t0 = time.perf_counter()
        result = run_full_diagnostic(
            count=int(args.count),
            output_dir=output_dir,
            base_seed=int(args.base_seed),
            run_tests_first=not bool(args.skip_tests),
            test_modules=test_modules,
            parallel_diagnostics=bool(args.parallel_diagnostics),
            max_concurrency=int(args.max_concurrency),
            tests_non_blocking=bool(args.tests_non_blocking),
            test_timeout_s=timeout_s,
        )
        dt = time.perf_counter() - t0

        tests_summary = result.get("tests_summary")
        if tests_summary is not None:
            _print_test_summary(_coerce_test_run_summary(tests_summary))

        summary = result["summary"]
        paths = result["paths"]

        print("\nDiagnostic terminé.")
        print(f"Candidats analysés : {summary['total_candidates']}")
        print(f"Acceptés           : {summary['accepted']}")
        print(f"Rejetés            : {summary['rejected']}")
        print(f"Taux acceptation   : {summary['acceptance_rate']:.4%}")
        print(f"CSV                : {paths['csv']}")
        print(f"JSON               : {paths['json']}")
        print(f"TXT                : {paths['txt']}")
        print(f"Runtime log        : {paths['runtime_log']}")
        print(f"Snapshot runtime   : {paths['runtime_snapshot']}")
        print(f"Durée totale       : {dt:.2f} s")
    finally:
        if subscriber is not None:
            LOG_MANAGER.unsubscribe(subscriber)


async def async_main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _apply_fast_mode(args)

    if _handle_tail_runtime(args):
        return

    subscriber = None
    if args.live_console:
        subscriber = attach_live_console_printer(args.console_level)

    try:
        output_dir = ensure_dir(Path(args.output))
        test_modules = _normalize_module_names(args.test_modules)
        timeout_s = _normalize_timeout(args.test_timeout)

        if args.tests_only:
            if args.parallel_tests:
                summary = await async_run_preflight_tests(
                    module_names=test_modules,
                    output_dir=output_dir,
                    timeout_s=timeout_s,
                )
            else:
                summary = await asyncio.to_thread(
                    run_preflight_tests,
                    tuple(test_modules),
                    output_dir,
                    timeout_s,
                )
            _print_test_summary(summary)
            return

        t0 = time.perf_counter()
        result = await async_run_full_diagnostic(
            count=int(args.count),
            output_dir=output_dir,
            base_seed=int(args.base_seed),
            run_tests_first=not bool(args.skip_tests),
            test_modules=test_modules,
            parallel_tests=bool(args.parallel_tests),
            parallel_diagnostics=bool(args.parallel_diagnostics),
            max_concurrency=int(args.max_concurrency),
            tests_non_blocking=bool(args.tests_non_blocking),
            test_timeout_s=timeout_s,
        )
        dt = time.perf_counter() - t0

        tests_summary = result.get("tests_summary")
        if tests_summary is not None:
            _print_test_summary(_coerce_test_run_summary(tests_summary))

        summary = result["summary"]
        paths = result["paths"]

        print("\nDiagnostic terminé.")
        print(f"Candidats analysés : {summary['total_candidates']}")
        print(f"Acceptés           : {summary['accepted']}")
        print(f"Rejetés            : {summary['rejected']}")
        print(f"Taux acceptation   : {summary['acceptance_rate']:.4%}")
        print(f"CSV                : {paths['csv']}")
        print(f"JSON               : {paths['json']}")
        print(f"TXT                : {paths['txt']}")
        print(f"Runtime log        : {paths['runtime_log']}")
        print(f"Snapshot runtime   : {paths['runtime_snapshot']}")
        print(f"Durée totale       : {dt:.2f} s")
    finally:
        if subscriber is not None:
            LOG_MANAGER.unsubscribe(subscriber)


if __name__ == "__main__":
    parsed = parse_args()
    _apply_fast_mode(parsed)

    argv = sys.argv[1:]
    if parsed.use_async:
        asyncio.run(async_main(argv))
    else:
        main(argv)
