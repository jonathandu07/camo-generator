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
- exporter des diagnostics exploitables pour main.py et start.py.

Sorties principales :
- logs_generation/diagnostic_candidates.csv
- logs_generation/diagnostic_summary.json
- logs_generation/diagnostic_summary.txt
- logs_generation/runtime.log
- logs_generation/tests_summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import logging
import statistics
import threading
import time
import unittest
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence

import numpy as np

import main as camo


# ============================================================
# CONFIG
# ============================================================

DEFAULT_ANALYSIS_COUNT = 300
DEFAULT_OUTPUT_DIR = Path("logs_generation")
DEFAULT_TEST_MODULES: tuple[str, ...] = ("test_main", "test_start")
DEFAULT_RUNTIME_LOG_FILE = "runtime.log"
DEFAULT_TEST_SUMMARY_FILE = "tests_summary.json"
DEFAULT_HISTORY_LIMIT = 5000

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
            return f"{stamp} | {self.level:<8} | {self.source} | {self.message} | {json.dumps(self.payload, ensure_ascii=False, sort_keys=True)}"
        return f"{stamp} | {self.level:<8} | {self.source} | {self.message}"


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

    def short_text(self) -> str:
        if self.ok:
            return f"{self.total} tests OK"
        return (
            f"{self.total} tests exécutés | "
            f"{self.failures} échec(s) | {self.errors} erreur(s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
                log_message = f"{log_message} | payload={json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
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


def generate_diagnostics(
    count: int,
    base_seed: int = camo.DEFAULT_BASE_SEED,
) -> List[CandidateDiagnostic]:
    diagnostics: List[CandidateDiagnostic] = []

    log_event("INFO", "generate_diagnostics", "Début diagnostic synchrone", count=int(count), base_seed=int(base_seed))
    for i in range(1, count + 1):
        seed = camo.build_seed(target_index=1, local_attempt=i, base_seed=base_seed)
        candidate = camo.generate_candidate_from_seed(seed)
        diagnostic = analyze_candidate(candidate, target_index=1, local_attempt=i)
        diagnostics.append(diagnostic)

    log_event("INFO", "generate_diagnostics", "Fin diagnostic synchrone", count=len(diagnostics))
    return diagnostics


async def async_generate_diagnostics(
    count: int,
    base_seed: int = camo.DEFAULT_BASE_SEED,
) -> List[CandidateDiagnostic]:
    diagnostics: List[CandidateDiagnostic] = []

    await async_log_event("INFO", "async_generate_diagnostics", "Début diagnostic asynchrone", count=int(count), base_seed=int(base_seed))
    for i in range(1, count + 1):
        seed = camo.build_seed(target_index=1, local_attempt=i, base_seed=base_seed)
        candidate = await camo.async_generate_candidate_from_seed(seed)
        diagnostic = await async_analyze_candidate(candidate, target_index=1, local_attempt=i)
        diagnostics.append(diagnostic)

    await async_log_event("INFO", "async_generate_diagnostics", "Fin diagnostic asynchrone", count=len(diagnostics))
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
# TESTS UNITAIRES — ORCHESTRATION
# ============================================================


def _extract_test_case_tuples(items: Sequence[tuple[Any, str]]) -> List[Dict[str, str]]:
    extracted: List[Dict[str, str]] = []
    for case, trace in items:
        extracted.append(
            {
                "test": str(case),
                "trace": str(trace),
            }
        )
    return extracted


def discover_test_log_files(module_names: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            log_file = getattr(module, "LOG_FILE", None)
            if log_file is not None:
                out[module_name] = str(Path(log_file).resolve())
        except Exception:
            continue
    return out


def run_preflight_tests(
    module_names: Sequence[str] = DEFAULT_TEST_MODULES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> TestRunSummary:
    ensure_dir(Path(output_dir))
    log_event("INFO", "tests", "Début préflight", modules=list(module_names))

    t0 = time.perf_counter()
    importlib.invalidate_caches()

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    loaded_modules: List[str] = []

    for module_name in module_names:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        suite.addTests(loader.loadTestsFromModule(module))
        loaded_modules.append(module_name)

    result = unittest.TestResult()
    suite.run(result)
    dt = time.perf_counter() - t0

    summary = TestRunSummary(
        ok=result.wasSuccessful(),
        total=int(result.testsRun),
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(getattr(result, "skipped", [])),
        expected_failures=len(getattr(result, "expectedFailures", [])),
        unexpected_successes=len(getattr(result, "unexpectedSuccesses", [])),
        duration_s=float(dt),
        modules=list(loaded_modules),
        log_files=discover_test_log_files(loaded_modules),
        details={
            "failures": _extract_test_case_tuples(result.failures),
            "errors": _extract_test_case_tuples(result.errors),
        },
    )

    write_test_summary_json(summary, output_dir=output_dir)
    log_event(
        "INFO" if summary.ok else "ERROR",
        "tests",
        "Fin préflight",
        ok=summary.ok,
        total=summary.total,
        failures=summary.failures,
        errors=summary.errors,
        duration_s=round(summary.duration_s, 6),
    )
    return summary


async def async_run_preflight_tests(
    module_names: Sequence[str] = DEFAULT_TEST_MODULES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> TestRunSummary:
    return await asyncio.to_thread(run_preflight_tests, tuple(module_names), output_dir)


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
) -> Dict[str, Any]:
    output_dir = ensure_dir(Path(output_dir))

    tests_summary: Optional[TestRunSummary] = None
    if run_tests_first:
        tests_summary = run_preflight_tests(output_dir=output_dir)

    diagnostics = generate_diagnostics(count=count, base_seed=base_seed)
    summary = build_summary(diagnostics)

    csv_path = write_candidates_csv(diagnostics, output_dir)
    json_path = write_summary_json(summary, output_dir)
    txt_path = write_summary_txt(summary, output_dir)
    runtime_snapshot_path = export_runtime_snapshot(output_dir)

    result = {
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
    return result


async def async_run_full_diagnostic(
    count: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_seed: int = camo.DEFAULT_BASE_SEED,
    run_tests_first: bool = True,
) -> Dict[str, Any]:
    output_dir = ensure_dir(Path(output_dir))

    tests_summary: Optional[TestRunSummary] = None
    if run_tests_first:
        tests_summary = await async_run_preflight_tests(output_dir=output_dir)

    diagnostics = await async_generate_diagnostics(count=count, base_seed=base_seed)
    summary = build_summary(diagnostics)

    csv_path = await async_write_candidates_csv(diagnostics, output_dir)
    json_path = await async_write_summary_json(summary, output_dir)
    txt_path = await async_write_summary_txt(summary, output_dir)
    runtime_snapshot_path = await asyncio.to_thread(export_runtime_snapshot, output_dir)

    result = {
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
    return result


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic des rejets, logs continus et orchestration de tests pour la génération camouflage."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_ANALYSIS_COUNT, help="Nombre de candidats à analyser.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Dossier de sortie des logs.")
    parser.add_argument("--base-seed", type=int, default=camo.DEFAULT_BASE_SEED, help="Seed de base.")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Utiliser la version asynchrone.")
    parser.add_argument("--skip-tests", action="store_true", help="Ne pas lancer test_main.py et test_start.py avant le diagnostic.")
    parser.add_argument("--tests-only", action="store_true", help="Lancer uniquement les tests unitaires.")
    return parser.parse_args()


def _print_test_summary(summary: TestRunSummary) -> None:
    print("\nPréflight tests terminé.")
    print(f"Modules            : {', '.join(summary.modules)}")
    print(f"Résultat           : {'OK' if summary.ok else 'KO'}")
    print(f"Tests exécutés     : {summary.total}")
    print(f"Échecs             : {summary.failures}")
    print(f"Erreurs            : {summary.errors}")
    print(f"Skips              : {summary.skipped}")
    print(f"Durée              : {summary.duration_s:.2f} s")
    if summary.log_files:
        print("Logs modules       :")
        for module_name, path in summary.log_files.items():
            print(f"  - {module_name}: {path}")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output))

    if args.tests_only:
        summary = run_preflight_tests(output_dir=output_dir)
        _print_test_summary(summary)
        return

    t0 = time.perf_counter()
    result = run_full_diagnostic(
        count=int(args.count),
        output_dir=output_dir,
        base_seed=int(args.base_seed),
        run_tests_first=not bool(args.skip_tests),
    )
    dt = time.perf_counter() - t0

    tests_summary = result.get("tests_summary")
    if tests_summary is not None:
        _print_test_summary(TestRunSummary(**tests_summary))

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


async def async_main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output))

    if args.tests_only:
        summary = await async_run_preflight_tests(output_dir=output_dir)
        _print_test_summary(summary)
        return

    t0 = time.perf_counter()
    result = await async_run_full_diagnostic(
        count=int(args.count),
        output_dir=output_dir,
        base_seed=int(args.base_seed),
        run_tests_first=not bool(args.skip_tests),
    )
    dt = time.perf_counter() - t0

    tests_summary = result.get("tests_summary")
    if tests_summary is not None:
        _print_test_summary(TestRunSummary(**tests_summary))

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


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.use_async:
        asyncio.run(async_main())
    else:
        main()
