# -*- coding: utf-8 -*-
"""
log.py
Supervision continue, analyse des rejets et préflight pour le générateur
de camouflage organique 8K.

Rôles :
- centraliser les logs runtime continus ;
- analyser précisément pourquoi un candidat est rejeté ;
- exécuter un préflight fort avant lancement ;
- fournir des corrections en direct à main.py ;
- exporter des snapshots exploitables par le front.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import subprocess
import sys
import shutil
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import numpy as np

import main as camo


# ============================================================
# CONFIG
# ============================================================

DEFAULT_OUTPUT_DIR = Path(os.getenv("LOG_OUTPUT_DIR", "logs")).resolve()
DEFAULT_RUNTIME_LOG_FILE = "runtime.log"
DEFAULT_RUNTIME_SNAPSHOT_FILE = "runtime_snapshot.json"
DEFAULT_DIAG_CSV = "diagnostic_candidates.csv"
DEFAULT_DIAG_JSON = "diagnostic_summary.json"
DEFAULT_TEST_SUMMARY_FILE = "tests_summary.json"
DEFAULT_COMPILED_LOGS_FILE = "compiled_logs.txt"
DEFAULT_COMPILED_LOGS_JSON = "compiled_logs.json"
DEFAULT_HISTORY_LIMIT = 5000
DEFAULT_ANALYSIS_COUNT = 64
DEFAULT_TEST_MODULES: tuple[str, ...] = ("test_main", "test_start", "test_camouflage_ml_dl_precise")
DEFAULT_TEST_TIMEOUT_S: float | None = None

LOGGER_NAME = "camo_supervisor"
ensure_output_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True) or Path(p)


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

    def to_csv_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "seed": self.seed,
            "target_index": self.target_index,
            "local_attempt": self.local_attempt,
            "accepted": int(self.accepted),
            "fail_count": len(self.failures),
            "fail_rules": " | ".join(f.rule for f in self.failures),
        }
        row.update({k: round(float(v), 8) for k, v in self.ratios.items()})
        row.update({k: round(float(v), 8) for k, v in self.metrics.items()})
        return row


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
    stdout: str
    stderr: str
    timed_out: bool = False
    timeout_s: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SupervisorDecision:
    max_workers: int | None = None
    attempt_batch_size: int | None = None
    parallel_attempts: bool | None = None
    machine_intensity: float | None = None
    pause_s: float | None = None
    reason: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if v is not None}
        return out


# ============================================================
# OUTILS
# ============================================================

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


def _normalize_timeout(timeout_s: float | None) -> float | None:
    if timeout_s is None:
        return None
    try:
        value = float(timeout_s)
    except Exception:
        return None
    return None if value <= 0 else value


def _subprocess_env(output_dir: Path | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["LOG_OUTPUT_DIR"] = str(Path(output_dir or DEFAULT_OUTPUT_DIR).resolve())
    env.setdefault("MUT_MODULE", "main")
    env.setdefault("CAMO_MLDL_MODULE", "camouflage_ml_dl")
    return env


def _module_specific_env(module_name: str, output_dir: Path | None = None) -> Dict[str, str]:
    env = _subprocess_env(output_dir)
    name = str(module_name).strip().lower()
    if name.startswith("test_main"):
        env["MUT_MODULE"] = "main"
    if "camouflage_ml_dl" in name:
        env["CAMO_MLDL_MODULE"] = "camouflage_ml_dl"
    return env


# ============================================================
# LOGGER CENTRAL CONTINU
# ============================================================

class ContinuousLogManager:
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        runtime_filename: str = DEFAULT_RUNTIME_LOG_FILE,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_path = self.output_dir / runtime_filename
        self.snapshot_path = self.output_dir / DEFAULT_RUNTIME_SNAPSHOT_FILE
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
            fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
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
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

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
            msg = event.message
            if payload:
                msg = f"{msg} | payload={json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"
            self.logger.log(getattr(logging, event.level, logging.INFO), f"[{event.source}] {msg}")
            subscribers = list(self._subscribers)
            self._write_snapshot_unlocked()

        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                pass
        return event

    def _write_snapshot_unlocked(self) -> None:
        payload = [evt.to_dict() for evt in list(self.history)[-200:]]
        self.snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def last_lines(self, n: int = 50) -> List[str]:
        with self._lock:
            return [evt.format_line() for evt in list(self.history)[-max(1, int(n)):]]

    def snapshot(self, n: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            return [evt.to_dict() for evt in list(self.history)[-max(1, int(n)):]]


LOG_MANAGER = ContinuousLogManager()


def log_event(level: str, source: str, message: str, **payload: Any) -> RuntimeEvent:
    return LOG_MANAGER.emit(level, source, message, **payload)


async def async_log_event(level: str, source: str, message: str, **payload: Any) -> RuntimeEvent:
    return await asyncio.to_thread(log_event, level, source, message, **payload)


def get_recent_runtime_lines(n: int = 50) -> List[str]:
    return LOG_MANAGER.last_lines(n)


def get_runtime_snapshot(n: int = 200) -> List[Dict[str, Any]]:
    return LOG_MANAGER.snapshot(n)


# ============================================================
# ANALYSE D'UN CANDIDAT
# ============================================================

def _fail_min(name: str, actual: float, min_value: float) -> RuleFailure | None:
    if actual >= min_value:
        return None
    return RuleFailure(name, actual, None, min_value, None, min_value - actual)


def _fail_max(name: str, actual: float, max_value: float) -> RuleFailure | None:
    if actual <= max_value:
        return None
    return RuleFailure(name, actual, None, None, max_value, actual - max_value)


def _fail_range(name: str, actual: float, min_value: float, max_value: float) -> RuleFailure | None:
    if actual < min_value:
        return RuleFailure(name, actual, None, min_value, max_value, min_value - actual)
    if actual > max_value:
        return RuleFailure(name, actual, None, min_value, max_value, actual - max_value)
    return None


def _fail_target_abs(name: str, actual: float, target: float, max_abs_error: float) -> RuleFailure | None:
    delta = abs(actual - target)
    if delta <= max_abs_error:
        return None
    return RuleFailure(name, actual, target, None, max_abs_error, delta - max_abs_error)


def analyze_candidate(candidate: camo.CandidateResult, target_index: int, local_attempt: int) -> CandidateDiagnostic:
    rs = np.asarray(candidate.ratios, dtype=float)
    m = dict(candidate.metrics)
    failures: List[RuleFailure] = []

    abs_err = np.abs(rs - camo.TARGET)
    mean_abs_err = float(np.mean(abs_err))

    per_color_rules = [
        ("abs_err_coyote", float(rs[camo.IDX_COYOTE]), float(camo.TARGET[camo.IDX_COYOTE]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_COYOTE])),
        ("abs_err_olive", float(rs[camo.IDX_OLIVE]), float(camo.TARGET[camo.IDX_OLIVE]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_OLIVE])),
        ("abs_err_terre", float(rs[camo.IDX_TERRE]), float(camo.TARGET[camo.IDX_TERRE]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_TERRE])),
        ("abs_err_gris", float(rs[camo.IDX_GRIS]), float(camo.TARGET[camo.IDX_GRIS]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_GRIS])),
    ]
    for name, actual, target, max_abs in per_color_rules:
        fail = _fail_target_abs(name, actual, target, max_abs)
        if fail is not None:
            failures.append(fail)

    fail = _fail_max("mean_abs_error", mean_abs_err, float(camo.MAX_MEAN_ABS_ERROR))
    if fail is not None:
        failures.append(fail)

    range_checks = [
        ("boundary_density", _metric(m, "boundary_density"), float(camo.MIN_BOUNDARY_DENSITY), float(camo.MAX_BOUNDARY_DENSITY)),
        ("boundary_density_small", _metric(m, "boundary_density_small"), float(camo.MIN_BOUNDARY_DENSITY_SMALL), float(camo.MAX_BOUNDARY_DENSITY_SMALL)),
        ("boundary_density_tiny", _metric(m, "boundary_density_tiny"), float(camo.MIN_BOUNDARY_DENSITY_TINY), float(camo.MAX_BOUNDARY_DENSITY_TINY)),
    ]
    for name, actual, min_v, max_v in range_checks:
        fail = _fail_range(name, actual, min_v, max_v)
        if fail is not None:
            failures.append(fail)

    fail = _fail_max("mirror_similarity", _metric(m, "mirror_similarity"), float(camo.MAX_MIRROR_SIMILARITY))
    if fail is not None:
        failures.append(fail)

    largest_metric_name = "largest_component_ratio_class_1" if "largest_component_ratio_class_1" in m else "largest_olive_component_ratio"
    largest_metric_min = float(getattr(camo, "MIN_LARGEST_COMPONENT_RATIO_CLASS_1", getattr(camo, "MIN_LARGEST_OLIVE_COMPONENT_RATIO", 0.0)))
    fail = _fail_min(largest_metric_name, _metric(m, largest_metric_name), largest_metric_min)
    if fail is not None:
        failures.append(fail)

    fail = _fail_max("edge_contact_ratio", _metric(m, "edge_contact_ratio"), float(camo.MAX_EDGE_CONTACT_RATIO))
    if fail is not None:
        failures.append(fail)

    motif_scale = _metric(m, "motif_scale", float(getattr(camo, "MOTIF_SCALE", getattr(camo, "DEFAULT_MOTIF_SCALE", 1.0))))
    if motif_scale <= 0.0:
        failures.append(RuleFailure("motif_scale", motif_scale, None, 0.0, None, abs(motif_scale)))

    return CandidateDiagnostic(
        seed=int(candidate.seed),
        target_index=int(target_index),
        local_attempt=int(local_attempt),
        accepted=not failures,
        ratios={(getattr(camo, "CLASS_NAMES", getattr(camo, "COLOR_NAMES", [f"class_{i}" for i in range(4)]))[i]): float(rs[i]) for i in range(4)},
        metrics={k: _safe_float(v) for k, v in m.items()},
        failures=failures,
    )


def deep_rejection_analysis(candidate: camo.CandidateResult, target_index: int = 0, local_attempt: int = 0) -> CandidateDiagnostic:
    return analyze_candidate(candidate, target_index=target_index, local_attempt=local_attempt)


def export_candidate_diagnostics(diags: Sequence[CandidateDiagnostic], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / DEFAULT_DIAG_CSV
    json_path = output_dir / DEFAULT_DIAG_JSON

    rows = [d.to_csv_row() for d in diags]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    summary = {
        "count": len(diags),
        "accepted": int(sum(1 for d in diags if d.accepted)),
        "rejected": int(sum(1 for d in diags if not d.accepted)),
        "top_rules": Counter(rule.rule for d in diags for rule in d.failures).most_common(15),
        "items": [d.to_dict() for d in diags[-DEFAULT_ANALYSIS_COUNT:]],
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


# ============================================================
# SUPERVISEUR CONTINU
# ============================================================

class RuntimeSupervisor:
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.attempt_history: Deque[Dict[str, Any]] = deque(maxlen=256)
        self.resource_history: Deque[Dict[str, Any]] = deque(maxlen=64)
        self.last_decision: Dict[str, Any] = {}
        self.last_tuning: Dict[str, Any] = {}
        self.last_snapshot_path = self.output_dir / DEFAULT_RUNTIME_SNAPSHOT_FILE
        self.tests_summary_path = self.output_dir / DEFAULT_TEST_SUMMARY_FILE

    def _record_attempt(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.attempt_history.append(dict(payload))

    def _record_resource(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.resource_history.append(dict(payload))

    def _recent_attempts(self, n: int = 32) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.attempt_history)[-max(1, int(n)):]

    def _latest_resource(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.resource_history[-1]) if self.resource_history else {}

    def _write_runtime_controller_snapshot(self) -> None:
        with self._lock:
            payload = {
                "attempt_history_tail": list(self.attempt_history)[-64:],
                "resource_history_tail": list(self.resource_history)[-16:],
                "last_decision": dict(self.last_decision),
                "last_tuning": dict(self.last_tuning),
                "runtime_log_tail": LOG_MANAGER.snapshot(80),
            }
        self.last_snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _decide_from_state(self, tuning: Dict[str, Any]) -> Dict[str, Any]:
        recent = self._recent_attempts(32)
        latest_resource = self._latest_resource()
        if not recent:
            return {}

        accepted_ratio = sum(1 for x in recent if bool(x.get("accepted"))) / max(1, len(recent))
        reject_ratio = 1.0 - accepted_ratio
        avg_duration = sum(_safe_float(x.get("duration_s"), 0.0) for x in recent) / max(1, len(recent))

        max_workers = int(tuning.get("max_workers", camo.DEFAULT_MAX_WORKERS))
        attempt_batch_size = int(tuning.get("attempt_batch_size", max_workers))
        parallel_attempts = bool(tuning.get("parallel_attempts", True))
        machine_intensity = float(tuning.get("machine_intensity", camo.DEFAULT_MACHINE_INTENSITY))

        available_mb = _safe_float(latest_resource.get("system_available_mb"), 0.0)
        process_rss_mb = _safe_float(latest_resource.get("process_rss_mb"), 0.0)
        disk_free_mb = _safe_float(latest_resource.get("disk_free_mb"), 0.0)

        decision = SupervisorDecision(reason="stable")

        if available_mb > 0.0 and available_mb < 1536:
            decision.max_workers = max(1, min(max_workers, max_workers // 2 or 1))
            decision.attempt_batch_size = max(1, min(attempt_batch_size, decision.max_workers))
            decision.parallel_attempts = decision.max_workers > 1
            decision.reason = "memory_pressure"
        elif process_rss_mb > 4096:
            decision.max_workers = max(1, min(max_workers, max_workers - 1))
            decision.attempt_batch_size = max(1, min(attempt_batch_size, max_workers))
            decision.parallel_attempts = decision.max_workers > 1
            decision.reason = "process_rss_high"
        elif reject_ratio > 0.88 and len(recent) >= 12:
            decision.attempt_batch_size = max(1, int(round(attempt_batch_size * 0.70)))
            decision.max_workers = max(1, min(max_workers, decision.attempt_batch_size))
            decision.parallel_attempts = decision.max_workers > 1
            decision.reason = "rejection_storm"
        elif accepted_ratio > 0.22 and avg_duration < 6.0 and available_mb > 4096:
            decision.max_workers = min(camo.CPU_COUNT, max_workers + 1)
            decision.attempt_batch_size = min(camo.CPU_COUNT * 2, max(attempt_batch_size, decision.max_workers))
            decision.parallel_attempts = decision.max_workers > 1
            decision.reason = "safe_ramp_up"
        elif disk_free_mb and disk_free_mb < 512:
            decision.parallel_attempts = False
            decision.max_workers = 1
            decision.attempt_batch_size = 1
            decision.reason = "disk_pressure"

        if decision.reason == "stable":
            return {}

        if decision.machine_intensity is None:
            decision.machine_intensity = machine_intensity
        return decision.to_dict()

    def feedback(self, event_type: str, **payload: Any) -> Dict[str, Any]:
        tuning = dict(payload.get("tuning") or {})
        if tuning:
            with self._lock:
                self.last_tuning = dict(tuning)

        if event_type == "resource_snapshot":
            snap = dict(payload.get("snapshot") or {})
            if snap:
                self._record_resource(snap)

        if event_type == "attempt_finished":
            self._record_attempt({
                "accepted": bool(payload.get("accepted")),
                "duration_s": _safe_float(payload.get("duration_s"), 0.0),
                "seed": payload.get("seed"),
                "target_index": payload.get("target_index"),
                "local_attempt": payload.get("local_attempt"),
            })

        decision = self._decide_from_state(tuning=tuning) if tuning else {}

        if event_type == "attempt_finished":
            accepted = bool(payload.get("accepted"))
            metrics = dict(payload.get("metrics") or {})
            ratios = dict(payload.get("ratios") or {})
            log_event(
                "INFO" if accepted else "WARNING",
                "supervisor",
                "Tentative terminée",
                target_index=payload.get("target_index"),
                local_attempt=payload.get("local_attempt"),
                accepted=accepted,
                duration_s=payload.get("duration_s"),
                tuning=tuning,
                decision=decision,
                ratios=ratios,
                metrics_subset={
                    "boundary_density": metrics.get("boundary_density"),
                    "boundary_density_small": metrics.get("boundary_density_small"),
                    "mirror_similarity": metrics.get("mirror_similarity"),
                    "edge_contact_ratio": metrics.get("edge_contact_ratio"),
                    "motif_scale": metrics.get("motif_scale"),
                },
            )
        elif event_type in {"generation_started", "generation_finished", "candidate_accepted", "resource_snapshot", "batch_finished"}:
            log_event("INFO", "supervisor", f"Événement {event_type}", payload=payload, decision=decision)

        with self._lock:
            self.last_decision = dict(decision)

        self._write_runtime_controller_snapshot()
        return decision


SUPERVISOR = RuntimeSupervisor(DEFAULT_OUTPUT_DIR)


def feedback_runtime_event(event_type: str, **payload: Any) -> Dict[str, Any]:
    return SUPERVISOR.feedback(event_type=event_type, **payload)


# ============================================================
# PRÉFLIGHT
# ============================================================

def _run_test_module(module_name: str, timeout_s: float | None, output_dir: Path) -> TestModuleSummary:
    started = time.time()
    cmd = [sys.executable, "-m", "unittest", "-v", module_name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_module_specific_env(module_name, output_dir),
            timeout=_normalize_timeout(timeout_s),
        )
        return TestModuleSummary(
            module=module_name,
            ok=proc.returncode == 0,
            returncode=int(proc.returncode),
            duration_s=float(time.time() - started),
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            timeout_s=_normalize_timeout(timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        return TestModuleSummary(
            module=module_name,
            ok=False,
            returncode=124,
            duration_s=float(time.time() - started),
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
            timed_out=True,
            timeout_s=_normalize_timeout(timeout_s),
        )


def _parse_unittest_counts(text: str) -> Dict[str, int]:
    raw = str(text or "")
    if not raw:
        return {"total": 0, "failures": 0, "errors": 0}

    clean = raw.replace("\r", "\n")
    total = 0
    failures = 0
    errors = 0

    ran_matches = re.findall(r"Ran\s+(\d+)\s+tests?\s+in", clean, flags=re.IGNORECASE)
    if ran_matches:
        total = max(int(x) for x in ran_matches)

    failed_match = re.search(r"FAILED\s*\((.*?)\)", clean, flags=re.IGNORECASE | re.DOTALL)
    if failed_match:
        details = failed_match.group(1)
        fail_match = re.search(r"failures\s*=\s*(\d+)", details, flags=re.IGNORECASE)
        err_match = re.search(r"errors\s*=\s*(\d+)", details, flags=re.IGNORECASE)
        if fail_match:
            failures = int(fail_match.group(1))
        if err_match:
            errors = int(err_match.group(1))

    return {"total": int(total), "failures": int(failures), "errors": int(errors)}


def _collect_preflight_counts(results: Sequence[TestModuleSummary]) -> Dict[str, Any]:
    total = 0
    failures = 0
    errors = 0
    per_module: List[Dict[str, Any]] = []

    for item in results:
        parsed = _parse_unittest_counts("\n".join([str(item.stdout or ""), str(item.stderr or "")]))
        total += int(parsed["total"])
        failures += int(parsed["failures"])
        errors += int(parsed["errors"])
        per_module.append({
            "module": item.module,
            "total": int(parsed["total"]),
            "failures": int(parsed["failures"]),
            "errors": int(parsed["errors"]),
            "ok": bool(item.ok),
        })

    return {
        "total": int(total),
        "failures": int(failures),
        "errors": int(errors),
        "per_module": per_module,
    }


def _truncate_text(text: str, limit: int = 4000) -> str:
    raw = str(text or "")
    return raw if len(raw) <= limit else raw[:limit] + "\n...<truncated>..."


def compile_test_and_runtime_logs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    candidates = [
        out / DEFAULT_RUNTIME_LOG_FILE,
        out / "test_main.log",
        out / "test_start.log",
        out / "test_camouflage_ml_dl_guided.log",
        out / "test_camouflage_ml_dl_precise.log",
    ]

    sections: List[str] = []
    files_meta: List[Dict[str, Any]] = []
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            text = f"<unreadable: {exc}>"
        files_meta.append({
            "path": str(path.resolve()),
            "name": path.name,
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            "line_count": int(len(text.splitlines())),
            "tail_preview": _truncate_text("\n".join(text.splitlines()[-80:])),
        })
        sections.append(f"===== {path.name} =====\n{text}\n")

    compiled_txt = out / DEFAULT_COMPILED_LOGS_FILE
    compiled_json = out / DEFAULT_COMPILED_LOGS_JSON
    compiled_txt.write_text("\n".join(sections), encoding="utf-8")
    compiled_json.write_text(json.dumps({"files": files_meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "compiled_text": str(compiled_txt),
        "compiled_json": str(compiled_json),
        "files": files_meta,
    }


def run_preflight_tests(
    module_names: Sequence[str] | None = None,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_s: float | None = DEFAULT_TEST_TIMEOUT_S,
) -> Dict[str, Any]:
    modules = _normalize_module_names(module_names)
    results = [_run_test_module(name, timeout_s=timeout_s, output_dir=output_dir) for name in modules]
    ok = all(item.ok for item in results)
    counts = _collect_preflight_counts(results)
    compiled = compile_test_and_runtime_logs(output_dir=output_dir)
    summary = {
        "ok": bool(ok),
        "modules": [r.to_dict() for r in results],
        "count": len(results),
        "module_count": len(results),
        "failed_modules": [r.module for r in results if not r.ok],
        "total": int(counts["total"]),
        "failures": int(counts["failures"]),
        "errors": int(counts["errors"]),
        "per_module": counts["per_module"],
        "compiled_logs": compiled,
        "short_text": f"{int(counts['total'])} tests | {int(counts['failures'])} échec(s) | {int(counts['errors'])} erreur(s)",
    }
    path = Path(output_dir) / DEFAULT_TEST_SUMMARY_FILE
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event("INFO" if ok else "ERROR", "preflight", "Préflight tests terminé", summary={
        "ok": summary["ok"],
        "failed_modules": summary["failed_modules"],
        "total": summary["total"],
        "failures": summary["failures"],
        "errors": summary["errors"],
        "compiled_logs": {
            "compiled_text": compiled["compiled_text"],
            "compiled_json": compiled["compiled_json"],
            "file_count": len(compiled["files"]),
        },
    })
    return summary


async def async_run_preflight_tests(
    module_names: Sequence[str] | None = None,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_s: float | None = DEFAULT_TEST_TIMEOUT_S,
) -> Dict[str, Any]:
    return await asyncio.to_thread(run_preflight_tests, module_names, output_dir=output_dir, timeout_s=timeout_s)


def run_generation_preflight(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    strict: bool = True,
    module_names: Sequence[str] | None = DEFAULT_TEST_MODULES,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checks: Dict[str, Any] = {
        "ok": True,
        "message": "Préflight validé",
        "output_dir": str(out),
        "write_ok": False,
        "disk_free_mb": 0.0,
        "tests_ok": True,
        "tests": None,
    }

    probe = out / ".preflight_probe.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    checks["write_ok"] = True

    disk = shutil.disk_usage(str(out))
    checks["disk_free_mb"] = float(disk.free / (1024 * 1024))
    if checks["disk_free_mb"] < 256:
        checks["ok"] = False
        checks["message"] = "Espace disque insuffisant"

    tests = run_preflight_tests(module_names=module_names, output_dir=out)
    checks["tests"] = tests
    checks["tests_ok"] = bool(tests.get("ok", False))
    if strict and not checks["tests_ok"]:
        checks["ok"] = False
        checks["message"] = "Les tests de préflight ont échoué"

    log_event("INFO" if checks["ok"] else "ERROR", "preflight", "Préflight génération", checks=checks)
    return checks
