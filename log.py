# -*- coding: utf-8 -*-
"""
log.py
Analyseur de rejets pour la génération de camouflage.

But :
- comprendre pourquoi les candidats échouent,
- conserver une validation stricte dans main.py,
- fournir des logs exploitables pour affiner main.py et l'interface start.py.

Sorties :
- logs_generation/diagnostic_candidates.csv
- logs_generation/diagnostic_summary.json
- logs_generation/diagnostic_summary.txt

Usage simple :
    python log.py

Usage avancé :
    python log.py --count 500 --output logs_generation
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

import main as camo


# ============================================================
# CONFIG
# ============================================================

DEFAULT_ANALYSIS_COUNT = 300
DEFAULT_OUTPUT_DIR = Path("logs_generation")


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


# ============================================================
# OUTILS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(x: Any) -> float:
    return float(x)


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

def analyze_candidate(candidate: camo.CandidateResult, target_index: int, local_attempt: int) -> CandidateDiagnostic:
    rs = candidate.ratios
    m = candidate.metrics

    failures: List[RuleFailure] = []

    abs_err = np.abs(rs - camo.TARGET)
    mean_abs_err = float(np.mean(abs_err))

    # erreurs absolues par couleur
    per_color_rules = [
        ("abs_err_coyote", float(rs[camo.IDX_COYOTE]), float(camo.TARGET[camo.IDX_COYOTE]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_COYOTE])),
        ("abs_err_olive", float(rs[camo.IDX_OLIVE]), float(camo.TARGET[camo.IDX_OLIVE]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_OLIVE])),
        ("abs_err_terre", float(rs[camo.IDX_TERRE]), float(camo.TARGET[camo.IDX_TERRE]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_TERRE])),
        ("abs_err_gris", float(rs[camo.IDX_GRIS]), float(camo.TARGET[camo.IDX_GRIS]), float(camo.MAX_ABS_ERROR_PER_COLOR[camo.IDX_GRIS])),
    ]
    for name, actual, target, max_abs in per_color_rules:
        fail = rule_fail_target_abs(name, actual, target, max_abs)
        if fail is not None:
            failures.append(fail)

    fail = rule_fail_max("mean_abs_error", mean_abs_err, float(camo.MAX_MEAN_ABS_ERROR))
    if fail is not None:
        failures.append(fail)

    # ratios globaux
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

    # métriques strictes
    metric_min_checks = [
        ("largest_olive_component_ratio", m["largest_olive_component_ratio"], camo.MIN_OLIVE_CONNECTED_COMPONENT_RATIO),
        ("largest_olive_component_ratio_small", m["largest_olive_component_ratio_small"], 0.12),
        ("olive_multizone_share", m["olive_multizone_share"], camo.MIN_OLIVE_MULTIZONE_SHARE),
        ("boundary_density", m["boundary_density"], camo.MIN_BOUNDARY_DENSITY),
        ("boundary_density_small", m["boundary_density_small"], camo.MIN_BOUNDARY_DENSITY_SMALL),
        ("oblique_share", m["oblique_share"], camo.MIN_OBLIQUE_SHARE),
        ("vert_olive_macro_share", m["vert_olive_macro_share"], camo.MIN_VISIBLE_OLIVE_MACRO_SHARE),
        ("terre_de_france_transition_share", m["terre_de_france_transition_share"], camo.MIN_VISIBLE_TERRE_TRANS_SHARE),
        ("vert_de_gris_micro_share", m["vert_de_gris_micro_share"], camo.MIN_VISIBLE_GRIS_MICRO_SHARE),
    ]
    for name, actual, min_v in metric_min_checks:
        fail = rule_fail_min(name, float(actual), float(min_v))
        if fail is not None:
            failures.append(fail)

    metric_max_checks = [
        ("center_empty_ratio", m["center_empty_ratio"], camo.MAX_COYOTE_CENTER_EMPTY_RATIO),
        ("center_empty_ratio_small", m["center_empty_ratio_small"], camo.MAX_COYOTE_CENTER_EMPTY_RATIO_SMALL),
        ("boundary_density", m["boundary_density"], camo.MAX_BOUNDARY_DENSITY),
        ("boundary_density_small", m["boundary_density_small"], camo.MAX_BOUNDARY_DENSITY_SMALL),
        ("mirror_similarity", m["mirror_similarity"], camo.MAX_MIRROR_SIMILARITY),
        ("angle_dominance_ratio", m["angle_dominance_ratio"], camo.MAX_ANGLE_DOMINANCE_RATIO),
        ("vert_de_gris_macro_share", m["vert_de_gris_macro_share"], camo.MAX_VISIBLE_GRIS_MACRO_SHARE),
    ]
    for name, actual, max_v in metric_max_checks:
        fail = rule_fail_max(name, float(actual), float(max_v))
        if fail is not None:
            failures.append(fail)

    vertical_fail = rule_fail_range(
        "vertical_share",
        float(m["vertical_share"]),
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

    metrics = {k: float(v) for k, v in m.items()}

    return CandidateDiagnostic(
        seed=int(candidate.seed),
        target_index=int(target_index),
        local_attempt=int(local_attempt),
        accepted=accepted,
        ratios=ratios,
        metrics=metrics,
        failures=failures,
    )


# ============================================================
# GÉNÉRATION DE DIAGNOSTIC
# ============================================================

def generate_diagnostics(
    count: int,
    base_seed: int = camo.DEFAULT_BASE_SEED,
) -> List[CandidateDiagnostic]:
    diagnostics: List[CandidateDiagnostic] = []

    for i in range(1, count + 1):
        seed = camo.build_seed(target_index=1, local_attempt=i, base_seed=base_seed)
        candidate = camo.generate_candidate_from_seed(seed)
        diagnostic = analyze_candidate(candidate, target_index=1, local_attempt=i)
        diagnostics.append(diagnostic)

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

    return {
        "total_candidates": int(total),
        "accepted": int(accepted),
        "rejected": int(rejected),
        "acceptance_rate": float(accepted / total) if total else 0.0,
        "top_failure_rules": fail_stats,
        "top_failure_combinations": top_fail_combinations,
        "ratios_distribution": ratio_summary,
        "metrics_distribution": metric_summary,
    }


# ============================================================
# EXPORTS
# ============================================================

def write_candidates_csv(diagnostics: List[CandidateDiagnostic], output_dir: Path) -> Path:
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

    return path


def write_summary_json(summary: Dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "diagnostic_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_summary_txt(summary: Dict[str, Any], output_dir: Path) -> Path:
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
    return path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic des rejets de génération camouflage.")
    parser.add_argument("--count", type=int, default=DEFAULT_ANALYSIS_COUNT, help="Nombre de candidats à analyser.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Dossier de sortie des logs.")
    parser.add_argument("--base-seed", type=int, default=camo.DEFAULT_BASE_SEED, help="Seed de base.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output))

    t0 = time.perf_counter()
    diagnostics = generate_diagnostics(
        count=int(args.count),
        base_seed=int(args.base_seed),
    )
    summary = build_summary(diagnostics)

    csv_path = write_candidates_csv(diagnostics, output_dir)
    json_path = write_summary_json(summary, output_dir)
    txt_path = write_summary_txt(summary, output_dir)
    dt = time.perf_counter() - t0

    print("\nDiagnostic terminé.")
    print(f"Candidats analysés : {summary['total_candidates']}")
    print(f"Acceptés           : {summary['accepted']}")
    print(f"Rejetés            : {summary['rejected']}")
    print(f"Taux acceptation   : {summary['acceptance_rate']:.4%}")
    print(f"CSV                : {csv_path.resolve()}")
    print(f"JSON               : {json_path.resolve()}")
    print(f"TXT                : {txt_path.resolve()}")
    print(f"Durée              : {dt:.2f} s")


if __name__ == "__main__":
    main()