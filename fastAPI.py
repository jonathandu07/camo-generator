from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

import main as camo


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

BASE_OUTPUT_DIR = Path(os.getenv("CAMO_API_OUTPUT_DIR", "service_outputs")).resolve()
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS_RAW = os.getenv("CAMO_API_CORS_ORIGINS", "*")
CORS_ORIGINS = [x.strip() for x in CORS_ORIGINS_RAW.split(",") if x.strip()] or ["*"]

# Important : main.py repose sur des variables globales mutables.
# Pour éviter les corruptions entre jobs aux configurations différentes,
# on sérialise l'exécution effective du moteur.
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("CAMO_API_MAX_CONCURRENT_JOBS", "1")))
RECENT_EVENT_LIMIT = max(20, int(os.getenv("CAMO_API_RECENT_EVENT_LIMIT", "500")))
JOB_PUBLIC_EVENT_LIMIT = max(10, int(os.getenv("CAMO_API_PUBLIC_EVENT_LIMIT", "80")))
MAX_SAVED_PREVIEWS = max(1, int(os.getenv("CAMO_API_MAX_SAVED_PREVIEWS", "40")))

job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
generation_state_lock = asyncio.Lock()

REPORT_FILENAME = "rapport_textures.csv"
SUMMARY_FILENAME = "run_summary.json"
PREVIEW_DIRNAME = "_previews"
DEFAULT_SERVICE_NAME = "camo-fastapi-service-advanced"

SAFE_GLOBAL_OVERRIDES: dict[str, type] = {
    "DEFAULT_OVERSCAN": float,
    "MIN_MOTIF_SCALE": float,
    "MAX_MOTIF_SCALE": float,
    "MIN_PATCH_PX": float,
    "STRAY_CLEANUP_PASSES": int,
    "ORPHAN_MAX_SAME_NEIGHBORS": int,
    "ORPHAN_MIN_WINNER_NEIGHBORS": int,
    "MIN_COMPONENT_PIXELS": tuple,
    "MACRO_GUIDE_MIN_SIDE": int,
    "MACRO_GUIDE_MAX_SIDE": int,
    "MACRO_GUIDE_BONUS": float,
    "MACRO_GUIDE_PENALTY": float,
    "MIN_DST_NEIGHBORS_FOR_FLIP": int,
    "COHERENCE_GAIN_WEIGHT": float,
    "BESTOF_REQUIRED": bool,
    "BESTOF_MIN_SCORE": float,
    "DEFAULT_DYNAMIC_TOLERANCE_ENABLED": bool,
    "DEFAULT_REJECTION_RATE_WINDOW": int,
    "DEFAULT_REJECTION_RATE_HIGH": float,
    "DEFAULT_REJECTION_RATE_LOW": float,
    "DEFAULT_TOLERANCE_MIN_ATTEMPTS": int,
    "DEFAULT_TOLERANCE_RELAX_STEP": float,
    "MAX_TOLERANCE_RELAX": float,
    "DEFAULT_ENABLE_ANTI_PIXEL": bool,
    "ANTI_PIXEL_MODE_FILTER_SIZE": int,
    "ANTI_PIXEL_PASSES": int,
    "MAX_ABS_ERROR_PER_COLOR": tuple,
}


# ============================================================
# OUTILS BAS NIVEAU
# ============================================================


def _safe_round(value: Any, digits: int = 6) -> Any:
    try:
        return round(float(value), digits)
    except Exception:
        return value



def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "oui"}
    return bool(value)



def _normalize_hex(hex_color: str) -> str:
    raw = str(hex_color or "").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        raise ValueError(f"Couleur hex invalide: {hex_color}")
    return f"#{raw.upper()}"



def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hx = _normalize_hex(hex_color)[1:]
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))



def _rgb_to_hex(rgb: Sequence[int]) -> str:
    r, g, b = [max(0, min(255, int(x))) for x in rgb[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"



def _normalize_ratios(raw_ratios: Sequence[float], *, label: str = "ratios") -> List[float]:
    vals = [float(x) for x in raw_ratios]
    if any((not math.isfinite(x)) or x <= 0 for x in vals):
        raise ValueError(f"{label} doit contenir uniquement des valeurs > 0")
    total = float(sum(vals))
    if total <= 0:
        raise ValueError(f"{label} doit avoir une somme > 0")
    vals = [x / total for x in vals]
    correction = 1.0 - float(sum(vals[:-1]))
    vals[-1] = correction
    return vals



def _resolve_ratios_from_colors(colors: Sequence["ColorSpec"]) -> List[float]:
    specified = [c.ratio for c in colors if c.ratio is not None]
    if not specified:
        return [1.0 / len(colors)] * len(colors)

    consumed = sum(float(c.ratio or 0.0) for c in colors if c.ratio is not None)
    unspecified_count = sum(1 for c in colors if c.ratio is None)
    if consumed >= 1.0 and unspecified_count > 0:
        raise ValueError("Les ratios déjà fournis atteignent ou dépassent 1.0 alors qu'il reste des couleurs sans ratio")

    remaining = max(0.0, 1.0 - consumed)
    shared = (remaining / unspecified_count) if unspecified_count else 0.0
    ratios = [float(c.ratio) if c.ratio is not None else float(shared) for c in colors]
    return _normalize_ratios(ratios, label="ratios_palette")



def _ensure_under(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return path



def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _jsonable(vars(obj))
    return str(obj)


# ============================================================
# MODÈLES Pydantic
# ============================================================


class ColorSpec(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)
    hex: str = Field(..., description="Couleur au format #RRGGBB")
    ratio: Optional[float] = Field(default=None, gt=0.0, description="Poids relatif. Si omis partout, répartition uniforme.")


class GenerationSettings(BaseModel):
    target_count: int = Field(default=10, ge=1, le=500)
    base_seed: int = Field(default=camo.DEFAULT_BASE_SEED, ge=0)

    width: int = Field(default=int(getattr(camo, "DEFAULT_WIDTH", camo.WIDTH)), ge=64, le=16384)
    height: int = Field(default=int(getattr(camo, "DEFAULT_HEIGHT", camo.HEIGHT)), ge=64, le=16384)
    physical_width_cm: float = Field(default=float(getattr(camo, "DEFAULT_PHYSICAL_WIDTH_CM", 768.0)), gt=0.0)
    physical_height_cm: float = Field(default=float(getattr(camo, "DEFAULT_PHYSICAL_HEIGHT_CM", 432.0)), gt=0.0)
    motif_scale: float = Field(default=float(getattr(camo, "DEFAULT_MOTIF_SCALE", 0.55)), gt=0.0, le=5.0)

    machine_intensity: float = Field(default=float(getattr(camo, "DEFAULT_MACHINE_INTENSITY", 0.90)), ge=0.10, le=1.00)
    max_workers: Optional[int] = Field(default=None, ge=1)
    attempt_batch_size: Optional[int] = Field(default=None, ge=1)
    parallel_attempts: bool = True
    resource_sample_every_batches: int = Field(default=int(getattr(camo, "DEFAULT_RESOURCE_SAMPLE_EVERY_BATCHES", 1)), ge=1, le=1000)

    anti_pixel: bool = Field(default=bool(getattr(camo, "DEFAULT_ENABLE_ANTI_PIXEL", True)))
    max_repair_rounds: int = Field(default=int(getattr(camo, "MAX_REPAIR_ROUNDS", 3)), ge=0, le=32)

    overscan_override: Optional[float] = Field(default=None, ge=1.0, le=2.0)
    shift_strength_override: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    palette_bias_override: Optional[List[float]] = Field(default=None, description="4 biais backend, un par classe")

    extra_cleanup_passes: int = Field(default=0, ge=0, le=16)
    merge_micro: bool = False
    rebalance_edges: bool = False
    anti_mirror: bool = False


class ToleranceSettings(BaseModel):
    dynamic_tolerance_enabled: bool = Field(default=bool(getattr(camo, "DEFAULT_DYNAMIC_TOLERANCE_ENABLED", True)))
    rejection_rate_window: int = Field(default=int(getattr(camo, "DEFAULT_REJECTION_RATE_WINDOW", 24)), ge=1, le=5000)
    rejection_rate_high: float = Field(default=float(getattr(camo, "DEFAULT_REJECTION_RATE_HIGH", 0.90)), ge=0.0, le=1.0)
    rejection_rate_low: float = Field(default=float(getattr(camo, "DEFAULT_REJECTION_RATE_LOW", 0.55)), ge=0.0, le=1.0)
    tolerance_min_attempts: int = Field(default=int(getattr(camo, "DEFAULT_TOLERANCE_MIN_ATTEMPTS", 24)), ge=1, le=100000)
    tolerance_relax_step: float = Field(default=float(getattr(camo, "DEFAULT_TOLERANCE_RELAX_STEP", 0.08)), ge=0.0, le=1.0)
    relax_level_override: Optional[float] = Field(default=None, ge=0.0, le=float(getattr(camo, "MAX_TOLERANCE_RELAX", 0.40)))

    max_abs_error_per_color: Optional[List[float]] = None
    max_mean_abs_error: Optional[float] = None
    min_boundary_density: Optional[float] = None
    max_boundary_density: Optional[float] = None
    min_boundary_density_small: Optional[float] = None
    max_boundary_density_small: Optional[float] = None
    min_boundary_density_tiny: Optional[float] = None
    max_boundary_density_tiny: Optional[float] = None
    max_mirror_similarity: Optional[float] = None
    min_largest_component_ratio_class_1: Optional[float] = None
    max_edge_contact_ratio: Optional[float] = None
    bestof_min_score: Optional[float] = None
    max_orphan_ratio: Optional[float] = None
    max_micro_islands_per_mp: Optional[float] = None


class GenerateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)
    palette_mode: Literal["strict_backend", "extended_postprocess"] = Field(
        default="strict_backend",
        description=(
            "strict_backend = vrai backend 4 classes. "
            "extended_postprocess = backend 4 classes + remap post-traité vers palette étendue."
        ),
    )
    colors: List[ColorSpec] = Field(default_factory=list)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    tolerances: ToleranceSettings = Field(default_factory=ToleranceSettings)
    advanced_globals: Dict[str, Any] = Field(default_factory=dict)


class PreviewRequest(GenerateRequest):
    save_preview: bool = True


class JobEvent(BaseModel):
    ts: float
    level: str
    message: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class JobPublic(BaseModel):
    job_id: str
    label: Optional[str]
    status: str
    created_at: float
    started_at: Optional[float]
    ended_at: Optional[float]
    target_count: int
    accepted_count: int
    rejected_count: int
    total_attempts: int
    current_index: int
    current_local_attempt: int
    progress_ratio: float
    output_dir: str
    report_path: Optional[str]
    summary_path: Optional[str]
    error_message: Optional[str]
    palette_mode: str
    colors_count: int
    max_workers: int
    attempt_batch_size: int
    parallel_attempts: bool
    machine_intensity: float
    last_candidate: Optional[Dict[str, Any]]
    recent_events: List[JobEvent]


class JobListResponse(BaseModel):
    jobs: List[JobPublic]


class CancelResponse(BaseModel):
    job_id: str
    cancel_requested: bool
    status: str


# ============================================================
# STRUCTURES INTERNE
# ============================================================


@dataclass
class NormalizedColor:
    name: str
    hex: str
    ratio: float
    rgb: Tuple[int, int, int]


@dataclass
class BackendGroup:
    backend_index: int
    ratio: float
    anchor_hex: str
    anchor_rgb: Tuple[int, int, int]
    source_colors: List[NormalizedColor]


@dataclass
class RuntimeConfig:
    label: Optional[str]
    palette_mode: str
    requested_colors: List[NormalizedColor]
    backend_groups: List[BackendGroup]
    generation: GenerationSettings
    tolerances: ToleranceSettings
    advanced_globals: Dict[str, Any]

    @property
    def backend_rgb(self) -> np.ndarray:
        return np.asarray([grp.anchor_rgb for grp in self.backend_groups], dtype=np.uint8)

    @property
    def backend_target(self) -> np.ndarray:
        return np.asarray([grp.ratio for grp in self.backend_groups], dtype=float)

    @property
    def backend_class_names(self) -> List[str]:
        names: List[str] = []
        for grp in self.backend_groups:
            if len(grp.source_colors) == 1:
                base = grp.source_colors[0].name
            else:
                base = f"group_{grp.backend_index + 1}"
            names.append(base)
        return names

    @property
    def requested_palette_hex(self) -> List[str]:
        return [c.hex for c in self.requested_colors]

    @property
    def requested_palette_ratios(self) -> List[float]:
        return [float(c.ratio) for c in self.requested_colors]


@dataclass
class JobState:
    job_id: str
    runtime: RuntimeConfig
    created_at: float
    output_dir: Path
    report_path: Optional[Path] = None
    summary_path: Optional[Path] = None

    status: str = "queued"
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error_message: Optional[str] = None
    cancel_requested: bool = False

    accepted_count: int = 0
    rejected_count: int = 0
    total_attempts: int = 0
    current_index: int = 0
    current_local_attempt: int = 0
    last_candidate: Optional[Dict[str, Any]] = None
    rows: List[Dict[str, Any]] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    task: Optional[asyncio.Task] = None

    _accepted_target_indices: set[int] = field(default_factory=set, repr=False)

    def add_event(self, level: str, message: str, **payload: Any) -> None:
        self.recent_events.append(
            {
                "ts": time.time(),
                "level": level.upper(),
                "message": message,
                "payload": _jsonable(payload),
            }
        )
        if len(self.recent_events) > RECENT_EVENT_LIMIT:
            self.recent_events = self.recent_events[-RECENT_EVENT_LIMIT:]

    def recompute_counters(self) -> None:
        self.accepted_count = len(self._accepted_target_indices)
        self.rejected_count = max(0, self.total_attempts - self.accepted_count)

    def mark_attempt(self, target_index: int, accepted: bool) -> None:
        if accepted:
            self._accepted_target_indices.add(int(target_index))
        self.recompute_counters()

    def to_public(self) -> JobPublic:
        target_count = int(self.runtime.generation.target_count)
        progress_ratio = (self.accepted_count / target_count) if target_count else 0.0
        return JobPublic(
            job_id=self.job_id,
            label=self.runtime.label,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            ended_at=self.ended_at,
            target_count=target_count,
            accepted_count=self.accepted_count,
            rejected_count=self.rejected_count,
            total_attempts=self.total_attempts,
            current_index=self.current_index,
            current_local_attempt=self.current_local_attempt,
            progress_ratio=progress_ratio,
            output_dir=str(self.output_dir),
            report_path=str(self.report_path) if self.report_path else None,
            summary_path=str(self.summary_path) if self.summary_path else None,
            error_message=self.error_message,
            palette_mode=self.runtime.palette_mode,
            colors_count=len(self.runtime.requested_colors),
            max_workers=int(self.runtime.generation.max_workers or camo.DEFAULT_MAX_WORKERS),
            attempt_batch_size=int(self.runtime.generation.attempt_batch_size or max(1, int(self.runtime.generation.max_workers or camo.DEFAULT_MAX_WORKERS))),
            parallel_attempts=bool(self.runtime.generation.parallel_attempts),
            machine_intensity=float(self.runtime.generation.machine_intensity),
            last_candidate=self.last_candidate,
            recent_events=[JobEvent(**evt) for evt in self.recent_events[-JOB_PUBLIC_EVENT_LIMIT:]],
        )


jobs: Dict[str, JobState] = {}


# ============================================================
# NORMALISATION PALETTE
# ============================================================


def _normalize_colors(colors: Sequence[ColorSpec]) -> List[NormalizedColor]:
    if not colors:
        base_rgb = np.asarray(getattr(camo, "RGB", np.zeros((4, 3), dtype=np.uint8)), dtype=np.uint8)
        base_target = np.asarray(getattr(camo, "TARGET", np.array([0.32, 0.28, 0.22, 0.18])), dtype=float)
        colors = [
            ColorSpec(name=f"class_{idx}", hex=_rgb_to_hex(base_rgb[idx]), ratio=float(base_target[idx]))
            for idx in range(int(base_rgb.shape[0]))
        ]

    ratios = _resolve_ratios_from_colors(colors)
    out: List[NormalizedColor] = []
    for idx, (color, ratio) in enumerate(zip(colors, ratios), start=1):
        hx = _normalize_hex(color.hex)
        rgb = _hex_to_rgb(hx)
        out.append(
            NormalizedColor(
                name=(str(color.name).strip() if color.name else f"color_{idx}"),
                hex=hx,
                ratio=float(ratio),
                rgb=rgb,
            )
        )
    return out



def _weighted_average_rgb(colors: Sequence[NormalizedColor]) -> Tuple[int, int, int]:
    if not colors:
        raise ValueError("Groupe de couleurs vide")
    weights = np.asarray([c.ratio for c in colors], dtype=float)
    weights = weights / max(1e-12, float(weights.sum()))
    rgbs = np.asarray([c.rgb for c in colors], dtype=float)
    avg = np.average(rgbs, axis=0, weights=weights)
    return tuple(int(round(x)) for x in avg.tolist())



def _split_largest_color_group(colors: List[NormalizedColor]) -> List[NormalizedColor]:
    if not colors:
        return []
    idx = max(range(len(colors)), key=lambda i: colors[i].ratio)
    chosen = colors[idx]
    half_a = max(chosen.ratio / 2.0, 1e-6)
    half_b = max(chosen.ratio - half_a, 1e-6)
    repl = [
        NormalizedColor(name=f"{chosen.name}_a", hex=chosen.hex, ratio=half_a, rgb=chosen.rgb),
        NormalizedColor(name=f"{chosen.name}_b", hex=chosen.hex, ratio=half_b, rgb=chosen.rgb),
    ]
    return colors[:idx] + repl + colors[idx + 1 :]



def _build_backend_groups(requested_colors: List[NormalizedColor], palette_mode: str) -> List[BackendGroup]:
    if palette_mode == "strict_backend":
        if len(requested_colors) != 4:
            raise ValueError("Le mode strict_backend exige exactement 4 couleurs")
        return [
            BackendGroup(
                backend_index=i,
                ratio=float(color.ratio),
                anchor_hex=color.hex,
                anchor_rgb=color.rgb,
                source_colors=[color],
            )
            for i, color in enumerate(requested_colors)
        ]

    working = list(requested_colors)
    while len(working) < 4:
        working = _split_largest_color_group(working)
        working = [
            NormalizedColor(name=c.name, hex=c.hex, ratio=r, rgb=c.rgb)
            for c, r in zip(working, _normalize_ratios([x.ratio for x in working], label="ratios_palette_extended"))
        ]

    if len(working) == 4:
        groups = [[c] for c in working]
    else:
        target_group_sum = 1.0 / 4.0
        groups: List[List[NormalizedColor]] = []
        current: List[NormalizedColor] = []
        running = 0.0
        remaining = list(working)
        for idx, color in enumerate(working):
            current.append(color)
            running += float(color.ratio)
            left_items = len(working) - (idx + 1)
            groups_left = 4 - (len(groups) + 1)
            must_close = groups_left > 0 and left_items == groups_left
            should_close = running >= target_group_sum and groups_left > 0
            if must_close or should_close:
                groups.append(current)
                current = []
                running = 0.0
        if current:
            groups.append(current)
        while len(groups) > 4:
            tail = groups.pop()
            groups[-1].extend(tail)
        while len(groups) < 4:
            # si une partition a produit <4 groupes, on scinde le plus gros
            idx_big = max(range(len(groups)), key=lambda i: sum(c.ratio for c in groups[i]))
            big = groups.pop(idx_big)
            if len(big) == 1:
                c = big[0]
                a = NormalizedColor(name=f"{c.name}_a", hex=c.hex, ratio=c.ratio / 2.0, rgb=c.rgb)
                b = NormalizedColor(name=f"{c.name}_b", hex=c.hex, ratio=c.ratio - a.ratio, rgb=c.rgb)
                groups.insert(idx_big, [b])
                groups.insert(idx_big, [a])
            else:
                cut = max(1, len(big) // 2)
                groups.insert(idx_big, big[cut:])
                groups.insert(idx_big, big[:cut])

    backend_groups: List[BackendGroup] = []
    for backend_index, group in enumerate(groups):
        group_ratio = float(sum(c.ratio for c in group))
        anchor_rgb = _weighted_average_rgb(group)
        backend_groups.append(
            BackendGroup(
                backend_index=backend_index,
                ratio=group_ratio,
                anchor_hex=_rgb_to_hex(anchor_rgb),
                anchor_rgb=anchor_rgb,
                source_colors=[
                    NormalizedColor(name=c.name, hex=c.hex, ratio=float(c.ratio), rgb=c.rgb)
                    for c in group
                ],
            )
        )

    normalized_group_ratios = _normalize_ratios([g.ratio for g in backend_groups], label="ratios_backend")
    for grp, ratio in zip(backend_groups, normalized_group_ratios):
        grp.ratio = float(ratio)

    return backend_groups



def normalize_runtime_config(payload: GenerateRequest) -> RuntimeConfig:
    colors = _normalize_colors(payload.colors)
    generation = payload.generation
    tolerances = payload.tolerances

    if generation.palette_bias_override is not None and len(generation.palette_bias_override) != 4:
        raise ValueError("generation.palette_bias_override doit contenir exactement 4 valeurs")
    if tolerances.max_abs_error_per_color is not None and len(tolerances.max_abs_error_per_color) != 4:
        raise ValueError("tolerances.max_abs_error_per_color doit contenir exactement 4 valeurs")

    backend_groups = _build_backend_groups(colors, payload.palette_mode)

    for key in payload.advanced_globals.keys():
        if key not in SAFE_GLOBAL_OVERRIDES:
            raise ValueError(
                f"advanced_globals contient une clé non autorisée: {key}. "
                f"Clés autorisées: {', '.join(sorted(SAFE_GLOBAL_OVERRIDES.keys()))}"
            )

    return RuntimeConfig(
        label=payload.label,
        palette_mode=payload.palette_mode,
        requested_colors=colors,
        backend_groups=backend_groups,
        generation=generation,
        tolerances=tolerances,
        advanced_globals=dict(payload.advanced_globals),
    )


# ============================================================
# PATCH TEMPORAIRE DU MODULE main.py
# ============================================================


@contextmanager
def patched_camo_state(runtime: RuntimeConfig):
    saved: Dict[str, Any] = {
        "WIDTH": camo.WIDTH,
        "HEIGHT": camo.HEIGHT,
        "PHYSICAL_WIDTH_CM": camo.PHYSICAL_WIDTH_CM,
        "PHYSICAL_HEIGHT_CM": camo.PHYSICAL_HEIGHT_CM,
        "PX_PER_CM": camo.PX_PER_CM,
        "MOTIF_SCALE": camo.MOTIF_SCALE,
        "RGB": np.array(camo.RGB, copy=True),
        "TARGET": np.array(camo.TARGET, copy=True),
        "CLASS_NAMES": list(camo.CLASS_NAMES),
        "N_CLASSES": int(camo.N_CLASSES),
        "MIN_COMPONENT_PIXELS": tuple(camo.MIN_COMPONENT_PIXELS),
        "BESTOF_REQUIRED": bool(camo.BESTOF_REQUIRED),
    }
    for key in runtime.advanced_globals:
        saved[key] = getattr(camo, key)

    try:
        camo.set_canvas_geometry(
            width=int(runtime.generation.width),
            height=int(runtime.generation.height),
            physical_width_cm=float(runtime.generation.physical_width_cm),
            physical_height_cm=float(runtime.generation.physical_height_cm),
            motif_scale=float(runtime.generation.motif_scale),
        )
        camo.RGB = np.asarray(runtime.backend_rgb, dtype=np.uint8)
        camo.TARGET = np.asarray(runtime.backend_target, dtype=float)
        camo.CLASS_NAMES = list(runtime.backend_class_names)
        camo.N_CLASSES = 4

        min_component_pixels = runtime.advanced_globals.get("MIN_COMPONENT_PIXELS", camo.MIN_COMPONENT_PIXELS)
        if isinstance(min_component_pixels, (list, tuple)):
            if len(min_component_pixels) != 4:
                raise ValueError("MIN_COMPONENT_PIXELS doit contenir exactement 4 valeurs")
            camo.MIN_COMPONENT_PIXELS = tuple(int(x) for x in min_component_pixels)

        for key, value in runtime.advanced_globals.items():
            expected = SAFE_GLOBAL_OVERRIDES[key]
            if expected is tuple:
                if key == "MAX_ABS_ERROR_PER_COLOR":
                    if not isinstance(value, (list, tuple)) or len(value) != 4:
                        raise ValueError("MAX_ABS_ERROR_PER_COLOR doit contenir exactement 4 valeurs")
                    cast_value = np.asarray([float(x) for x in value], dtype=float)
                else:
                    if not isinstance(value, (list, tuple)):
                        raise ValueError(f"{key} doit être une liste ou un tuple")
                    cast_value = tuple(value)
            elif expected is bool:
                cast_value = _as_bool(value)
            elif expected is int:
                cast_value = int(value)
            else:
                cast_value = float(value)
            setattr(camo, key, cast_value)

        yield
    finally:
        camo.WIDTH = int(saved["WIDTH"])
        camo.HEIGHT = int(saved["HEIGHT"])
        camo.PHYSICAL_WIDTH_CM = float(saved["PHYSICAL_WIDTH_CM"])
        camo.PHYSICAL_HEIGHT_CM = float(saved["PHYSICAL_HEIGHT_CM"])
        camo.PX_PER_CM = float(saved["PX_PER_CM"])
        camo.MOTIF_SCALE = float(saved["MOTIF_SCALE"])
        camo.RGB = np.asarray(saved["RGB"], dtype=np.uint8)
        camo.TARGET = np.asarray(saved["TARGET"], dtype=float)
        camo.CLASS_NAMES = list(saved["CLASS_NAMES"])
        camo.N_CLASSES = int(saved["N_CLASSES"])
        camo.MIN_COMPONENT_PIXELS = tuple(saved["MIN_COMPONENT_PIXELS"])
        camo.BESTOF_REQUIRED = bool(saved["BESTOF_REQUIRED"])

        for key in runtime.advanced_globals:
            setattr(camo, key, saved[key])


# ============================================================
# TOLÉRANCES ET PROFILS
# ============================================================


def apply_tolerance_overrides(
    profile: camo.ValidationToleranceProfile,
    tolerances: ToleranceSettings,
) -> camo.ValidationToleranceProfile:
    if tolerances.max_abs_error_per_color is not None:
        profile.max_abs_error_per_color = tuple(float(x) for x in tolerances.max_abs_error_per_color)
    if tolerances.max_mean_abs_error is not None:
        profile.max_mean_abs_error = float(tolerances.max_mean_abs_error)
    if tolerances.min_boundary_density is not None:
        profile.min_boundary_density = float(tolerances.min_boundary_density)
    if tolerances.max_boundary_density is not None:
        profile.max_boundary_density = float(tolerances.max_boundary_density)
    if tolerances.min_boundary_density_small is not None:
        profile.min_boundary_density_small = float(tolerances.min_boundary_density_small)
    if tolerances.max_boundary_density_small is not None:
        profile.max_boundary_density_small = float(tolerances.max_boundary_density_small)
    if tolerances.min_boundary_density_tiny is not None:
        profile.min_boundary_density_tiny = float(tolerances.min_boundary_density_tiny)
    if tolerances.max_boundary_density_tiny is not None:
        profile.max_boundary_density_tiny = float(tolerances.max_boundary_density_tiny)
    if tolerances.max_mirror_similarity is not None:
        profile.max_mirror_similarity = float(tolerances.max_mirror_similarity)
    if tolerances.min_largest_component_ratio_class_1 is not None:
        profile.min_largest_component_ratio_class_1 = float(tolerances.min_largest_component_ratio_class_1)
    if tolerances.max_edge_contact_ratio is not None:
        profile.max_edge_contact_ratio = float(tolerances.max_edge_contact_ratio)
    if tolerances.bestof_min_score is not None:
        profile.bestof_min_score = float(tolerances.bestof_min_score)
    if tolerances.max_orphan_ratio is not None:
        profile.max_orphan_ratio = float(tolerances.max_orphan_ratio)
    if tolerances.max_micro_islands_per_mp is not None:
        profile.max_micro_islands_per_mp = float(tolerances.max_micro_islands_per_mp)
    return profile



def build_tolerance_profile(runtime: RuntimeConfig, relax_level: float) -> camo.ValidationToleranceProfile:
    if runtime.tolerances.relax_level_override is not None:
        relax_level = float(runtime.tolerances.relax_level_override)
    profile = camo.build_validation_tolerance_profile(relax_level)
    return apply_tolerance_overrides(profile, runtime.tolerances)



def build_variant_profile(seed: int, runtime: RuntimeConfig) -> camo.VariantProfile:
    base_profile = camo.make_profile(seed)
    overscan = float(runtime.generation.overscan_override) if runtime.generation.overscan_override is not None else float(base_profile.overscan)
    shift_strength = float(runtime.generation.shift_strength_override) if runtime.generation.shift_strength_override is not None else float(base_profile.shift_strength)

    if runtime.generation.palette_bias_override is not None:
        palette_bias = tuple(float(x) for x in runtime.generation.palette_bias_override)
    else:
        palette_bias = tuple(float(x) for x in base_profile.palette_bias)

    if len(palette_bias) != 4:
        raise ValueError("Le profile backend exige exactement 4 biais de palette")

    return camo.VariantProfile(
        seed=int(seed),
        overscan=float(overscan),
        shift_strength=float(shift_strength),
        palette_bias=tuple(palette_bias),
    )


# ============================================================
# GÉNÉRATION / REMAP PALETTE
# ============================================================


def render_final_image(candidate: camo.CandidateResult, runtime: RuntimeConfig, *, target_index: int) -> Image.Image:
    if runtime.palette_mode == "strict_backend":
        return candidate.image.copy()

    label_map = np.asarray(candidate.label_map, dtype=np.uint8)
    flat_labels = label_map.ravel()
    final_rgb = np.zeros((flat_labels.shape[0], 3), dtype=np.uint8)
    rng = np.random.default_rng(int(candidate.seed) ^ int(target_index * 10007) ^ 0x6A09E667)

    for group in runtime.backend_groups:
        idx = np.flatnonzero(flat_labels == int(group.backend_index))
        if idx.size == 0:
            continue
        rng.shuffle(idx)

        local_ratios = _normalize_ratios([c.ratio for c in group.source_colors], label=f"group_{group.backend_index}_ratios")
        counts = [int(round(r * idx.size)) for r in local_ratios]
        delta = idx.size - sum(counts)
        if counts:
            counts[-1] += delta

        cursor = 0
        for color, count in zip(group.source_colors, counts):
            if count <= 0:
                continue
            chosen = idx[cursor : cursor + count]
            final_rgb[chosen] = np.asarray(color.rgb, dtype=np.uint8)
            cursor += count

        if cursor < idx.size and group.source_colors:
            final_rgb[idx[cursor:]] = np.asarray(group.source_colors[-1].rgb, dtype=np.uint8)

    out = final_rgb.reshape(label_map.shape[0], label_map.shape[1], 3)
    return Image.fromarray(out, mode="RGB")



def build_last_candidate_payload(
    target_index: int,
    local_attempt: int,
    candidate: camo.CandidateResult,
    outcome: camo.ValidationOutcome,
    runtime: RuntimeConfig,
) -> Dict[str, Any]:
    return {
        "seed": int(candidate.seed),
        "target_index": int(target_index),
        "local_attempt": int(local_attempt),
        "accepted": bool(outcome.accepted),
        "ratios_backend": [_safe_round(x, 6) for x in candidate.ratios.tolist()],
        "metrics": {k: _safe_round(v, 6) for k, v in candidate.metrics.items()},
        "reasons": list(outcome.reasons),
        "bestof_score": _safe_round(outcome.bestof_score, 6),
        "palette_mode": runtime.palette_mode,
        "requested_palette_hex": runtime.requested_palette_hex,
        "requested_palette_ratios": [_safe_round(x, 6) for x in runtime.requested_palette_ratios],
        "backend_palette_hex": [grp.anchor_hex for grp in runtime.backend_groups],
    }



def generate_and_validate_custom(
    seed: int,
    runtime: RuntimeConfig,
    tolerance_profile: camo.ValidationToleranceProfile,
) -> Tuple[camo.CandidateResult, camo.ValidationOutcome]:
    profile = build_variant_profile(seed, runtime)
    candidate = camo.generate_one_variant(
        profile,
        motif_scale_override=float(runtime.generation.motif_scale),
        extra_cleanup_passes=int(runtime.generation.extra_cleanup_passes),
        merge_micro=bool(runtime.generation.merge_micro),
        rebalance_edges=bool(runtime.generation.rebalance_edges),
        anti_mirror=bool(runtime.generation.anti_mirror),
        anti_pixel=bool(runtime.generation.anti_pixel),
    )
    outcome = camo.validate_with_reasons(candidate, tolerance_profile=tolerance_profile)

    trace: List[Dict[str, Any]] = [
        {
            "round": 0,
            "seed": int(candidate.seed),
            "accepted": bool(outcome.accepted),
            "bestof_score": float(outcome.bestof_score),
            "reasons": list(outcome.reasons),
            "metrics": {k: float(v) for k, v in candidate.metrics.items()},
        }
    ]

    best_candidate = candidate
    best_outcome = outcome
    max_repair_rounds = int(runtime.generation.max_repair_rounds)

    for repair_round in range(1, max_repair_rounds + 1):
        if best_outcome.accepted:
            break

        plan = camo.derive_repair_plan(best_candidate, best_outcome, repair_round)
        repaired_profile = camo.VariantProfile(
            seed=int(plan.seed),
            overscan=float(plan.overscan),
            shift_strength=float(plan.shift_strength),
            palette_bias=tuple(plan.palette_bias),
        )

        repaired_candidate = camo.generate_one_variant(
            repaired_profile,
            motif_scale_override=float(plan.motif_scale),
            extra_cleanup_passes=int(plan.extra_cleanup_passes + runtime.generation.extra_cleanup_passes),
            merge_micro=bool(plan.merge_micro or runtime.generation.merge_micro),
            rebalance_edges=bool(plan.rebalance_edges or runtime.generation.rebalance_edges),
            anti_mirror=bool(plan.anti_mirror or runtime.generation.anti_mirror),
            anti_pixel=bool(runtime.generation.anti_pixel),
        )
        repaired_candidate.metrics["repair_round"] = float(repair_round)
        repaired_outcome = camo.validate_with_reasons(repaired_candidate, tolerance_profile=tolerance_profile)

        trace.append(
            {
                "round": int(repair_round),
                "seed": int(repaired_candidate.seed),
                "accepted": bool(repaired_outcome.accepted),
                "bestof_score": float(repaired_outcome.bestof_score),
                "reasons": list(repaired_outcome.reasons),
                "metrics": {k: float(v) for k, v in repaired_candidate.metrics.items()},
                "applied_plan": {
                    "motif_scale": float(plan.motif_scale),
                    "overscan": float(plan.overscan),
                    "shift_strength": float(plan.shift_strength),
                    "palette_bias": [float(x) for x in plan.palette_bias],
                    "extra_cleanup_passes": int(plan.extra_cleanup_passes),
                    "merge_micro": bool(plan.merge_micro),
                    "rebalance_edges": bool(plan.rebalance_edges),
                    "anti_mirror": bool(plan.anti_mirror),
                },
            }
        )

        if camo.validation_rank(repaired_outcome) > camo.validation_rank(best_outcome):
            best_candidate = repaired_candidate
            best_outcome = repaired_outcome

    best_outcome.repair_trace = trace
    return best_candidate, best_outcome



def save_image(path: Path, image: Image.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path



def build_output_path(output_dir: Path, target_index: int, local_attempt: int, global_attempt: int, seed: int) -> Path:
    return camo.build_unique_pattern_path(
        output_dir=output_dir,
        target_index=int(target_index),
        seed=int(seed),
        local_attempt=int(local_attempt),
        global_attempt=int(global_attempt),
    )



def build_row(
    *,
    target_index: int,
    local_attempt: int,
    global_attempt: int,
    candidate: camo.CandidateResult,
    outcome: camo.ValidationOutcome,
    saved_path: Path,
    tolerance_profile: camo.ValidationToleranceProfile,
    runtime: RuntimeConfig,
) -> Dict[str, Any]:
    row = camo.candidate_row(
        target_index=target_index,
        local_attempt=local_attempt,
        global_attempt=global_attempt,
        candidate=candidate,
        outcome=outcome,
        image_name=saved_path.name,
        image_path=str(saved_path),
        tolerance_profile=tolerance_profile,
    )
    row["palette_mode"] = runtime.palette_mode
    row["requested_palette_hex"] = "|".join(runtime.requested_palette_hex)
    row["requested_palette_ratios"] = "|".join(f"{x:.8f}" for x in runtime.requested_palette_ratios)
    row["backend_palette_hex"] = "|".join(grp.anchor_hex for grp in runtime.backend_groups)
    row["backend_palette_ratios"] = "|".join(f"{grp.ratio:.8f}" for grp in runtime.backend_groups)
    row["colors_count"] = len(runtime.requested_colors)
    row["max_repair_rounds"] = int(runtime.generation.max_repair_rounds)
    row["anti_pixel"] = int(bool(runtime.generation.anti_pixel))
    row["machine_intensity"] = float(runtime.generation.machine_intensity)
    row["requested_label"] = runtime.label or ""
    return row


# ============================================================
# JOBS / EXÉCUTION
# ============================================================


async def run_generation_job(job: JobState) -> None:
    async with job_semaphore:
        async with generation_state_lock:
            runtime = job.runtime
            output_dir = job.output_dir

            if job.cancel_requested:
                job.status = "cancelled"
                job.started_at = time.time()
                job.ended_at = job.started_at
                job.report_path = output_dir / REPORT_FILENAME
                job.report_path.write_text("", encoding="utf-8")
                job.summary_path = output_dir / SUMMARY_FILENAME
                job.summary_path.write_text(json.dumps({"status": "cancelled_before_start"}, ensure_ascii=False, indent=2), encoding="utf-8")
                job.add_event("warning", "Job annulé avant démarrage")
                return

            job.status = "running"
            job.started_at = time.time()
            job.add_event(
                "info",
                "Job démarré",
                palette_mode=runtime.palette_mode,
                requested_palette_hex=runtime.requested_palette_hex,
                backend_palette_hex=[grp.anchor_hex for grp in runtime.backend_groups],
                target_count=int(runtime.generation.target_count),
            )

            worker_count = int(runtime.generation.max_workers or max(1, min(camo.CPU_COUNT, 4)))
            batch_size = int(runtime.generation.attempt_batch_size or worker_count)
            batch_size = max(1, batch_size)
            use_parallel = bool(runtime.generation.parallel_attempts and batch_size > 1 and worker_count > 1)

            tolerance_outcomes: List[bool] = []
            tolerance_relax_level = 0.0
            tolerance_runtime = {
                "rejection_rate": 0.0,
                "window_count": 0.0,
                "relax_before": 0.0,
                "relax_after": 0.0,
            }

            rows: List[Dict[str, Any]] = []
            target_count = int(runtime.generation.target_count)

            try:
                with patched_camo_state(runtime):
                    camo.validate_generation_request(
                        target_count=target_count,
                        output_dir=output_dir,
                        base_seed=int(runtime.generation.base_seed),
                        machine_intensity=float(runtime.generation.machine_intensity),
                        max_workers=worker_count,
                        attempt_batch_size=batch_size,
                    )

                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        for target_index in range(1, target_count + 1):
                            if job.cancel_requested:
                                break

                            local_attempt = 1
                            while True:
                                if job.cancel_requested:
                                    break

                                previous_relax = tolerance_relax_level
                                if runtime.tolerances.relax_level_override is None:
                                    tolerance_relax_level, _, tolerance_runtime = camo.adapt_tolerance_relax_level(
                                        tolerance_relax_level,
                                        tolerance_outcomes,
                                        window=int(runtime.tolerances.rejection_rate_window),
                                        rejection_rate_high=float(runtime.tolerances.rejection_rate_high),
                                        rejection_rate_low=float(runtime.tolerances.rejection_rate_low),
                                        min_attempts=int(runtime.tolerances.tolerance_min_attempts),
                                        relax_step=float(runtime.tolerances.tolerance_relax_step),
                                        enabled=bool(runtime.tolerances.dynamic_tolerance_enabled),
                                    )
                                tolerance_profile = build_tolerance_profile(runtime, tolerance_relax_level)

                                actual_batch_size = batch_size if use_parallel else 1
                                batch = camo.build_batch(
                                    target_index=target_index,
                                    start_attempt=local_attempt,
                                    batch_size=actual_batch_size,
                                    base_seed=int(runtime.generation.base_seed),
                                )

                                async def _await_attempt(executor_future: asyncio.Future, attempt_no: int) -> Tuple[int, camo.CandidateResult, camo.ValidationOutcome]:
                                    candidate, outcome = await executor_future
                                    return int(attempt_no), candidate, outcome

                                tasks: List[asyncio.Task] = []
                                for attempt_no, seed in batch:
                                    fut = loop.run_in_executor(
                                        executor,
                                        generate_and_validate_custom,
                                        int(seed),
                                        runtime,
                                        tolerance_profile,
                                    )
                                    tasks.append(asyncio.create_task(_await_attempt(fut, int(attempt_no))))

                                ordered_results: List[Tuple[int, camo.CandidateResult, camo.ValidationOutcome]] = []

                                for done in asyncio.as_completed(tasks):
                                    attempt_no, candidate, outcome = await done

                                    job.current_index = int(target_index)
                                    job.current_local_attempt = int(attempt_no)
                                    job.total_attempts += 1
                                    job.mark_attempt(target_index=target_index, accepted=bool(outcome.accepted))

                                    if outcome.accepted:
                                        tolerance_outcomes.append(True)
                                    else:
                                        tolerance_outcomes.append(False)

                                    ordered_results.append((int(attempt_no), candidate, outcome))
                                    job.last_candidate = build_last_candidate_payload(
                                        target_index=target_index,
                                        local_attempt=attempt_no,
                                        candidate=candidate,
                                        outcome=outcome,
                                        runtime=runtime,
                                    )
                                    job.add_event(
                                        "info" if outcome.accepted else "warning",
                                        "Tentative validée" if outcome.accepted else "Tentative rejetée",
                                        target_index=target_index,
                                        local_attempt=attempt_no,
                                        total_attempts=job.total_attempts,
                                        seed=int(candidate.seed),
                                        reasons=list(outcome.reasons),
                                        bestof_score=_safe_round(outcome.bestof_score, 6),
                                        tolerance_relax_before=_safe_round(previous_relax, 6),
                                        tolerance_relax_after=_safe_round(tolerance_profile.relax_level, 6),
                                    )

                                ordered_results.sort(key=lambda x: x[0])
                                accepted_item = next(((a, c, o) for a, c, o in ordered_results if o.accepted), None)

                                if accepted_item is None:
                                    local_attempt += actual_batch_size
                                    continue

                                accepted_attempt, accepted_candidate, accepted_outcome = accepted_item
                                final_img = render_final_image(accepted_candidate, runtime, target_index=target_index)
                                save_path = build_output_path(
                                    output_dir=output_dir,
                                    target_index=target_index,
                                    local_attempt=accepted_attempt,
                                    global_attempt=job.total_attempts,
                                    seed=int(accepted_candidate.seed),
                                )
                                save_image(save_path, final_img)

                                rows.append(
                                    build_row(
                                        target_index=target_index,
                                        local_attempt=accepted_attempt,
                                        global_attempt=job.total_attempts,
                                        candidate=accepted_candidate,
                                        outcome=accepted_outcome,
                                        saved_path=save_path,
                                        tolerance_profile=tolerance_profile,
                                        runtime=runtime,
                                    )
                                )
                                break

                        job.rows = rows
                        job.report_path = camo.write_report(rows, output_dir, REPORT_FILENAME)

                        summary_payload = {
                            "service": DEFAULT_SERVICE_NAME,
                            "status": "cancelled" if job.cancel_requested else "done",
                            "label": runtime.label,
                            "palette_mode": runtime.palette_mode,
                            "requested_palette": [
                                {"name": c.name, "hex": c.hex, "ratio": c.ratio} for c in runtime.requested_colors
                            ],
                            "backend_palette": [
                                {
                                    "backend_index": grp.backend_index,
                                    "anchor_hex": grp.anchor_hex,
                                    "ratio": grp.ratio,
                                    "source_colors": [
                                        {"name": c.name, "hex": c.hex, "ratio": c.ratio} for c in grp.source_colors
                                    ],
                                }
                                for grp in runtime.backend_groups
                            ],
                            "generation": _jsonable(runtime.generation.dict()),
                            "tolerances": _jsonable(runtime.tolerances.dict()),
                            "advanced_globals": _jsonable(runtime.advanced_globals),
                            "accepted_count": len(rows),
                            "rejected_count": int(job.rejected_count),
                            "total_attempts": int(job.total_attempts),
                            "report": str((output_dir / REPORT_FILENAME).resolve()),
                            "output_dir": str(output_dir.resolve()),
                            "note": (
                                "main.py reste structurellement un moteur backend à 4 classes. "
                                "Le mode extended_postprocess permet d'utiliser plus de 4 couleurs en sortie, "
                                "mais la structure générative profonde reste basée sur 4 groupes backend."
                            ),
                        }
                        job.summary_path = output_dir / SUMMARY_FILENAME
                        job.summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                job.status = "cancelled" if job.cancel_requested else "done"
                job.accepted_count = len(job.rows)
                job.rejected_count = max(0, job.total_attempts - job.accepted_count)
                job.add_event(
                    "info",
                    "Job terminé",
                    accepted_count=job.accepted_count,
                    rejected_count=job.rejected_count,
                    total_attempts=job.total_attempts,
                    status=job.status,
                )
            except Exception as exc:
                job.status = "error"
                job.error_message = f"{type(exc).__name__}: {exc}"
                job.add_event("error", "Erreur job", error=job.error_message)
            finally:
                job.ended_at = time.time()


# ============================================================
# APP FASTAPI
# ============================================================


app = FastAPI(
    title="Camouflage Armée Fédérale Europe API - avancée",
    version="2.0.0",
    description=(
        "API FastAPI avancée pour main.py : jobs, preview, palette libre, remap palette étendue, "
        "géométrie, tolérances, paramètres de génération et téléchargement des résultats."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def shutdown_event() -> None:
    try:
        camo.shutdown_process_pool()
    except Exception:
        pass


# ============================================================
# HELPERS ROUTES
# ============================================================


def get_job_or_404(job_id: str) -> JobState:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job introuvable: {job_id}")
    return job



def make_output_dir(job_id: str) -> Path:
    out = BASE_OUTPUT_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)
    return out



def list_relative_files(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    root = root.resolve()
    if not root.exists():
        return items
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        items.append(
            {
                "relative_path": rel,
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "url": rel,
            }
        )
    return items


# ============================================================
# ROUTES INFO / CONFIG
# ============================================================


@app.get("/health")
def health() -> Dict[str, Any]:
    running = sum(1 for job in jobs.values() if job.status == "running")
    queued = sum(1 for job in jobs.values() if job.status == "queued")
    return {
        "ok": True,
        "service": DEFAULT_SERVICE_NAME,
        "running_jobs": running,
        "queued_jobs": queued,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "cpu_count": camo.CPU_COUNT,
        "default_max_workers": camo.DEFAULT_MAX_WORKERS,
        "default_attempt_batch_size": camo.DEFAULT_ATTEMPT_BATCH_SIZE,
        "engine_limits": {
            "backend_classes": 4,
            "strict_backend_requires_exactly_4_colors": True,
            "extended_postprocess_supports_output_palettes_with_more_than_4_colors": True,
        },
    }


@app.get("/config")
def config() -> Dict[str, Any]:
    return {
        "defaults": {
            "generation": GenerationSettings().dict(),
            "tolerances": ToleranceSettings().dict(),
            "colors": [
                {"name": f"class_{idx}", "hex": _rgb_to_hex(rgb), "ratio": float(ratio)}
                for idx, (rgb, ratio) in enumerate(zip(np.asarray(camo.RGB).tolist(), np.asarray(camo.TARGET).tolist()))
            ],
            "palette_mode": "strict_backend",
        },
        "capabilities": {
            "strict_backend": {
                "description": "Utilise réellement le backend 4 classes avec 4 couleurs exactes.",
                "requires_exactly_4_colors": True,
            },
            "extended_postprocess": {
                "description": (
                    "Génère en backend 4 groupes puis remappe l'image finale vers une palette étendue. "
                    "Permet d'ajouter des couleurs sans réécrire main.py."
                ),
                "supports_more_than_4_output_colors": True,
            },
            "advanced_globals_allowed": sorted(SAFE_GLOBAL_OVERRIDES.keys()),
        },
        "backend_current_state": {
            "width": int(camo.WIDTH),
            "height": int(camo.HEIGHT),
            "physical_width_cm": float(camo.PHYSICAL_WIDTH_CM),
            "physical_height_cm": float(camo.PHYSICAL_HEIGHT_CM),
            "target_ratios": np.asarray(camo.TARGET, dtype=float).tolist(),
            "rgb": np.asarray(camo.RGB, dtype=np.uint8).tolist(),
            "bestof_required": bool(camo.BESTOF_REQUIRED),
            "bestof_min_score": float(camo.BESTOF_MIN_SCORE),
        },
        "note": (
            "main.py est structurellement un moteur 4 classes. "
            "L'API permet donc soit un vrai mode backend 4 couleurs, soit un mode palette étendue en post-traitement."
        ),
    }


# ============================================================
# ROUTE PREVIEW
# ============================================================


@app.post("/preview")
async def preview(payload: PreviewRequest) -> Dict[str, Any]:
    try:
        runtime = normalize_runtime_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    preview_root = BASE_OUTPUT_DIR / PREVIEW_DIRNAME
    preview_root.mkdir(parents=True, exist_ok=True)

    async with job_semaphore:
        async with generation_state_lock:
            try:
                with patched_camo_state(runtime):
                    output_dir = preview_root
                    camo.validate_generation_request(
                        target_count=1,
                        output_dir=output_dir,
                        base_seed=int(runtime.generation.base_seed),
                        machine_intensity=float(runtime.generation.machine_intensity),
                        max_workers=int(runtime.generation.max_workers or 1),
                        attempt_batch_size=int(runtime.generation.attempt_batch_size or 1),
                    )
                    tolerance_profile = build_tolerance_profile(runtime, runtime.tolerances.relax_level_override or 0.0)
                    candidate, outcome = await asyncio.to_thread(
                        generate_and_validate_custom,
                        int(runtime.generation.base_seed),
                        runtime,
                        tolerance_profile,
                    )
                    final_img = render_final_image(candidate, runtime, target_index=1)

                    save_path: Optional[Path] = None
                    if payload.save_preview:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        save_path = preview_root / f"preview_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                        save_image(save_path, final_img)

                        old_previews = sorted(preview_root.glob("preview_*.png"))
                        if len(old_previews) > MAX_SAVED_PREVIEWS:
                            for old in old_previews[: len(old_previews) - MAX_SAVED_PREVIEWS]:
                                old.unlink(missing_ok=True)

                    bio = io.BytesIO()
                    final_img.save(bio, format="PNG")
                    size_bytes = bio.tell()

                    return {
                        "accepted": bool(outcome.accepted),
                        "reasons": list(outcome.reasons),
                        "bestof_score": float(outcome.bestof_score),
                        "metrics": {k: _safe_round(v, 6) for k, v in candidate.metrics.items()},
                        "ratios_backend": [_safe_round(x, 6) for x in candidate.ratios.tolist()],
                        "requested_palette": [
                            {"name": c.name, "hex": c.hex, "ratio": c.ratio} for c in runtime.requested_colors
                        ],
                        "backend_palette": [
                            {
                                "backend_index": grp.backend_index,
                                "anchor_hex": grp.anchor_hex,
                                "ratio": grp.ratio,
                                "sources": [
                                    {"name": c.name, "hex": c.hex, "ratio": c.ratio} for c in grp.source_colors
                                ],
                            }
                            for grp in runtime.backend_groups
                        ],
                        "saved_preview_path": str(save_path) if save_path else None,
                        "saved_preview_url": (
                            f"/preview/files/{save_path.name}" if save_path is not None else None
                        ),
                        "image_size_bytes": int(size_bytes),
                    }
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Erreur preview: {type(exc).__name__}: {exc}") from exc


@app.get("/preview/files/{filename}")
def get_preview_file(filename: str) -> FileResponse:
    preview_root = (BASE_OUTPUT_DIR / PREVIEW_DIRNAME).resolve()
    path = _ensure_under(preview_root, preview_root / filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Preview introuvable")
    return FileResponse(path)


# ============================================================
# ROUTES JOBS
# ============================================================


@app.post("/jobs", response_model=JobPublic)
async def create_job(payload: GenerateRequest) -> JobPublic:
    try:
        runtime = normalize_runtime_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    output_dir = make_output_dir(job_id)
    job = JobState(
        job_id=job_id,
        runtime=runtime,
        created_at=time.time(),
        output_dir=output_dir,
    )
    job.add_event(
        "info",
        "Job créé",
        palette_mode=runtime.palette_mode,
        colors_count=len(runtime.requested_colors),
        requested_palette_hex=runtime.requested_palette_hex,
        target_count=int(runtime.generation.target_count),
    )
    jobs[job_id] = job
    job.task = asyncio.create_task(run_generation_job(job))
    return job.to_public()


@app.get("/jobs", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    ordered = sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)
    return JobListResponse(jobs=[job.to_public() for job in ordered])


@app.get("/jobs/{job_id}", response_model=JobPublic)
def get_job(job_id: str) -> JobPublic:
    return get_job_or_404(job_id).to_public()


@app.get("/jobs/{job_id}/events")
def get_job_events(job_id: str, limit: int = Query(default=80, ge=1, le=1000)) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    return {
        "job_id": job.job_id,
        "count": min(limit, len(job.recent_events)),
        "events": job.recent_events[-limit:],
    }


@app.post("/jobs/{job_id}/cancel", response_model=CancelResponse)
def cancel_job(job_id: str) -> CancelResponse:
    job = get_job_or_404(job_id)
    if job.status in {"done", "error", "cancelled"}:
        return CancelResponse(job_id=job.job_id, cancel_requested=job.cancel_requested, status=job.status)
    job.cancel_requested = True
    job.add_event("warning", "Annulation demandée")
    return CancelResponse(job_id=job.job_id, cancel_requested=True, status=job.status)


@app.get("/jobs/{job_id}/rows")
def get_rows(job_id: str) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    return {
        "job_id": job.job_id,
        "count": len(job.rows),
        "rows": job.rows,
    }


@app.get("/jobs/{job_id}/files")
def list_job_files(job_id: str) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    files = list_relative_files(job.output_dir)
    return {
        "job_id": job.job_id,
        "files": [
            {
                **meta,
                "url": f"/jobs/{job.job_id}/files/{meta['relative_path']}",
            }
            for meta in files
        ],
        "report_url": f"/jobs/{job.job_id}/report" if job.report_path and job.report_path.exists() else None,
        "summary_url": f"/jobs/{job.job_id}/summary" if job.summary_path and job.summary_path.exists() else None,
        "archive_url": f"/jobs/{job.job_id}/archive",
    }


@app.get("/jobs/{job_id}/files/{file_path:path}")
def get_job_file(job_id: str, file_path: str) -> FileResponse:
    job = get_job_or_404(job_id)
    root = job.output_dir.resolve()
    path = _ensure_under(root, root / file_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return FileResponse(path)


@app.get("/jobs/{job_id}/report")
def get_report(job_id: str) -> FileResponse:
    job = get_job_or_404(job_id)
    if job.report_path is None or not job.report_path.exists():
        raise HTTPException(status_code=404, detail="Rapport introuvable")
    return FileResponse(job.report_path)


@app.get("/jobs/{job_id}/summary")
def get_summary(job_id: str) -> FileResponse:
    job = get_job_or_404(job_id)
    if job.summary_path is None or not job.summary_path.exists():
        raise HTTPException(status_code=404, detail="Résumé introuvable")
    return FileResponse(job.summary_path)


@app.get("/jobs/{job_id}/archive")
def get_archive(job_id: str) -> StreamingResponse:
    job = get_job_or_404(job_id)
    if not job.output_dir.exists():
        raise HTTPException(status_code=404, detail="Dossier de sortie introuvable")

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(job.output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(job.output_dir).as_posix())
    mem.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{job.job_id}.zip"'}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)


@app.delete("/jobs/{job_id}")
def forget_job(job_id: str, delete_files: bool = Query(default=False)) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    if job.status == "running":
        raise HTTPException(status_code=409, detail="Job encore en cours")

    jobs.pop(job_id, None)

    deleted_files = False
    if delete_files and job.output_dir.exists():
        shutil.rmtree(job.output_dir, ignore_errors=True)
        deleted_files = not job.output_dir.exists()

    return {"deleted": True, "job_id": job_id, "deleted_files": deleted_files}


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        f"{Path(__file__).stem}:app",
        host=os.getenv("CAMO_API_HOST", "127.0.0.1"),
        port=int(os.getenv("CAMO_API_PORT", "8000")),
        reload=False,
    )
