from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import main as camo


# ============================================================
# CONFIG
# ============================================================

BASE_OUTPUT_DIR = Path(os.getenv("CAMO_API_OUTPUT_DIR", "service_outputs")).resolve()
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS_RAW = os.getenv("CAMO_API_CORS_ORIGINS", "*")
CORS_ORIGINS = [x.strip() for x in CORS_ORIGINS_RAW.split(",") if x.strip()] or ["*"]
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("CAMO_API_MAX_CONCURRENT_JOBS", "2")))
RECENT_EVENT_LIMIT = max(10, int(os.getenv("CAMO_API_RECENT_EVENT_LIMIT", "300")))
JOB_PUBLIC_EVENT_LIMIT = max(10, int(os.getenv("CAMO_API_PUBLIC_EVENT_LIMIT", "50")))

job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


# ============================================================
# Pydantic models
# ============================================================

class GenerateRequest(BaseModel):
    target_count: int = Field(default=10, ge=1, le=500)
    base_seed: int = Field(default=camo.DEFAULT_BASE_SEED)
    max_workers: Optional[int] = Field(default=None, ge=1)
    attempt_batch_size: Optional[int] = Field(default=None, ge=1)
    parallel_attempts: bool = True
    adaptive_rejection_correction: bool = True
    label: Optional[str] = Field(default=None, max_length=120)


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
    error_message: Optional[str]
    parallel_attempts: bool
    adaptive_rejection_correction: bool
    max_workers: int
    attempt_batch_size: int
    last_candidate: Optional[Dict[str, Any]]
    recent_events: List[JobEvent]


class JobListResponse(BaseModel):
    jobs: List[JobPublic]


class CancelResponse(BaseModel):
    job_id: str
    cancel_requested: bool
    status: str


# ============================================================
# Internal structures
# ============================================================

def _safe_round(value: Any, digits: int = 6) -> Any:
    try:
        return round(float(value), digits)
    except Exception:
        return value


@dataclass
class JobState:
    job_id: str
    label: Optional[str]
    created_at: float
    target_count: int
    base_seed: int
    max_workers: int
    attempt_batch_size: int
    parallel_attempts: bool
    adaptive_rejection_correction: bool
    output_dir: Path
    report_path: Optional[Path] = None

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
                "payload": payload,
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
        progress_ratio = (self.accepted_count / self.target_count) if self.target_count else 0.0
        return JobPublic(
            job_id=self.job_id,
            label=self.label,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            ended_at=self.ended_at,
            target_count=self.target_count,
            accepted_count=self.accepted_count,
            rejected_count=self.rejected_count,
            total_attempts=self.total_attempts,
            current_index=self.current_index,
            current_local_attempt=self.current_local_attempt,
            progress_ratio=progress_ratio,
            output_dir=str(self.output_dir),
            report_path=str(self.report_path) if self.report_path else None,
            error_message=self.error_message,
            parallel_attempts=self.parallel_attempts,
            adaptive_rejection_correction=self.adaptive_rejection_correction,
            max_workers=self.max_workers,
            attempt_batch_size=self.attempt_batch_size,
            last_candidate=self.last_candidate,
            recent_events=[JobEvent(**evt) for evt in self.recent_events[-JOB_PUBLIC_EVENT_LIMIT:]],
        )


jobs: Dict[str, JobState] = {}


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Camouflage Armée Fédérale Europe API",
    version="1.1.1",
    description="Service FastAPI pour générer des camouflages via main.py et les exposer à une interface Vue 3.",
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
# Helpers
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


def normalize_attempt_batch_size(max_workers: int, attempt_batch_size: Optional[int]) -> int:
    if attempt_batch_size is not None:
        return max(1, int(attempt_batch_size))
    return max(1, int(max_workers))


def build_last_candidate_payload(
    target_index: int,
    local_attempt: int,
    accepted: bool,
    candidate: camo.CandidateResult,
) -> Dict[str, Any]:
    ratios = candidate.ratios.tolist() if hasattr(candidate.ratios, "tolist") else list(candidate.ratios)
    return {
        "seed": int(candidate.seed),
        "target_index": int(target_index),
        "local_attempt": int(local_attempt),
        "accepted": bool(accepted),
        "ratios": [_safe_round(x, 6) for x in ratios],
        "metrics": {k: _safe_round(v, 6) for k, v in candidate.metrics.items()},
    }


async def run_generation_job(job: JobState) -> None:
    async with job_semaphore:
        if job.cancel_requested:
            job.status = "cancelled"
            job.started_at = time.time()
            job.ended_at = job.started_at
            job.report_path = job.output_dir / "rapport_camouflages.csv"
            if not job.report_path.exists():
                job.report_path.write_text("", encoding="utf-8")
            job.add_event("warning", "Job annulé avant démarrage")
            return

        job.status = "running"
        job.started_at = time.time()
        job.add_event(
            "info",
            "Job démarré",
            target_count=job.target_count,
            max_workers=job.max_workers,
            attempt_batch_size=job.attempt_batch_size,
            parallel_attempts=job.parallel_attempts,
            adaptive_rejection_correction=job.adaptive_rejection_correction,
        )
        if job.adaptive_rejection_correction:
            job.add_event(
                "info",
                "Champ adaptive_rejection_correction reçu mais ignoré : main.py ne l'expose pas.",
            )

        async def progress_callback(
            target_index: int,
            local_attempt: int,
            total_attempts: int,
            target_count: int,
            candidate: camo.CandidateResult,
            accepted: bool,
        ) -> None:
            job.current_index = int(target_index)
            job.current_local_attempt = int(local_attempt)
            job.total_attempts = int(total_attempts)
            job.mark_attempt(target_index=target_index, accepted=accepted)

            verdict = "accepted" if accepted else "rejected"
            job.last_candidate = build_last_candidate_payload(
                target_index=target_index,
                local_attempt=local_attempt,
                accepted=accepted,
                candidate=candidate,
            )
            job.add_event(
                "info" if accepted else "warning",
                f"Tentative {verdict}",
                target_index=target_index,
                local_attempt=local_attempt,
                total_attempts=total_attempts,
                accepted_count=job.accepted_count,
                rejected_count=job.rejected_count,
                seed=int(candidate.seed),
            )

        async def stop_requested() -> bool:
            return bool(job.cancel_requested)

        try:
            job.rows = await camo.async_generate_all(
                target_count=job.target_count,
                output_dir=job.output_dir,
                base_seed=job.base_seed,
                progress_callback=progress_callback,
                stop_requested=stop_requested,
                max_workers=job.max_workers,
                attempt_batch_size=job.attempt_batch_size,
                parallel_attempts=job.parallel_attempts,
                live_console=False,
            )

            job.report_path = job.output_dir / "rapport_camouflages.csv"
            job.accepted_count = len(job.rows)
            job.rejected_count = max(0, job.total_attempts - job.accepted_count)
            job.status = "cancelled" if job.cancel_requested else "done"
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
# Routes
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    running = sum(1 for job in jobs.values() if job.status == "running")
    queued = sum(1 for job in jobs.values() if job.status == "queued")
    return {
        "ok": True,
        "service": "camo-fastapi-service",
        "running_jobs": running,
        "queued_jobs": queued,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "cpu_count": camo.CPU_COUNT,
        "default_max_workers": camo.DEFAULT_MAX_WORKERS,
        "default_attempt_batch_size": camo.DEFAULT_ATTEMPT_BATCH_SIZE,
    }


@app.get("/config")
def config() -> Dict[str, Any]:
    return {
        "width": camo.WIDTH,
        "height": camo.HEIGHT,
        "target_ratios": camo.TARGET.tolist(),
        "rgb": camo.RGB.tolist(),
        "default_target_count": 10,
        "default_max_workers": camo.DEFAULT_MAX_WORKERS,
        "default_attempt_batch_size": camo.DEFAULT_ATTEMPT_BATCH_SIZE,
        "default_parallel_attempts": True,
        "default_adaptive_rejection_correction": True,
        "output_root": str(BASE_OUTPUT_DIR),
    }


@app.post("/jobs", response_model=JobPublic)
async def create_job(payload: GenerateRequest) -> JobPublic:
    job_id = uuid.uuid4().hex
    output_dir = make_output_dir(job_id)

    resolved_max_workers = int(payload.max_workers or camo.DEFAULT_MAX_WORKERS)
    resolved_attempt_batch_size = normalize_attempt_batch_size(
        max_workers=resolved_max_workers,
        attempt_batch_size=payload.attempt_batch_size,
    )

    job = JobState(
        job_id=job_id,
        label=payload.label,
        created_at=time.time(),
        target_count=payload.target_count,
        base_seed=payload.base_seed,
        max_workers=resolved_max_workers,
        attempt_batch_size=resolved_attempt_batch_size,
        parallel_attempts=payload.parallel_attempts,
        adaptive_rejection_correction=payload.adaptive_rejection_correction,
        output_dir=output_dir,
    )
    job.add_event(
        "info",
        "Job créé",
        max_workers=job.max_workers,
        attempt_batch_size=job.attempt_batch_size,
        parallel_attempts=job.parallel_attempts,
        adaptive_rejection_correction=job.adaptive_rejection_correction,
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
def get_job_events(job_id: str, limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
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
        return CancelResponse(
            job_id=job.job_id,
            cancel_requested=job.cancel_requested,
            status=job.status,
        )
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
    output_root = job.output_dir.resolve()
    images = sorted(output_root.glob("camouflage_*.png"))
    return {
        "job_id": job.job_id,
        "images": [
            {
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "url": f"/jobs/{job.job_id}/images/{p.name}",
            }
            for p in images
        ],
        "report_url": f"/jobs/{job.job_id}/report"
        if job.report_path and job.report_path.exists()
        else None,
    }


@app.get("/jobs/{job_id}/images/{filename}")
def get_generated_image(job_id: str, filename: str) -> FileResponse:
    job = get_job_or_404(job_id)
    output_root = job.output_dir.resolve()
    path = (output_root / filename).resolve()
    if not path.exists() or path.parent != output_root:
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(path)


@app.get("/jobs/{job_id}/report")
def get_report(job_id: str) -> FileResponse:
    job = get_job_or_404(job_id)
    if job.report_path is None or not job.report_path.exists():
        raise HTTPException(status_code=404, detail="Rapport introuvable")
    return FileResponse(job.report_path)


@app.delete("/jobs/{job_id}")
def forget_job(job_id: str, delete_files: bool = Query(default=False)) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    if job.status == "running":
        raise HTTPException(status_code=409, detail="Job encore en cours")

    jobs.pop(job_id, None)

    deleted_files = False
    if delete_files and job.output_dir.exists():
        for path in sorted(job.output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            job.output_dir.rmdir()
        except OSError:
            pass
        deleted_files = not job.output_dir.exists()

    return {"deleted": True, "job_id": job_id, "deleted_files": deleted_files}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        f"{Path(__file__).stem}:app",
        host=os.getenv("CAMO_API_HOST", "127.0.0.1"),
        port=int(os.getenv("CAMO_API_PORT", "8000")),
        reload=False,
    )
