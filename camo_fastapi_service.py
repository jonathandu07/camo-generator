from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
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
    best_of_dir: Optional[str]
    error_message: Optional[str]
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
    output_dir: Path
    report_path: Optional[Path] = None
    best_of_dir: Optional[Path] = None

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
            best_of_dir=str(self.best_of_dir) if self.best_of_dir else None,
            error_message=self.error_message,
            last_candidate=self.last_candidate,
            recent_events=[JobEvent(**evt) for evt in self.recent_events[-50:]],
        )


jobs: Dict[str, JobState] = {}


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Camouflage Armée Fédérale Europe API",
    version="1.0.0",
    description="Service FastAPI pour générer des camouflages via main.py et les exposer à une interface Vue 3.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


async def run_generation_job(job: JobState) -> None:
    async with job_semaphore:
        job.status = "running"
        job.started_at = time.time()
        job.add_event("info", "Job démarré", target_count=job.target_count)

        async def progress_callback(
            target_index: int,
            local_attempt: int,
            total_attempts: int,
            target_count: int,
            candidate: camo.CandidateResult,
            accepted: bool,
        ) -> None:
            job.current_index = target_index
            job.current_local_attempt = local_attempt
            job.total_attempts = total_attempts

            if accepted:
                job.accepted_count += 1
                verdict = "accepted"
            else:
                job.rejected_count += 1
                verdict = "rejected"

            ratios = candidate.ratios.tolist() if hasattr(candidate.ratios, "tolist") else list(candidate.ratios)
            job.last_candidate = {
                "seed": candidate.seed,
                "target_index": target_index,
                "local_attempt": local_attempt,
                "accepted": accepted,
                "ratios": [round(float(x), 6) for x in ratios],
                "metrics": {k: round(float(v), 6) for k, v in candidate.metrics.items()},
            }
            job.add_event(
                "info" if accepted else "warning",
                f"Tentative {verdict}",
                target_index=target_index,
                local_attempt=local_attempt,
                total_attempts=total_attempts,
                seed=candidate.seed,
            )

        async def stop_requested() -> bool:
            return job.cancel_requested

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
            )

            job.report_path = job.output_dir / "rapport_camouflages.csv"
            best_of_dir = job.output_dir / "best_of"
            job.best_of_dir = best_of_dir if best_of_dir.exists() else None
            job.status = "cancelled" if job.cancel_requested else "done"
            job.add_event(
                "info",
                "Job terminé",
                accepted_count=job.accepted_count,
                rejected_count=job.rejected_count,
                total_attempts=job.total_attempts,
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
        "output_root": str(BASE_OUTPUT_DIR),
    }


@app.post("/jobs", response_model=JobPublic)
async def create_job(payload: GenerateRequest) -> JobPublic:
    job_id = uuid.uuid4().hex
    output_dir = make_output_dir(job_id)

    job = JobState(
        job_id=job_id,
        label=payload.label,
        created_at=time.time(),
        target_count=payload.target_count,
        base_seed=payload.base_seed,
        max_workers=payload.max_workers or camo.DEFAULT_MAX_WORKERS,
        attempt_batch_size=payload.attempt_batch_size or camo.DEFAULT_ATTEMPT_BATCH_SIZE,
        parallel_attempts=payload.parallel_attempts,
        output_dir=output_dir,
    )
    job.add_event(
        "info",
        "Job créé",
        max_workers=job.max_workers,
        attempt_batch_size=job.attempt_batch_size,
        parallel_attempts=job.parallel_attempts,
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
    images = sorted(job.output_dir.glob("camouflage_*.png"))
    return {
        "job_id": job.job_id,
        "images": [
            {
                "name": p.name,
                "url": f"/jobs/{job.job_id}/images/{p.name}",
            }
            for p in images
        ],
        "report_url": f"/jobs/{job.job_id}/report" if job.report_path and job.report_path.exists() else None,
    }


@app.get("/jobs/{job_id}/images/{filename}")
def get_generated_image(job_id: str, filename: str) -> FileResponse:
    job = get_job_or_404(job_id)
    path = (job.output_dir / filename).resolve()
    if not path.exists() or path.parent != job.output_dir.resolve():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(path)


@app.get("/jobs/{job_id}/report")
def get_report(job_id: str) -> FileResponse:
    job = get_job_or_404(job_id)
    if job.report_path is None or not job.report_path.exists():
        raise HTTPException(status_code=404, detail="Rapport introuvable")
    return FileResponse(job.report_path)


@app.delete("/jobs/{job_id}")
def forget_job(job_id: str) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    if job.status == "running":
        raise HTTPException(status_code=409, detail="Job encore en cours")
    jobs.pop(job_id, None)
    return {"deleted": True, "job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "camo_fastapi_service:app",
        host=os.getenv("CAMO_API_HOST", "127.0.0.1"),
        port=int(os.getenv("CAMO_API_PORT", "8000")),
        reload=False,
    )
