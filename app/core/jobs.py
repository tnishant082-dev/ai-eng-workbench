from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: Any = None
    error: str | None = None


class JobQueue:
    """Lightweight in-process async job queue (no Redis required)."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    def submit_sync(self, name: str, fn: Callable[[], Any]) -> Job:
        job = Job(id=str(uuid.uuid4())[:8], name=name, status=JobStatus.RUNNING)
        self._jobs[job.id] = job
        try:
            job.result = fn()
            job.status = JobStatus.DONE
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
        job.finished_at = time.time()
        return job

    async def submit(self, name: str, coro_factory: Callable[[], Awaitable[Any]]) -> Job:
        job = Job(id=str(uuid.uuid4())[:8], name=name, status=JobStatus.PENDING)
        self._jobs[job.id] = job

        async def _run() -> None:
            job.status = JobStatus.RUNNING
            try:
                job.result = await coro_factory()
                job.status = JobStatus.DONE
            except Exception as exc:  # noqa: BLE001
                job.status = JobStatus.FAILED
                job.error = str(exc)
            job.finished_at = time.time()

        asyncio.create_task(_run())
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]
