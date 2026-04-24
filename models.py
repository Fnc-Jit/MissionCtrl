"""Shared API schemas for HF Space observability endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class BuildMetadata(BaseModel):
    service: str = "missionctrl"
    version: str = "1.0.0"
    container_id: str = "unknown"
    build_id: str = "unknown"
    git_sha: str = "unknown"
    runtime: str = "huggingface-space"
    started_at: datetime


class HeartbeatResponse(BaseModel):
    status: str = "ok"
    service: str = "missionctrl"
    version: str = "1.0.0"
    container_id: str = "unknown"
    build_id: str = "unknown"
    git_sha: str = "unknown"
    runtime: str = "huggingface-space"
    host: str = "0.0.0.0"
    port: int = 8000
    uptime_seconds: float = 0.0
    timestamp_utc: datetime
    details: Dict[str, Any] = Field(default_factory=dict)


class RequestLogEntry(BaseModel):
    timestamp: datetime
    method: str
    path: str
    status_code: int
    duration_ms: float
    container_id: str = "unknown"


class StateResponse(BaseModel):
    status: str = "ok"
    build: BuildMetadata
    observation: Dict[str, object]


class LogsSummaryResponse(BaseModel):
    status: str = "ok"
    build: BuildMetadata
    totals: Dict[str, int] = Field(default_factory=dict)
    statuses: Dict[str, int] = Field(default_factory=dict)
    paths: Dict[str, int] = Field(default_factory=dict)
    entries: List[RequestLogEntry] = Field(default_factory=list)
