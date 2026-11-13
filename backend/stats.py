"""Thread-safe performance telemetry and statistics collector for HiFiSampler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Optional
from aiohttp import web


@dataclass(slots=True)
class ResampleStats:
    """Per-request timing and cache statistics for a single resample call."""

    note_name: str
    feature_ms: float = 0.0
    synthesis_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    cache_hit: bool = False
    hnsep_cache_hit: bool = False
    input_samples: int = 0
    output_samples: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats into a JSON-compatible dictionary."""
        return {
            "note_name": self.note_name,
            "feature_ms": self.feature_ms,
            "synthesis_ms": self.synthesis_ms,
            "postprocess_ms": self.postprocess_ms,
            "total_ms": self.total_ms,
            "cache_hit": self.cache_hit,
            "hnsep_cache_hit": self.hnsep_cache_hit,
            "input_samples": self.input_samples,
            "output_samples": self.output_samples,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute nearest-rank percentile on an already-sorted non-empty list."""
    if not sorted_values:
        return 0.0
    idx = int(round((p / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


class StatsCollector:
    """Thread-safe bounded collector with microsecond-scale lock hold times."""

    def __init__(self, max_history: int = 200) -> None:
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        self._max_history = int(max_history)
        self._lock = threading.Lock()
        self._history: deque[ResampleStats] = deque(maxlen=self._max_history)
        self._total_requests = 0
        self._success_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._hnsep_cache_hits = 0
        self._start_time = time.monotonic()

    @property
    def max_history(self) -> int:
        """Maximum number of recent records retained in memory."""
        return self._max_history

    def record(self, stats: ResampleStats) -> None:
        """Record a completed (or failed) resample stats entry under short lock."""
        with self._lock:
            self._total_requests += 1
            if stats.success:
                self._success_count += 1
            else:
                self._error_count += 1
            if stats.cache_hit:
                self._cache_hits += 1
            if stats.hnsep_cache_hit:
                self._hnsep_cache_hits += 1
            self._history.append(stats)

    def record_error(self, note_name: str, error: str, total_ms: float = 0.0) -> None:
        """Convenience helper to record a failed resample request."""
        entry = ResampleStats(
            note_name=note_name,
            total_ms=total_ms,
            success=False,
            error=error,
        )
        self.record(entry)

    def reset(self) -> None:
        """Clear all accumulated counters and history."""
        with self._lock:
            self._history.clear()
            self._total_requests = 0
            self._success_count = 0
            self._error_count = 0
            self._cache_hits = 0
            self._hnsep_cache_hits = 0
            self._start_time = time.monotonic()

    def reset_start_time(self) -> None:
        """Reset server uptime baseline to current monotonic clock."""
        with self._lock:
            self._start_time = time.monotonic()

    def snapshot(
        self,
        active_jobs: int = 0,
        ready: bool = True,
        uptime_sec: Optional[float] = None,
    ) -> dict[str, Any]:
        """Return aggregated metrics and recent history; heavy work runs lock-free."""
        with self._lock:
            total_requests = self._total_requests
            success_count = self._success_count
            error_count = self._error_count
            cache_hits = self._cache_hits
            hnsep_cache_hits = self._hnsep_cache_hits
            start_time = self._start_time
            history = list(self._history)

        resolved_uptime = (
            float(uptime_sec)
            if uptime_sec is not None
            else round(max(0.0, time.monotonic() - start_time), 3)
        )
        cache_hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0

        # Compute latency aggregates only over successful renders so 0ms errors do not skew averages/p95
        lat_items = [item for item in history if item.success]
        n = len(lat_items)
        if n == 0:
            avg_total_ms = 0.0
            p95_total_ms = 0.0
            avg_feature_ms = 0.0
            avg_synthesis_ms = 0.0
            avg_postprocess_ms = 0.0
        else:
            avg_total_ms = sum(item.total_ms for item in lat_items) / n
            avg_feature_ms = sum(item.feature_ms for item in lat_items) / n
            avg_synthesis_ms = sum(item.synthesis_ms for item in lat_items) / n
            avg_postprocess_ms = sum(item.postprocess_ms for item in lat_items) / n
            sorted_totals = sorted(item.total_ms for item in lat_items)
            p95_total_ms = _percentile(sorted_totals, 95.0)
        recent = [item.to_dict() for item in reversed(history[-20:])] if history else []

        return {
            "ready": bool(ready),
            "active_jobs": int(active_jobs),
            "uptime_sec": resolved_uptime,
            "total_requests": total_requests,
            "success_count": success_count,
            "error_count": error_count,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "hnsep_cache_hits": hnsep_cache_hits,
            "avg_total_ms": avg_total_ms,
            "p95_total_ms": p95_total_ms,
            "avg_feature_ms": avg_feature_ms,
            "avg_synthesis_ms": avg_synthesis_ms,
            "avg_postprocess_ms": avg_postprocess_ms,
            "recent": recent,
        }


stats_collector = StatsCollector()


def _resolve_server_state(
    active_jobs_fn: Optional[Callable[[], int]],
    ready_fn: Optional[Callable[[], bool]],
) -> tuple[int, bool]:
    active_jobs = int(active_jobs_fn()) if callable(active_jobs_fn) else 0
    ready = bool(ready_fn()) if callable(ready_fn) else True
    return active_jobs, ready


async def handle_stats(
    request: web.Request,
    collector: StatsCollector = stats_collector,
    active_jobs_fn: Optional[Callable[[], int]] = None,
    ready_fn: Optional[Callable[[], bool]] = None,
) -> web.Response:
    """GET /stats handler returning JSON performance telemetry."""
    if "Origin" in request.headers or "Sec-Fetch-Site" in request.headers:
        return web.json_response({"error": "Forbidden"}, status=403)
    active_jobs, ready = _resolve_server_state(active_jobs_fn, ready_fn)
    payload = collector.snapshot(active_jobs=active_jobs, ready=ready)
    return web.json_response(payload, status=200)


async def handle_health(
    request: web.Request,
    active_jobs_fn: Optional[Callable[[], int]] = None,
    ready_fn: Optional[Callable[[], bool]] = None,
) -> web.Response:
    """GET /health handler returning 200 when ready or 503 while initializing."""
    active_jobs, ready = _resolve_server_state(active_jobs_fn, ready_fn)
    status = 200 if ready else 503
    return web.json_response({"ready": ready, "active_jobs": active_jobs}, status=status)
