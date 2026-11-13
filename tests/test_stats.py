from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from backend.stats import (
    ResampleStats,
    StatsCollector,
    handle_health,
    handle_stats,
    stats_collector,
)


class TestResampleStats:
    def test_slots_and_to_dict(self):
        entry = ResampleStats(
            note_name="C4",
            feature_ms=12.5,
            synthesis_ms=25.0,
            postprocess_ms=2.5,
            total_ms=40.0,
            cache_hit=True,
            hnsep_cache_hit=False,
            input_samples=44100,
            output_samples=44100,
            success=True,
            error=None,
            timestamp=1700000000.0,
        )
        assert not hasattr(entry, "__dict__")
        d = entry.to_dict()
        assert d == {
            "note_name": "C4",
            "feature_ms": 12.5,
            "synthesis_ms": 25.0,
            "postprocess_ms": 2.5,
            "total_ms": 40.0,
            "cache_hit": True,
            "hnsep_cache_hit": False,
            "input_samples": 44100,
            "output_samples": 44100,
            "success": True,
            "error": None,
            "timestamp": 1700000000.0,
        }


class TestStatsCollector:
    def test_invalid_max_history(self):
        with pytest.raises(ValueError):
            StatsCollector(max_history=0)

    def test_empty_snapshot_and_reset(self):
        collector = StatsCollector()
        snap = collector.snapshot(active_jobs=2, ready=False, uptime_sec=12.34)
        assert snap["ready"] is False
        assert snap["active_jobs"] == 2
        assert snap["uptime_sec"] == 12.34
        assert snap["total_requests"] == 0
        assert snap["success_count"] == 0
        assert snap["error_count"] == 0
        assert snap["cache_hits"] == 0
        assert snap["cache_hit_rate"] == 0.0
        assert snap["hnsep_cache_hits"] == 0
        assert snap["avg_total_ms"] == 0.0
        assert snap["p95_total_ms"] == 0.0
        assert snap["avg_feature_ms"] == 0.0
        assert snap["avg_synthesis_ms"] == 0.0
        assert snap["avg_postprocess_ms"] == 0.0
        assert snap["recent"] == []

        collector.record(ResampleStats(note_name="A4", total_ms=10.0))
        collector.record_error("B4", "boom", total_ms=5.0)
        assert collector.snapshot()["total_requests"] == 2
        collector.reset()
        after = collector.snapshot()
        assert after["total_requests"] == 0
        assert after["success_count"] == 0
        assert after["error_count"] == 0
        assert after["recent"] == []

    def test_stage_averages_and_p95_accuracy(self):
        collector = StatsCollector(max_history=200)
        # 100 entries with total_ms = 1.0 .. 100.0
        for i in range(1, 101):
            collector.record(
                ResampleStats(
                    note_name=f"N{i}",
                    feature_ms=float(i) * 0.2,
                    synthesis_ms=float(i) * 0.6,
                    postprocess_ms=float(i) * 0.2,
                    total_ms=float(i),
                    cache_hit=(i % 4 == 0),
                    hnsep_cache_hit=(i % 5 == 0),
                )
            )

        snap = collector.snapshot()
        assert snap["total_requests"] == 100
        assert snap["success_count"] == 100
        assert snap["error_count"] == 0
        assert snap["cache_hits"] == 25
        assert snap["hnsep_cache_hits"] == 20
        assert snap["cache_hit_rate"] == pytest.approx(0.25)

        assert snap["avg_total_ms"] == pytest.approx(50.5)
        assert snap["avg_feature_ms"] == pytest.approx(10.1)
        assert snap["avg_synthesis_ms"] == pytest.approx(30.3)
        assert snap["avg_postprocess_ms"] == pytest.approx(10.1)
        # nearest-rank index round(0.95 * 99) = 94 -> value 95.0
        assert snap["p95_total_ms"] == pytest.approx(95.0)

        # recent returns latest 20 entries, newest first
        assert len(snap["recent"]) == 20
        assert snap["recent"][0]["note_name"] == "N100"
        assert snap["recent"][-1]["note_name"] == "N81"

    def test_cache_hit_rate_with_errors_and_bounded_history(self):
        collector = StatsCollector(max_history=5)
        for i in range(8):
            collector.record(
                ResampleStats(
                    note_name=f"S{i}",
                    total_ms=10.0,
                    cache_hit=(i < 4),
                    hnsep_cache_hit=(i % 2 == 0),
                )
            )
        collector.record_error("E0", "failed", total_ms=2.0)
        collector.record_error("E1", "failed2", total_ms=4.0)

        snap = collector.snapshot()
        assert snap["total_requests"] == 10
        assert snap["success_count"] == 8
        assert snap["error_count"] == 2
        assert snap["cache_hits"] == 4
        assert snap["hnsep_cache_hits"] == 4
        assert snap["cache_hit_rate"] == pytest.approx(0.4)
        # History bounded to last 5 items: S6, S7, E0, E1... wait: S5, S6, S7, E0, E1
        assert len(snap["recent"]) == 5
        assert [r["note_name"] for r in snap["recent"]] == ["E1", "E0", "S7", "S6", "S5"]

    def test_multithreaded_concurrent_record_accuracy(self):
        max_hist = 50
        collector = StatsCollector(max_history=max_hist)
        num_threads = 8
        per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker(tid: int):
            barrier.wait()
            for i in range(per_thread):
                if i % 10 == 0:
                    collector.record_error(f"t{tid}_{i}", "err", total_ms=1.0)
                else:
                    collector.record(
                        ResampleStats(
                            note_name=f"t{tid}_{i}",
                            feature_ms=2.0,
                            synthesis_ms=5.0,
                            postprocess_ms=1.0,
                            total_ms=8.0,
                            cache_hit=(i % 2 == 0),
                            hnsep_cache_hit=(i % 5 == 0),
                        )
                    )

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker, t) for t in range(num_threads)]
            for f in futures:
                f.result()

        snap = collector.snapshot()
        assert snap["total_requests"] == num_threads * per_thread
        assert snap["error_count"] == num_threads * 10
        assert snap["success_count"] == num_threads * 90
        assert len(collector._history) == max_hist
        assert len(snap["recent"]) == 20


class TestStatsAiohttpHandlers:
    @pytest.mark.asyncio
    async def test_handle_stats_endpoint(self):
        local_collector = StatsCollector(max_history=100)
        local_collector.record(
            ResampleStats(
                note_name="D4",
                feature_ms=5.0,
                synthesis_ms=15.0,
                postprocess_ms=5.0,
                total_ms=25.0,
                cache_hit=True,
                hnsep_cache_hit=True,
            )
        )
        state = {"active": 3, "ready": True}

        app = web.Application()
        app.router.add_get(
            "/stats",
            partial(
                handle_stats,
                collector=local_collector,
                active_jobs_fn=lambda: state["active"],
                ready_fn=lambda: state["ready"],
            ),
        )

        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/stats")
            assert resp.status == 200
            data = await resp.json()
            assert data["ready"] is True
            assert data["active_jobs"] == 3
            assert data["total_requests"] == 1
            assert data["success_count"] == 1
            assert data["error_count"] == 0
            assert data["cache_hits"] == 1
            assert data["hnsep_cache_hits"] == 1
            assert data["cache_hit_rate"] == pytest.approx(1.0)
            assert data["avg_total_ms"] == pytest.approx(25.0)
            assert data["p95_total_ms"] == pytest.approx(25.0)
            assert len(data["recent"]) == 1
            assert data["recent"][0]["note_name"] == "D4"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_stats_default_singleton(self):
        stats_collector.reset()
        stats_collector.record(ResampleStats(note_name="E4", total_ms=12.0))

        app = web.Application()
        app.router.add_get("/stats", handle_stats)

        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/stats")
            assert resp.status == 200
            data = await resp.json()
            assert data["total_requests"] == 1
            assert data["ready"] is True
            assert data["active_jobs"] == 0
        finally:
            stats_collector.reset()
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_health_ready_200_and_not_ready_503(self):
        state = {"ready": False, "active": 0}
        app = web.Application()
        app.router.add_get(
            "/health",
            partial(
                handle_health,
                active_jobs_fn=lambda: state["active"],
                ready_fn=lambda: state["ready"],
            ),
        )

        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp_503 = await client.get("/health")
            assert resp_503.status == 503
            body_503 = await resp_503.json()
            assert body_503 == {"ready": False, "active_jobs": 0}

            state["ready"] = True
            state["active"] = 4
            resp_200 = await client.get("/health")
            assert resp_200.status == 200
            body_200 = await resp_200.json()
            assert body_200 == {"ready": True, "active_jobs": 4}
        finally:
            await client.close()
