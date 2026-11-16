use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const DEFAULT_MAX_HISTORY: usize = 200;
const RECENT_LIMIT: usize = 20;

/// Per-request timing and cache statistics for a single resample call.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResampleStats {
    pub note_name: String,
    pub feature_ms: f64,
    pub synthesis_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
    pub cache_hit: bool,
    pub hnsep_cache_hit: bool,
    pub input_samples: usize,
    pub output_samples: usize,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: f64,
}

impl ResampleStats {
    pub fn now_unix_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn success(
        note_name: impl Into<String>,
        feature_ms: f64,
        synthesis_ms: f64,
        postprocess_ms: f64,
        total_ms: f64,
        cache_hit: bool,
        hnsep_cache_hit: bool,
        input_samples: usize,
        output_samples: usize,
    ) -> Self {
        Self {
            note_name: note_name.into(),
            feature_ms,
            synthesis_ms,
            postprocess_ms,
            total_ms,
            cache_hit,
            hnsep_cache_hit,
            input_samples,
            output_samples,
            success: true,
            error: None,
            timestamp: Self::now_unix_secs(),
        }
    }
}

/// JSON payload returned by `GET /stats`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StatsSnapshot {
    pub ready: bool,
    pub active_jobs: usize,
    pub uptime_sec: f64,
    pub total_requests: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub cache_hits: u64,
    pub cache_hit_rate: f64,
    pub hnsep_cache_hits: u64,
    pub avg_total_ms: f64,
    pub p95_total_ms: f64,
    pub avg_feature_ms: f64,
    pub avg_synthesis_ms: f64,
    pub avg_postprocess_ms: f64,
    pub recent: Vec<ResampleStats>,
}

struct StatsInner {
    history: VecDeque<ResampleStats>,
    total_requests: u64,
    success_count: u64,
    error_count: u64,
    cache_hits: u64,
    hnsep_cache_hits: u64,
    start_time: Instant,
}

/// Thread-safe bounded telemetry collector with 200-item ring buffer.
pub struct StatsCollector {
    max_history: usize,
    inner: Mutex<StatsInner>,
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_HISTORY).expect("DEFAULT_MAX_HISTORY >= 1")
    }
}

impl StatsCollector {
    pub fn new(max_history: usize) -> Result<Self, String> {
        if max_history < 1 {
            return Err("max_history must be at least 1".to_string());
        }
        Ok(Self {
            max_history,
            inner: Mutex::new(StatsInner {
                history: VecDeque::with_capacity(max_history),
                total_requests: 0,
                success_count: 0,
                error_count: 0,
                cache_hits: 0,
                hnsep_cache_hits: 0,
                start_time: Instant::now(),
            }),
        })
    }

    pub fn max_history(&self) -> usize {
        self.max_history
    }

    pub fn record(&self, stats: ResampleStats) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.total_requests += 1;
        if stats.success {
            guard.success_count += 1;
        } else {
            guard.error_count += 1;
        }
        if stats.cache_hit {
            guard.cache_hits += 1;
        }
        if stats.hnsep_cache_hit {
            guard.hnsep_cache_hits += 1;
        }
        if guard.history.len() == self.max_history {
            guard.history.pop_front();
        }
        guard.history.push_back(stats);
    }

    pub fn record_error(
        &self,
        note_name: impl Into<String>,
        error: impl Into<String>,
        total_ms: f64,
    ) {
        self.record(ResampleStats {
            note_name: note_name.into(),
            feature_ms: 0.0,
            synthesis_ms: 0.0,
            postprocess_ms: 0.0,
            total_ms,
            cache_hit: false,
            hnsep_cache_hit: false,
            input_samples: 0,
            output_samples: 0,
            success: false,
            error: Some(error.into()),
            timestamp: ResampleStats::now_unix_secs(),
        });
    }

    pub fn reset(&self) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.history.clear();
        guard.total_requests = 0;
        guard.success_count = 0;
        guard.error_count = 0;
        guard.cache_hits = 0;
        guard.hnsep_cache_hits = 0;
        guard.start_time = Instant::now();
    }

    pub fn reset_start_time(&self) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.start_time = Instant::now();
    }

    /// Generate telemetry snapshot; aggregation and sorting run outside the mutex.
    pub fn snapshot(
        &self,
        active_jobs: usize,
        ready: bool,
        uptime_override: Option<f64>,
    ) -> StatsSnapshot {
        let (
            total_requests,
            success_count,
            error_count,
            cache_hits,
            hnsep_cache_hits,
            start_time,
            history,
        ) = {
            let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            (
                guard.total_requests,
                guard.success_count,
                guard.error_count,
                guard.cache_hits,
                guard.hnsep_cache_hits,
                guard.start_time,
                guard.history.iter().cloned().collect::<Vec<_>>(),
            )
        };

        let resolved_uptime = match uptime_override {
            Some(v) => v,
            None => (start_time.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
        };
        let cache_hit_rate = if total_requests > 0 {
            cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let lat_items: Vec<&ResampleStats> = history.iter().filter(|item| item.success).collect();
        let n = lat_items.len();
        let (avg_total_ms, p95_total_ms, avg_feature_ms, avg_synthesis_ms, avg_postprocess_ms) =
            if n == 0 {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            } else {
                let sum_total: f64 = lat_items.iter().map(|i| i.total_ms).sum();
                let sum_feat: f64 = lat_items.iter().map(|i| i.feature_ms).sum();
                let sum_synth: f64 = lat_items.iter().map(|i| i.synthesis_ms).sum();
                let sum_post: f64 = lat_items.iter().map(|i| i.postprocess_ms).sum();
                let mut totals: Vec<f64> = lat_items.iter().map(|i| i.total_ms).collect();
                totals.sort_by(|a, b| a.total_cmp(b));
                (
                    sum_total / n as f64,
                    nearest_rank_percentile(&totals, 95.0),
                    sum_feat / n as f64,
                    sum_synth / n as f64,
                    sum_post / n as f64,
                )
            };

        let recent: Vec<ResampleStats> = history.into_iter().rev().take(RECENT_LIMIT).collect();

        StatsSnapshot {
            ready,
            active_jobs,
            uptime_sec: resolved_uptime,
            total_requests,
            success_count,
            error_count,
            cache_hits,
            cache_hit_rate,
            hnsep_cache_hits,
            avg_total_ms,
            p95_total_ms,
            avg_feature_ms,
            avg_synthesis_ms,
            avg_postprocess_ms,
            recent,
        }
    }
}

/// Compute nearest-rank percentile on an already-sorted slice: `idx = round(p/100 * (n - 1))`.
pub fn nearest_rank_percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_and_reset() {
        let collector = StatsCollector::default();
        let snap = collector.snapshot(2, false, Some(12.34));
        assert!(!snap.ready);
        assert_eq!(snap.active_jobs, 2);
        assert_eq!(snap.uptime_sec, 12.34);
        assert_eq!(snap.total_requests, 0);
        assert_eq!(snap.p95_total_ms, 0.0);
        assert!(snap.recent.is_empty());

        collector.record_error("B4", "boom", 5.0);
        assert_eq!(collector.snapshot(0, true, None).error_count, 1);
        collector.reset();
        assert_eq!(collector.snapshot(0, true, None).total_requests, 0);
    }

    #[test]
    fn averages_p95_and_recent_newest_first() {
        let collector = StatsCollector::new(200).unwrap();
        for i in 1..=100 {
            collector.record(ResampleStats::success(
                format!("N{i}"),
                i as f64 * 0.2,
                i as f64 * 0.6,
                i as f64 * 0.2,
                i as f64,
                i % 4 == 0,
                i % 5 == 0,
                44100,
                22050,
            ));
        }

        let snap = collector.snapshot(0, true, None);
        assert_eq!(snap.total_requests, 100);
        assert_eq!(snap.success_count, 100);
        assert_eq!(snap.cache_hits, 25);
        assert_eq!(snap.hnsep_cache_hits, 20);
        assert!((snap.cache_hit_rate - 0.25).abs() < 1e-9);
        assert!((snap.avg_total_ms - 50.5).abs() < 1e-9);
        assert!((snap.p95_total_ms - 95.0).abs() < 1e-9);
        assert_eq!(snap.recent.len(), 20);
        assert_eq!(snap.recent[0].note_name, "N100");
        assert_eq!(snap.recent[19].note_name, "N81");
    }

    #[test]
    fn errors_do_not_skew_latency_and_ring_buffer_bounds() {
        let collector = StatsCollector::new(5).unwrap();
        for i in 0..8 {
            collector.record(ResampleStats::success(
                format!("S{i}"),
                1.0,
                8.0,
                1.0,
                10.0,
                i < 4,
                i % 2 == 0,
                100,
                100,
            ));
        }
        collector.record_error("E0", "failed", 2.0);
        collector.record_error("E1", "failed2", 4.0);

        let snap = collector.snapshot(0, true, None);
        assert_eq!(snap.total_requests, 10);
        assert_eq!(snap.success_count, 8);
        assert_eq!(snap.error_count, 2);
        assert!((snap.avg_total_ms - 10.0).abs() < 1e-9);
        let names: Vec<&str> = snap.recent.iter().map(|r| r.note_name.as_str()).collect();
        assert_eq!(names, vec!["E1", "E0", "S7", "S6", "S5"]);
    }
}
