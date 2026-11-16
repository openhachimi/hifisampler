# HiFiSampler E2E Testing Infrastructure Specification (`TEST_INFRA.md`)

## 1. Overview & Architecture

The HiFiSampler End-to-End (E2E) Test Suite provides an independent, opaque-box verification framework for testing both the Python reference implementation and the Rust port (`hifisampler-rs`).

The testing methodology treats the synthesis engine strictly as a black box:
- **Zero internal white-box coupling**: Tests never depend on private variables, internal crate structures, or non-public APIs.
- **Strict contract verification**: All assertions validate observed behavior at public system boundaries:
  1. **UTAU 13-Positional CLI Argument Protocol**: `hifisampler <in> <out> <pitch> <vel> [flags] [offset] [length] [consonant] [cutoff] [volume] [mod] [tempo] [pitchbend]`
  2. **HTTP Server API & Managed Lifecycle**: `GET /` readiness, `POST /` job dispatch, `GET /health`, `GET /stats`, locator beacon `~/.hifisampler/server.json`, single-instance file locks, and `--managed` idle auto-shutdown.
  3. **Binary Data Formats**: 44.1 kHz mono/stereo WAV files and HIFIBIN1 `<8sIIId` (28-byte packed header) feature cache files.
  4. **DSP & Acoustic Correctness**: BS.1770-4 LUFS integrated loudness, Fant LF glottal tension ($H_t$ v3), Klatt resonators ($H_o$, $H_r$), Vocal Fry creaky voice, and Growl 80 Hz square LFO.

---

## 2. Directory Layout & Module Structure

```text
tests/e2e/
├── harness.py                 # E2EHarness core class, process supervision, HTTP client, binary parser
├── runner.py                  # Standalone CLI test runner with per-feature metric aggregation
├── test_tier1_features.py     # Tier 1: 85 Foundational Feature Coverage Tests (F1..F17, 5 per feature)
├── test_tier2_boundaries.py   # Tier 2: 85 Boundary & Corner Cases Tests (F1..F17, 5 per feature)
├── test_tier3_combinations.py # Tier 3: 17 Pairwise Cross-Feature Integration Tests (T3)
└── test_tier4_scenarios.py    # Tier 4: 9 Real-World OpenUtau Vocal Render Scenarios (T4)
```

### Module Responsibilities

| Module | Scope | Test Count | Description |
|---|---|---|---|
| `harness.py` | Fixture / Driver | — | Provides `E2EHarness` with deadlock-safe background process management, port allocation, HTTP GET/POST wrappers, binary cache header inspection, and synthetic WAV generators. |
| `runner.py` | CLI Entrypoint | — | Standalone test runner supporting `--tier [1..4]`, `--all`, `--feature <ID>`, and `--target [python\|rust]`. Generates tabular reporting grouped by feature. |
| `test_tier1_features.py` | Tier 1 | 85 | Tests the 17 core features from `PROJECT.md § Feature Inventory` under nominal, canonical conditions. |
| `test_tier2_boundaries.py` | Tier 2 | 85 | Tests the 17 core features under extreme conditions: empty/zero-byte files, corrupted headers, boundary pitchbends, invalid arguments, security headers, and EP fallback. |
| `test_tier3_combinations.py`| Tier 3 | 17 | Tests pairwise and triple feature combinations (e.g. Ht v3 + Vocal Fry, Ht v3 + Growl, Ho + Hr + Ht tension, LUFS + pitchbend, null sink + cache pre-warm). |
| `test_tier4_scenarios.py`   | Tier 4 | 9 | Tests realistic end-to-end OpenUtau multi-note workloads: 8-note scales, expressive ballad phrases, staccato notes, multi-pitch vibrato, rock growl, and batch render sessions. |

---

## 3. Concurrency & Windows Safety Controls

Windows processes require specific safety measures to prevent test deadlocks:

1. **Subprocess Pipe Deadlock Prevention**:
   - In `harness.py`, background server processes (`managed_server`) redirect `stdout` and `stderr` to isolated temporary log files on disk (`tempfile.gettempdir() / hifi_server_<port>_<pid>_<ts>.log`).
   - Standard `subprocess.PIPE` has an OS buffer limit (~4 KB). Heavy ONNX or HTTP access logs fill this buffer rapidly, causing child processes to block permanently on `write()` syscalls. Disk file logging completely eliminates pipe deadlocks.
2. **Reliable Process Tree Cleanup**:
   - `_terminate_process` executes `taskkill /F /T /PID <pid>` on Windows to recursively kill spawned child processes, worker pools, and detached runners before removing log files.
3. **Lock & Port Contention Avoidance**:
   - `managed_server` checks for existing healthy servers on the target port before attempting to spawn a new process. If a healthy server is already running, it attaches immediately to avoid `FileLock` timeouts.
   - For ephemeral test servers, `find_free_port()` binds to `127.0.0.1:0` to guarantee conflict-free port selection.

---

## 4. Test Runner Usage Guide

The standalone runner `tests/e2e/runner.py` is invoked with the virtual environment Python interpreter:

### Execute Entire Test Suite (196 Tests)
```powershell
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --all
```

### Execute Specific Test Tiers
```powershell
# Tier 1: Foundational Feature Coverage (85 tests)
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --tier 1

# Tier 2: Boundary & Corner Cases (85 tests)
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --tier 2

# Tier 3: Cross-Feature Combinations (17 tests)
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --tier 3

# Tier 4: Real-World Vocal Scenarios (9 tests)
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --tier 4
```

### Target Selection & Feature Filtering
```powershell
# Target Python baseline explicitly
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --target python --tier 1

# Target Rust port (when compiled in rust/target/release/)
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --target rust --all

# Filter to specific feature across tiers
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --feature F12 --tier 1
& "C:\Project\hifi\.venv\Scripts\python.exe" tests/e2e/runner.py --feature F14 --tier 2
```

---

## 5. Verification Metrics & Quality Bar

Every test run produces a structured execution summary:
- **Pass Rate**: Must achieve **100.0%** across all executed test tiers.
- **Failures / Deadlocks**: 0 tolerated.
- **Total Test Minimum**: 196 automated tests (85 Tier 1 + 85 Tier 2 + 17 Tier 3 + 9 Tier 4).
