#!/usr/bin/env python3
"""Standalone E2E test runner for HiFiSampler.

Executes E2E test suites (Tier 1 through Tier 3) against the Python baseline
or the Rust port, reporting structured pass/fail metrics grouped by feature.

Usage:
    python tests/e2e/runner.py --tier 1
    python tests/e2e/runner.py --tier 1 --target python
    python tests/e2e/runner.py --tier 1 --feature F12
    python tests/e2e/runner.py --help
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pytest

FEATURE_NAMES: dict[str, str] = {
    "F1": "Branch & Toolchain Setup",
    "F2": "Audio I/O",
    "F3": "HIFIBIN1 Feature Cache",
    "F4": "BS.1770-4 LUFS Normalization",
    "F5": "Pitch & Expression Parsing",
    "F6": "Ht v3 Glottal Tension Engine",
    "F7": "Ho & Hr Klatt Resonators",
    "F8": "Vocal Fry & Growl",
    "F9": "Neural Vocoder ONNX Inference",
    "F10": "HNSEP Voice/Breath Separation",
    "F11": "Cross-Platform EP Fallback",
    "F12": "CLI Client 13 Arguments",
    "F13": "HTTP Server & Routes",
    "F14": "--managed Lifecycle & Beacon",
    "F15": "Telemetry Endpoints",
    "F16": "Multi-Tier Audio Equivalence",
    "F17": "Latency & Throughput Benchmark",
    "T3": "Cross-Feature Combinations",
    "T4": "Real-World Vocal Scenarios",
}


class FeatureSummaryPlugin:
    """Pytest plugin to capture and aggregate test outcomes per feature."""

    def __init__(self) -> None:
        self.features: dict[str, dict[str, Any]] = {}
        for fid in FEATURE_NAMES:
            self.features[fid] = {
                "name": FEATURE_NAMES[fid],
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "failures": [],
            }
        self.unmatched: dict[str, Any] = {
            "name": "General / Miscellaneous",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failures": [],
        }

    def _extract_feature_id(self, nodeid: str) -> str | None:
        """Extract feature or tier category ID from test nodeid or function name."""
        match = re.search(r"test_f(\d{1,2})_", nodeid, re.IGNORECASE)
        if match:
            return f"F{match.group(1)}"
        if "tier3" in nodeid.lower() or "comb_" in nodeid.lower() or "test_t3_" in nodeid.lower():
            return "T3"
        if "tier4" in nodeid.lower() or "scen_" in nodeid.lower() or "test_t4_" in nodeid.lower():
            return "T4"
        return None

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # Capture call phase or failure during setup/teardown
        if report.when == "call" or (report.when == "setup" and report.skipped) or (report.when != "call" and report.failed):
            fid = self._extract_feature_id(report.nodeid)
            target_dict = self.features.get(fid, self.unmatched) if fid else self.unmatched

            target_dict["total"] += 1
            if report.passed:
                target_dict["passed"] += 1
            elif report.failed:
                target_dict["failed"] += 1
                target_dict["failures"].append((report.nodeid, str(report.longrepr)))
            elif report.skipped:
                target_dict["skipped"] += 1


def parse_runner_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiFiSampler E2E Test Suite Runner")
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Test tier to run (1: foundational, 2: boundaries, 3: combinations, 4: scenarios).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tiers (Tiers 1, 2, 3, and 4).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=["python", "rust"],
        help="Target implementation: 'python' baseline or 'rust' port (auto-detects if omitted).",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Filter execution to specific feature (e.g. 'F1', 'F12').",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose test output.",
    )
    return parser.parse_args()


def run_e2e_tests() -> int:
    args = parse_runner_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    # Propagate target to environment
    if args.target:
        os.environ["HIFISAMPLER_TEST_TARGET"] = args.target

    # Ensure PYTHONPATH contains project_root
    py_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{py_path}" if py_path else str(project_root)
    )

    # Locate test target files
    tier_map = {
        1: project_root / "tests" / "e2e" / "test_tier1_features.py",
        2: project_root / "tests" / "e2e" / "test_tier2_boundaries.py",
        3: project_root / "tests" / "e2e" / "test_tier3_combinations.py",
        4: project_root / "tests" / "e2e" / "test_tier4_scenarios.py",
    }

    if args.all:
        test_files = [p for p in tier_map.values() if p.exists()]
        tier_title = "All Tiers (1 - 4)"
    else:
        tier_idx = args.tier if args.tier is not None else 1
        target_file = tier_map.get(tier_idx)
        if not target_file or not target_file.exists():
            print(f"Error: Test file for Tier {tier_idx} not found at {target_file}", file=sys.stderr)
            return 1
        test_files = [target_file]
        tier_title = f"Tier {tier_idx}"

    pytest_args = [str(p) for p in test_files] + ["-q"]
    if args.verbose:
        pytest_args.append("-v")

    if args.feature:
        feat_clean = args.feature.upper().strip()
        num_match = re.search(r"\d+", feat_clean)
        if num_match:
            filter_pattern = f"test_f{num_match.group(0)}_"
            pytest_args.extend(["-k", filter_pattern])
        else:
            pytest_args.extend(["-k", args.feature])

    plugin = FeatureSummaryPlugin()
    target_display = args.target or os.environ.get("HIFISAMPLER_TEST_TARGET", "python (auto)")

    print("=" * 88)
    print(f"HiFiSampler E2E Test Suite — {tier_title} Coverage")
    print(f"Target: {target_display} | Working Directory: {project_root}")
    print("=" * 88)

    t0 = time.time()
    ret_code = pytest.main(pytest_args, plugins=[plugin])
    elapsed = time.time() - t0

    # Print Formatted Report Table
    print("\n" + "=" * 88)
    print(f"{'Feature':<9} {'Description':<35} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Skip':>6} {'Status':>8}")
    print("-" * 88)

    total_all = 0
    passed_all = 0
    failed_all = 0
    skipped_all = 0

    # Determine which features to display
    display_features = (
        [args.feature.upper()]
        if args.feature and args.feature.upper() in plugin.features
        else list(FEATURE_NAMES.keys())
    )

    for fid in display_features:
        info = plugin.features.get(fid)
        if not info:
            continue
        tot = info["total"]
        pas = info["passed"]
        fai = info["failed"]
        skp = info["skipped"]

        total_all += tot
        passed_all += pas
        failed_all += fai
        skipped_all += skp

        status = "PASS" if fai == 0 and tot > 0 else ("FAIL" if fai > 0 else "SKIP")
        print(f"{fid:<9} {info['name']:<35} {tot:>6} {pas:>6} {fai:>6} {skp:>6} {status:>8}")

    if plugin.unmatched["total"] > 0:
        u = plugin.unmatched
        total_all += u["total"]
        passed_all += u["passed"]
        failed_all += u["failed"]
        skipped_all += u["skipped"]
        u_status = "PASS" if u["failed"] == 0 else "FAIL"
        print(f"{'OTHER':<9} {u['name']:<35} {u['total']:>6} {u['passed']:>6} {u['failed']:>6} {u['skipped']:>6} {u_status:>8}")

    print("=" * 88)
    success_rate = (passed_all / total_all * 100.0) if total_all > 0 else 0.0
    overall_status = "ALL PASS" if failed_all == 0 and total_all > 0 else "FAILURES DETECTED"
    print(
        f"{tier_title} Summary: {total_all} Total | {passed_all} Passed | "
        f"{failed_all} Failed | {skipped_all} Skipped | {elapsed:.2f}s | {overall_status} ({success_rate:.1f}%)"
    )
    print("=" * 88)

    if failed_all > 0:
        print("\nFailures Detail:")
        for fid, info in plugin.features.items():
            for node, err in info["failures"]:
                print(f"[{fid}] {node}\n{err}\n")
        return 1

    return 0 if (failed_all == 0 and total_all > 0) else ret_code


if __name__ == "__main__":
    sys.exit(run_e2e_tests())
