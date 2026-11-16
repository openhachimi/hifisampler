"""Helper server launcher for Phase C A/B benchmarking of Python baseline.

Overrides CONFIG.hnsep_model_path in memory (without modifying config.yaml) so
Python baseline runs against the exact same ONNX models (`pc_nsf_hifigan.../model.onnx`
and `hnsep/vr/model.onnx`) as Rust, exposing identical HTTP routes and `/stats`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="HiFiSampler Python benchmark server launcher")
    parser.add_argument("--port", type=int, required=True, help="HTTP port to bind")
    parser.add_argument(
        "--hnsep-onnx",
        action="store_true",
        help="Override CONFIG.hnsep_model_path to ./hnsep/vr/model.onnx in memory",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=0,
        help="Runtime idle timeout override in seconds",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    from config import CONFIG
    from backend import models, server

    if args.hnsep_onnx:
        CONFIG.hnsep_model_path = "./hnsep/vr/model.onnx"
    CONFIG.idle_timeout = max(0, int(args.idle_timeout))

    models.initialize_models()
    server.run(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
