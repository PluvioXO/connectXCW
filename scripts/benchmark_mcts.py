"""Benchmark MCTS speed for tiny/small/full model configs.

Tests PyTorch forward pass, ONNX Runtime forward pass, and actual MCTS
simulation throughput to determine Kaggle time budget feasibility.

The "sims/2s" column is an estimate: Python tree overhead is measured via
PyTorch MCTS, then projected onto ONNX inference time.

Usage:
    python scripts/benchmark_mcts.py
    python scripts/benchmark_mcts.py --models full --sims 400
    python scripts/benchmark_mcts.py --models tiny small full --sims 400
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from src.game.board import Connect4Board  # noqa: E402
from src.mcts.search import MCTS  # noqa: E402
from src.neural_net.model import Connect4Net  # noqa: E402
from src.utils.config import MCTSConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Model configs: name → (num_blocks, num_filters)
_MODEL_CONFIGS: dict[str, tuple[int, int]] = {
    "tiny": (2, 32),
    "small": (3, 64),
    "full": (5, 128),
}

_KAGGLE_BUDGET_SECS: float = 2.0
_SIMS_GO_THRESHOLD: int = 80
_SIMS_MARGINAL_THRESHOLD: int = 40


def _build_model(num_blocks: int, num_filters: int) -> Connect4Net:
    """Create a random-weight model for benchmarking."""
    model = Connect4Net(num_blocks=num_blocks, num_filters=num_filters)
    model.eval()
    return model


def _benchmark_pytorch_forward(
    model: Connect4Net,
    num_calls: int,
    warmup: int,
) -> float:
    """Return average PyTorch forward pass time in milliseconds.

    Args:
        model: The model to benchmark.
        num_calls: Number of timed forward calls.
        warmup: Number of warmup calls (not timed).

    Returns:
        Milliseconds per forward call.
    """
    dummy = torch.zeros(1, 3, 6, 7)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        start = time.perf_counter()
        for _ in range(num_calls):
            model(dummy)
    elapsed = time.perf_counter() - start
    return (elapsed / num_calls) * 1000


def _benchmark_onnx_forward(
    model: Connect4Net,
    num_calls: int,
    warmup: int,
) -> tuple[float, float]:
    """Return (avg_ms_per_call, model_size_mb) for ONNX Runtime inference.

    Exports the model to a temp file, benchmarks, then cleans up.

    Args:
        model: The model to export and benchmark.
        num_calls: Number of timed inference calls.
        warmup: Number of warmup calls (not timed).

    Returns:
        Tuple of (milliseconds_per_call, onnx_file_size_mb).
    """
    dummy = torch.zeros(1, 3, 6, 7)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        torch.onnx.export(
            model,
            dummy,
            str(tmp_path),
            opset_version=11,
            input_names=["board_state"],
            output_names=["policy_logits", "value"],
            dynamic_axes={"board_state": {0: "batch_size"}},
        )

        size_mb = tmp_path.stat().st_size / (1024 * 1024)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(str(tmp_path), sess_options=opts)

        inp = np.zeros((1, 3, 6, 7), dtype=np.float32)
        feed = {"board_state": inp}

        for _ in range(warmup):
            session.run(None, feed)

        start = time.perf_counter()
        for _ in range(num_calls):
            session.run(None, feed)
        elapsed = time.perf_counter() - start

        ms_per_call = (elapsed / num_calls) * 1000
        return ms_per_call, size_mb

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _benchmark_mcts(model: Connect4Net, num_sims: int) -> float:
    """Run one full MCTS search from an empty board. Return elapsed seconds.

    Args:
        model: Neural network for MCTS evaluation.
        num_sims: Number of MCTS simulations.

    Returns:
        Wall-clock seconds for the complete search.
    """
    config = MCTSConfig(num_simulations=num_sims)
    mcts = MCTS(model, config)
    board = Connect4Board()
    start = time.perf_counter()
    mcts.search(board, add_dirichlet_noise=False)
    return time.perf_counter() - start


def _verdict(sims_per_budget: float) -> str:
    """Return GO / MARGINAL / NO-GO based on estimated sims in Kaggle budget."""
    if sims_per_budget >= _SIMS_GO_THRESHOLD:
        return "GO"
    if sims_per_budget >= _SIMS_MARGINAL_THRESHOLD:
        return "MARGINAL"
    return "NO-GO"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MCTS speed for Connect4 AlphaZero models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(_MODEL_CONFIGS.keys()),
        default=list(_MODEL_CONFIGS.keys()),
        help="Which model configs to benchmark (default: all three).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=400,
        help="Number of MCTS simulations to measure (default: 400).",
    )
    parser.add_argument(
        "--forward-calls",
        type=int,
        default=200,
        help="Number of forward pass calls for timing (default: 200).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup calls before timing (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmarks and print a summary table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    results: list[dict] = []

    for model_name in args.models:
        num_blocks, num_filters = _MODEL_CONFIGS[model_name]
        logger.info(
            "Benchmarking %s (%d blocks, %d filters) — %d sims ...",
            model_name,
            num_blocks,
            num_filters,
            args.sims,
        )

        model = _build_model(num_blocks, num_filters)

        pt_ms = _benchmark_pytorch_forward(model, args.forward_calls, args.warmup)
        logger.info("  PyTorch forward: %.2f ms/call", pt_ms)

        ort_ms, onnx_mb = _benchmark_onnx_forward(model, args.forward_calls, args.warmup)
        logger.info("  ONNX RT forward: %.2f ms/call (%.2f MB)", ort_ms, onnx_mb)

        mcts_elapsed = _benchmark_mcts(model, args.sims)
        sims_per_sec_pt = args.sims / mcts_elapsed
        logger.info(
            "  MCTS (%d sims): %.2f s → %.1f sims/s (PyTorch backend)",
            args.sims,
            mcts_elapsed,
            sims_per_sec_pt,
        )

        # Estimate ONNX MCTS time:
        #   python_overhead ≈ mcts_elapsed - N * pytorch_ms_per_sim
        #   est_onnx_mcts   ≈ python_overhead + N * onnx_ms_per_sim
        python_overhead = mcts_elapsed - args.sims * (pt_ms / 1000)
        python_overhead = max(0.0, python_overhead)
        est_onnx_mcts = python_overhead + args.sims * (ort_ms / 1000)

        # sims/2s: how many sims fit in Kaggle's 2s budget with ONNX
        if est_onnx_mcts > 0:
            sims_per_budget = args.sims * _KAGGLE_BUDGET_SECS / est_onnx_mcts
        else:
            sims_per_budget = 0.0

        results.append(
            {
                "name": model_name,
                "pt_ms": pt_ms,
                "ort_ms": ort_ms,
                "onnx_mb": onnx_mb,
                "sims_per_sec_pt": sims_per_sec_pt,
                "sims_per_budget": sims_per_budget,
            }
        )

    # Print summary table
    budget = _KAGGLE_BUDGET_SECS
    print()
    print("=" * 72)
    print(
        f"BENCHMARK SUMMARY  ({args.sims} MCTS sims, Kaggle budget = {budget:.1f}s)"
    )
    print("=" * 72)
    print(
        f"{'Model':<8} {'PT ms':>6} {'ORT ms':>7} {'ONNX MB':>8}"
        f" {'sims/s (PT)':>12} {'sims/2s':>8}  Verdict"
    )
    print("-" * 72)
    for r in results:
        verdict = _verdict(r["sims_per_budget"])
        print(
            f"{r['name']:<8} {r['pt_ms']:>6.2f} {r['ort_ms']:>7.2f}"
            f" {r['onnx_mb']:>8.2f} {r['sims_per_sec_pt']:>12.1f}"
            f" {r['sims_per_budget']:>8.0f}  {verdict}"
        )
    print("=" * 72)
    print("Notes:")
    print(
        "  - sims/2s = estimated (python overhead + N * onnx_ms per sim)."
    )
    print(
        "  - Quantized ONNX NOT recommended on Apple Silicon (slower, not faster)."
    )
    print(
        f"  - sims/2s >= {_SIMS_GO_THRESHOLD} = GO, "
        f">= {_SIMS_MARGINAL_THRESHOLD} = MARGINAL, "
        f"< {_SIMS_MARGINAL_THRESHOLD} = NO-GO."
    )


if __name__ == "__main__":
    main()
