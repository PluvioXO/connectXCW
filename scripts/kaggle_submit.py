"""Package a Kaggle Connect-X submission.

Modes:

  --tar (RECOMMENDED):
    Pure numpy inference, no onnxruntime needed.
    Extracts weights from a .pt checkpoint into weights.npz.
    Bundles main.py + weights.npz as submission.tar.gz.
    Kaggle extracts the archive to /kaggle_simulations/agent/ and runs main.py.

  --base64:
    ONNX-based inference, model embedded as base64 in a single main.py.
    Requires onnxruntime in the Kaggle environment — currently NOT available
    in Connect-X simulation environment.

  --zip (BROKEN for Connect-X):
    Kaggle does not extract zip archives — it tries to compile the raw zip
    bytes as Python, causing a SyntaxError. Kept for reference only.

  (default, no flag):
    Directory mode — submission.py + model.onnx for notebook submissions
    that attach the model as a Kaggle Dataset.

Usage:
    # Tar — pure numpy, no external deps (RECOMMENDED for Connect-X)
    python scripts/kaggle_submit.py --checkpoint checkpoints/baseline_b2_f32.pt \\
        --output submission/ --tar

    # Base64 (if onnxruntime were available)
    python scripts/kaggle_submit.py --model model.onnx --output submission/ --base64
"""

import argparse
import base64
import io
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_AGENT_SOURCE: Path = _PROJECT_ROOT / "src" / "export" / "kaggle_agent.py"
_NUMPY_AGENT_SOURCE: Path = _PROJECT_ROOT / "src" / "export" / "kaggle_agent_numpy.py"

# Sentinels in kaggle_agent.py (onnx-based)
_MODEL_PATH_SENTINEL: str = '_MODEL_PATH: str = "model.onnx"'
_NUM_SIMS_SENTINEL: str = "_NUM_MCTS_SIMS: int = 200"

# Sentinel in kaggle_agent_numpy.py
_NUMPY_TIME_BUDGET_SENTINEL: str = "_TIME_BUDGET_SECS: float = 2.0  # sentinel"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Package a Kaggle Connect-X submission."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tar",
        action="store_true",
        default=False,
        help=(
            "Pure numpy inference. Extracts weights from --checkpoint to weights.npz, "
            "bundles with main.py as submission.tar.gz. (RECOMMENDED)"
        ),
    )
    mode.add_argument(
        "--base64",
        action="store_true",
        default=False,
        help="Embed ONNX model as base64 in a single main.py (requires onnxruntime on Kaggle).",
    )
    mode.add_argument(
        "--zip",
        action="store_true",
        default=False,
        help="(BROKEN for Connect-X) submission.zip with main.py + model.onnx.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (required for --tar).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .onnx model file (required for --base64, --zip, default).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the submission package.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=2.0,
        help="Time budget in seconds per move (default: 2.0; Kaggle allows 2s per move).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/kaggle/input/connect4-alphazero-model/model.onnx",
        help="Model path on Kaggle (directory mode only).",
    )
    return parser.parse_args()


# -------------------------------------------------------------------------
# --tar mode: numpy inference, tar.gz archive
# -------------------------------------------------------------------------

def _extract_weights_npz(checkpoint_path: Path) -> bytes:
    """Load a .pt checkpoint and return compressed npz bytes of its weights.

    Filters out num_batches_tracked (bookkeeping only) and includes
    num_blocks and num_filters so the inference code can reconstruct
    the architecture without a separate config.
    """
    import torch
    import numpy as np

    ckpt = torch.load(str(checkpoint_path), weights_only=False, map_location="cpu")
    weights = {}
    for k, v in ckpt["model_state_dict"].items():
        if "num_batches_tracked" not in k:
            # Use .tolist() to work around missing NumPy C bridge in this build
            array = np.array(v.tolist(), dtype=np.float32)
            # Subnormal weights stall CPU inference and make negligible contributions.
            array[np.abs(array) < 1e-30] = 0
            weights[k] = array
    weights["num_blocks"] = np.array(ckpt["num_blocks"])
    weights["num_filters"] = np.array(ckpt["num_filters"])

    buf = io.BytesIO()
    np.savez_compressed(buf, **weights)
    return buf.getvalue()


def _build_tar_submission(
    output_dir: Path,
    checkpoint_path: Path,
    time_budget: float,
) -> Path:
    """Create submission.tar.gz: main.py (numpy agent) + weights.npz.

    Kaggle extracts the archive to /kaggle_simulations/agent/ and runs main.py.
    weights.npz lands in the same directory, so _WEIGHTS_PATH resolves correctly.
    """
    if not _NUMPY_AGENT_SOURCE.exists():
        print(f"Error: numpy agent source not found: {_NUMPY_AGENT_SOURCE}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting weights from {checkpoint_path} ...")
    npz_bytes = _extract_weights_npz(checkpoint_path)

    import numpy as np
    npz_size_kb = len(npz_bytes) / 1024
    npz = np.load(io.BytesIO(npz_bytes))
    total_params = sum(npz[k].size for k in npz.files if k not in ("num_blocks", "num_filters"))
    print(f"  {total_params:,} parameters → {npz_size_kb:.0f} KB compressed npz")

    # Build main.py from the numpy agent template
    source = _NUMPY_AGENT_SOURCE.read_text(encoding="utf-8")
    if _NUMPY_TIME_BUDGET_SENTINEL not in source:
        print(f"Error: sentinel not found in {_NUMPY_AGENT_SOURCE}:\n  {_NUMPY_TIME_BUDGET_SENTINEL}", file=sys.stderr)
        sys.exit(1)

    source = source.replace(
        _NUMPY_TIME_BUDGET_SENTINEL,
        f"_TIME_BUDGET_SECS: float = {time_budget}",
    )

    # Build tar.gz
    tar_path = output_dir / "submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        # main.py
        main_bytes = source.encode("utf-8")
        info = tarfile.TarInfo(name="main.py")
        info.size = len(main_bytes)
        tf.addfile(info, io.BytesIO(main_bytes))
        # weights.npz
        info2 = tarfile.TarInfo(name="weights.npz")
        info2.size = len(npz_bytes)
        tf.addfile(info2, io.BytesIO(npz_bytes))

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {tar_path}  ({size_mb:.2f} MB)")
    print(f"\nSubmission ready: {tar_path}")
    print(f"  main.py     (numpy ResNet + MCTS, {time_budget}s budget)")
    print(f"  weights.npz ({npz_size_kb:.0f} KB)")
    print("\nNext step: upload submission.tar.gz to Kaggle → Submit Predictions.")
    return tar_path


# -------------------------------------------------------------------------
# --base64 mode: onnxruntime, base64-embedded model
# -------------------------------------------------------------------------

def _read_and_validate_onnx_source() -> str:
    source = _AGENT_SOURCE.read_text(encoding="utf-8")
    for sentinel in (_MODEL_PATH_SENTINEL, _NUM_SIMS_SENTINEL):
        if sentinel not in source:
            print(f"Error: sentinel not found in {_AGENT_SOURCE}:\n  {sentinel}", file=sys.stderr)
            sys.exit(1)
    return source


def _build_base64_submission(output_dir: Path, model_path: Path, num_sims: int) -> Path:
    """Single main.py with ONNX model embedded as base64. Requires onnxruntime on Kaggle."""
    model_b64 = base64.b64encode(model_path.read_bytes()).decode("ascii")
    source = _read_and_validate_onnx_source()
    source = source.replace(_NUM_SIMS_SENTINEL, f"_NUM_MCTS_SIMS: int = {num_sims}")
    b64_block = (
        "# ONNX model embedded as base64 — no external file needed\n"
        "import base64 as _b64m, tempfile as _tmpm, atexit as _atm, os as _osm\n"
        f'_MODEL_DATA: str = "{model_b64}"\n'
        "_b64_tmp = _tmpm.NamedTemporaryFile(suffix='.onnx', delete=False)\n"
        "_b64_tmp.write(_b64m.b64decode(_MODEL_DATA))\n"
        "_b64_tmp.close()\n"
        "_MODEL_PATH: str = _b64_tmp.name\n"
        "_atm.register(lambda: _osm.unlink(_MODEL_PATH) if _osm.path.exists(_MODEL_PATH) else None)"
    )
    source = source.replace(_MODEL_PATH_SENTINEL, b64_block)

    output_path = output_dir / "main.py"
    output_path.write_text(source, encoding="utf-8")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path}  ({size_mb:.2f} MB)")
    print("\nWARNING: onnxruntime is not available in Kaggle Connect-X environment.")
    print("Use --tar for a numpy-only submission that works without onnxruntime.")
    return output_path


# -------------------------------------------------------------------------
# --zip mode (broken) and directory mode
# -------------------------------------------------------------------------

def _build_zip(output_dir: Path, model_path: Path, num_sims: int) -> Path:
    source = _read_and_validate_onnx_source()
    source = source.replace(_NUM_SIMS_SENTINEL, f"_NUM_MCTS_SIMS: int = {num_sims}")
    source = source.replace(_MODEL_PATH_SENTINEL, '_MODEL_PATH: str = "model.onnx"')
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.py", source)
        zf.write(model_path, arcname="model.onnx")
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path}  ({size_mb:.2f} MB)")
    print("\nWARNING: Kaggle does not extract zip archives for Connect-X. Use --tar.")
    return zip_path


def _build_directory(output_dir: Path, model_path: Path, dataset_path: str, num_sims: int) -> None:
    source = _read_and_validate_onnx_source()
    source = source.replace(_NUM_SIMS_SENTINEL, f"_NUM_MCTS_SIMS: int = {num_sims}")
    source = source.replace(_MODEL_PATH_SENTINEL, f'_MODEL_PATH: str = "{dataset_path}"')
    submission_py = output_dir / "submission.py"
    submission_py.write_text(source, encoding="utf-8")
    dest_model = output_dir / model_path.name
    shutil.copy2(model_path, dest_model)
    print(f"Wrote {submission_py}")
    print(f"Copied model to {dest_model}")
    print(f"\nNext steps:\n  1. Upload {model_path.name} as a Kaggle Dataset.\n  2. Submit submission.py via Kaggle notebook.")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main() -> None:
    """Build the submission package."""
    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tar:
        if not args.checkpoint:
            print("Error: --tar requires --checkpoint <path/to/checkpoint.pt>", file=sys.stderr)
            sys.exit(1)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        _build_tar_submission(output_dir, checkpoint_path, args.time_budget)

    elif args.base64:
        model_path = Path(args.model) if args.model else None
        if not model_path or not model_path.exists():
            print("Error: --base64 requires --model <path/to/model.onnx>", file=sys.stderr)
            sys.exit(1)
        _build_base64_submission(output_dir, model_path, 200)

    elif args.zip:
        model_path = Path(args.model) if args.model else None
        if not model_path or not model_path.exists():
            print("Error: --zip requires --model <path/to/model.onnx>", file=sys.stderr)
            sys.exit(1)
        _build_zip(output_dir, model_path, 200)

    else:
        model_path = Path(args.model) if args.model else None
        if not model_path or not model_path.exists():
            print("Error: directory mode requires --model <path/to/model.onnx>", file=sys.stderr)
            sys.exit(1)
        _build_directory(output_dir, model_path, args.dataset_path, 200)


if __name__ == "__main__":
    main()
