"""CLI script to export a Connect4Net checkpoint to ONNX format.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/baseline_b2_f32.pt --output model.onnx
    python scripts/export_onnx.py --checkpoint ... --output ... --quantize
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.export.onnx_export import export_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a trained Connect4Net checkpoint to ONNX format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint file (e.g. checkpoints/baseline_b2_f32.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .onnx file path (e.g. model.onnx).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Apply dynamic int8 quantization. WARNING: slower on Apple Silicon.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to load model onto (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    """Export the model to ONNX."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    output = export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        quantize=args.quantize,
        device=args.device,
    )
    print(f"Model exported to: {output}")


if __name__ == "__main__":
    main()
