"""ONNX export utilities for the Connect 4 AlphaZero model.

Exports a trained Connect4Net checkpoint to ONNX format for deployment.
The exported model accepts (batch, 3, 6, 7) float32 input and returns
(policy_logits, value) with shapes (batch, 7) and (batch, 1).

Output names are locked to ["policy_logits", "value"] — kaggle_agent.py
depends on this ordering when calling session.run().
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch
import onnxruntime as ort

from src.neural_net.model import Connect4Net

logger = logging.getLogger(__name__)


def export_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    quantize: bool = False,
    device: str = "cpu",
) -> Path:
    """Export a trained Connect4Net checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to a .pt checkpoint file saved by Coach.
        output_path: Destination .onnx file path.
        quantize: If True, apply dynamic int8 quantization. WARNING: slower
            on Apple Silicon — keep False for MPS/CPU on Mac.
        device: PyTorch device to load the model onto (default "cpu").

    Returns:
        Path to the exported ONNX file.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if quantize:
        logger.warning(
            "Quantization is slower on Apple Silicon. "
            "float32 ONNX is typically faster on this platform."
        )

    # Load checkpoint using same keys as alphazero_agent.py
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = Connect4Net(
        num_blocks=ckpt["num_blocks"],
        num_filters=ckpt["num_filters"],
        input_planes=ckpt.get("input_planes", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    # Use zeros (deterministic) rather than randn
    dummy_input = torch.zeros(1, 3, 6, 7, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if quantize:
        # Export to a temp file first, then quantize to the final path
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            _export_to_path(model, dummy_input, tmp_path)
            _quantize_model(tmp_path, output_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        _export_to_path(model, dummy_input, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exported ONNX model to %s (%.2f MB)", output_path, size_mb)
    return output_path


def load_onnx_session(
    model_path: str | Path,
    intra_op_num_threads: int = 1,
) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session.

    Args:
        model_path: Path to the .onnx model file.
        intra_op_num_threads: Number of threads per operator (default 1 for
            Kaggle's single-core environment).

    Returns:
        Configured InferenceSession ready for inference.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_num_threads
    return ort.InferenceSession(str(model_path), sess_options=opts)


def _export_to_path(
    model: Connect4Net,
    dummy_input: torch.Tensor,
    path: Path,
) -> None:
    """Run torch.onnx.export to the given path.

    Output names are fixed to ["policy_logits", "value"] so that downstream
    consumers (kaggle_agent.py, benchmarks) can rely on this order.
    """
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        opset_version=11,
        input_names=["board_state"],
        output_names=["policy_logits", "value"],
        dynamic_axes={"board_state": {0: "batch_size"}},
    )


def _quantize_model(input_path: Path, output_path: Path) -> None:
    """Apply dynamic int8 quantization to a saved ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QUInt8)
