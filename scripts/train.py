"""Entry point for AlphaZero Connect 4 training.

Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/full.yaml --resume checkpoints/checkpoint_iter_005.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from src.training.coach import Coach  # noqa: E402
from src.utils.config import Config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an AlphaZero Connect 4 agent via self-play."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g. configs/tiny.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    config = Config.from_yaml(args.config)

    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        start_iteration = checkpoint.get("iteration", 0) + 1
        logging.info("Resuming from iteration %d", start_iteration)

    coach = Coach(config)

    if args.resume:
        # Load saved model weights into the coach's model
        checkpoint = torch.load(args.resume, weights_only=False)
        coach._model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Loaded model weights from %s", args.resume)

        # Ensure best_model.pt exists so the arena has a baseline to compare
        # against. Without this, the first resumed iteration auto-accepts the
        # candidate unconditionally (the "no previous best" branch).
        import shutil
        best_path = coach._best_checkpoint_path
        if not best_path.exists():
            shutil.copy(args.resume, best_path)
            logging.info("Copied resume checkpoint to %s as best model baseline", best_path)

    coach.train(start_iteration=start_iteration)


if __name__ == "__main__":
    # Must be set before any ProcessPoolExecutor is created.
    # 'spawn' is required for PyTorch (avoids CUDA context inheritance on Linux).
    # force=True is safe on macOS where 'spawn' is already the default.
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
