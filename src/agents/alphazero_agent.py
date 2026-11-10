"""AlphaZero agent: neural network-guided MCTS.

Loads a trained Connect4Net checkpoint and uses it to guide MCTS search.
Wraps the existing MCTS class from src/mcts/search.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.agents.base_agent import Agent
from src.game.board import Connect4Board
from src.mcts.search import MCTS, select_move
from src.neural_net.model import Connect4Net
from src.utils.config import MCTSConfig

# Default simulations for evaluation (faster than training's 600)
_DEFAULT_SIMS: int = 200
# Near-deterministic temperature for evaluation play
_DEFAULT_TEMPERATURE: float = 0.1


class AlphaZeroAgent(Agent):
    """Agent combining a trained neural network with MCTS search.

    Args:
        checkpoint_path: Path to a .pt checkpoint file saved by Coach.
        mcts_config: Optional MCTS configuration. If None, uses default
            with num_simulations=200.
        temperature: Move selection temperature (default 0.1, near-deterministic).
        device: PyTorch device string (default "cpu").
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        mcts_config: MCTSConfig | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        device: str = "cpu",
    ) -> None:
        """Load a checkpoint and initialize MCTS.

        Args:
            checkpoint_path: Path to a .pt checkpoint file.
            mcts_config: MCTS hyperparameters. Defaults to 200 simulations.
            temperature: Temperature for move selection.
            device: PyTorch device to load the model onto.
        """
        self._temperature = temperature
        self._checkpoint_path = Path(checkpoint_path)

        ckpt = torch.load(self._checkpoint_path, weights_only=False, map_location=device)
        model = Connect4Net(
            num_blocks=ckpt["num_blocks"],
            num_filters=ckpt["num_filters"],
            input_planes=ckpt.get("input_planes", 3),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.to(device)

        config = mcts_config or MCTSConfig(num_simulations=_DEFAULT_SIMS)
        self._mcts = MCTS(model, config)
        self._num_sims = config.num_simulations

    @property
    def name(self) -> str:
        """Return the agent's display name."""
        return f"AlphaZero-{self._num_sims}sims"

    def select_move(self, board: Connect4Board) -> int:
        """Select the best move using NN-guided MCTS.

        Args:
            board: Current game state.

        Returns:
            Best column index selected by AlphaZero.
        """
        visit_counts = self._mcts.search(board, add_dirichlet_noise=False)
        return select_move(visit_counts, temperature=self._temperature)
