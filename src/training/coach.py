"""AlphaZero training loop orchestrator.

Coordinates self-play game generation, neural network training, and
arena evaluation to iteratively improve the model.

The training loop per iteration:
  1. Self-play: current model generates games and fills the replay buffer.
  2. Train: update the network on samples from the replay buffer.
  3. Arena: new candidate vs. current best; accept if win rate >= threshold.
  4. Save checkpoint for the iteration.
"""

import copy
import logging
from pathlib import Path

import torch

from src.agents.minimax_agent import MinimaxAgent
from src.game.board import Connect4Board
from src.game.constants import PLAYER_1, PLAYER_2
from src.mcts.search import MCTS, select_move
from src.neural_net.model import Connect4Net
from src.training.arena import pit
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import BatchedSelfPlay, SelfPlay
from src.training.trainer import Trainer
from src.utils.config import Config, MCTSConfig

logger = logging.getLogger(__name__)


class Coach:
    """Orchestrates the full AlphaZero self-play training loop.

    Args:
        config: Full configuration (game, model, mcts, training sub-configs).
    """

    def __init__(self, config: Config) -> None:
        """Initialize the coach.

        Args:
            config: Training configuration including model architecture,
                MCTS settings, and training hyperparameters.
        """
        self._config = config
        self._replay_buffer = ReplayBuffer(config.training.replay_buffer_max_size)

        # Create the model and move to available device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self._device)

        self._model = Connect4Net(
            num_blocks=config.model.num_residual_blocks,
            num_filters=config.model.num_filters,
            input_planes=config.model.input_planes,
        )
        self._model.to(self._device)

        checkpoint_dir = Path(config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir
        self._best_checkpoint_path = checkpoint_dir / "best_model.pt"

    def train(self, start_iteration: int = 0) -> None:
        """Run the full training loop.

        Args:
            start_iteration: Iteration index to start from (0 for fresh
                training, or loaded from a checkpoint for resumption).
        """
        cfg = self._config.training
        model = self._model

        for iteration in range(start_iteration, cfg.num_iterations):
            logger.info("=== Iteration %d / %d ===", iteration + 1, cfg.num_iterations)

            # ----------------------------------------------------------------
            # 1. Self-play: generate games and fill the replay buffer
            # ----------------------------------------------------------------
            model.eval()
            if cfg.mcts_batch_size > 1:
                self_play = BatchedSelfPlay(
                    model,
                    self._config.mcts,
                    batch_size=cfg.mcts_batch_size,
                )
            else:
                self_play = SelfPlay(
                    model,
                    self._config.mcts,
                    num_workers=cfg.num_self_play_workers,
                    model_config=self._config.model,
                )
            logger.info(
                "Self-play: generating %d games...", cfg.self_play_games_per_iteration
            )
            samples = self_play.generate_games(cfg.self_play_games_per_iteration)
            for sample in samples:
                self._replay_buffer.add(sample)
            logger.info("Replay buffer size: %d", len(self._replay_buffer))

            # ----------------------------------------------------------------
            # 2. Train: update the network on replay buffer data
            # ----------------------------------------------------------------
            trainer = Trainer(model, cfg)
            logger.info("Training for %d epochs...", cfg.training_epochs)
            metrics = trainer.train(self._replay_buffer, cfg.training_epochs)
            logger.info(
                "Losses — policy: %.4f  value: %.4f  total: %.4f",
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["total_loss"],
            )
            model.eval()

            # ----------------------------------------------------------------
            # 3. Arena: compare candidate vs. current best
            # ----------------------------------------------------------------
            candidate = copy.deepcopy(model)

            if self._best_checkpoint_path.exists():
                best = self._load_best_model(self._best_checkpoint_path)
                arena_mcts = self._config.mcts
                if cfg.arena_num_simulations is not None:
                    import dataclasses
                    arena_mcts = dataclasses.replace(
                        self._config.mcts, num_simulations=cfg.arena_num_simulations
                    )
                logger.info(
                    "Arena: %d games (candidate vs. best, %d sims)...",
                    cfg.arena_num_games, arena_mcts.num_simulations,
                )
                wins, losses, draws = pit(
                    candidate, best, cfg.arena_num_games, arena_mcts
                )
                total = wins + losses + draws
                decisive = wins + losses
                win_rate = wins / decisive if decisive > 0 else 0.5
                logger.info(
                    "Arena result — wins: %d  losses: %d  draws: %d  win_rate: %.3f (decisive games only)",
                    wins,
                    losses,
                    draws,
                    win_rate,
                )

                if win_rate >= cfg.update_threshold:
                    logger.info("Accepted new model (win_rate %.3f >= %.3f)", win_rate, cfg.update_threshold)
                    model = candidate
                    self._model = model
                    self._save_checkpoint(
                        model, trainer.optimizer, self._best_checkpoint_path, iteration
                    )
                else:
                    logger.info(
                        "Rejected new model (win_rate %.3f < %.3f), reverting to best",
                        win_rate,
                        cfg.update_threshold,
                    )
                    model = best
                    self._model = model
            else:
                # First iteration: accept unconditionally
                logger.info("No previous best model — accepting candidate unconditionally.")
                self._model = candidate
                model = candidate
                self._save_checkpoint(
                    model, trainer.optimizer, self._best_checkpoint_path, iteration
                )

            # ----------------------------------------------------------------
            # 4. Save iteration checkpoint
            # ----------------------------------------------------------------
            iter_checkpoint_path = (
                self._checkpoint_dir / f"checkpoint_iter_{iteration:03d}.pt"
            )
            self._save_checkpoint(model, trainer.optimizer, iter_checkpoint_path, iteration)
            logger.info("Saved checkpoint: %s", iter_checkpoint_path)

            # ----------------------------------------------------------------
            # 5. Quick benchmark: current best vs random and minimax(d=1)
            # ----------------------------------------------------------------
            self._log_benchmark(model)

    def _log_benchmark(self, model: Connect4Net, num_games: int = 30) -> None:
        """Run a quick benchmark of the current model vs random and minimax agents.

        Plays num_games against each opponent (half as P1, half as P2) and
        logs win rates. Uses 50 MCTS simulations for speed.

        Args:
            model: The current best model to benchmark.
            num_games: Games per opponent (default 30).
        """
        bench_config = MCTSConfig(num_simulations=50)
        mcts = MCTS(model, bench_config)
        half = num_games // 2

        opponents = [
            ("Minimax(d=7)", MinimaxAgent(max_depth=7)),
            ("Minimax(d=9)", MinimaxAgent(max_depth=9)),
        ]

        for opp_name, opp in opponents:
            wins = losses = draws = 0

            for p1_is_az in [True, False]:
                count = half if p1_is_az else (num_games - half)
                for _ in range(count):
                    board = Connect4Board()
                    while not board.is_terminal():
                        if board.current_player == PLAYER_1:
                            if p1_is_az:
                                col = select_move(mcts.search(board, add_dirichlet_noise=False), 0.1)
                            else:
                                col = opp.select_move(board)
                        else:
                            if p1_is_az:
                                col = opp.select_move(board)
                            else:
                                col = select_move(mcts.search(board, add_dirichlet_noise=False), 0.1)
                        board = board.make_move(col)

                    winner = board.get_winner()
                    az_player = PLAYER_1 if p1_is_az else PLAYER_2
                    if winner == az_player:
                        wins += 1
                    elif winner is None:
                        draws += 1
                    else:
                        losses += 1

            logger.info(
                "Benchmark vs %s: %d/%d wins, %d draws, %d losses (%.0f%% win rate)",
                opp_name, wins, num_games, draws, losses, 100 * wins / num_games,
            )

    def _save_checkpoint(
        self,
        model: Connect4Net,
        optimizer: torch.optim.Adam,
        path: Path,
        iteration: int,
    ) -> None:
        """Save model + optimizer state to disk.

        Args:
            model: The neural network to save.
            optimizer: The Adam optimizer whose state to save.
            path: Destination file path.
            iteration: Current iteration index (stored in checkpoint).
        """
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_blocks": self._config.model.num_residual_blocks,
                "num_filters": self._config.model.num_filters,
            },
            path,
        )

    def _load_best_model(self, path: Path) -> Connect4Net:
        """Load the best model from a checkpoint file.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Connect4Net loaded from the checkpoint, in eval mode.
        """
        checkpoint = torch.load(path, weights_only=False)
        num_blocks = checkpoint.get("num_blocks", self._config.model.num_residual_blocks)
        num_filters = checkpoint.get("num_filters", self._config.model.num_filters)

        model = Connect4Net(
            num_blocks=num_blocks,
            num_filters=num_filters,
            input_planes=self._config.model.input_planes,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self._device)
        return model
