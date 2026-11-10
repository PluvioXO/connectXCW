"""Configuration dataclasses for Connect 4 AlphaZero training."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GameConfig:
    """Connect 4 board dimensions and win condition."""

    rows: int = 6
    cols: int = 7
    win_length: int = 4


@dataclass
class ModelConfig:
    """Neural network architecture parameters."""

    num_residual_blocks: int = 5
    num_filters: int = 128
    input_planes: int = 3


@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search hyperparameters."""

    num_simulations: int = 600
    c_puct: float = 2.0
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 20
    temperature_high: float = 1.0
    temperature_low: float = 0.3


@dataclass
class TrainingConfig:
    """Self-play and training loop parameters."""

    num_iterations: int = 25
    self_play_games_per_iteration: int = 5000
    training_epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_max_size: int = 1_000_000
    arena_num_games: int = 128
    arena_num_simulations: int | None = None  # if None, uses mcts.num_simulations
    update_threshold: float = 0.55
    checkpoint_dir: str = "checkpoints"
    num_self_play_workers: int = 1  # >1 enables parallel self-play via multiprocessing
    mcts_batch_size: int = 1  # >1 enables BatchedSelfPlay (GPU batch inference); mutually exclusive with num_self_play_workers > 1


@dataclass
class Config:
    """Top-level configuration combining all sub-configs."""

    game: GameConfig = field(default_factory=GameConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file, overriding defaults.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A Config instance with values from the YAML file merged over defaults.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        game_data = data.get("game", {})
        model_data = data.get("model", {})
        mcts_data = data.get("mcts", {})
        training_data = data.get("training", {})

        return cls(
            game=GameConfig(**game_data),
            model=ModelConfig(**model_data),
            mcts=MCTSConfig(**mcts_data),
            training=TrainingConfig(**training_data),
        )
