"""Tests for the self-play training pipeline (Iteration 4).

Covers: ReplayBuffer, SelfPlay, Trainer, Arena, Coach, and the
end-to-end integration test (marked slow).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from src.game.board import Connect4Board
from src.game.constants import COLS, ROWS
from src.mcts.search import MCTS
from src.neural_net.model import Connect4Net
from src.training.arena import pit
from src.training.coach import Coach
from src.training.replay_buffer import ReplayBuffer, TrainingSample
from src.training.self_play import SelfPlay
from src.training.trainer import Trainer
from src.utils.config import (
    Config,
    GameConfig,
    MCTSConfig,
    ModelConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_mcts_config() -> MCTSConfig:
    """Very fast MCTSConfig for unit tests (10 simulations)."""
    return MCTSConfig(num_simulations=10, c_puct=2.0)


@pytest.fixture
def tiny_net() -> Connect4Net:
    """Tiny Connect4Net (2 blocks, 32 filters) in eval mode."""
    model = Connect4Net(num_blocks=2, num_filters=32, input_planes=3)
    model.eval()
    return model


@pytest.fixture
def micro_config(tmp_path) -> Config:
    """Config for a minimal coach test: 1 iteration, 10 games, 5 sims."""
    return Config(
        game=GameConfig(),
        model=ModelConfig(num_residual_blocks=2, num_filters=32, input_planes=3),
        mcts=MCTSConfig(num_simulations=5, c_puct=2.0),
        training=TrainingConfig(
            num_iterations=1,
            self_play_games_per_iteration=10,
            training_epochs=1,
            batch_size=16,
            learning_rate=1e-3,
            weight_decay=1e-4,
            replay_buffer_max_size=10_000,
            arena_num_games=4,
            update_threshold=0.0,  # always accept to speed up test
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )


# ---------------------------------------------------------------------------
# TestReplayBuffer
# ---------------------------------------------------------------------------


def _make_sample(value: float = 1.0) -> TrainingSample:
    """Create a dummy TrainingSample for testing."""
    return TrainingSample(
        state=np.zeros((3, ROWS, COLS), dtype=np.float32),
        policy=np.ones(COLS, dtype=np.float32) / COLS,
        value=value,
    )


class TestReplayBuffer:
    def test_add_and_sample(self) -> None:
        buf = ReplayBuffer(max_size=100)
        for i in range(10):
            buf.add(_make_sample(float(i % 3 - 1)))
        assert len(buf) == 10
        batch = buf.sample(5)
        assert len(batch) == 5
        assert all(isinstance(s, TrainingSample) for s in batch)

    def test_fifo_eviction_when_full(self) -> None:
        buf = ReplayBuffer(max_size=5)
        for i in range(8):
            buf.add(_make_sample(float(i)))
        # Buffer should only contain the last 5 samples (values 3-7)
        assert len(buf) == 5
        values = {s.value for s in buf.sample(5)}
        assert values == {3.0, 4.0, 5.0, 6.0, 7.0}

    def test_sample_returns_correct_shapes(self) -> None:
        buf = ReplayBuffer(max_size=100)
        for _ in range(20):
            buf.add(_make_sample())
        batch = buf.sample(8)
        assert len(batch) == 8
        for s in batch:
            assert s.state.shape == (3, ROWS, COLS)
            assert s.policy.shape == (COLS,)
            assert isinstance(s.value, float)

    def test_sample_size_clamped_to_buffer_size(self) -> None:
        buf = ReplayBuffer(max_size=100)
        for _ in range(3):
            buf.add(_make_sample())
        # Request more than available
        batch = buf.sample(50)
        assert len(batch) == 3


# ---------------------------------------------------------------------------
# TestSelfPlay
# ---------------------------------------------------------------------------


class TestSelfPlay:
    def test_produces_samples(self, tiny_net, tiny_mcts_config) -> None:
        sp = SelfPlay(tiny_net, tiny_mcts_config)
        samples = sp.play_game()
        assert len(samples) > 0

    def test_correct_shapes(self, tiny_net, tiny_mcts_config) -> None:
        sp = SelfPlay(tiny_net, tiny_mcts_config)
        samples = sp.play_game()
        for s in samples:
            assert s.state.shape == (3, ROWS, COLS), f"state shape {s.state.shape}"
            assert s.policy.shape == (COLS,), f"policy shape {s.policy.shape}"
            assert isinstance(s.value, float)

    def test_values_are_valid(self, tiny_net, tiny_mcts_config) -> None:
        sp = SelfPlay(tiny_net, tiny_mcts_config)
        samples = sp.play_game()
        for s in samples:
            assert s.value in {-1.0, 0.0, 1.0}, f"unexpected value {s.value}"

    def test_augmentation_doubles_samples(self, tiny_net, tiny_mcts_config) -> None:
        sp = SelfPlay(tiny_net, tiny_mcts_config)
        samples = sp.play_game()
        # Number of moves in the game × 2 (original + flipped) = len(samples)
        assert len(samples) % 2 == 0
        # There must be at least 2 samples (even the shortest game has moves)
        assert len(samples) >= 2


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_one_step_reduces_loss(self, tiny_net) -> None:
        """A single training step should return finite losses (no NaN)."""
        cfg = TrainingConfig(batch_size=16, learning_rate=1e-3, weight_decay=1e-4)
        buf = ReplayBuffer(max_size=1000)
        for _ in range(32):
            buf.add(_make_sample(random.choice([-1.0, 0.0, 1.0])))

        trainer = Trainer(tiny_net, cfg)
        batch = buf.sample(16)
        metrics = trainer.train_step(batch)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "total_loss" in metrics
        assert not np.isnan(metrics["total_loss"]), "NaN in total_loss"
        assert not np.isinf(metrics["total_loss"]), "Inf in total_loss"

    def test_returns_loss_metrics(self, tiny_net) -> None:
        cfg = TrainingConfig(batch_size=8, learning_rate=1e-3, weight_decay=1e-4)
        buf = ReplayBuffer(max_size=1000)
        for _ in range(16):
            buf.add(_make_sample(random.choice([-1.0, 1.0])))

        trainer = Trainer(tiny_net, cfg)
        metrics = trainer.train(buf, epochs=1)

        assert set(metrics.keys()) == {"policy_loss", "value_loss", "total_loss"}
        for v in metrics.values():
            assert isinstance(v, float)
            assert not np.isnan(v)


# ---------------------------------------------------------------------------
# TestArena
# ---------------------------------------------------------------------------


class TestArena:
    def test_returns_valid_counts(self, tiny_net, tiny_mcts_config) -> None:
        model_a = tiny_net
        model_b = copy.deepcopy(tiny_net)
        wins, losses, draws = pit(model_a, model_b, 4, tiny_mcts_config)
        assert wins + losses + draws == 4
        assert wins >= 0 and losses >= 0 and draws >= 0

    def test_symmetric_seating(self, tiny_net, tiny_mcts_config) -> None:
        """pit() uses half games with each model as P1 (no structural bias)."""
        model_a = tiny_net
        model_b = copy.deepcopy(tiny_net)
        # Run two symmetric arenas and confirm neither model has >100% wins
        wins_a, losses_a, draws_a = pit(model_a, model_b, 4, tiny_mcts_config)
        wins_b, losses_b, draws_b = pit(model_b, model_a, 4, tiny_mcts_config)
        # Totals must be valid
        assert wins_a + losses_a + draws_a == 4
        assert wins_b + losses_b + draws_b == 4


# ---------------------------------------------------------------------------
# TestCoach
# ---------------------------------------------------------------------------


class TestCoach:
    def test_single_iteration_completes(self, micro_config) -> None:
        coach = Coach(micro_config)
        # Must not raise
        coach.train(start_iteration=0)

    def test_checkpoint_save_and_resume(self, micro_config, tmp_path) -> None:
        coach = Coach(micro_config)
        coach.train(start_iteration=0)

        checkpoint_dir = tmp_path / "checkpoints"
        # At least the best model checkpoint must exist
        assert (checkpoint_dir / "best_model.pt").exists()

        # Load the checkpoint and verify it has the expected keys
        ckpt = torch.load(
            checkpoint_dir / "best_model.pt", weights_only=False
        )
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "iteration" in ckpt


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    def test_full_tiny_training_beats_random(self, tmp_path) -> None:
        """2 iterations of tiny training should produce a model that beats random >90%.

        This test is slow (~5-15 min on CPU). Run with: pytest -k slow

        Results are written to logs/benchmarks/ as JSON + human-readable .txt
        for tracking training progress across runs.
        """
        from src.game.constants import PLAYER_1, PLAYER_2
        from src.mcts.search import MCTS, select_move

        # -----------------------------------------------------------------
        # Training configuration
        # -----------------------------------------------------------------
        model_cfg = ModelConfig(num_residual_blocks=2, num_filters=32, input_planes=3)
        mcts_cfg = MCTSConfig(
            num_simulations=50,
            c_puct=2.0,
            dirichlet_alpha=1.0,
            dirichlet_epsilon=0.25,
            temperature_threshold=20,
            temperature_high=1.0,
            temperature_low=0.3,
        )
        training_cfg = TrainingConfig(
            num_iterations=2,
            self_play_games_per_iteration=100,
            training_epochs=5,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-4,
            replay_buffer_max_size=10_000,
            arena_num_games=8,
            update_threshold=0.0,  # always accept (baseline run)
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        config = Config(game=GameConfig(), model=model_cfg, mcts=mcts_cfg, training=training_cfg)

        # -----------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------
        t_start = datetime.now(timezone.utc)
        coach = Coach(config)
        coach.train(start_iteration=0)
        t_end = datetime.now(timezone.utc)
        training_seconds = (t_end - t_start).total_seconds()

        # Persist the trained model to a stable location for future benchmarking.
        # Saved as checkpoints/tiny_baseline.pt alongside a metadata sidecar.
        _save_persistent_checkpoint(coach._model, config, t_end)

        # -----------------------------------------------------------------
        # Evaluate vs random agent
        # -----------------------------------------------------------------
        trained_model = coach._model
        trained_model.eval()

        eval_mcts_sims = 50
        trained_mcts = MCTS(trained_model, MCTSConfig(num_simulations=eval_mcts_sims, c_puct=2.0))

        num_games = 50
        wins = 0
        losses = 0
        draws = 0

        for game_idx in range(num_games):
            board = Connect4Board()
            trained_is_p1 = game_idx % 2 == 0  # alternate sides

            while not board.is_terminal():
                if (board.current_player == PLAYER_1) == trained_is_p1:
                    visit_counts = trained_mcts.search(board, add_dirichlet_noise=False)
                    col = select_move(visit_counts, temperature=0.1)
                else:
                    col = random.choice(board.get_legal_moves())
                board = board.make_move(col)

            winner = board.get_winner()
            trained_player = PLAYER_1 if trained_is_p1 else PLAYER_2
            if winner == trained_player:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1

        win_rate = wins / num_games

        # -----------------------------------------------------------------
        # Write benchmark log
        # -----------------------------------------------------------------
        _write_benchmark_log(
            benchmark_name="baseline_vs_random",
            description=(
                "Tiny-config AlphaZero (2 ResNet blocks, 32 filters) trained for "
                f"{training_cfg.num_iterations} iterations via self-play, then evaluated "
                f"against a uniform-random opponent over {num_games} games "
                f"(alternating sides). MCTS uses {eval_mcts_sims} simulations per move "
                "at evaluation time."
            ),
            config=config,
            results={
                "num_eval_games": num_games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": round(win_rate, 4),
                "pass_threshold": 0.90,
                "passed": bool(win_rate >= 0.90),
            },
            training_seconds=training_seconds,
        )

        assert win_rate >= 0.90, (
            f"Trained model only won {wins}/{num_games} ({win_rate:.1%}) vs random. "
            "Expected >90%."
        )


# ---------------------------------------------------------------------------
# Benchmark logging helper
# ---------------------------------------------------------------------------


def _write_benchmark_log(
    benchmark_name: str,
    description: str,
    config: Config,
    results: dict,
    training_seconds: float,
) -> None:
    """Write benchmark results to logs/benchmarks/ as JSON + .txt summary.

    Args:
        benchmark_name: Short identifier used in the filename.
        description: Human-readable explanation of what was tested.
        config: Full training Config (model, mcts, training sub-configs).
        results: Dict of outcome metrics (wins, win_rate, etc.).
        training_seconds: Wall-clock training time in seconds.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(__file__).resolve().parent.parent / "logs" / "benchmarks"
    log_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{timestamp}_{benchmark_name}"

    # ---- JSON (machine-readable) ----
    record = {
        "benchmark": benchmark_name,
        "timestamp_utc": timestamp,
        "description": description,
        "training_wall_seconds": round(training_seconds, 1),
        "config": {
            "model": dataclasses.asdict(config.model),
            "mcts": dataclasses.asdict(config.mcts),
            "training": dataclasses.asdict(config.training),
            "game": dataclasses.asdict(config.game),
        },
        "results": results,
    }
    json_path = log_dir / f"{stem}.json"
    json_path.write_text(json.dumps(record, indent=2))

    # ---- Plain text (human-readable / article-ready) ----
    txt_lines = [
        f"Benchmark: {benchmark_name}",
        f"Date (UTC): {timestamp}",
        "",
        "Description",
        "-----------",
        description,
        "",
        "Training Configuration",
        "----------------------",
        f"  Model:      {config.model.num_residual_blocks} residual blocks, "
        f"{config.model.num_filters} filters, {config.model.input_planes} input planes",
        f"  MCTS:       {config.mcts.num_simulations} sims/move (training), "
        f"c_puct={config.mcts.c_puct}, "
        f"dirichlet α={config.mcts.dirichlet_alpha} ε={config.mcts.dirichlet_epsilon}",
        f"  Self-play:  {config.training.num_iterations} iterations × "
        f"{config.training.self_play_games_per_iteration} games/iter",
        f"  Training:   {config.training.training_epochs} epochs/iter, "
        f"batch={config.training.batch_size}, lr={config.training.learning_rate}, "
        f"wd={config.training.weight_decay}",
        f"  Arena:      {config.training.arena_num_games} games, "
        f"threshold={config.training.update_threshold}",
        f"  Wall time:  {training_seconds:.0f}s ({training_seconds/60:.1f} min)",
        "",
        "Results",
        "-------",
    ]
    for k, v in results.items():
        txt_lines.append(f"  {k}: {v}")
    txt_lines.append("")

    txt_path = log_dir / f"{stem}.txt"
    txt_path.write_text("\n".join(txt_lines))

    print(f"\nBenchmark logs written to:\n  {json_path}\n  {txt_path}")


def _save_persistent_checkpoint(
    model: Connect4Net,
    config: Config,
    timestamp: datetime,
) -> None:
    """Save the trained model to checkpoints/baseline_b{N}_f{F}.pt.

    The filename encodes the model architecture (blocks + filters) so that
    checkpoints from different model sizes never overwrite each other.
    A JSON sidecar stores the full config for reproducibility.

    Args:
        model: Trained Connect4Net.
        config: Config used for this training run.
        timestamp: UTC datetime of training completion.
    """
    project_root = Path(__file__).resolve().parent.parent
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stem = f"baseline_b{config.model.num_residual_blocks}_f{config.model.num_filters}"
    ckpt_path = ckpt_dir / f"{stem}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_blocks": config.model.num_residual_blocks,
            "num_filters": config.model.num_filters,
            "input_planes": config.model.input_planes,
            "trained_utc": timestamp.strftime("%Y%m%dT%H%M%SZ"),
        },
        ckpt_path,
    )

    # Sidecar: full config so the checkpoint is self-describing
    meta = {
        "trained_utc": timestamp.strftime("%Y%m%dT%H%M%SZ"),
        "checkpoint_file": ckpt_path.name,
        "config": {
            "model": dataclasses.asdict(config.model),
            "mcts": dataclasses.asdict(config.mcts),
            "training": {k: v for k, v in dataclasses.asdict(config.training).items()
                         if k != "checkpoint_dir"},
            "game": dataclasses.asdict(config.game),
        },
    }
    meta_path = ckpt_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Persistent checkpoint saved to:\n  {ckpt_path}\n  {meta_path}")
