"""Tests for the Connect 4 neural network model (Iteration 2).

Covers: ResidualBlock, Connect4Net output shapes, value range, NaN/Inf safety,
gradients, save/load, get_policy helper, multi-config forward passes,
and real board encoding integration.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.game.board import Connect4Board
from src.game.constants import COLS, ROWS
from src.neural_net.model import (
    POLICY_HEAD_FILTERS,
    Connect4Net,
    ResidualBlock,
    VALUE_HEAD_FILTERS,
    get_policy,
)
from src.utils.config import ModelConfig


# ---------------------------------------------------------------------------
# TestResidualBlock
# ---------------------------------------------------------------------------


class TestResidualBlock:
    """Tests for the ResidualBlock module in isolation."""

    @pytest.mark.parametrize(
        "batch_size, num_filters",
        [(1, 32), (4, 64), (1, 128)],
    )
    def test_output_shape_matches_input(self, batch_size: int, num_filters: int) -> None:
        """ResidualBlock must not change the tensor shape."""
        block = ResidualBlock(num_filters)
        x = torch.randn(batch_size, num_filters, ROWS, COLS)
        out = block(x)
        assert out.shape == x.shape

    def test_no_nan_in_output(self) -> None:
        """ResidualBlock output must not contain NaN values."""
        block = ResidualBlock(64)
        x = torch.randn(2, 64, ROWS, COLS)
        out = block(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# TestConnect4NetOutputShapes
# ---------------------------------------------------------------------------


class TestConnect4NetOutputShapes:
    """Tests that the model produces tensors of the expected shapes."""

    def test_policy_logits_shape_batch_1(self, tiny_model: Connect4Net) -> None:
        """Policy logits must be (1, 7) for a single input."""
        x = torch.randn(1, 3, ROWS, COLS)
        policy_logits, _ = tiny_model(x)
        assert policy_logits.shape == (1, COLS)

    def test_value_shape_batch_1(self, tiny_model: Connect4Net) -> None:
        """Value must be (1, 1) for a single input."""
        x = torch.randn(1, 3, ROWS, COLS)
        _, value = tiny_model(x)
        assert value.shape == (1, 1)

    @pytest.mark.parametrize("batch_size", [2, 8, 32])
    def test_policy_shape_batch_N(self, tiny_model: Connect4Net, batch_size: int) -> None:
        """Policy logits must be (N, 7) for batch size N."""
        x = torch.randn(batch_size, 3, ROWS, COLS)
        policy_logits, _ = tiny_model(x)
        assert policy_logits.shape == (batch_size, COLS)

    @pytest.mark.parametrize("batch_size", [2, 8, 32])
    def test_value_shape_batch_N(self, tiny_model: Connect4Net, batch_size: int) -> None:
        """Value must be (N, 1) for batch size N."""
        x = torch.randn(batch_size, 3, ROWS, COLS)
        _, value = tiny_model(x)
        assert value.shape == (batch_size, 1)


# ---------------------------------------------------------------------------
# TestConnect4NetValueRange
# ---------------------------------------------------------------------------


class TestConnect4NetValueRange:
    """Tests that the value head output stays within [-1, 1]."""

    def test_value_in_minus_one_to_plus_one(self, tiny_model: Connect4Net) -> None:
        """Value must be in [-1, 1] for 10 random batches."""
        for _ in range(10):
            x = torch.randn(4, 3, ROWS, COLS)
            _, value = tiny_model(x)
            assert value.min().item() >= -1.0 - 1e-6
            assert value.max().item() <= 1.0 + 1e-6

    def test_value_range_with_extreme_inputs(self, tiny_model: Connect4Net) -> None:
        """Tanh must clamp the value to [-1, 1] even with extreme inputs (~±100)."""
        x = torch.full((2, 3, ROWS, COLS), 100.0)
        _, value_pos = tiny_model(x)
        assert value_pos.min().item() >= -1.0 - 1e-6
        assert value_pos.max().item() <= 1.0 + 1e-6

        x_neg = torch.full((2, 3, ROWS, COLS), -100.0)
        _, value_neg = tiny_model(x_neg)
        assert value_neg.min().item() >= -1.0 - 1e-6
        assert value_neg.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TestConnect4NetNaNSafety
# ---------------------------------------------------------------------------


class TestConnect4NetNaNSafety:
    """Tests that no NaN or Inf values appear in model outputs."""

    def test_no_nan_in_policy_logits(self, tiny_model: Connect4Net) -> None:
        """Policy logits must not contain NaN."""
        x = torch.randn(4, 3, ROWS, COLS)
        policy_logits, _ = tiny_model(x)
        assert not torch.isnan(policy_logits).any()

    def test_no_nan_in_value(self, tiny_model: Connect4Net) -> None:
        """Value must not contain NaN."""
        x = torch.randn(4, 3, ROWS, COLS)
        _, value = tiny_model(x)
        assert not torch.isnan(value).any()

    def test_no_inf_in_policy_logits(self, tiny_model: Connect4Net) -> None:
        """Policy logits must not contain Inf."""
        x = torch.randn(4, 3, ROWS, COLS)
        policy_logits, _ = tiny_model(x)
        assert not torch.isinf(policy_logits).any()


# ---------------------------------------------------------------------------
# TestConnect4NetGradients
# ---------------------------------------------------------------------------


class TestConnect4NetGradients:
    """Tests that backpropagation works correctly through the full network."""

    def test_backward_pass_runs(self, tiny_model: Connect4Net) -> None:
        """Calling .backward() must not raise an exception."""
        model = Connect4Net(num_blocks=2, num_filters=32)
        model.train()
        x = torch.randn(2, 3, ROWS, COLS)
        policy_logits, value = model(x)
        loss = policy_logits.sum() + value.sum()
        loss.backward()  # must not raise

    def test_all_params_have_gradients(self, tiny_model: Connect4Net) -> None:
        """Every parameter must receive a gradient after backward."""
        model = Connect4Net(num_blocks=2, num_filters=32)
        model.train()
        x = torch.randn(2, 3, ROWS, COLS)
        policy_logits, value = model(x)
        loss = policy_logits.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_are_finite(self, tiny_model: Connect4Net) -> None:
        """All gradients must be finite (no NaN or Inf)."""
        model = Connect4Net(num_blocks=2, num_filters=32)
        model.train()
        x = torch.randn(2, 3, ROWS, COLS)
        policy_logits, value = model(x)
        loss = policy_logits.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# TestConnect4NetSaveLoad
# ---------------------------------------------------------------------------


class TestConnect4NetSaveLoad:
    """Tests for model checkpoint save/load correctness."""

    def test_save_load_preserves_policy_output(self, tmp_path: pytest.TempPathFactory) -> None:
        """Policy logits after loading must match the original output."""
        model = Connect4Net(num_blocks=2, num_filters=32)
        model.eval()
        x = torch.randn(1, 3, ROWS, COLS)

        with torch.no_grad():
            policy_before, _ = model(x)

        checkpoint_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        loaded = Connect4Net(num_blocks=2, num_filters=32)
        loaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        loaded.eval()

        with torch.no_grad():
            policy_after, _ = loaded(x)

        assert torch.allclose(policy_before, policy_after, atol=1e-6)

    def test_save_load_preserves_value_output(self, tmp_path: pytest.TempPathFactory) -> None:
        """Value output after loading must match the original output."""
        model = Connect4Net(num_blocks=2, num_filters=32)
        model.eval()
        x = torch.randn(1, 3, ROWS, COLS)

        with torch.no_grad():
            _, value_before = model(x)

        checkpoint_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        loaded = Connect4Net(num_blocks=2, num_filters=32)
        loaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        loaded.eval()

        with torch.no_grad():
            _, value_after = loaded(x)

        assert torch.allclose(value_before, value_after, atol=1e-6)

    def test_mismatched_config_raises_on_load(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loading state dict from a different architecture must raise RuntimeError."""
        big_model = Connect4Net(num_blocks=5, num_filters=128)
        checkpoint_path = tmp_path / "big_model.pt"
        torch.save(big_model.state_dict(), checkpoint_path)

        small_model = Connect4Net(num_blocks=2, num_filters=32)
        with pytest.raises(RuntimeError):
            small_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))


# ---------------------------------------------------------------------------
# TestGetPolicy
# ---------------------------------------------------------------------------


class TestGetPolicy:
    """Tests for the get_policy helper function."""

    def test_policy_sums_to_one_all_legal(self, tiny_model: Connect4Net) -> None:
        """Policy must sum to 1.0 when all moves are legal."""
        x = torch.randn(1, 3, ROWS, COLS)
        mask = np.ones(COLS, dtype=bool)
        policy, _ = get_policy(tiny_model, x, mask)
        assert abs(policy.sum().item() - 1.0) < 1e-5

    def test_illegal_moves_get_zero_probability(self, tiny_model: Connect4Net) -> None:
        """Columns marked illegal must have exactly 0 probability."""
        x = torch.randn(1, 3, ROWS, COLS)
        mask = np.array([True, False, True, False, True, False, True])
        policy, _ = get_policy(tiny_model, x, mask)
        illegal_indices = [1, 3, 5]
        for idx in illegal_indices:
            assert policy[idx].item() == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize(
        "legal_cols",
        [[3], [0, 6], [1, 2, 3, 4, 5]],
    )
    def test_legal_moves_sum_to_one(
        self, tiny_model: Connect4Net, legal_cols: list[int]
    ) -> None:
        """Legal move probabilities must sum to 1.0."""
        x = torch.randn(1, 3, ROWS, COLS)
        mask = np.zeros(COLS, dtype=bool)
        for col in legal_cols:
            mask[col] = True
        policy, _ = get_policy(tiny_model, x, mask)
        legal_sum = sum(policy[col].item() for col in legal_cols)
        assert abs(legal_sum - 1.0) < 1e-5

    def test_single_legal_move_gets_probability_one(self, tiny_model: Connect4Net) -> None:
        """When only one move is legal, it must receive probability 1.0."""
        x = torch.randn(1, 3, ROWS, COLS)
        mask = np.zeros(COLS, dtype=bool)
        mask[3] = True
        policy, _ = get_policy(tiny_model, x, mask)
        assert policy[3].item() == pytest.approx(1.0, abs=1e-6)

    def test_output_shapes(self, tiny_model: Connect4Net) -> None:
        """get_policy must return policy of shape (7,) and scalar value."""
        x = torch.randn(1, 3, ROWS, COLS)
        mask = np.ones(COLS, dtype=bool)
        policy, value = get_policy(tiny_model, x, mask)
        assert policy.shape == (COLS,)
        assert value.shape == ()  # scalar

    def test_accepts_real_board_encoding(
        self, tiny_model: Connect4Net, empty_board_tensor: torch.Tensor
    ) -> None:
        """get_policy must run on an actual encoded board without error."""
        mask = np.ones(COLS, dtype=bool)
        policy, value = get_policy(tiny_model, empty_board_tensor, mask)
        assert policy.shape == (COLS,)
        assert abs(policy.sum().item() - 1.0) < 1e-5
        assert -1.0 <= value.item() <= 1.0


# ---------------------------------------------------------------------------
# TestConnect4NetConfigs
# ---------------------------------------------------------------------------


class TestConnect4NetConfigs:
    """Tests that multiple configs produce valid forward passes."""

    @pytest.mark.parametrize(
        "num_blocks, num_filters",
        [
            (2, 32),   # tiny
            (3, 64),   # small
            (5, 128),  # full
        ],
    )
    def test_all_configs_forward_pass(self, num_blocks: int, num_filters: int) -> None:
        """All standard configs must produce correctly shaped outputs."""
        model = Connect4Net(num_blocks=num_blocks, num_filters=num_filters)
        model.eval()
        x = torch.randn(1, 3, ROWS, COLS)
        with torch.no_grad():
            policy_logits, value = model(x)
        assert policy_logits.shape == (1, COLS)
        assert value.shape == (1, 1)

    def test_parameter_count_scales_with_size(self) -> None:
        """Full config must have more parameters than small, which has more than tiny."""
        tiny = Connect4Net(num_blocks=2, num_filters=32)
        small = Connect4Net(num_blocks=3, num_filters=64)
        full = Connect4Net(num_blocks=5, num_filters=128)

        def count_params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        assert count_params(tiny) < count_params(small) < count_params(full)


# ---------------------------------------------------------------------------
# TestConnect4NetWithRealBoardEncoding
# ---------------------------------------------------------------------------


class TestConnect4NetWithRealBoardEncoding:
    """Integration tests using real board encodings from the game engine."""

    def test_forward_on_empty_board(
        self, tiny_model: Connect4Net, empty_board_tensor: torch.Tensor
    ) -> None:
        """Model must produce valid outputs on an empty board encoding."""
        with torch.no_grad():
            policy_logits, value = tiny_model(empty_board_tensor)
        assert policy_logits.shape == (1, COLS)
        assert value.shape == (1, 1)
        assert not torch.isnan(policy_logits).any()
        assert not torch.isnan(value).any()
        assert -1.0 <= value.item() <= 1.0

    def test_batch_of_board_encodings(self, tiny_model: Connect4Net) -> None:
        """Model must handle a batched stack of board encodings."""
        boards = [
            Connect4Board(),
            Connect4Board().make_move(3),
            Connect4Board().make_move(3).make_move(4),
            Connect4Board().make_move(0).make_move(6).make_move(3),
        ]
        import numpy as np

        encoded = np.stack([b.encode() for b in boards])  # (4, 3, 6, 7)
        # Convert via tolist() because this PyTorch build has no NumPy C bridge.
        batch_tensor = torch.tensor(encoded.tolist(), dtype=torch.float32)  # (4, 3, 6, 7)

        with torch.no_grad():
            policy_logits, value = tiny_model(batch_tensor)

        assert policy_logits.shape == (4, COLS)
        assert value.shape == (4, 1)
        assert not torch.isnan(policy_logits).any()
        assert (value >= -1.0).all() and (value <= 1.0).all()
