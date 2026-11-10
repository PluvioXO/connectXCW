"""Neural network model for Connect 4 AlphaZero agent.

Architecture: ResNet tower with separate policy and value heads.
Input: (batch, 3, 6, 7) board encoding
Output: policy_logits (batch, 7), value (batch, 1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.constants import COLS, ROWS

# Head architecture constants
POLICY_HEAD_FILTERS: int = 32  # channels after policy head 1×1 conv
VALUE_HEAD_FILTERS: int = 32   # channels after value head 1×1 conv
VALUE_HEAD_HIDDEN: int = 256   # hidden units in value MLP


class ResidualBlock(nn.Module):
    """A single residual block in the ResNet tower.

    Architecture: Conv-BN-ReLU-Conv-BN + skip connection → ReLU.
    Preserves spatial dimensions (6×7) throughout.

    Args:
        num_filters: Number of convolutional filters (channels).
    """

    def __init__(self, num_filters: int) -> None:
        """Initialize the residual block.

        Args:
            num_filters: Number of input and output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block to input tensor.

        Args:
            x: Input tensor of shape (batch, num_filters, 6, 7).

        Returns:
            Output tensor of same shape as input.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out, inplace=True)


class Connect4Net(nn.Module):
    """ResNet-based neural network for Connect 4 policy and value prediction.

    The network takes a board state encoding and outputs:
    - Policy logits: unnormalized log-probabilities for each of the 7 columns
    - Value: a scalar estimate in [-1, 1] of the current position's outcome

    Args:
        num_blocks: Number of residual blocks in the tower (default: 5).
        num_filters: Number of convolutional filters in the tower (default: 128).
        input_planes: Number of input feature planes (default: 3).
    """

    def __init__(
        self,
        num_blocks: int = 5,
        num_filters: int = 128,
        input_planes: int = 3,
    ) -> None:
        """Initialize the Connect4Net.

        Args:
            num_blocks: Number of residual blocks in the residual tower.
            num_filters: Number of channels in the residual tower.
            input_planes: Number of input feature planes.
        """
        super().__init__()

        # Stem: project input planes to num_filters
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_blocks)])

        # Policy head: 1×1 conv → flatten → linear → 7 logits
        policy_flat = POLICY_HEAD_FILTERS * ROWS * COLS
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, POLICY_HEAD_FILTERS, kernel_size=1, bias=False),
            nn.BatchNorm2d(POLICY_HEAD_FILTERS),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_flat, COLS),
        )

        # Value head: 1×1 conv → flatten → hidden linear → tanh output
        value_flat = VALUE_HEAD_FILTERS * ROWS * COLS
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, VALUE_HEAD_FILTERS, kernel_size=1, bias=False),
            nn.BatchNorm2d(VALUE_HEAD_FILTERS),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_flat, VALUE_HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(VALUE_HEAD_HIDDEN, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass.

        Args:
            x: Board state tensor of shape (batch, input_planes, ROWS, COLS).

        Returns:
            Tuple of (policy_logits, value):
                policy_logits: (batch, 7) raw logits, NOT softmaxed.
                value: (batch, 1) tanh-activated, range [-1, 1].
        """
        features = self.tower(self.stem(x))
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


def get_policy(
    model: Connect4Net,
    board_tensor: torch.Tensor,
    legal_moves_mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get move probabilities with illegal moves masked to zero.

    Runs the model, masks illegal moves to -inf, then applies softmax
    so the output sums to 1.0 over legal moves only.

    Args:
        model: The Connect4Net to evaluate.
        board_tensor: Encoded board state of shape (1, 3, 6, 7).
        legal_moves_mask: Boolean array of shape (7,), True for legal columns.

    Returns:
        Tuple of (policy, value):
            policy: (7,) probability distribution; illegal columns are 0.
            value: Scalar tensor in [-1, 1].

    Raises:
        AssertionError: If legal_moves_mask has no legal moves (would produce NaN).
    """
    assert legal_moves_mask.any(), "legal_moves_mask must have at least one legal move"

    policy_logits, value = model(board_tensor)

    # Clone to avoid in-place mutation on the computation graph
    logits = policy_logits.clone().squeeze(0)  # (7,)

    # Build mask tensor on the same device as the model.
    # Convert via tolist() to avoid issues with numpy.bool_ dtype in older torch/numpy combos.
    mask = torch.tensor(legal_moves_mask.tolist(), dtype=torch.bool, device=logits.device)

    # Set illegal columns to -inf before softmax
    logits[~mask] = float("-inf")

    policy = F.softmax(logits, dim=-1)
    return policy, value.squeeze()
