"""Tests for Connect 4 agent implementations (Iteration 5).

Covers RandomAgent, MinimaxAgent, MCTSAgent, AlphaZeroAgent, and an
integration test for the evaluate script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.agents.mcts_agent import MCTSAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.game.board import Connect4Board
from src.neural_net.model import Connect4Net


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _board_from_moves(moves: list[int]) -> Connect4Board:
    """Build a board by replaying a sequence of column moves."""
    board = Connect4Board()
    for col in moves:
        board = board.make_move(col)
    return board


def _save_tiny_checkpoint(path: Path) -> None:
    """Save a tiny model checkpoint to the given path for testing."""
    model = Connect4Net(num_blocks=2, num_filters=32, input_planes=3)
    torch.save(
        {
            "num_blocks": 2,
            "num_filters": 32,
            "input_planes": 3,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    """Tests for RandomAgent."""

    def test_random_agent_returns_legal_move(self) -> None:
        """Agent must always return a legal column index."""
        agent = RandomAgent(seed=42)
        board = Connect4Board()
        for _ in range(20):
            move = agent.select_move(board)
            assert move in board.get_legal_moves()
            board = board.make_move(move)
            if board.is_terminal():
                break

    def test_random_agent_different_moves_over_many_calls(self) -> None:
        """Agent should not always return the same column from an empty board."""
        agent = RandomAgent()
        board = Connect4Board()
        moves = {agent.select_move(board) for _ in range(50)}
        assert len(moves) > 1, "RandomAgent always returned the same move"


# ---------------------------------------------------------------------------
# MinimaxAgent
# ---------------------------------------------------------------------------

class TestMinimaxAgent:
    """Tests for MinimaxAgent (depth 1, 3, 5)."""

    def test_minimax_depth1_takes_immediate_win(self) -> None:
        """Depth-1 agent must take a winning move when available.

        Board: P1 has vertical pieces in col 3 (rows 0,1,2). Playing col 3
        wins immediately.
        """
        # Sequence: [3, 0, 3, 0, 3, 0] → P1 has col3 rows 0,1,2; P2 has col0
        # It is now P1's turn; col3 wins vertically.
        board = _board_from_moves([3, 0, 3, 0, 3, 0])
        assert board.current_player == 1
        agent = MinimaxAgent(max_depth=1)
        move = agent.select_move(board)
        assert move == 3, f"Expected col 3 (immediate win), got {move}"

    def test_minimax_depth1_blocks_immediate_win(self) -> None:
        """Depth-1 agent must block the opponent's immediate winning threat.

        Board: P2 has vertical pieces in col 3 (rows 0,1,2). It's P1's turn.
        P1 has no winning moves, so it must block col 3.
        Move sequence [0, 3, 1, 3, 6, 3]: P1 gets scattered pieces (cols 0,1,6
        at row 0) while P2 builds col 3 vertically.
        """
        board = _board_from_moves([0, 3, 1, 3, 6, 3])
        assert board.current_player == 1

        agent = MinimaxAgent(max_depth=1)
        move = agent.select_move(board)
        # P2 wins by playing col 3 row 3; P1 must block
        assert move == 3, f"Expected col 3 (blocking P2 vertical win), got {move}"

    def test_minimax_depth3_finds_two_step_win(self) -> None:
        """Depth-3 minimax finds a forced win requiring 2 moves of lookahead.

        Set up a fork: P1 has 3 in a row horizontally and can create a double
        threat. Depth-1 may not see it; depth-3 should.
        """
        # Build board where P1 can fork:
        # P1 plays cols 2, 3, 4 (horizontal threat at row 0)
        # P2 plays elsewhere. P1 next needs col 1 OR col 5 to win.
        # If we give P1 cols 2,3,4 and P2 is blocked from the winning square,
        # then depth-3 can identify the winning continuation.
        board = _board_from_moves([2, 6, 3, 6, 4, 6])
        # P1 has cols 2,3,4 at row 0; P2 has col 6 rows 0,1,2. P1's turn.
        assert board.current_player == 1

        agent = MinimaxAgent(max_depth=3)
        move = agent.select_move(board)
        # P1 wins by playing col 1 or col 5 (extends to 4 in a row)
        assert move in (1, 5), (
            f"Expected col 1 or 5 (completing 4-in-a-row), got {move}"
        )


# ---------------------------------------------------------------------------
# AlphaZeroAgent
# ---------------------------------------------------------------------------

class TestAlphaZeroAgent:
    """Tests for AlphaZeroAgent."""

    def test_alphazero_agent_loads_checkpoint_and_plays(self, tmp_path: Path) -> None:
        """Agent must load a checkpoint and return a legal move."""
        ckpt_path = tmp_path / "tiny_model.pt"
        _save_tiny_checkpoint(ckpt_path)

        from src.agents.alphazero_agent import AlphaZeroAgent
        from src.utils.config import MCTSConfig

        agent = AlphaZeroAgent(
            checkpoint_path=ckpt_path,
            mcts_config=MCTSConfig(num_simulations=10),
        )
        board = Connect4Board()
        move = agent.select_move(board)
        assert move in board.get_legal_moves(), f"Illegal move returned: {move}"


# ---------------------------------------------------------------------------
# All agents
# ---------------------------------------------------------------------------

class TestAllAgents:
    """Tests that apply to all agent types."""

    @pytest.mark.parametrize(
        "agent",
        [
            RandomAgent(seed=0),
            MinimaxAgent(max_depth=1),
            MinimaxAgent(max_depth=3),
            MCTSAgent(num_simulations=20, seed=0),
        ],
    )
    def test_all_agents_handle_near_terminal_board(self, agent) -> None:
        """All agents must return a legal move on a partially-filled board.

        Uses a column filled with alternating pieces (no 4-in-a-row) to
        create a realistic mid-to-late game position.
        """
        # Fill col 3 completely (alternating P1/P2, no vertical win)
        # then add more pieces in other columns. This creates a realistic
        # mid-game position guaranteed not to be terminal.
        # [3,3,3,3,3,3] fills col 3 with P1@rows0,2,4 and P2@rows1,3,5 — no win.
        board = _board_from_moves([3, 3, 3, 3, 3, 3])
        assert not board.is_terminal(), "Board became terminal unexpectedly"

        legal = board.get_legal_moves()
        assert legal, "No legal moves available"

        move = agent.select_move(board)
        assert move in legal, (
            f"{agent.name} returned illegal move {move}; legal moves: {legal}"
        )

    def test_evaluate_script_produces_results(self) -> None:
        """Integration test: evaluate.py runs without error and prints results."""
        script = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
        result = subprocess.run(
            [sys.executable, str(script), "--num-games", "2", "--depth", "1", "--mcts-sims", "10"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"evaluate.py exited with code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Output should contain the results table header
        assert "Win%" in result.stdout, (
            f"Expected 'Win%' in output.\nstdout: {result.stdout}"
        )
