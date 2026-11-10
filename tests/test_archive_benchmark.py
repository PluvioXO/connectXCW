from types import SimpleNamespace

from archive_benchmark import openings, play_game
from src.game.board import Connect4Board


def test_benchmark_openings_are_unique_legal_and_reproducible():
    starts = openings(24, 82026)
    assert starts == openings(24, 82026)
    keys = []
    for moves in starts:
        board = Connect4Board()
        for col in moves:
            board = board.make_move(col)
            assert not board.is_terminal()
        keys.append((board._mask, board._position))
    assert len(set(keys)) == 24


def test_identical_agents_balance_when_players_are_swapped():
    def agent(obs, config):
        return next(c for c in range(7) if obs.board[c] == 0)
    modules = [SimpleNamespace(my_agent=agent)] * 2
    first = play_game(modules, [3, 2, 4, 1], 0)
    second = play_game(modules, [3, 2, 4, 1], 1)
    assert first["actions"] == second["actions"]
    assert first["candidate_reward"] + second["candidate_reward"] == 0
    assert first["failure"] is second["failure"] is None


def test_invalid_move_is_recorded_as_a_loss():
    bad = SimpleNamespace(my_agent=lambda obs, config: 7)
    good = SimpleNamespace(my_agent=lambda obs, config: next(c for c in range(7) if obs.board[c] == 0))
    result = play_game([good, bad], [], 0)
    assert result["candidate_reward"] == -1
    assert result["failure"]["agent"] == "candidate"
