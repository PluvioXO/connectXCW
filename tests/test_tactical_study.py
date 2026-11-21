import argparse
import json

from scripts import tactical_study
from src.training.double_dqn import make_network


def test_evaluation_keeps_moves_and_training_game_count_separate(tmp_path, monkeypatch):
    testpack = tmp_path / "testpack.json"
    testpack.write_text(json.dumps({"curve": [[]], "curve_seed": 1}))
    training = tmp_path / "training"
    training.mkdir()
    (training / "move-020000.pt").write_bytes(b"test checkpoint")
    monkeypatch.setattr(tactical_study.torch, "load", lambda *args, **kwargs: {
        "weights": make_network().state_dict(), "steps": 20000, "updates": 4745,
        "games": 931, "seed": 61, "variant": "plain",
    })
    traces = [{"opening": [], "moves_after_opening": [3], "player": 1}]
    monkeypatch.setattr(tactical_study, "evaluate_games", lambda *args: {
        "games": traces, "wins": 1, "draws": 0, "losses": 0, "score": 1,
    })
    output = tmp_path / "output"
    tactical_study.evaluate(argparse.Namespace(training=training, output=output, testpack=testpack,
                                              step=20000))
    result = json.loads((output / "curve-greedy-20000-minimax-2.json").read_text())
    assert result["games"] == traces
    assert result["completed_training_games"] == 931
