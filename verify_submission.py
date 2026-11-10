"""Verify the exported archive, then play both sides in Kaggle's local engine."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time

import numpy as np
import torch

from kaggle_environments import make
from src.game.board import Connect4Board
from src.neural_net.model import Connect4Net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive, checkpoint, output = args.archive.resolve(), args.checkpoint.resolve(), args.output.resolve()
    torch.set_num_threads(1)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference = Connect4Net(num_blocks=ckpt["num_blocks"], num_filters=ckpt["num_filters"]).eval()
    reference.load_state_dict(ckpt["model_state_dict"])
    report = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "checkpoint": str(checkpoint), "matches": []}
    with tempfile.TemporaryDirectory() as folder:
        with tarfile.open(archive) as handle:
            assert set(handle.getnames()) == {"main.py", "weights.npz"}
            for name in handle.getnames():
                Path(folder, name).write_bytes(handle.extractfile(name).read())
        os.chdir(folder)
        spec = importlib.util.spec_from_file_location("submitted_agent", Path(folder, "main.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(2026)
        board = Connect4Board()
        max_logit_error = max_value_error = 0.0
        for _ in range(40):
            if board.is_terminal():
                board = Connect4Board()
            state = board.encode()[None]
            logits, value = module._predict(state)
            with torch.no_grad():
                expected_logits, expected_value = reference(torch.from_numpy(state))
            max_logit_error = max(max_logit_error, float(np.max(np.abs(logits - expected_logits[0].numpy()))))
            max_value_error = max(max_value_error, abs(value - expected_value.item()))
            np.testing.assert_allclose(logits, expected_logits[0].numpy(), atol=1e-4, rtol=1e-4)
            assert abs(value - expected_value.item()) < 1e-4
            board = board.make_move(int(rng.choice(board.get_legal_moves())))
        report["parity"] = {"positions": 40, "max_logit_error": max_logit_error,
                            "max_value_error": max_value_error}
        for opponent in ("random", "negamax"):
            for side in (0, 1):
                durations = []
                module._weights = None
                def timed_agent(observation, configuration):
                    started = time.perf_counter()
                    action = module.my_agent(observation, configuration)
                    durations.append(time.perf_counter() - started)
                    assert isinstance(action, int) and 0 <= action < 7 and observation.board[action] == 0
                    return action
                agents = [timed_agent, opponent] if side == 0 else [opponent, timed_agent]
                env = make("connectx", debug=True)
                env.run(agents)
                assert all(s.status == "DONE" for s in env.state), [s.status for s in env.state]
                assert max(durations) < 2.0, durations
                result = {"opponent": opponent, "player": side + 1, "reward": env.state[side].reward,
                          "agent_moves": len(durations), "max_move_seconds": max(durations),
                          "average_move_seconds": sum(durations) / len(durations)}
                report["matches"].append(result)
                print("MATCH", json.dumps(result), flush=True)
    output.write_text(json.dumps(report, indent=2))
    print("VERIFIED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
