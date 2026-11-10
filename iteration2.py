"""Generate deeper self-play targets and conservatively fine-tune the large model."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from finetune_20min import load_model, save_checkpoint
from src.training.self_play import BatchedSelfPlay
from src.utils.config import MCTSConfig


def training_targets(states, policies, outcomes, teacher_logits, teacher_values):
    legal = (states[:, 0, -1] + states[:, 1, -1]) == 0
    teacher_policy = torch.softmax(teacher_logits.masked_fill(~legal, -float("inf")), -1)
    policy = 0.8 * policies + 0.2 * teacher_policy
    value = 0.5 * outcomes + 0.5 * teacher_values
    return policy, value


def generate(args):
    model = load_model("models/large.pt")
    config = MCTSConfig(num_simulations=512, dirichlet_epsilon=0.15,
                        temperature_threshold=12)
    started = time.monotonic()
    for chunk in range(2):
        samples = BatchedSelfPlay(model, config, batch_size=64).generate_games(64)
        states = np.stack([s.state for s in samples])
        policies = np.stack([s.policy for s in samples])
        outcomes = np.array([s.value for s in samples], dtype=np.float32)
        assert np.isfinite(states).all() and np.isfinite(policies).all()
        assert np.allclose(policies.sum(1), 1) and np.isin(outcomes, [-1, 0, 1]).all()
        path = args.output / f"selfplay_{args.seed}_{chunk}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, states=states, policies=policies, outcomes=outcomes)
        temporary.replace(path)
        print("SHARD", str(path), "games", 64, "samples", len(samples),
              "elapsed", time.monotonic() - started, flush=True)


def train(args):
    paths = sorted(args.data.glob("selfplay_*.npz"))
    assert len(paths) >= 5, "Need at least one complete shard from each generation worker"
    chunks = []
    for path in paths:
        with np.load(path) as data:
            chunks.append({key: data[key] for key in ("states", "policies", "outcomes")})
    data = {key: torch.from_numpy(np.concatenate([c[key] for c in chunks])).cuda()
            for key in ("states", "policies", "outcomes")}
    baseline = load_model("models/large.pt")
    policy_targets, value_targets = [], []
    with torch.no_grad():
        for start in range(0, len(data["states"]), 512):
            batch = slice(start, start + 512)
            logits, values = baseline(data["states"][batch])
            policy, value = training_targets(data["states"][batch], data["policies"][batch],
                                             data["outcomes"][batch], logits, values[:, 0])
            policy_targets.append(policy)
            value_targets.append(value)
    policies, values = torch.cat(policy_targets), torch.cat(value_targets)
    del baseline
    model = load_model("models/large.pt")
    original_stats = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    report = {"learning_rate": args.lr, "seed": args.seed,
              "shards": [str(p) for p in paths], "samples": len(policies),
              "selfplay_simulations": 512, "frozen_batchnorm_statistics": True, "epochs": []}
    started = time.monotonic()
    # eval() freezes BatchNorm statistics; autograd still updates the weights.
    model.eval()
    for epoch in range(1, 7):
        order = torch.randperm(len(policies), device="cuda")
        losses = []
        for indices in order.split(512):
            logits, value = model(data["states"][indices])
            loss = -(policies[indices] * F.log_softmax(logits, -1)).sum(-1).mean()
            loss += F.mse_loss(value[:, 0], values[indices])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        for name, value in model.named_buffers():
            assert torch.equal(value, original_stats[name]), name
        result = {"epoch": epoch, "loss": float(np.mean(losses)),
                  "elapsed_seconds": time.monotonic() - started}
        report["epochs"].append(result)
        print("TRAIN", json.dumps(result), flush=True)
        if epoch in (3, 6):
            save_checkpoint(model, args.output / f"candidate_{epoch}.pt", epoch=epoch,
                            learning_rate=args.lr, optimizer_state_dict=optimizer.state_dict())
        (args.output / "report.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["generate", "train"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    assert args.task != "train" or args.data is not None
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    (generate if args.task == "generate" else train)(args)


if __name__ == "__main__":
    main()
