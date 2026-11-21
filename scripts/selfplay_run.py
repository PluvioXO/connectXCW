"""Persistent self-play branch with paired candidate evaluation and saved data."""
import argparse
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from finetune_20min import arena, load_model
from iteration2 import train
from scripts.parallel_selfplay import run_workers
from src.training.self_play import BatchedSelfPlay
from src.utils.config import MCTSConfig


def initialize_learner(checkpoint, destination):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    seed = {key: saved[key] for key in ('model_state_dict', 'num_blocks', 'num_filters')}
    seed.update(training_mode='persistent_raw_v1', optimizer_steps=0)
    temporary = destination.with_suffix('.tmp')
    torch.save(seed, temporary)
    temporary.replace(destination)


def promotion(scores):
    scores = np.asarray(scores, dtype=float)
    assert len(scores) >= 64 and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    rng = np.random.default_rng(715)
    lower, upper = np.quantile(rng.choice(scores, (10000, len(scores))).mean(1), [.025, .975])
    return {"score": float(scores.mean()), "paired_ci95": [float(lower), float(upper)],
            "promoted": bool(scores.mean() >= .55 and lower > .5)}


def resume_progress(output):
    """Restore the last committed champion; incomplete iterations may be replayed."""
    path = output / 'metrics.jsonl'
    raw = path.read_bytes()
    complete = raw[:raw.rfind(b'\n') + 1]
    events = [json.loads(line) for line in complete.splitlines()]
    finished = [e for e in events if e['phase'] == 'iteration_complete']
    completed = finished[-1]['iteration'] if finished else 0
    assert [e['iteration'] for e in finished] == list(range(1, completed + 1))
    promoted = [e for e in finished if e['promoted']]
    champion = Path(promoted[-1]['candidate']) if promoted else Path('models/large.pt')
    expected = promoted[-1]['best_sha256'] if promoted else json.loads((output / 'plan.json').read_text())['starting_sha256']
    assert hashlib.sha256(champion.read_bytes()).hexdigest() == expected
    if raw != complete:
        (output / f'metrics_interrupted_{time.time_ns()}.jsonl').write_bytes(raw)
        path.write_bytes(complete)
    shutil.copy2(champion, output / 'best.tmp')
    (output / 'best.tmp').replace(output / 'best.pt')
    plan = json.loads((output / 'plan.json').read_text())
    if plan.get('persistent_learner', False):
        if completed >= plan['persistent_start_iteration']:
            assert finished[-1]['training_mode'] == 'persistent_raw_v1'
            learner = Path(finished[-1]['candidate'])
            expected_learner = finished[-1]['learner_sha256']
        else:
            learner = output / 'learner_seed.pt'
            expected_learner = plan['learner_seed_sha256']
        assert hashlib.sha256(learner.read_bytes()).hexdigest() == expected_learner
        shutil.copy2(learner, output / 'learner.tmp')
        (output / 'learner.tmp').replace(output / 'learner.pt')
    shards = list(output.glob('iteration_*/selfplay_*.npz'))
    assert all(int(p.parent.name.split('_')[1]) <= completed + 1 for p in shards)
    return completed, len(shards) * 64, max((e['elapsed_seconds'] for e in events), default=0.)


def run(args):
    resume = getattr(args, 'resume', False)
    persistent = getattr(args, 'persistent_learner', False)
    replay_shards = getattr(args, 'replay_shards', 16)
    epochs = getattr(args, 'epochs', 4)
    max_games = getattr(args, 'max_selfplay_games', None)
    if max_games is not None and (max_games <= 0 or max_games % 64):
        raise ValueError('--max-selfplay-games must be a positive multiple of 64')
    assert replay_shards > 0 and epochs > 0
    devices = [int(x) for x in args.gpus.split(',')] if getattr(args, 'gpus', None) else []
    full_chunks = max(4, len(devices))
    if devices:
        assert 1 <= len(devices) <= 6 and len(devices) == len(set(devices)) and min(devices) >= 0
        assert args.gpus == os.environ.get('CUDA_VISIBLE_DEVICES')
    if resume:
        plan = json.loads((args.output / 'plan.json').read_text())
        assert all(plan[k] == getattr(args, k) for k in ('seed', 'lr', 'iterations'))
        assert plan['gpu'] == os.environ.get('CUDA_VISIBLE_DEVICES')
        assert plan.get('persistent_learner', False) == persistent
        assert plan['replay_shards'] == replay_shards and plan['epochs'] == epochs
        assert plan.get('max_selfplay_games') == max_games
        completed, games, elapsed = resume_progress(args.output)
        if max_games is not None and games > max_games:
            raise ValueError('Saved self-play games already exceed the game limit')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        completed, games, elapsed = 0, 0, 0.
    seed = args.seed + completed * 1000 if resume else args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    best = args.output / "best.pt"
    learner = args.output / 'learner.pt'
    if not resume:
        shutil.copy2("models/large.pt", best)
        if persistent:
            initialize_learner(best, args.output / 'learner_seed.pt')
            shutil.copy2(args.output / 'learner_seed.pt', learner)
    plan = {**vars(args), "output": str(args.output), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "starting_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
            "games_per_iteration": full_chunks * 64, "search_simulations": 512, "epochs": epochs,
            "persistent_learner": persistent, "replay_shards": replay_shards,
            "training_policy": "100% search visits" if persistent else "80% search / 20% retained model",
            "training_value": "100% game outcome" if persistent else "50% game outcome / 50% retained model",
            "promotion": "paired score >= .55 and bootstrap 95% lower bound > .5",
            "arena_games": 128, "arena_simulations": 512}
    if not resume:
        if persistent:
            plan.update(persistent_start_iteration=1,
                        learner_seed_sha256=hashlib.sha256(learner.read_bytes()).hexdigest())
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    started = time.monotonic() - elapsed

    def record(phase, **fields):
        event = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "phase": phase, "iteration": iteration, "completed_selfplay_games": games,
                 "elapsed_seconds": time.monotonic() - started, **fields}
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(event, indent=2))
        temporary.replace(args.output / "status.json")
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    iteration = completed
    model = load_model(best)
    check = arena(model, model, 2, 8, 123, float("inf"))
    assert check["score"] == .5 and check["pair_scores"] == [.5, .5], check
    del model
    record("resume_self_check" if resume else "arena_self_check", result=check)
    for next_iteration in range(completed + 1, args.iterations + 1):
        folder = args.output / f"iteration_{next_iteration:04d}"
        observed = set(folder.glob('selfplay_*.npz'))
        chunks = full_chunks
        if max_games is not None:
            # The resume count already includes any shards from this unfinished iteration.
            chunks = min(full_chunks, len(observed) + (max_games - games) // 64)
            if not observed.issubset(folder / f'selfplay_{chunk}.npz' for chunk in range(chunks)):
                raise ValueError('Saved self-play shards do not fit the remaining game budget')
            if chunks == 0:
                break
        iteration = next_iteration
        source = learner if persistent else best
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        folder.mkdir(exist_ok=resume)
        if devices:
            def progress():
                nonlocal games
                for path in sorted(set(folder.glob('selfplay_*.npz')) - observed):
                    with np.load(path) as shard:
                        count = len(shard['outcomes'])
                    observed.add(path)
                    games += 64
                    record('self_play_complete', chunk=int(path.stem.split('_')[-1]), samples=count)
            generation_devices = devices[:chunks]
            record('self_play', gpus=generation_devices, source_checkpoint=str(source), source_sha256=source_sha)
            run_workers(generation_devices, 'generate', source, folder, args.seed + iteration * 1000,
                        progress, chunks=chunks)
            assert len(observed) == chunks
        else:
            model = load_model(source)
            generator = BatchedSelfPlay(model, MCTSConfig(num_simulations=512,
                                       dirichlet_epsilon=.15, temperature_threshold=12), batch_size=64)
            for chunk in range(chunks):
                if (folder / f"selfplay_{chunk}.npz").exists():
                    continue
                record("self_play", chunk=chunk)
                samples = generator.generate_games(64)
                states = np.stack([s.state for s in samples])
                policies = np.stack([s.policy for s in samples])
                outcomes = np.array([s.value for s in samples], dtype=np.float32)
                assert np.isfinite(states).all() and np.isfinite(policies).all()
                assert np.allclose(policies.sum(1), 1) and np.isin(outcomes, [-1, 0, 1]).all()
                with (folder / f"selfplay_{chunk}.tmp").open("wb") as stream:
                    np.savez_compressed(stream, states=states, policies=policies, outcomes=outcomes)
                (folder / f"selfplay_{chunk}.tmp").replace(folder / f"selfplay_{chunk}.npz")
                games += 64
                record("self_play_complete", chunk=chunk, samples=len(samples))
            del generator, model
        paths = sorted(args.output.glob("iteration_*/selfplay_*.npz"))[-replay_shards:]
        record("training", shards=[str(p) for p in paths])
        extra = {'persistent': True} if persistent else {}
        training = train(argparse.Namespace(data=folder, output=folder, lr=args.lr, seed=args.seed),
                         checkpoint=source, epochs=epochs, shard_paths=paths, **extra)
        candidate = folder / f"candidate_{epochs}.pt"
        if devices:
            record('arena', gpus=devices)
            run_workers(devices, 'arena', candidate, folder, args.seed + iteration * 10,
                        lambda: None, opponent=best)
            results = [json.loads((folder / f'arena_batch_{i}.json').read_text()) for i in range(len(devices))]
            for result in results:
                assert result['checkpoint_sha256'] == hashlib.sha256(candidate.read_bytes()).hexdigest()
                assert result['opponent_sha256'] == hashlib.sha256(best.read_bytes()).hexdigest()
                assert result['games'] == 2 * len(result['pair_scores'])
            assert sum(result['games'] for result in results) == 128
            keys = [tuple(key) for result in results for key in result['opening_keys']]
            assert len(keys) == len(set(keys)) == 64, 'Arena openings must be distinct across workers'
        else:
            model, champion = load_model(candidate), load_model(best)
            results = []
            for batch in range(2):
                record("arena", batch=batch)
                results.append(arena(model, champion, 32, 512,
                                     args.seed + iteration * 10 + batch, float("inf")))
            del model, champion
        outcome = promotion([score for r in results for score in r["pair_scores"]])
        (folder / "arena.json").write_text(json.dumps({"batches": results, **outcome}, indent=2))
        if outcome["promoted"]:
            shutil.copy2(candidate, args.output / "best.tmp")
            (args.output / "best.tmp").replace(best)
        fields = {}
        if persistent:
            shutil.copy2(candidate, args.output / 'learner.tmp')
            (args.output / 'learner.tmp').replace(learner)
            fields = dict(training_mode='persistent_raw_v1', learner_sha256=hashlib.sha256(learner.read_bytes()).hexdigest(),
                          optimizer_steps=training['optimizer_steps_after'], parent_learner_sha256=source_sha)
        record("iteration_complete", **outcome, candidate=str(candidate),
               best_sha256=hashlib.sha256(best.read_bytes()).hexdigest(), **fields)
    if max_games is not None and games == max_games:
        record("complete", stop_reason="selfplay_game_limit")
    else:
        record("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-selfplay-games", type=int, help="Cumulative self-play limit in multiples of 64")
    parser.add_argument("--gpus", help="Physical GPUs sharing self-play and arena work, e.g. 0,1,2,3,4,5")
    parser.add_argument("--resume", action="store_true", help="Recover committed champion and completed replay shards")
    parser.add_argument('--persistent-learner', action='store_true', help='Carry learner weights and Adam state across candidate evaluations')
    parser.add_argument('--replay-shards', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    run(parser.parse_args())
