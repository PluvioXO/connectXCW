"""One GPU's share of a common champion's self-play or paired arena."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from finetune_20min import arena, load_model, opening_boards
from src.training.self_play import BatchedSelfPlay
from src.utils.config import MCTSConfig


def run(args):
    os.environ.setdefault('CONNECTX_NATIVE_MCTS', '1')
    os.environ.setdefault('CONNECTX_CUDA_GRAPHS', '1')
    native = os.environ['CONNECTX_NATIVE_MCTS'] == '1'
    if native:
        from src.mcts.native import library
        library()  # Fail before generating data if the compiled backend is missing or incompatible.
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    search_dir = Path(__file__).resolve().parents[1] / 'src/mcts'
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(json.dumps({'event': 'worker_start', 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'task': args.task, 'worker': args.worker, 'torch_threads': torch.get_num_threads(),
                      'checkpoint_sha256': checkpoint_sha,
                      'native_mcts': native, 'cuda_graphs': os.environ['CONNECTX_CUDA_GRAPHS'] == '1',
                      'native_binary_sha256': hashlib.sha256((search_dir / '_native_search.so').read_bytes()).hexdigest() if native else None,
                      'search_sha256': hashlib.sha256((search_dir / 'search.py').read_bytes()).hexdigest()}), flush=True)
    model = load_model(args.checkpoint)
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.task == 'generate':
        generator = BatchedSelfPlay(model, MCTSConfig(num_simulations=args.simulations,
                                    dirichlet_epsilon=.15, temperature_threshold=12), batch_size=64)
        for chunk in range(args.worker, args.chunks, args.workers):
            path = args.folder / f'selfplay_{chunk}.npz'
            if path.exists():
                with np.load(path) as saved:
                    if 'source_checkpoint_sha256' in saved:
                        assert str(saved['source_checkpoint_sha256']) == checkpoint_sha
                continue
            started = time.monotonic()
            previous_stats = generator._mcts.stats.copy()
            samples = generator.generate_games(args.games)
            states = np.stack([s.state for s in samples])
            policies = np.stack([s.policy for s in samples])
            outcomes = np.array([s.value for s in samples], dtype=np.float32)
            assert np.isfinite(states).all() and np.isfinite(policies).all()
            assert np.allclose(policies.sum(1), 1) and np.isin(outcomes, [-1, 0, 1]).all()
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, states=states, policies=policies, outcomes=outcomes,
                                    source_checkpoint_sha256=checkpoint_sha, games=args.games)
            path.with_suffix('.tmp').replace(path)
            elapsed = time.monotonic() - started
            print(json.dumps({'chunk': chunk, 'games': args.games, 'samples': len(samples),
                              'seconds': elapsed, 'games_per_minute': args.games * 60 / elapsed,
                              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                              'search': {key: value - previous_stats[key]
                                         for key, value in generator._mcts.stats.items()}}), flush=True)
    else:
        opponent = load_model(args.opponent)
        started = time.monotonic()
        seed = args.seed - args.worker if args.total_pairs else args.seed
        boards = opening_boards(args.total_pairs, seed)[args.worker::args.workers] if args.total_pairs else None
        result = arena(model, opponent, args.pairs, args.simulations, seed, float('inf'), boards=boards)
        result['seconds'] = time.monotonic() - started
        result.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      opponent_sha256=hashlib.sha256(args.opponent.read_bytes()).hexdigest())
        path = args.folder / f'arena_batch_{args.worker}.json'
        path.with_suffix('.tmp').write_text(json.dumps(result, indent=2))
        path.with_suffix('.tmp').replace(path)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['generate', 'arena'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--opponent', type=Path)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunks', type=int, default=4)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--simulations', type=int, default=512)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--pairs', type=int, default=32)
    parser.add_argument('--total-pairs', type=int, help='Partition one shared opening set across arena workers')
    args = parser.parse_args()
    assert min(args.simulations, args.games, args.pairs) > 0
    assert 0 <= args.worker < args.workers <= 6 and args.chunks > 0
    assert args.task == 'generate' or args.opponent is not None
    run(args)
