import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import parallel_selfplay, selfplay_run


@pytest.mark.parametrize('devices,chunks', [([4, 5], 4), (list(range(6)), None), (list(range(3)), 3)])
@pytest.mark.parametrize('failure', [False, True])
def test_workers_share_checkpoint_and_use_distinct_gpus(tmp_path, monkeypatch, failure, devices, chunks):
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            self.code = 7 if failure and not calls else 0
            calls.append((command, kwargs))
        def poll(self): return self.code
        def wait(self, **kwargs): return self.code
    monkeypatch.setattr(parallel_selfplay.subprocess, 'Popen', Process)
    if failure:
        with pytest.raises(RuntimeError, match='worker failed'):
            parallel_selfplay.run_workers(devices, 'generate', 'same.pt', tmp_path, 99, lambda: None,
                                         chunks=chunks)
    else:
        parallel_selfplay.run_workers(devices, 'generate', 'same.pt', tmp_path, 99, lambda: None,
                                     chunks=chunks)
    assert [kwargs['env']['CUDA_VISIBLE_DEVICES'] for _, kwargs in calls] == list(map(str, devices))
    assert all(command[1:4] == ['-u', '-m', 'scripts.selfplay_worker'] for command, _ in calls)
    assert all(command[command.index('--checkpoint')+1] == 'same.pt' for command, _ in calls)
    assert [c[c.index('--seed')+1] for c, _ in calls] == [str(99+i) for i in range(len(devices))]
    assert all(c[c.index('--chunks')+1] == str(chunks or max(4, len(devices))) for c, _ in calls)


@pytest.mark.parametrize('devices', [[4, 5], list(range(6))])
@pytest.mark.parametrize('persistent', [False, True])
@pytest.mark.parametrize('capped', [False, True])
def test_shared_run_trains_one_candidate_from_all_shards(tmp_path, monkeypatch, devices, persistent, capped):
    expected_devices = devices
    iterations = 3 if persistent else 1
    expected_chunks = [max(4, len(devices))] * iterations
    if capped:
        expected_chunks[-1] -= 1
    final_games = sum(expected_chunks) * 64
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ','.join(map(str, devices)))
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/large.pt').write_bytes(b'baseline')
    monkeypatch.setattr(selfplay_run, 'load_model', lambda path: None)
    monkeypatch.setattr(selfplay_run, 'arena', lambda *args: {'score': .5, 'pair_scores': [.5, .5]})
    monkeypatch.setattr(selfplay_run, 'initialize_learner', lambda source, dest: dest.write_bytes(source.read_bytes()))
    tasks, trainings = [], []
    def workers(devices, task, checkpoint, folder, seed, progress, opponent=None, *, chunks=None):
        tasks.append(task)
        if task == 'generate':
            assert chunks == expected_chunks[len(trainings)]
            assert devices == expected_devices[:chunks]
            expected = f'candidate{len(trainings)}'.encode() if persistent and trainings else b'baseline'
            assert checkpoint.read_bytes() == expected
            for chunk in range(chunks):
                path = folder / f'selfplay_{chunk}.npz'
                if not path.exists():
                    np.savez(path, outcomes=np.ones(128))
            progress()
        else:
            assert devices == expected_devices
            for i in range(len(devices)):
                pairs = 64 // len(devices) + int(i < 64 % len(devices))
                result = {'games': pairs * 2, 'pair_scores': [.5] * pairs,
                          'opening_keys': [(key, 0) for key in range(i, 64, len(devices))],
                          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          'opponent_sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}
                (folder / f'arena_batch_{i}.json').write_text(json.dumps(result))
    def train(args, checkpoint, epochs, shard_paths, **extra):
        assert len(shard_paths) == sum(expected_chunks[:len(trainings) + 1])
        assert epochs == (1 if persistent else 4) and args.lr == 8e-5
        assert extra.get('persistent', False) == persistent
        trainings.append(shard_paths)
        (args.output / f'candidate_{epochs}.pt').write_bytes(f'candidate{len(trainings)}'.encode())
        return {'optimizer_steps_after': len(trainings)}
    monkeypatch.setattr(selfplay_run, 'run_workers', workers)
    monkeypatch.setattr(selfplay_run, 'train', train)
    args = SimpleNamespace(output=tmp_path / 'run', seed=5101, lr=8e-5, iterations=iterations + int(capped),
                           max_selfplay_games=final_games if capped else None,
                           resume=False, gpus=','.join(map(str, devices)), persistent_learner=persistent,
                           epochs=1 if persistent else 4, replay_shards=64)
    selfplay_run.run(args)
    assert tasks == ['generate', 'arena'] * iterations and len(trainings) == iterations
    status = json.loads((args.output / 'status.json').read_text())
    assert status['phase'] == 'complete' and status['completed_selfplay_games'] == final_games
    assert status['iteration'] == iterations
    assert (args.output / 'best.pt').read_bytes() == b'baseline'
    if persistent:
        assert (args.output / 'learner.pt').read_bytes() == b'candidate3'
        events = [json.loads(line) for line in (args.output / 'metrics.jsonl').read_text().splitlines()]
        finished = [e for e in events if e['phase'] == 'iteration_complete']
        assert [e['optimizer_steps'] for e in finished] == [1, 2, 3]
        assert finished[1]['parent_learner_sha256'] == finished[0]['learner_sha256']
    if capped:
        assert status['stop_reason'] == 'selfplay_game_limit'
        args.resume = True
        tasks.clear()
        selfplay_run.run(args)
        assert tasks == []

        # Recover an interruption after the final shards were written and before training.
        metrics = args.output / 'metrics.jsonl'
        lines = metrics.read_text().splitlines(keepends=True)
        cutoff = next(i for i, line in enumerate(lines)
                      if json.loads(line)['iteration'] == iterations and json.loads(line)['phase'] == 'training')
        metrics.write_text(''.join(lines[:cutoff]))
        trainings.pop()
        selfplay_run.run(args)
        status = json.loads((args.output / 'status.json').read_text())
        assert tasks == ['generate', 'arena']
        assert status['stop_reason'] == 'selfplay_game_limit' and status['completed_selfplay_games'] == final_games


def test_six_workers_partition_all_arena_pairs(tmp_path, monkeypatch):
    commands = []
    class Process:
        def __init__(self, command, **kwargs): commands.append(command)
        def poll(self): return 0
        def wait(self, **kwargs): return 0
    monkeypatch.setattr(parallel_selfplay.subprocess, 'Popen', Process)
    parallel_selfplay.run_workers(list(range(6)), 'arena', 'candidate.pt', tmp_path, 99,
                                 lambda: None, opponent='best.pt')
    counts = [int(c[c.index('--pairs')+1]) for c in commands]
    assert counts == [11, 11, 11, 11, 10, 10] and sum(counts) == 64
    from finetune_20min import opening_boards
    assigned = []
    for command in commands:
        value = lambda flag: int(command[command.index(flag) + 1])
        worker, workers = value('--worker'), value('--workers')
        boards = opening_boards(value('--total-pairs'), value('--seed') - worker)[worker::workers]
        assert len(boards) == value('--pairs')
        assigned.extend((b._mask, b._position) for b in boards)
    expected = {(b._mask, b._position) for b in opening_boards(64, 99)}
    assert len(assigned) == len(set(assigned)) == 64 and set(assigned) == expected
