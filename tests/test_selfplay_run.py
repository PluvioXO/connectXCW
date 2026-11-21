from scripts.selfplay_run import promotion, resume_progress
import hashlib
import json
import numpy as np
import finetune_20min
import pytest


def test_promotion_requires_mean_and_paired_confidence():
    assert not promotion([.5] * 64)["promoted"]
    assert not promotion([1.] * 36 + [0.] * 28)["promoted"]
    assert promotion([1.] * 48 + [0.] * 16)["promoted"]


def test_arena_tracks_both_seats_per_opening(monkeypatch):
    class Search:
        def __init__(self, *args):
            self.stats = {}

        def search_batch(self, boards):
            return [np.array([int(c in b.get_legal_moves()) for c in range(7)]) for b in boards]

    monkeypatch.setattr(finetune_20min, "BatchedMCTS", Search)
    result = finetune_20min.arena(None, None, 16, 1, 100, float("inf"))
    assert result["score"] == .5 and result["pair_scores"] == [.5] * 16
    assert result["wins"] == result["losses"]


def test_resume_restores_committed_champion_and_retains_partial_shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'models').mkdir()
    initial = tmp_path / 'models/large.pt'
    initial.write_bytes(b'initial')
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'plan.json').write_text(json.dumps({'starting_sha256': hashlib.sha256(b'initial').hexdigest()}))
    candidate = out / 'candidate.pt'
    candidate.write_bytes(b'promoted')
    (out / 'best.pt').write_bytes(b'uncommitted promotion')
    event = {'phase': 'iteration_complete', 'iteration': 1, 'promoted': True,
             'candidate': str(candidate), 'best_sha256': hashlib.sha256(b'promoted').hexdigest(),
             'elapsed_seconds': 12.}
    (out / 'metrics.jsonl').write_text(json.dumps(event) + '\n{"interrupted":')
    for iteration, chunks in [(1, 4), (2, 2)]:
        folder = out / f'iteration_{iteration:04d}'
        folder.mkdir()
        for chunk in range(chunks):
            (folder / f'selfplay_{chunk}.npz').write_bytes(b'shard')
    assert resume_progress(out) == (1, 384, 12.)
    assert (out / 'best.pt').read_bytes() == b'promoted'
    assert (out / 'metrics.jsonl').read_text() == json.dumps(event) + '\n'
    assert list(out.glob('metrics_interrupted_*.jsonl'))


@pytest.mark.parametrize('started', [False, True])
def test_resume_restores_learner_without_promoting_it(tmp_path, monkeypatch, started):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/large.pt').write_bytes(b'initial')
    out = tmp_path / 'run'
    out.mkdir()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    (out / 'learner_seed.pt').write_bytes(b'clean seed')
    plan = {'starting_sha256': sha(b'initial'), 'persistent_learner': True,
            'persistent_start_iteration': 1 if started else 2, 'learner_seed_sha256': sha(b'clean seed')}
    (out / 'plan.json').write_text(json.dumps(plan))
    candidate = out / 'candidate.pt'
    candidate.write_bytes(b'unpromoted learner')
    (out / 'best.pt').write_bytes(b'uncommitted champion')
    (out / 'learner.pt').write_bytes(b'uncommitted learner')
    event = {'phase': 'iteration_complete', 'iteration': 1, 'promoted': False,
             'candidate': str(candidate), 'best_sha256': sha(b'initial'), 'elapsed_seconds': 12.,
             'training_mode': 'persistent_raw_v1', 'learner_sha256': sha(b'unpromoted learner')}
    (out / 'metrics.jsonl').write_text(json.dumps(event) + '\n')
    assert resume_progress(out) == (1, 0, 12.)
    assert (out / 'best.pt').read_bytes() == b'initial'
    assert (out / 'learner.pt').read_bytes() == (b'unpromoted learner' if started else b'clean seed')
