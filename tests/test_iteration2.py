import torch
import numpy as np
import pytest
from types import SimpleNamespace

import iteration2
from finetune_20min import opening_boards, save_checkpoint
from iteration2 import training_targets
from scripts.selfplay_run import initialize_learner
from src.neural_net.model import Connect4Net


def test_openings_are_unique_and_reproducible():
    first = opening_boards(64, 2001)
    second = opening_boards(64, 2001)
    keys = lambda boards: [(b._mask, b._position) for b in boards]
    assert keys(first) == keys(second)
    assert len(set(keys(first))) == 64
    assert sum(b._num_moves == 0 for b in first) == 1
    assert all(not b.is_terminal() and b.current_player == 1 for b in first)


def test_training_targets_mask_illegal_moves_and_stay_normalized():
    states = torch.zeros(2, 3, 6, 7)
    states[1, 0, -1, 0] = 1
    policies = torch.ones(2, 7)
    policies[1, 0] = 0
    policies /= policies.sum(-1, keepdim=True)
    logits = torch.zeros(2, 7)
    logits[1, 0] = 100
    policy, value = training_targets(states, policies, torch.tensor([1., -1.]),
                                     logits, torch.tensor([0.5, -0.5]))
    assert torch.allclose(policy.sum(-1), torch.ones(2))
    assert policy[1, 0] == 0
    assert torch.allclose(value, torch.tensor([0.75, -0.75]))


def test_eval_mode_preserves_batchnorm_statistics_during_learning(tiny_model):
    buffers = {name: value.clone() for name, value in tiny_model.named_buffers()}
    old_weights = tiny_model.policy_head[4].weight.detach().clone()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
    logits, value = tiny_model(torch.randn(4, 3, 6, 7))
    loss = logits.square().mean() + value.square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(old_weights, tiny_model.policy_head[4].weight)
    assert all(torch.equal(value, buffers[name]) for name, value in tiny_model.named_buffers())


def test_persistent_training_matches_uninterrupted_adam_and_uses_raw_targets(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(72)
    model = Connect4Net(num_blocks=1, num_filters=8).eval()
    base = tmp_path / 'base.pt'
    save_checkpoint(model, base, optimizer_state_dict={'old_objective': True})
    seed = tmp_path / 'seed.pt'
    initialize_learner(base, seed)
    assert 'optimizer_state_dict' not in torch.load(seed, weights_only=False)
    states = torch.randn(24, 3, 6, 7)
    policies = torch.softmax(torch.randn(24, 7), -1)
    outcomes = torch.tensor([1., 0., -1.] * 8)
    shard = tmp_path / 'selfplay_0.npz'
    np.savez(shard, states=states.numpy(), policies=policies.numpy(), outcomes=outcomes.numpy())
    with torch.no_grad():
        logits, value = model(states)
        expected_policy_loss = -(policies * torch.log_softmax(logits, -1)).sum(-1).mean().item()
        expected_value_loss = (value[:, 0] - outcomes).square().mean().item()

    def load_cpu(path):
        saved = torch.load(path, weights_only=False)
        net = Connect4Net(num_blocks=saved['num_blocks'], num_filters=saved['num_filters']).eval()
        net.load_state_dict(saved['model_state_dict'])
        return net

    monkeypatch.setattr(iteration2, 'load_model', load_cpu)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    def run(name, checkpoint, epochs):
        folder = tmp_path / name
        folder.mkdir()
        args = SimpleNamespace(data=tmp_path, output=folder, lr=3e-5, seed=12)
        report = iteration2.train(args, checkpoint=checkpoint, epochs=epochs,
                                  shard_paths=[shard], persistent=True)
        return folder / f'candidate_{epochs}.pt', report

    torch.manual_seed(123)
    continuous, full_report = run('continuous', seed, 2)
    torch.manual_seed(123)
    first, report = run('first', seed, 1)
    resumed, resumed_report = run('resumed', first, 1)
    assert report['epochs'][0]['policy_loss'] == pytest.approx(expected_policy_loss, abs=1e-6)
    assert report['epochs'][0]['value_loss'] == pytest.approx(expected_value_loss, abs=1e-6)
    assert report['optimizer_steps_after'] == resumed_report['optimizer_steps_before'] == 1
    assert full_report['optimizer_steps_after'] == resumed_report['optimizer_steps_after'] == 2
    full = torch.load(continuous, weights_only=False)
    recovered = torch.load(resumed, weights_only=False)
    assert any(not torch.equal(full['model_state_dict'][key], value) for key, value in model.state_dict().items())
    for key in full['model_state_dict']:
        assert torch.equal(full['model_state_dict'][key], recovered['model_state_dict'][key]), key
    assert torch.equal(full['torch_rng_state'], recovered['torch_rng_state'])
    for parameter, state in full['optimizer_state_dict']['state'].items():
        for key, value in state.items():
            assert torch.equal(value, recovered['optimizer_state_dict']['state'][parameter][key])
    with pytest.raises(AssertionError, match='clean optimizer'):
        run('reject_old_momentum', base, 1)
