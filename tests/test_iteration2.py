import torch

from finetune_20min import opening_boards
from iteration2 import training_targets


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
