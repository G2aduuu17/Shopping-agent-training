import pytest
import torch

from shopmini.objective import advantages, clipped_loss, token_logprobs


def test_group_centering_and_rejections():
    assert torch.equal(advantages([1, 0]), torch.tensor([0.5, -0.5]))
    assert torch.equal(advantages([1, 1]), torch.zeros(2))
    for rewards in ([1], [float("nan"), 0], [float("inf"), 1]):
        with pytest.raises(ValueError):
            advantages(rewards)


@pytest.mark.parametrize("ratio,adv,expected_zero,sign", [
    (1.0, 1.0, False, -1), (1.0, -1.0, False, 1),
    (1.3, 1.0, True, 0), (0.7, -1.0, True, 0),
    (0.7, 1.0, False, -1), (1.3, -1.0, False, 1),
])
def test_clip_gradient_direction(ratio, adv, expected_zero, sign):
    new = torch.tensor([ratio]).log().requires_grad_()
    old = torch.zeros(1, requires_grad=True)
    advantage = torch.tensor(adv, requires_grad=True)
    clipped_loss(new, old, advantage).backward()
    assert old.grad is None and advantage.grad is None
    assert bool(new.grad == 0) == expected_zero
    if not expected_zero:
        assert int(new.grad.sign()) == sign


def test_mask_and_length_normalization():
    new = torch.zeros(3, requires_grad=True)
    loss = clipped_loss(new, torch.zeros(3), 0.5, mask=torch.tensor([1, 0, 1]))
    loss.backward()
    assert torch.allclose(new.grad, torch.tensor([-0.25, 0, -0.25]))
    assert loss == clipped_loss(torch.zeros(1), torch.zeros(1), 0.5)
    with pytest.raises(ValueError):
        clipped_loss(new, new.detach(), 1, mask=torch.zeros(3))


def test_logprob_matches_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(5, 11, requires_grad=True)
    tokens = torch.tensor([1, 4, 0, 2, 8])
    actual = token_logprobs(logits, tokens)
    expected = -torch.nn.functional.cross_entropy(logits, tokens, reduction="none")
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                          torch.autograd.grad(expected.sum(), logits)[0], atol=1e-6)
