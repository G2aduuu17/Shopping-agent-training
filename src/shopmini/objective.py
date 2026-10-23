"""Trajectory-mean clipped group-relative objective, no KL or std normalization."""
import torch


def advantages(rewards):
    rewards = torch.as_tensor(rewards, dtype=torch.float32).detach()
    if rewards.ndim != 1 or rewards.numel() < 2 or not torch.isfinite(rewards).all():
        raise ValueError("A group requires at least two finite rewards")
    return rewards - rewards.mean()


def clipped_loss(new_logp, old_logp, advantage, epsilon=0.2, mask=None):
    if new_logp.ndim != 1 or new_logp.shape != old_logp.shape or new_logp.numel() == 0:
        raise ValueError("Expected matching nonempty token vectors")
    if not 0 < epsilon < 1:
        raise ValueError("Invalid clipping epsilon")
    if not torch.isfinite(new_logp).all() or not torch.isfinite(old_logp).all():
        raise ValueError("Non-finite log probability")
    mask = torch.ones_like(new_logp, dtype=torch.bool) if mask is None else mask.bool()
    if mask.shape != new_logp.shape or not mask.any():
        raise ValueError("Invalid/empty action mask")
    adv = torch.as_tensor(advantage, device=new_logp.device).float().detach()
    if adv.numel() != 1 or not torch.isfinite(adv).all():
        raise ValueError("Expected a finite scalar trajectory advantage")
    ratio = (new_logp.float() - old_logp.detach().float()).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Non-finite policy ratio")
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-epsilon, 1+epsilon) * adv)
    return -surrogate[mask].mean()


def token_logprobs(logits, tokens):
    # logits and tokens must already be causally aligned.
    if logits.shape[:-1] != tokens.shape:
        raise ValueError("Logits/token shape mismatch")
    values = logits.float()
    return values.gather(-1, tokens.unsqueeze(-1)).squeeze(-1) - values.logsumexp(-1)
