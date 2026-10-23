"""Safety checks for reward grouping before an optimizer can consume rollouts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage_c_grpo_entry import group_signal


def samples(rewards, *, invalid=(), bad_audit=()):
    return [{"sample_index": index, "reward": reward,
             "reward_valid": index not in invalid,
             "probability_audit_passed": index not in bad_audit}
            for index, reward in enumerate(rewards)]


def test_g4_reward_varying_group_is_eligible():
    result = group_signal(samples([1.0, 0.0, 0.0, 0.0]))
    assert result["eligible"] is True
    assert result["reward_difference"] == 1.0


def test_equal_or_invalid_reward_never_enters_update():
    equal = group_signal(samples([1.0] * 4))
    invalid = group_signal(samples([1.0, 0.0, 0.0, 0.0], invalid=(1,)))
    unaudited = group_signal(samples([1.0, 0.0, 0.0, 0.0], bad_audit=(2,)))
    assert equal["drop_reason"] == "same_reward"
    assert invalid["drop_reason"] == "invalid_reward"
    assert unaudited["drop_reason"] == "probability_audit_failed"
    assert not any(group["eligible"] for group in (equal, invalid, unaudited))


def test_missing_or_duplicate_g4_sample_rejected():
    with pytest.raises(ValueError, match="four distinct"):
        group_signal(samples([1.0, 0.0, 0.0]))
    duplicate = samples([1.0, 0.0, 0.0, 0.0])
    duplicate[3]["sample_index"] = 2
    with pytest.raises(ValueError, match="four distinct"):
        group_signal(duplicate)
