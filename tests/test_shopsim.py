from copy import deepcopy
from types import SimpleNamespace

import pytest

from shopmini.shopsim import ENVIRONMENT_VERSION, REWARD_VERSION, ShopSession, terminal_outcome


def terminal(valid=True):
    fields = {"reward": -0.35, "reward_valid": valid, "termination_reason": "early_abstain"}
    return {"done": True, **fields, "reward_detail": {**fields, "reward_version": REWARD_VERSION}}


def test_negative_reward_is_valid_but_not_success():
    outcome = terminal_outcome(terminal())
    assert outcome["training_eligible"]
    assert not outcome["strict_success"]
    assert outcome["reward"] < 0
    assert not terminal_outcome(terminal(False))["training_eligible"]


@pytest.mark.parametrize("field,value", [
    ("done", False), ("reward", float("nan")), ("reward", True),
    ("reward", 0.5), ("reward_valid", False), ("termination_reason", "gold_purchase"),
    ("reward_detail", {}),
])
def test_reject_broken_terminal_contract(field, value):
    result = terminal()
    result[field] = value
    with pytest.raises(ValueError):
        terminal_outcome(result)


def fake_session():
    session = ShopSession.__new__(ShopSession)
    events = []
    session.env = SimpleNamespace(env_idx=None, done=False)
    session.event_sink = events.append
    session.render = lambda state: "public page"

    def reset(task_id):
        session.env.env_idx = 0
        return {"environment_version": ENVIRONMENT_VERSION, "instruction": "public request",
                "observation_state": {}, "goal_options": "secret", "persona": "secret"}

    def release():
        session.env.env_idx = None

    session.env.reset = reset
    session.env.release = release
    return session, events


def test_reset_excludes_private_fields_and_exception_releases():
    session, events = fake_session()
    with pytest.raises(RuntimeError, match="actor failure"), session:
        public = session.reset(53)
        assert public == {"instruction": "public request", "observation": "public page"}
        assert "secret" not in str(events)
        raise RuntimeError("actor failure")
    assert session.env.env_idx is None
    assert events[-1]["type"] == "released"
    before = deepcopy(events)
    session.close()
    assert events == before


def test_release_failure_preserves_original_error():
    session, events = fake_session()

    def fail_release():
        raise OSError("transport failed")

    session.env.release = fail_release
    with pytest.raises(RuntimeError, match="actor failure") as caught, session:
        session.reset(53)
        raise RuntimeError("actor failure")
    assert "transport failed" in caught.value.__notes__[0]
    assert events[-1]["type"] == "release_failed"
