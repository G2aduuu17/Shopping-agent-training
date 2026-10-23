"""Mini adapter for pinned ShopSimulator tools, public observations and terminal rewards."""
from __future__ import annotations

import math
import time

from .upstream import enable_upstream

ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
REWARD_VERSION = "shopsimulator-reward-v3"


def terminal_outcome(result):
    if result.get("done") is not True:
        raise ValueError("No terminal reward: environment has not finished")
    detail = result.get("reward_detail")
    if not isinstance(detail, dict) or detail.get("reward_version") != REWARD_VERSION:
        raise ValueError("Missing or incompatible reward version")
    valid = result.get("reward_valid")
    if type(valid) is not bool or detail.get("reward_valid") is not valid:
        raise ValueError("Missing or inconsistent reward validity")
    reward = result.get("reward")
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError("Non-finite or nonnumeric reward")
    if detail.get("reward") != reward:
        raise ValueError("Top-level reward disagrees with reward breakdown")
    reason = result.get("termination_reason")
    if not reason or detail.get("termination_reason") != reason:
        raise ValueError("Missing or inconsistent terminal reason")
    return {"reward": reward, "reward_valid": valid, "reward_version": REWARD_VERSION,
            "termination_reason": reason, "training_eligible": valid,
            "strict_success": valid and reason == "gold_purchase",
            "reward_detail": detail}


class ShopSession:
    def __init__(self, base_url="http://127.0.0.1:5700", timeout=60, event_sink=None):
        enable_upstream()
        from shopping_grpo.environment.actions import action_reject_reason
        from shopping_grpo.environment.client import ShopAgentEnv
        from shopping_grpo.environment.observation import render_structured_observation
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action

        self.env = ShopAgentEnv(base_url=base_url, timeout=timeout)
        self.guard = action_reject_reason
        self.render = render_structured_observation
        self.translate = tool_call_to_action
        self.tools = SHOP_TOOL_SCHEMAS
        self.schemas = {s["function"]["name"]: s["function"]["parameters"] for s in self.tools}
        self.event_sink = event_sink or (lambda event: None)
        self.observation = None
        self.state = None
        self.task_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as release_error:
            if exc is not None:
                exc.add_note(f"Session release also failed: {release_error}")
            else:
                raise
        return False

    def _accept(self, result):
        self.state = result["observation_state"]
        self.observation = self.render(self.state)

    def reset(self, task_id):
        start = time.perf_counter()
        result = self.env.reset(task_id)
        self.task_id = task_id
        if result.get("environment_version") != ENVIRONMENT_VERSION:
            raise ValueError("Incompatible ShopSimulator environment")
        self._accept(result)
        # Never pass reset goal_options, persona or terminal goal facts to the actor.
        public = {"instruction": result["instruction"], "observation": self.observation}
        self.event_sink({"type": "reset", "task_id": task_id, "env_idx": self.env.env_idx,
                         "seconds": time.perf_counter() - start, **public})
        return public

    def step(self, name, arguments):
        if self.env.env_idx is None or self.env.done:
            raise ValueError("Session must be reset and not terminal before step")
        schema = self.schemas.get(name)
        reason = None
        if schema is None or name == "think":
            reason = "unsupported_tool"
        elif not isinstance(arguments, dict):
            reason = "arguments_must_be_object"
        elif set(arguments) != set(schema.get("required", [])):
            reason = "missing_or_extra_arguments"
        elif any(not isinstance(v, str) or not v.strip() for v in arguments.values()):
            reason = "arguments_must_be_nonempty_strings"
        else:
            reason = self.guard(name, arguments, self.observation)
        if reason:
            event = {"type": "guard_rejected", "tool": name, "arguments": arguments,
                     "executed": False, "reason": reason}
            self.event_sink(event)
            return event
        start = time.perf_counter()
        result = self.env.step(self.translate(name, arguments))
        self._accept(result)
        event = {"type": "step", "tool": name, "arguments": arguments, "executed": True,
                 "done": result["done"], "observation": self.observation,
                 "seconds": time.perf_counter() - start}
        if result["done"]:
            event["outcome"] = terminal_outcome(result)
        self.event_sink(event)
        return event

    def close(self):
        if self.env.env_idx is None:
            return
        idx = self.env.env_idx
        try:
            self.env.release()
        except Exception as exc:
            self.event_sink({"type": "release_failed", "env_idx": idx, "error": str(exc)})
            raise
        self.event_sink({"type": "released", "env_idx": idx})
