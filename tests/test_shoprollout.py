from copy import deepcopy

import pytest

from shopmini.shoprollout import (
    PromptBudgetError,
    available_tools_for_observation,
    fit_prompt,
    parse_tool_call,
    run_task,
)


def test_qwen_tool_call_parser_rejects_ambiguous_or_broken_actions():
    xml = ("<tool_call>\n<function=search_products>\n<parameter=query>\n自动浇水器\n"
           "</parameter>\n</function>\n</tool_call>")
    assert parse_tool_call(xml) == ("search_products", {"query": "自动浇水器"})
    assert parse_tool_call("<tool_call><function=buy_now></function></tool_call>") == ("buy_now", {})
    assert parse_tool_call("I am done") is None
    for broken in (xml + xml, xml + "extra", "<tool_call><function=buy_now>",
                   ("<tool_call><function=search_products><parameter=query>a</parameter>"
                    "<parameter=query>b</parameter></function></tool_call>")):
        with pytest.raises(ValueError):
            parse_tool_call(broken)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(row["content"]) for row in messages)))


def test_context_compacts_complete_pairs_and_rejects_oversized_current_page():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a" * 10},
                {"role": "tool", "content": "t" * 10},
                {"role": "assistant", "content": "b"},
                {"role": "tool", "content": "c"}]
    ids, dropped = fit_prompt(FakeTokenizer(), messages, [], 5)
    assert len(ids) == 4 and dropped == 1
    assert len(messages) == 6  # The stored trajectory stays intact.
    with pytest.raises(PromptBudgetError):
        fit_prompt(FakeTokenizer(), messages[:2], [], 1)


def test_available_tools_follow_public_page_buttons():
    tools = [{"function": {"name": name}} for name in (
        "search_products", "open_product", "view_features", "view_attributes",
        "select_option", "back_to_search", "buy_now", "finish_without_purchase")]
    page = ("page_type: product_detail\navailable_options: {\"颜色分类\": [\"25米套装\"]}\n"
            "搜索功能是否可用: False\n"
            "可点击的按钮: [\"features\", \"25米套装\", \"back to search\", \"buy now\"]")
    offered = available_tools_for_observation(tools, page)
    assert [tool["function"]["name"] for tool in offered] == [
        "view_features", "select_option", "back_to_search", "buy_now",
        "finish_without_purchase"]


class FakeActor:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []
        self.offered = []

    def reset_peak(self):
        pass

    def peak_gib(self):
        return {"allocated_gib": 1.0, "reserved_gib": 1.2}

    def complete(self, messages, tools, observation=None):
        self.seen.append(deepcopy(messages))
        self.offered.append(deepcopy(tools))
        return {"text": next(self.responses), "prompt_tokens": 10, "generated_tokens": 5,
                "prompt_ids": [1], "generated_ids": [2], "history_pairs_dropped": 0,
                "generation_seconds": 0.1, "generation_truncated": False,
                "gpu_peak": self.peak_gib()}


class FakeSession:
    def __init__(self, event_sink):
        self.event_sink = event_sink
        self.tools = [{"function": {"name": "search_products"}},
                      {"function": {"name": "finish_without_purchase"}},
                      {"function": {"name": "think"}}]
        self.observation = "public home"
        self.released = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.released = True

    def reset(self, task_id):
        return {"instruction": "public instruction", "observation": self.observation}

    def step(self, name, arguments):
        assert name == "finish_without_purchase"
        assert arguments == {"reason": "no_suitable_product"}
        self.observation = "public terminal"
        return {"executed": True, "done": True, "observation": self.observation,
                "outcome": {"reward": -0.35, "reward_valid": True,
                            "termination_reason": "early_abstain", "strict_success": False,
                            "reward_detail": {"goal": "private"}}}


def test_rollout_exposes_only_public_observation_and_releases_after_terminal():
    actor = FakeActor([("<tool_call><function=finish_without_purchase>"
                        "<parameter=reason>no_suitable_product</parameter>"
                        "</function></tool_call>")])
    sessions = []
    turns = []

    def create(event_sink):
        session = FakeSession(event_sink)
        sessions.append(session)
        return session

    row = run_task(53, actor, create, turns.append, lambda event: None)
    assert row["status"] == "environment_terminal"
    assert row["outcome"]["reward"] == -0.35 and not row["strict_success"]
    assert sessions[0].released
    assert "private" not in str(actor.seen)
    assert all(tool["function"]["name"] != "think" for tool in sessions[0].tools[:2])
    assert turns[0]["tool"] == "finish_without_purchase"


def test_rollout_stops_at_turn_budget_and_releases():
    actor = FakeActor([("<tool_call><function=search_products><parameter=query>bad</parameter>"
                        "</function></tool_call>")])
    session = FakeSession(lambda event: None)
    session.step = lambda name, args: {"executed": False, "reason": "blocked"}
    row = run_task(53, actor, lambda event_sink: session, lambda turn: None,
                   lambda event: None, max_turns=1)
    assert row["status"] == "turn_budget"
    assert row["outcome"] is None and session.released


def test_teacher_profile_uses_upstream_training_prompt_and_full_tools():
    from shopmini.upstream import UPSTREAM_ROOT
    if not (UPSTREAM_ROOT / ".git").exists():
        pytest.skip("Pinned upstream checkout is optional for local unit tests")
    from shopmini.upstream import enable_upstream

    enable_upstream()
    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT as teacher_prompt

    actor = FakeActor([("<tool_call><function=finish_without_purchase>"
                        "<parameter=reason>no_suitable_product</parameter>"
                        "</function></tool_call>")])
    session = FakeSession(lambda event: None)
    run_task(53, actor, lambda event_sink: session, lambda turn: None,
             lambda event: None, prompt_profile="teacher")
    assert actor.seen[0] == [{"role": "system", "content": teacher_prompt},
                             {"role": "user", "content": "public instruction"}]
    assert [tool["function"]["name"] for tool in actor.offered[0]] == [
        "search_products", "finish_without_purchase", "think",
    ]


def test_teacher_profile_can_offer_only_tools_visible_on_current_page():
    from shopmini.upstream import UPSTREAM_ROOT
    if not (UPSTREAM_ROOT / ".git").exists():
        pytest.skip("Pinned upstream checkout is optional for local unit tests")
    actor = FakeActor([("<tool_call><function=finish_without_purchase>"
                        "<parameter=reason>no_suitable_product</parameter>"
                        "</function></tool_call>")])
    session = FakeSession(lambda event: None)
    session.observation = "搜索功能是否可用: False\n可点击的按钮: []"
    row = run_task(53, actor, lambda event_sink: session, lambda turn: None,
                   lambda event: None, prompt_profile="teacher",
                   restrict_tools_to_observation=True)
    assert row["status"] == "environment_terminal" and session.released
    assert [tool["function"]["name"] for tool in actor.offered[0]] == [
        "finish_without_purchase",
    ]
