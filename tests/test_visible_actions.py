"""The optional decoder must choose public values without rewriting tool calls."""
import pytest
import torch

from shopmini.upstream import UPSTREAM_ROOT

pytestmark = pytest.mark.skipif(
    not (UPSTREAM_ROOT / ".git").exists(),
    reason="Pinned upstream checkout is optional for local unit tests",
)

from shopmini.visible_actions import VisibleActionLogitsProcessor, visible_parameter_choices


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token_id) for token_id in ids)


def test_choices_use_only_current_actionable_page_values():
    page = ("[SHOPPING_OBSERVATION_V2]\npage_type: search_results\n"
            "1|1234567890|10|brand|category|attr|title\n"
            "2|2345678901|11|brand|category|attr|title\n"
            "available_options: {\"Color\": [\"Blue\", \"Red\", \"Hidden\"]}\n"
            "可点击的按钮: [\"1234567890\", \"Blue\", \"Red\"]")
    assert visible_parameter_choices(page) == {
        "open_product": ["1234567890"], "select_option": ["Blue", "Red"]}


def test_logits_constraint_forces_visible_asin_but_preserves_model_choice():
    page = ("1|1234567890|10|brand|category|attr|title\n"
            "2|2345678901|11|brand|category|attr|title\n"
            "可点击的按钮: [\"1234567890\", \"2345678901\"]")
    tokenizer = CharacterTokenizer()
    prefix = "<tool_call><function=open_product><parameter=asin>"
    generated = tokenizer.encode(prefix)
    processor = VisibleActionLogitsProcessor(
        tokenizer, 0, page, [{"function": {"name": "open_product"}}])
    for _ in range(40):
        scores = torch.zeros((1, 256))
        scores[0, ord("X")] = 100  # The unconstrained top token is illegal.
        scores[0, ord("2")] = 10    # Among legal ASINs, prefer the second.
        masked = processor(torch.tensor([generated]), scores)
        generated.append(int(masked.argmax().item()))
        if tokenizer.decode(generated).endswith("</parameter>"):
            break
    assert tokenizer.decode(generated) == prefix + "\n2345678901\n</parameter>"
    assert processor.report()["applied"] == [{"tool": "open_product", "candidate_count": 2}]
    assert processor.report()["raw_top1_blocked_steps"] > 0


def test_empty_public_choice_falls_back_to_guard_without_rewriting():
    page = "page_type: search_results\n可点击的按钮: []"
    tokenizer = CharacterTokenizer()
    prefix = "<tool_call><function=open_product><parameter=asin>"
    processor = VisibleActionLogitsProcessor(
        tokenizer, 0, page, [{"function": {"name": "open_product"}}])
    scores = torch.arange(256, dtype=torch.float32).unsqueeze(0)
    assert torch.equal(processor(torch.tensor([tokenizer.encode(prefix)]), scores), scores)
    assert processor.report()["unavailable"] == [
        {"tool": "function", "reason": "late_function_boundary"},
        {"tool": "open_product", "reason": "no_public_candidates"}]


def test_function_name_constraint_excludes_hidden_buttons():
    tokenizer = CharacterTokenizer()
    prefix = "<tool_call><function"
    generated = tokenizer.encode(prefix)
    processor = VisibleActionLogitsProcessor(
        tokenizer, 0, "可点击的按钮: [\"buy now\"]",
        [{"function": {"name": "buy_now"}},
         {"function": {"name": "finish_without_purchase"}}])
    for _ in range(30):
        scores = torch.zeros((1, 256))
        scores[0, ord("X")] = 100
        scores[0, ord("b")] = 10
        masked = processor(torch.tensor([generated]), scores)
        generated.append(int(masked.argmax().item()))
        if tokenizer.decode(generated).endswith("=buy_now>"):
            break
    assert tokenizer.decode(generated) == prefix + "=buy_now>"
    assert processor.report()["function_name_applied"]
    assert processor.report()["raw_top1_blocked_steps"] > 0
