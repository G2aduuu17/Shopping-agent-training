"""Optional, inference-only action constraints from the current public shop page.

The constraint never edits a completed tool call. Calls still go through the
ShopSimulator guard, and pages without an unambiguous public choice fall back
to ordinary generation so failures remain visible.
"""
from __future__ import annotations

import json
import re

from .upstream import enable_upstream

PARAMETER_OPEN = re.compile(
    r"<function=(open_product|select_option)>\s*<parameter=(asin|value)>",
    re.DOTALL,
)
PARAMETERS = {"open_product": "asin", "select_option": "value"}


def visible_parameter_choices(observation: str) -> dict[str, list[str]]:
    """Read only actionable ASINs and specification values on the latest page."""
    enable_upstream()
    from shopping_grpo.environment.actions import clickable_buttons, product_ids

    buttons = clickable_buttons(observation)
    button_keys = {button.casefold() for button in buttons}
    asins = [asin for asin in product_ids(observation) if asin.casefold() in button_keys]
    match = re.search(r"(?m)^available_options: (\{[^\n]*\})$", observation)
    options = json.loads(match.group(1)) if match else {}
    values = []
    if isinstance(options, dict):
        for axis_values in options.values():
            if isinstance(axis_values, list):
                values.extend(value for value in axis_values
                              if isinstance(value, str) and value
                              and value.casefold() in button_keys)
    return {"open_product": list(dict.fromkeys(asins)),
            "select_option": list(dict.fromkeys(values))}


def _insert(root: dict, ids: list[int]) -> None:
    node = root
    for token_id in ids:
        node = node.setdefault(token_id, {})


class VisibleActionLogitsProcessor:
    """Choose among offered tools and exact visible click parameters.

    The suffix includes the closing parameter tag, so every allowed token path
    emits one unchanged public value. Unsupported XML shapes remain unmasked and
    are checked by the downstream parser and guard.
    """

    def __init__(self, tokenizer, prompt_length: int, observation: str, tools: list):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.choices = visible_parameter_choices(observation)
        self.tool_names = list(dict.fromkeys(tool["function"]["name"] for tool in tools))
        self.name_trie = {}
        for name in self.tool_names:
            suffix = f"={name}>"
            ids = tokenizer.encode(suffix, add_special_tokens=False)
            if (not ids or tokenizer.decode(ids, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False) != suffix):
                raise ValueError(f"Cannot losslessly tokenize offered tool {name}")
            _insert(self.name_trie, ids)
        self.tries = {}
        for name, values in self.choices.items():
            root = {}
            for value in values:
                ids = tokenizer.encode(f"\n{value}\n</parameter>", add_special_tokens=False)
                if not ids or tokenizer.decode(ids, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False) != (
                                                   f"\n{value}\n</parameter>"):
                    raise ValueError(f"Cannot losslessly tokenize visible {name} value")
                _insert(root, ids)
            self.tries[name] = root
        self.active = None
        self.anchor = None
        self.name_anchor = None
        self.function_name_applied = False
        self.applied = []
        self.unavailable = []
        self._unavailable_keys = set()
        self.constrained_token_steps = 0
        self.raw_top1_blocked_steps = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError("Visible parameter constraints require batch size 1")
        generated = input_ids[0, self.prompt_length:].tolist()
        decoded = self.tokenizer.decode(generated, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
        if self.name_anchor is None and decoded.endswith("<function"):
            if self.name_trie:
                self.name_anchor = len(generated)
                self.function_name_applied = True
            else:
                self._record_unavailable("function", "no_offered_tools")
        if self.name_anchor is not None:
            node = self.name_trie
            for token_id in generated[self.name_anchor:]:
                if token_id not in node:
                    raise RuntimeError("Generated function left the offered-tool token trie")
                node = node[token_id]
            if node:
                return self._mask(scores, node)
            self.name_anchor = None
        elif not self.function_name_applied and "<function=" in decoded:
            self._record_unavailable("function", "late_function_boundary")
        matches = list(PARAMETER_OPEN.finditer(decoded))
        if not matches:
            return scores
        match = matches[-1]
        name, parameter = match.groups()
        if PARAMETERS[name] != parameter:
            return scores
        if "</parameter>" in decoded[match.end():]:
            self.active, self.anchor = None, None
            return scores
        if self.active is None:
            if decoded[match.end():]:
                # A token crossed the opening-tag boundary; leave it to the guard.
                self._record_unavailable(name, "late_parameter_boundary")
                return scores
            if not self.choices[name]:
                self._record_unavailable(name, "no_public_candidates")
                return scores
            self.active, self.anchor = name, len(generated)
            self.applied.append({"tool": name, "candidate_count": len(self.choices[name])})
        if self.active != name:
            return scores
        node = self.tries[name]
        for token_id in generated[self.anchor:]:
            if token_id not in node:
                raise RuntimeError("Generated parameter left the visible-value token trie")
            node = node[token_id]
        if not node:
            raise RuntimeError("Visible-value token trie ended before the parameter closed")
        return self._mask(scores, node)

    def _mask(self, scores, node: dict):
        allowed = list(node)
        self.constrained_token_steps += 1
        self.raw_top1_blocked_steps += int(scores[0].argmax().item() not in node)
        masked = scores.new_full(scores.shape, -float("inf"))
        masked[:, allowed] = scores[:, allowed]
        return masked

    def _record_unavailable(self, name: str, reason: str) -> None:
        key = (name, reason)
        if key not in self._unavailable_keys:
            self._unavailable_keys.add(key)
            self.unavailable.append({"tool": name, "reason": reason})

    def report(self) -> dict:
        return {"mode": "visible_actions", "offered_tool_count": len(self.tool_names),
                "function_name_applied": self.function_name_applied,
                "available_counts": {
                    name: len(values) for name, values in self.choices.items()},
                "applied": self.applied, "unavailable": self.unavailable,
                "constrained_token_steps": self.constrained_token_steps,
                "raw_top1_blocked_steps": self.raw_top1_blocked_steps}
