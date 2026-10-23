"""Frozen Qwen tool-use rollouts against the public ShopSimulator contract."""
from __future__ import annotations

import json
import re
import time


class PromptBudgetError(ValueError):
    """The current public observation cannot fit in the configured prompt budget."""


TOOL_RE = re.compile(
    r"<tool_call>\s*<function=([a-zA-Z_][a-zA-Z_0-9]*)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=([a-zA-Z_][a-zA-Z_0-9]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_call(text):
    """Parse one Qwen native XML tool call; return None for a plain final answer."""
    matches = list(TOOL_RE.finditer(text))
    if not matches:
        if "<tool_call" in text or "<function=" in text:
            raise ValueError("malformed_tool_call")
        return None
    if len(matches) != 1 or text[matches[0].end():].strip():
        raise ValueError("multiple_or_trailing_tool_calls")
    name, body = matches[0].groups()
    params = list(PARAM_RE.finditer(body))
    if PARAM_RE.sub("", body).strip():
        raise ValueError("malformed_tool_arguments")
    arguments = {}
    for item in params:
        key, value = item.groups()
        if key in arguments:
            raise ValueError("duplicate_tool_argument")
        arguments[key] = value.strip()
    return name, arguments


def fit_prompt(tokenizer, messages, tools, max_prompt_tokens):
    """Drop oldest complete assistant/tool pairs; never silently truncate tokens."""
    visible = list(messages)
    dropped = 0
    while True:
        ids = tokenizer.apply_chat_template(
            visible, tools=tools, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False,
        )
        if len(ids) <= max_prompt_tokens:
            return ids, dropped
        if len(visible) <= 4:
            raise PromptBudgetError(
                f"Public prompt needs {len(ids)} tokens; budget is {max_prompt_tokens}"
            )
        if visible[2]["role"] != "assistant" or visible[3]["role"] != "tool":
            raise ValueError("History is not a sequence of complete assistant/tool pairs")
        del visible[2:4]
        dropped += 1


BUTTON_TO_TOOL = {
    "view_description": "description", "view_features": "features",
    "view_reviews": "reviews", "view_attributes": "attributes",
    "next_page": "next >", "prev_page": "< prev",
    "back_to_search": "back to search", "buy_now": "buy now",
}


def available_tools_for_observation(tools, observation):
    """Offer only actions visible in the latest public page; keep the server guard authoritative."""
    match = re.search(r"(?m)^可点击的按钮: (.+)$", observation)
    if not match:
        return tools
    buttons = {str(label).casefold() for label in json.loads(match.group(1))}
    option_match = re.search(r"(?m)^available_options: (.+)$", observation)
    options = json.loads(option_match.group(1)) if option_match else {}
    option_visible = any(str(value).casefold() in buttons
                         for values in options.values() for value in values)
    can_open = any(re.fullmatch(r"\d{10,}", button) for button in buttons)
    allowed = {"finish_without_purchase"}
    if "搜索功能是否可用: True" in observation:
        allowed.add("search_products")
    if can_open:
        allowed.add("open_product")
    if option_visible:
        allowed.add("select_option")
    allowed.update(name for name, button in BUTTON_TO_TOOL.items() if button in buttons)
    return [tool for tool in tools if tool["function"]["name"] in allowed]


class FrozenQwen:
    def __init__(self, model_path, max_prompt_tokens=8192, max_new_tokens=192,
                 min_free_gpu_gib=8, device="cuda", adapter_path=None,
                 sample=False, record_logprobs=False, model_dtype="bf16",
                 temperature=1.0, top_p=1.0,
                 constrain_visible_actions=False):
        import torch
        from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration

        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        if model_dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported model dtype: {model_dtype}")
        if not (0.0 < temperature <= 1.0 and 0.0 < top_p <= 1.0):
            raise ValueError("Sampling temperature and top_p must be in (0, 1]")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the frozen Qwen baseline")
            if model_dtype == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA with BF16 is required for BF16 inference")
            free, _ = torch.cuda.mem_get_info()
            if free / 2**30 < min_free_gpu_gib:
                raise RuntimeError(f"Need at least {min_free_gpu_gib} GiB free GPU memory")
        self.device = "cuda:0" if device == "cuda" else "cpu"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=(torch.bfloat16 if device == "cuda" and model_dtype == "bf16"
                               else torch.float32),
            local_files_only=True, attn_implementation="sdpa", device_map={"": self.device},
        )
        if adapter_path is not None:
            from pathlib import Path

            from peft import PeftModel

            adapter_path = Path(adapter_path).resolve()
            if not (adapter_path / "adapter_config.json").is_file() or not (
                adapter_path / "adapter_model.safetensors"
            ).is_file():
                raise ValueError(f"Missing PEFT adapter files: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=False,
            )
        self.model.eval()
        self.model.requires_grad_(False)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.record_logprobs = record_logprobs
        self.constrain_visible_actions = constrain_visible_actions
        eos_ids = self.model.generation_config.eos_token_id
        self.eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        if not self.eos_ids:
            self.eos_ids = [self.tokenizer.eos_token_id]
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=sample,
            temperature=temperature if sample else 1.0,
            top_p=top_p if sample else 1.0, top_k=0,
            eos_token_id=self.eos_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.eos_ids[0],
            stop_strings=["</tool_call>"],
            use_cache=True, return_dict_in_generate=True,
            output_scores=record_logprobs, output_logits=record_logprobs,
        )

    def logprobs(self, prompt_ids, generated_ids):
        """Unwarped policy log-probabilities for the exact saved action tokens."""
        from .policy import completion_logprobs

        with self.torch.inference_mode():
            values = completion_logprobs(
                self.model, prompt_ids, generated_ids, self.device,
            )
        return values.float().cpu().tolist()

    def reset_peak(self):
        if self.device == "cuda:0":
            self.torch.cuda.reset_peak_memory_stats()

    def peak_gib(self):
        import psutil

        memory = psutil.Process().memory_info()
        process_peak = getattr(memory, "peak_wset", memory.rss) / 2**30
        if self.device == "cpu":
            return {"allocated_gib": None, "reserved_gib": None,
                    "process_peak_rss_gib": process_peak}
        return {
            "allocated_gib": self.torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gib": self.torch.cuda.max_memory_reserved() / 2**30,
            "process_peak_rss_gib": process_peak,
        }

    def complete(self, messages, tools, observation=None):
        prompt, dropped = fit_prompt(
            self.tokenizer, messages, tools, self.max_prompt_tokens,
        )
        inputs = self.torch.tensor([prompt], device=self.device)
        constraint = None
        generation_kwargs = {}
        if self.constrain_visible_actions:
            if observation is None:
                raise ValueError("Visible parameter constraints need the current public page")
            from transformers import LogitsProcessorList

            from .visible_actions import VisibleActionLogitsProcessor

            constraint = VisibleActionLogitsProcessor(
                self.tokenizer, len(prompt), observation, tools)
            generation_kwargs["logits_processor"] = LogitsProcessorList([constraint])
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs, attention_mask=self.torch.ones_like(inputs),
                generation_config=self.generation_config, logits_to_keep=1,
                tokenizer=self.tokenizer,
                **generation_kwargs,
            )
        generated = output.sequences[0, len(prompt):].cpu().tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        elapsed = time.perf_counter() - start
        result = {
            "text": text,
            "prompt_ids": prompt,
            "generated_ids": generated,
            "prompt_tokens": len(prompt),
            "generated_tokens": len(generated),
            "history_pairs_dropped": dropped,
            "generation_seconds": elapsed,
            "generation_truncated": (not generated or
                                     (generated[-1] not in self.eos_ids and
                                      not text.rstrip().endswith("</tool_call>"))),
            "gpu_peak": self.peak_gib(),
        }
        if constraint is not None:
            result["action_constraint"] = constraint.report()
        if self.record_logprobs:
            if not generated:
                raise RuntimeError("Cannot record policy probabilities for an empty generation")
            from .objective import token_logprobs

            if (len(output.scores) != len(generated)
                    or len(output.logits) != len(generated)):
                raise RuntimeError("Generation scores, logits, and tokens differ in length")
            result["old_logprobs"] = [
                float(token_logprobs(logits, token.reshape(1))[0].cpu())
                for token, logits in zip(output.sequences[0, len(prompt):],
                                         output.logits, strict=True)
            ]
            result["behavior_logprobs"] = [
                float(token_logprobs(scores, token.reshape(1))[0].cpu())
                for token, scores in zip(output.sequences[0, len(prompt):],
                                         output.scores, strict=True)
            ]
        del inputs, output
        return result


SYSTEM_PROMPT = (
    "你是 ShopSimulator 购物助手。只根据用户需求与工具返回的公开观察行动。"
    "每轮只输出一个完整工具调用，不要附加解释、分析、建议或模拟工具返回。"
    "参数严格符合 schema，ASIN、规格值和按钮必须来自最新页面。"
    "先搜索并核验商品与完整规格价格，只有确认品类和预算符合时才购买。"
    "要打开其他搜索结果时先返回结果页；信息子页先返回，不要在子页搜索或购买；不要重复无效动作。"
    "无法找到商品时可调用 finish_without_purchase。不要输出最终建议，继续调用工具直到环境结束。"
)


def run_task(task_id, actor, session_factory, on_turn, on_event, *, max_turns=8,
             max_invalid_calls=3, prompt_profile="mini",
             restrict_tools_to_observation=False):
    """Run one task; the session context always releases its environment lease."""
    start = time.perf_counter()
    actor.reset_peak()
    row = {"task_id": task_id, "status": "running", "turns": 0, "executed_actions": 0,
           "guard_rejections": 0, "parse_failures": 0, "prompt_tokens": 0,
           "generated_tokens": 0, "generation_seconds": 0.0,
           "environment_seconds": 0.0, "max_prompt_tokens_seen": 0,
           "history_pairs_dropped": 0, "generation_truncated_turns": 0,
           "outcome": None}
    invalid_in_a_row = 0
    with session_factory(event_sink=on_event) as session:
        initial = session.reset(task_id)
        if prompt_profile == "teacher":
            from .upstream import enable_upstream

            enable_upstream()
            from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT as teacher_prompt

            messages = [{"role": "system", "content": teacher_prompt},
                        {"role": "user", "content": initial["instruction"]}]
            tools = list(session.tools)
        elif prompt_profile == "mini":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial["instruction"] + "\n\n"
                 + initial["observation"]},
            ]
            tools = [tool for tool in session.tools if tool["function"]["name"] != "think"]
        else:
            raise ValueError(f"Unknown prompt profile: {prompt_profile}")
        for turn_index in range(max_turns):
            offered_tools = (available_tools_for_observation(tools, session.observation)
                             if prompt_profile == "mini" or restrict_tools_to_observation
                             else tools)
            try:
                generation = actor.complete(messages, offered_tools,
                                            observation=session.observation)
            except PromptBudgetError as exc:
                row.update(status="context_budget", budget_error=str(exc))
                break
            row["turns"] += 1
            row["prompt_tokens"] += generation["prompt_tokens"]
            row["generated_tokens"] += generation["generated_tokens"]
            row["generation_seconds"] += generation["generation_seconds"]
            row["max_prompt_tokens_seen"] = max(
                row["max_prompt_tokens_seen"], generation["prompt_tokens"])
            row["history_pairs_dropped"] += generation["history_pairs_dropped"]
            row["generation_truncated_turns"] += int(generation["generation_truncated"])
            turn = {"task_id": task_id, "turn_index": turn_index,
                    "offered_tools": [tool["function"]["name"] for tool in offered_tools],
                    **generation}
            text = generation["text"]
            messages.append({"role": "assistant", "content": text})
            try:
                call = parse_tool_call(text)
            except ValueError as exc:
                call = None
                turn["parse_error"] = str(exc)
            if call is None:
                if "parse_error" not in turn:
                    row["status"] = ("generation_budget" if generation["generation_truncated"]
                                     else "assistant_final")
                    turn["status"] = row["status"]
                    on_turn(turn)
                    break
                row["parse_failures"] += 1
                invalid_in_a_row += 1
                turn["status"] = "parse_rejected"
                messages.append({"role": "tool", "content":
                                 "工具格式无效；请按模板只输出一个完整 tool_call。当前页面：\n"
                                 + session.observation})
            else:
                name, arguments = call
                turn["tool"] = name
                turn["arguments"] = arguments
                event = session.step(name, arguments)
                if not event["executed"]:
                    row["guard_rejections"] += 1
                    invalid_in_a_row += 1
                    turn.update(status="guard_rejected", guard_reason=event["reason"])
                    response = ("动作未执行：" + event["reason"]
                                + "。请按最新页面选择合法动作：\n" + session.observation)
                else:
                    row["executed_actions"] += 1
                    row["environment_seconds"] += event.get("seconds", 0.0)
                    invalid_in_a_row = 0
                    turn["status"] = "executed"
                    response = event["observation"]
                    if event["done"]:
                        row.update(status="environment_terminal", outcome=event["outcome"])
                        turn["outcome"] = event["outcome"]
                messages.append({"role": "tool", "content": response})
            on_turn(turn)
            if row["status"] == "environment_terminal":
                break
            if invalid_in_a_row >= max_invalid_calls:
                row["status"] = "invalid_call_limit"
                break
        else:
            row["status"] = "turn_budget"
    row["wall_seconds"] = time.perf_counter() - start
    row["gpu_peak"] = actor.peak_gib()
    row["strict_success"] = bool(row["outcome"] and row["outcome"]["strict_success"])
    return row
