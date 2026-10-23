"""Collect frozen Stage A G=2 shopping rollouts and audit reward signal; no updates."""
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from shopmini.artifacts import append_jsonl, sha256, versions, write_json
from shopmini.shoprollout import FrozenQwen, run_task
from shopmini.shopsim import ShopSession
from shopmini.upstream import MINI_ROOT, UPSTREAM_COMMIT, require_upstream

AUDIT_MEAN_TOLERANCE = 0.03
AUDIT_MAX_TOLERANCE = 0.2


def group_signal(rows):
    """Never fabricate a reward for an invalid or unfinished rollout."""
    if len(rows) != 2:
        raise ValueError("Expected exactly two independent rollouts per task")
    rewards = [row["outcome"]["reward"] if row.get("outcome") and
               row["outcome"].get("reward_valid") else None for row in rows]
    valid = all(reward is not None and math.isfinite(reward) for reward in rewards)
    return {"rewards": rewards, "all_rewards_valid": valid,
            "reward_difference": abs(rewards[0] - rewards[1]) if valid else None,
            "nonzero_advantage": valid and not math.isclose(rewards[0], rewards[1],
                                                             abs_tol=1e-8)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--task-offset", type=int, default=0,
                        help="Zero-based starting row in the frozen GRPO training pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--model-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--constrain-visible-actions", action="store_true",
                        help="Inference-only token constraint for offered tools, ASINs and options")
    parser.add_argument("--restrict-tools-to-observation", action="store_true")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path,
                        help="Completed previous GRPO checkpoint for the next policy version")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.groups < 1 or args.task_offset < 0 or args.max_turns < 1
            or args.max_new_tokens < 1):
        parser.error("Use positive groups/turns/tokens and a nonnegative task offset")
    if args.max_prompt_tokens < args.max_new_tokens + 512:
        parser.error("Prompt budget is too small for the generation limit")
    if not (0.0 < args.temperature <= 1.0 and 0.0 < args.top_p <= 1.0):
        parser.error("Use sampling temperature and top-p in (0, 1]")
    training_eligible = (args.temperature == 1.0 and args.top_p == 1.0
                         and not args.constrain_visible_actions)
    upstream = require_upstream()
    split_path = MINI_ROOT / "data/shopsim_splits/manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_path = MINI_ROOT / "data/shopsim_splits/grpo_train.jsonl"
    ids = [json.loads(line)["task_id"] for line in train_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if (len(ids) != split["grpo"]["filtered_train_count"] or len(ids) != len(set(ids))
            or sha256(train_path) != split["grpo"]["filtered_train_sha256"]):
        raise RuntimeError("GRPO train pool differs from frozen split")
    if args.task_offset + args.groups > len(ids):
        parser.error("Requested range exceeds distinct GRPO training tasks")
    excluded = (set(split["sft"]["train_task_ids"])
                | set(split["sft"]["validation_task_ids"])
                | set(split["grpo"]["validation_task_ids"]))
    final_path = upstream / "data/evaluation/tasks.jsonl"
    excluded |= {json.loads(line)["task_id"] for line in final_path.read_text(
        encoding="utf-8").splitlines() if line.strip()}
    selected = ids[args.task_offset:args.task_offset + args.groups]
    if set(selected) & excluded:
        raise RuntimeError("Sampled tasks overlap SFT, development, or Final-200")
    adapter = args.adapter_path.resolve()
    adapter_file = adapter / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError(f"Missing Stage A adapter: {adapter_file}")
    parent = args.parent_checkpoint.resolve() if args.parent_checkpoint else None
    if parent:
        complete = json.loads((parent / "complete.json").read_text(encoding="utf-8"))
        if (adapter != (parent / "adapter").resolve()
                or sha256(adapter_file) != complete["adapter_sha256"]
                or sha256(parent / "state.pt") != complete["state_sha256"]
                or complete.get("optimizer_updates", 0) < 1):
            raise RuntimeError("Previous policy checkpoint is incomplete or changed")
    else:
        sft_summary = json.loads((adapter.parent / "summary.json").read_text(encoding="utf-8"))
        if not sft_summary.get("passed") or sft_summary.get("optimizer_updates") != 63:
            raise RuntimeError("Starting adapter is not the completed Stage A run")
    settings = {"group_size": 2, "task_ids": selected,
                "task_offset": args.task_offset, "seed": args.seed,
                "max_turns": args.max_turns, "max_prompt_tokens": args.max_prompt_tokens,
                "max_new_tokens": args.max_new_tokens, "model_dtype": args.model_dtype,
                "prompt_profile": "teacher",
                "restrict_tools_to_observation": args.restrict_tools_to_observation,
                "sampling": {"do_sample": True, "temperature": args.temperature,
                             "top_p": args.top_p, "top_k": 0},
                "action_constraint": ("visible_actions" if
                                      args.constrain_visible_actions else "none"),
                "training_eligible": training_eligible,
                "base_url": args.base_url}
    if args.dry_run:
        print(json.dumps({"dry_run": True, **settings}, ensure_ascii=False, indent=2))
        return

    import torch

    if not torch.cuda.is_available() or (args.model_dtype == "bf16"
                                         and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA and the requested dtype are required for grouped rollouts")
    min_free_gib = 18 if args.model_dtype == "fp32" else 8
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    if free_gib < min_free_gib:
        raise RuntimeError(f"Need {min_free_gib} GiB free GPU memory; found {free_gib:.2f}")
    out = (args.output_dir or MINI_ROOT / "outputs" / (
        "shopsim-grpo-signal-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        + "-" + uuid4().hex[:6])).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"Results: {out}", flush=True)
    model_path = args.model_path.resolve()
    model_files = ["config.json", "tokenizer_config.json", "tokenizer.json",
                   "chat_template.jinja", "model.safetensors.index.json"]
    model_index = json.loads((model_path / "model.safetensors.index.json").read_text(
        encoding="utf-8"))
    model_files.extend(sorted(set(model_index["weight_map"].values())))
    write_json(out / "manifest.json", {
        "kind": "frozen-stage-a-group-signal-probe" if not parent else
                "frozen-grpo-checkpoint-group-signal-probe",
        "training_started": False, "training_eligible": training_eligible,
        "upstream_commit": UPSTREAM_COMMIT, "settings": settings,
        "split_manifest_sha256": sha256(split_path),
        "train_pool_sha256": sha256(train_path),
        "adapter_path": str(adapter), "adapter_sha256": sha256(adapter_file),
        "model_file_sha256": {name: sha256(model_path / name) for name in model_files},
        "source_sha256": {name: sha256(MINI_ROOT / name) for name in (
            "src/shopmini/shoprollout.py", "src/shopmini/shopsim.py",
            "src/shopmini/visible_actions.py",
            "scripts/collect_shopsim_grpo_signal.py")},
        "parent_checkpoint": str(parent) if parent else None,
        "policy_version": complete["optimizer_updates"] if parent else 0,
        "versions": versions(),
    })
    groups = []
    errors = []
    constraint_counts = {"turns": 0, "function_name_turns": 0,
                         "parameter_turns": 0, "unavailable_turns": 0,
                         "masked_top1_steps": 0, "constrained_token_steps": 0}
    try:
        with FileLock(str(MINI_ROOT / ".gpu.lock"), timeout=0):
            actor = FrozenQwen(model_path, args.max_prompt_tokens, args.max_new_tokens,
                               adapter_path=adapter, sample=True, record_logprobs=True,
                               model_dtype=args.model_dtype,
                               temperature=args.temperature, top_p=args.top_p,
                               constrain_visible_actions=args.constrain_visible_actions,
                               min_free_gpu_gib=min_free_gib)
            for group_index, task_id in enumerate(selected):
                samples = []
                for sample_index in range(2):
                    seed = args.seed + (args.task_offset + group_index) * 2 + sample_index
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    turns = []

                    def on_turn(turn, group_index=group_index,
                                sample_index=sample_index, seed=seed, turns=turns):
                        constraint = turn.get("action_constraint")
                        if constraint:
                            function_applied = bool(constraint["function_name_applied"])
                            parameter_applied = bool(constraint["applied"])
                            constraint_counts["turns"] += int(function_applied or
                                                              parameter_applied)
                            constraint_counts["function_name_turns"] += int(
                                function_applied)
                            constraint_counts["parameter_turns"] += int(parameter_applied)
                            constraint_counts["unavailable_turns"] += int(
                                bool(constraint["unavailable"]))
                            constraint_counts["masked_top1_steps"] += (
                                constraint["raw_top1_blocked_steps"])
                            constraint_counts["constrained_token_steps"] += (
                                constraint["constrained_token_steps"])
                        turn.update(group_index=group_index, sample_index=sample_index,
                                    sampling_seed=seed)
                        turns.append(turn)
                        append_jsonl(out / "turns.jsonl", turn)

                    def on_event(event, group_index=group_index,
                                 sample_index=sample_index):
                        append_jsonl(out / "environment_events.jsonl", {
                            **event, "group_index": group_index,
                            "sample_index": sample_index,
                        })

                    row = run_task(
                        task_id, actor,
                        lambda event_sink: ShopSession(args.base_url, event_sink=event_sink),
                        on_turn, on_event, max_turns=args.max_turns,
                        prompt_profile="teacher",
                        restrict_tools_to_observation=args.restrict_tools_to_observation,
                    )
                    row.update(group_index=group_index, sample_index=sample_index,
                               sampling_seed=seed)
                    append_jsonl(out / "tasks.jsonl", row)
                    audit = []
                    for turn in turns:
                        old = turn["old_logprobs"]
                        new = actor.logprobs(turn["prompt_ids"], turn["generated_ids"])
                        if len(old) != len(new) or len(old) != len(turn["generated_ids"]):
                            raise RuntimeError("Saved token and log-probability lengths differ")
                        deltas = [abs(a - b) for a, b in zip(old, new, strict=True)]
                        audit.extend(deltas)
                        append_jsonl(out / "probability_audit.jsonl", {
                            "task_id": task_id, "group_index": group_index,
                            "sample_index": sample_index, "turn_index": turn["turn_index"],
                            "prompt_tokens": len(turn["prompt_ids"]),
                            "generated_tokens": len(deltas),
                            "mean_abs_error": sum(deltas) / len(deltas),
                            "max_abs_error": max(deltas),
                            "worst_token_index": max(range(len(deltas)), key=deltas.__getitem__),
                        })
                    if not audit or not all(math.isfinite(delta) for delta in audit):
                        raise RuntimeError("Missing or non-finite probability audit")
                    sample_mean_error = sum(audit) / len(audit)
                    sample_max_error = max(audit)
                    sample = {"task_id": task_id, "sample_index": sample_index,
                              "seed": seed, "status": row["status"],
                              "reward_valid": bool(row.get("outcome") and
                                                   row["outcome"].get("reward_valid")),
                              "reward": (row["outcome"]["reward"] if row.get("outcome")
                                         and row["outcome"].get("reward_valid") else None),
                              "strict_success": row["strict_success"],
                              "turns": row["turns"], "generated_tokens": row["generated_tokens"],
                              "audited_tokens": len(audit),
                              "logprob_mean_abs_error": sample_mean_error,
                              "logprob_max_abs_error": sample_max_error,
                              "probability_audit_passed": (
                                  sample_mean_error <= AUDIT_MEAN_TOLERANCE
                                  and sample_max_error <= AUDIT_MAX_TOLERANCE)}
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
                signal = group_signal([{"outcome": {"reward": sample["reward"],
                                                      "reward_valid": sample["reward_valid"]}}
                                       for sample in samples])
                group = {"task_id": task_id, "group_index": group_index,
                         "samples": samples, **signal}
                groups.append(group)
                append_jsonl(out / "groups.jsonl", group)
                print(json.dumps({"task_id": task_id, **signal}), flush=True)
                torch.cuda.empty_cache()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        write_json(out / "summary.json", {"completed": False, "errors": errors,
                                          "completed_groups": len(groups)})
        raise
    all_samples = [sample for group in groups for sample in group["samples"]]
    summary = {"completed": True, "training_started": False,
               "training_eligible": training_eligible,
               "groups": len(groups),
               "valid_reward_groups": sum(group["all_rewards_valid"] for group in groups),
               "nonzero_advantage_groups": sum(group["nonzero_advantage"] for group in groups),
               "censored_groups": sum(not group["all_rewards_valid"] for group in groups),
               "same_reward_groups": sum(group["all_rewards_valid"]
                                         and not group["nonzero_advantage"] for group in groups),
               "audit_tolerances": {"mean": AUDIT_MEAN_TOLERANCE,
                                    "max": AUDIT_MAX_TOLERANCE},
               "probability_audit_passed": all(
                   sample["probability_audit_passed"] for sample in all_samples),
               "audited_tokens": sum(sample["audited_tokens"] for sample in all_samples),
               "max_logprob_abs_error": max(sample["logprob_max_abs_error"]
                                            for sample in all_samples),
               "mean_logprob_abs_error": sum(sample["logprob_mean_abs_error"]
                                             * sample["audited_tokens"] for sample in all_samples)
               / sum(sample["audited_tokens"] for sample in all_samples),
               "action_constraint_counts": constraint_counts,
               "scope": "no-update group signal and probability probe; not GRPO training"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
