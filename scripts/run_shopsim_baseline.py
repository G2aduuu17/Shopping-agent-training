"""Run frozen local Qwen through serial ShopSimulator tools; no weight updates."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from shopmini.artifacts import append_jsonl, sha256, versions, write_json
from shopmini.shoprollout import FrozenQwen, run_task
from shopmini.shopsim import ShopSession
from shopmini.upstream import MINI_ROOT, UPSTREAM_COMMIT, require_upstream


def task_ids(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [int(row["task_id"]) for row in rows]


def summarize(rows):
    terminals = [row["outcome"] for row in rows if row.get("outcome")]
    valid = [outcome for outcome in terminals if outcome["reward_valid"]]
    calls = sum(row["executed_actions"] + row["guard_rejections"] for row in rows)
    turns = sum(row["turns"] for row in rows)
    return {
        "tasks": len(rows),
        "strict_successes": sum(row["strict_success"] for row in rows),
        "strict_success_rate": (sum(row["strict_success"] for row in rows) / len(rows)
                                if rows else None),
        "valid_terminals": len(valid),
        "valid_terminal_rate": len(valid) / len(rows) if rows else None,
        "mean_valid_reward": (sum(outcome["reward"] for outcome in valid) / len(valid)
                              if valid else None),
        "termination_reasons": {
            reason: sum(outcome["termination_reason"] == reason for outcome in terminals)
            for reason in sorted({outcome["termination_reason"] for outcome in terminals})
        },
        "run_statuses": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "tool_attempts": calls,
        "model_turns": turns,
        "executed_actions": sum(row["executed_actions"] for row in rows),
        "guard_rejections": sum(row["guard_rejections"] for row in rows),
        "parse_failures": sum(row["parse_failures"] for row in rows),
        "tool_parse_rate": calls / turns if turns else None,
        "valid_tool_rate_per_turn": (sum(row["executed_actions"] for row in rows) / turns
                                     if turns else None),
        "tool_execution_rate": (sum(row["executed_actions"] for row in rows) / calls
                                if calls else None),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "generation_seconds": sum(row["generation_seconds"] for row in rows),
        "environment_seconds": sum(row["environment_seconds"] for row in rows),
        "generation_truncated_turns": sum(row["generation_truncated_turns"] for row in rows),
        "max_prompt_tokens_seen": max((row["max_prompt_tokens_seen"] for row in rows),
                                      default=None),
        "history_pairs_dropped": sum(row["history_pairs_dropped"] for row in rows),
        "task_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "max_peak_allocated_gib": max((row["gpu_peak"]["allocated_gib"] for row in rows
                                       if row["gpu_peak"]["allocated_gib"] is not None),
                                      default=None),
        "max_process_peak_rss_gib": max((row["gpu_peak"].get("process_peak_rss_gib")
                                         for row in rows
                                         if row["gpu_peak"].get("process_peak_rss_gib") is not None),
                                        default=None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--prompt-profile", choices=["mini", "teacher"], default="mini")
    parser.add_argument("--restrict-tools-to-observation", action="store_true",
                        help="Offer only tools visible in the latest public observation")
    parser.add_argument("--adapter-path", type=Path,
                        help="Optional saved PEFT adapter for frozen SFT inference")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local standalone model directory")
    parser.add_argument("--model-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--min-free-gpu-gib", type=float)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    min_free_gpu_gib = (args.min_free_gpu_gib if args.min_free_gpu_gib is not None else
                        18 if args.model_dtype == "fp32" and args.device == "cuda" else 8)
    if (args.max_tasks < 1 or args.max_turns < 1 or args.max_new_tokens < 1
            or args.max_prompt_tokens < args.max_new_tokens + 512):
        parser.error("Use positive task/turn/token limits and enough prompt context")
    root = require_upstream()
    frozen_split = json.loads((MINI_ROOT / "data/shopsim_splits/manifest.json").read_text(
        encoding="utf-8"))
    tasks_path = (MINI_ROOT / "data/shopsim_splits/grpo_train.jsonl"
                  if args.split == "train" else root / "data/grpo/validation.jsonl")
    available = task_ids(tasks_path)
    expected = (frozen_split["grpo"]["filtered_train_count"] if args.split == "train"
                else frozen_split["grpo"]["validation_count"])
    if len(available) != expected or len(available) != len(set(available)):
        raise ValueError(f"{args.split} task list differs from frozen split")
    if args.split == "validation" and set(available) != set(
        frozen_split["grpo"]["validation_task_ids"]
    ):
        raise ValueError("GRPO validation IDs differ from frozen split")
    expected_hash = (frozen_split["grpo"]["filtered_train_sha256"]
                     if args.split == "train" else
                     frozen_split["sources"]["grpo_validation"]["sha256"])
    if sha256(tasks_path) != expected_hash:
        raise ValueError(f"{args.split} task file hash differs from frozen split")
    final_ids = set(task_ids(root / "data/evaluation/tasks.jsonl"))
    if set(available) & final_ids:
        raise ValueError(f"{args.split} overlaps final evaluation tasks")
    requested = args.task_ids or available[:args.max_tasks]
    if len(requested) != len(set(requested)) or len(requested) > args.max_tasks:
        raise ValueError("Duplicate task ID or task count exceeds --max-tasks")
    if not set(requested) <= set(available):
        raise ValueError(f"All task IDs must belong to {args.split}")
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Missing local model: {model_path}")
    adapter_path = args.adapter_path.resolve() if args.adapter_path else None
    if adapter_path and not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError(f"Missing local adapter: {adapter_path}")
    if adapter_path:
        adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(
            encoding="utf-8"
        ))
        if Path(adapter_config["base_model_name_or_path"]).resolve() != model_path.resolve():
            raise ValueError("Adapter base model path differs from local Qwen")
    settings = {"split": args.split, "task_ids": requested, "seed": args.seed,
                "device": args.device,
                "dtype": ("float32" if args.device == "cpu" or args.model_dtype == "fp32"
                          else "bfloat16"),
                "max_turns": args.max_turns, "max_prompt_tokens": args.max_prompt_tokens,
                "max_new_tokens": args.max_new_tokens, "min_free_gpu_gib": min_free_gpu_gib,
                "base_url": args.base_url, "model_path": str(model_path),
                "prompt_profile": args.prompt_profile,
                "restrict_tools_to_observation": args.restrict_tools_to_observation,
                "adapter_path": str(adapter_path) if adapter_path else None}
    if args.dry_run:
        print(json.dumps({"dry_run": True, **settings}, indent=2))
        return
    import torch

    if args.device == "cuda" and (not torch.cuda.is_available() or
                                  (args.model_dtype == "bf16" and
                                   not torch.cuda.is_bf16_supported())):
        raise RuntimeError("Frozen Qwen baseline requires CUDA and the requested dtype")
    out = args.output_dir or MINI_ROOT / "outputs" / (
        ("shopsim-qwen-sft-" if adapter_path else
         "shopsim-qwen-cpu-baseline-" if args.device == "cpu" else
         "shopsim-qwen-baseline-") + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        + "-" + uuid4().hex[:6])
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"Results: {out}", flush=True)
    model_files = ["config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        model_files.append(index_path.name)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        model_files.extend(sorted(set(index["weight_map"].values())))
    elif (model_path / "model.safetensors").is_file():
        model_files.append("model.safetensors")
    else:
        raise RuntimeError(f"No safetensors model weights in {model_path}")
    manifest = {"kind": ("frozen-qwen-shopsimulator-multiturn-sft" if adapter_path else
                         "frozen-qwen-shopsimulator-multiturn-baseline"),
                "model_trainable": False,
                "upstream_commit": UPSTREAM_COMMIT, "settings": settings,
                "task_file_sha256": sha256(tasks_path), "versions": versions(),
                "adapter_file_sha256": ({name: sha256(adapter_path / name) for name in
                                         ("adapter_config.json", "adapter_model.safetensors")}
                                        if adapter_path else None),
                "model_file_sha256": {name: sha256(model_path / name) for name in model_files},
                "source_sha256": {name: sha256(MINI_ROOT / name) for name in (
                    "src/shopmini/shoprollout.py", "src/shopmini/shopsim.py",
                    "scripts/run_shopsim_baseline.py")}}
    write_json(out / "manifest.json", manifest)
    rows = []
    started = time.perf_counter()
    try:
        lock = (FileLock(str(MINI_ROOT / ".gpu.lock"), timeout=0)
                if args.device == "cuda" else nullcontext())
        with lock:
            import torch

            torch.manual_seed(args.seed)
            if args.device == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            actor = FrozenQwen(model_path, args.max_prompt_tokens, args.max_new_tokens,
                               min_free_gpu_gib, device=args.device,
                               adapter_path=adapter_path, model_dtype=args.model_dtype)
            load_seconds = time.perf_counter() - started
            for task_id in requested:
                print(f"Frozen Qwen task {task_id}", flush=True)
                row = run_task(
                    task_id, actor,
                    lambda event_sink: ShopSession(args.base_url, event_sink=event_sink),
                    lambda turn: append_jsonl(out / "turns.jsonl", turn),
                    lambda event: append_jsonl(out / "environment_events.jsonl", event),
                    max_turns=args.max_turns, prompt_profile=args.prompt_profile,
                    restrict_tools_to_observation=args.restrict_tools_to_observation,
                )
                rows.append(row)
                append_jsonl(out / "tasks.jsonl", row)
                print(json.dumps({"task_id": task_id, "status": row["status"],
                                  "strict_success": row["strict_success"],
                                  "turns": row["turns"], "generated_tokens": row["generated_tokens"]}),
                      flush=True)
    except Exception as exc:
        write_json(out / "summary.json", {"completed": False, "error_type": type(exc).__name__,
                                          "error": str(exc), "completed_tasks": len(rows),
                                          "metrics": summarize(rows)})
        raise
    summary = {"completed": True, "split": args.split, "task_ids": requested,
               "device": args.device,
               "model_trainable": False, "model_load_seconds": load_seconds,
               "wall_seconds": time.perf_counter() - started,
               "metrics": summarize(rows)}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
