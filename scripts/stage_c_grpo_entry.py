"""Initialize Stage C GRPO LoRA and collect bounded G=4 on-policy groups.

This entry point does not perform an optimizer update. Each invocation freezes
one policy version and collects up to three two-task batches for the next update.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

from filelock import FileLock

from shopmini.artifacts import append_jsonl, sha256, versions, write_json
from shopmini.shoprollout import FrozenQwen, run_task
from shopmini.shopsim import ShopSession
from shopmini.upstream import MINI_ROOT, UPSTREAM_COMMIT, require_upstream

GROUP_SIZE = 4
GROUPS_PER_BATCH = 2
MAX_BATCHES = 3
AUDIT_MEAN_TOLERANCE = 0.03
AUDIT_MAX_TOLERANCE = 0.2


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[int]:
    return [int(json.loads(line)["task_id"]) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def task_window(train: list[int], offset: int, count: int) -> list[int]:
    """Cycle training prompts while the global cursor and rollout seeds keep advancing."""
    if not train or offset < 0 or count < 1:
        raise ValueError("Invalid cyclic training task window")
    return [train[(offset + index) % len(train)] for index in range(count)]


def group_signal(samples: list[dict]) -> dict:
    """Keep only complete, valid, reward-varying groups; never invent a reward."""
    if len(samples) != GROUP_SIZE or sorted(s["sample_index"] for s in samples) != list(
        range(GROUP_SIZE)):
        raise ValueError("A group must contain exactly four distinct sample indices")
    rewards = [s["reward"] if s["reward_valid"] else None for s in samples]
    valid = all(value is not None and math.isfinite(value) for value in rewards)
    if valid:
        spread = max(rewards) - min(rewards)
        varying = spread > 1e-8
    else:
        spread = None
        varying = False
    audited = all(s["probability_audit_passed"] for s in samples)
    reason = ("invalid_reward" if not valid else
              "probability_audit_failed" if not audited else
              "same_reward" if not varying else None)
    return {"rewards": rewards, "all_rewards_valid": valid,
            "reward_difference": spread, "nonzero_advantage": varying,
            "probability_audit_passed": audited, "eligible": reason is None,
            "drop_reason": reason}


def validate_start(path: Path) -> tuple[dict, list[int]]:
    start = read_json(path)
    if (start.get("kind") != "stage-c-grpo-start-freeze-v1" or
            start.get("passed") is not True or start.get("training_started") is not False or
            start.get("task_splits_disjoint") is not True):
        raise RuntimeError("Stage C GRPO start manifest is not a completed freeze")
    model = Path(start["model_path"]).resolve()
    if (sha256(model / "model.safetensors") != start["model_weights_sha256"] or
            sha256(model / "tokenizer.json") != start["tokenizer_sha256"] or
            sha256(model / "merge_complete.json") != start["merge_report_sha256"]):
        raise RuntimeError("Stage C model changed after start freeze")
    train_path = Path(start["grpo_train_path"]).resolve()
    validation_path = Path(start["grpo_validation_path"]).resolve()
    final_path = Path(start["final_200_path"]).resolve()
    for file, expected in ((train_path, start["grpo_train_sha256"]),
                           (validation_path, start["grpo_validation_sha256"]),
                           (final_path, start["final_200_sha256"])):
        if sha256(file) != expected:
            raise RuntimeError(f"Frozen task file changed: {file}")
    train, validation, final = map(read_ids, (train_path, validation_path, final_path))
    if (len(train) != start["grpo_train_tasks"] or
            len(validation) != start["grpo_validation_tasks"] or
            len(final) != start["final_200_tasks"] or
            validation != start["grpo_validation_task_ids"] or
            any(len(ids) != len(set(ids)) for ids in (train, validation, final)) or
            set(train) & (set(validation) | set(final)) or set(validation) & set(final)):
        raise RuntimeError("Task IDs differ from frozen disjoint split")
    return start, train


def policy_source(run: Path, parent: Path | None, start: dict) -> tuple[Path, int, str]:
    if parent is None:
        adapter = run / "initial_adapter"
        complete = run / "initial_complete.json"
        if not complete.is_file():
            raise RuntimeError("Initial adapter is missing; run --action init first")
        state = read_json(complete)
        merge = read_json(Path(start["model_path"]) / "merge_complete.json")
        expected_modules = sorted(read_json(
            Path(merge["stage_c_adapter"]) / "adapter_config.json")["target_modules"])
        adapter_config = read_json(adapter / "adapter_config.json")
        if (state.get("policy_version") != 0 or
                state.get("base_model_sha256") != start["model_weights_sha256"] or
                state.get("lora_r") != 16 or state.get("lora_alpha") != 32 or
                state.get("target_modules") != expected_modules or
                state.get("zero_initialized_lora_b_tensors", 0) < 1 or
                sorted(adapter_config["target_modules"]) != expected_modules or
                Path(adapter_config["base_model_name_or_path"]).resolve() !=
                Path(start["model_path"]).resolve() or
                sha256(adapter / "adapter_model.safetensors") != state.get("adapter_sha256")):
            raise RuntimeError("Initial Stage C LoRA changed or has wrong lineage")
        return adapter, 0, state["adapter_sha256"]
    complete = read_json(parent / "complete.json")
    adapter = parent / "adapter"
    if (complete.get("base_model_sha256") != start["model_weights_sha256"] or
            not isinstance(complete.get("optimizer_updates"), int) or
            complete["optimizer_updates"] < 1 or
            sha256(adapter / "adapter_model.safetensors") != complete.get("adapter_sha256") or
            sha256(parent / "state.pt") != complete.get("state_sha256")):
        raise RuntimeError("Parent checkpoint is incomplete or has wrong Stage C lineage")
    return adapter, complete["optimizer_updates"], complete["adapter_sha256"]


def initialize_adapter(run: Path, start: dict, seed: int) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import Qwen3_5ForConditionalGeneration

    complete = run / "initial_complete.json"
    adapter = run / "initial_adapter"
    if complete.exists():
        source, version, digest = policy_source(run, None, start)
        return {"policy_version": version, "adapter": str(source),
                "adapter_sha256": digest, "reused": True}
    if adapter.exists() or (run / "initial_adapter.incomplete").exists():
        raise RuntimeError("Uncommitted initial adapter exists; use a fresh output directory")
    staging = run / "initial_adapter.incomplete"
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        start["model_path"], dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", device_map={"": "cpu"})
    merge = read_json(Path(start["model_path"]) / "merge_complete.json")
    stage_c_adapter_config = read_json(
        Path(merge["stage_c_adapter"]) / "adapter_config.json")
    target_modules = sorted(stage_c_adapter_config["target_modules"])
    if not target_modules or stage_c_adapter_config["r"] != 16 or stage_c_adapter_config["lora_alpha"] != 32:
        raise RuntimeError("Stage C LoRA architecture differs from frozen recipe")
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
                        init_lora_weights=True)
    policy = get_peft_model(model, config)
    policy.save_pretrained(staging, safe_serialization=True)
    del policy, model
    gc.collect()
    weights = staging / "adapter_model.safetensors"
    b_count = 0
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for name in handle.keys():  # noqa: SIM118 - safe_open exposes keys(), not dict iteration
            if "lora_B" in name:
                if torch.count_nonzero(handle.get_tensor(name)).item() != 0:
                    raise RuntimeError(f"New LoRA is not a no-op: {name}")
                b_count += 1
    if b_count == 0:
        raise RuntimeError("New LoRA has no zero-initialized B matrices")
    staging.rename(adapter)
    state = {"policy_version": 0, "base_model_sha256": start["model_weights_sha256"],
             "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
             "lora_r": 16, "lora_alpha": 32, "target_modules": target_modules,
             "zero_initialized_lora_b_tensors": b_count, "seed": seed}
    write_json(complete, state)
    return {"policy_version": 0, "adapter": str(adapter),
            "adapter_sha256": state["adapter_sha256"], "reused": False,
            "zero_initialized_lora_b_tensors": b_count}


def run_collection(args, run: Path, start: dict, train: list[int]) -> None:
    import torch

    require_upstream()
    parent = args.parent_checkpoint.resolve() if args.parent_checkpoint else None
    adapter, policy_version, adapter_sha = policy_source(run, parent, start)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] / 2**30 < 18:
        raise RuntimeError("FP32 GRPO rollout needs CUDA with at least 18 GiB free")
    selected = task_window(train, args.task_offset, GROUPS_PER_BATCH * MAX_BATCHES)
    source = run / f"policy-{policy_version:04d}-offset-{args.task_offset:04d}"
    source.mkdir(exist_ok=False)
    settings = {"group_size": GROUP_SIZE, "groups_per_batch": GROUPS_PER_BATCH,
                "max_generation_batches": MAX_BATCHES, "required_eligible_groups": 2,
                "task_offset": args.task_offset, "candidate_task_ids": selected,
                "seed": args.seed, "max_turns": args.max_turns,
                "max_prompt_tokens": args.max_prompt_tokens,
                "max_new_tokens": args.max_new_tokens, "model_dtype": "fp32",
                "prompt_profile": "teacher", "restrict_tools_to_observation": False,
                "sampling": {"do_sample": True, "temperature": 0.7,
                             "top_p": 0.9, "top_k": 0},
                "action_constraint": "none", "base_url": args.base_url}
    write_json(source / "manifest.json", {
        "kind": "stage-c-grpo-g4-current-policy-rollouts-v1",
        "start_manifest": str(args.start_manifest.resolve()),
        "start_manifest_sha256": sha256(args.start_manifest),
        "upstream_commit": UPSTREAM_COMMIT, "training_started": False,
        "model_path": start["model_path"],
        "model_weights_sha256": start["model_weights_sha256"],
        "train_pool_sha256": start["grpo_train_sha256"],
        "adapter_path": str(adapter), "adapter_sha256": adapter_sha,
        "parent_checkpoint": str(parent) if parent else None,
        "policy_version": policy_version, "settings": settings,
        "source_sha256": {name: sha256(MINI_ROOT / name) for name in (
            "src/shopmini/shoprollout.py", "src/shopmini/shopsim.py",
            "scripts/stage_c_grpo_entry.py")}, "versions": versions()})
    print(f"Results: {source}", flush=True)
    groups = []
    try:
        with FileLock(str(MINI_ROOT / ".gpu.lock"), timeout=0):
            actor = FrozenQwen(start["model_path"], args.max_prompt_tokens,
                               args.max_new_tokens, adapter_path=adapter,
                               sample=True, record_logprobs=True,
                               model_dtype="fp32", temperature=0.7, top_p=0.9,
                               min_free_gpu_gib=18)
            for batch_index in range(MAX_BATCHES):
                for within_batch in range(GROUPS_PER_BATCH):
                    group_index = batch_index * GROUPS_PER_BATCH + within_batch
                    task_id = selected[group_index]
                    samples = []
                    for sample_index in range(GROUP_SIZE):
                        seed = args.seed + (args.task_offset + group_index) * GROUP_SIZE + sample_index
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        turns = []

                        def on_turn(turn, *, turns=turns, seed=seed,
                                    group_index=group_index, batch_index=batch_index,
                                    sample_index=sample_index):
                            turn.update(group_index=group_index, batch_index=batch_index,
                                        sample_index=sample_index, sampling_seed=seed,
                                        policy_version=policy_version)
                            turns.append(turn)
                            append_jsonl(source / "turns.jsonl", turn)

                        def on_event(event, *, group_index=group_index,
                                     sample_index=sample_index):
                            append_jsonl(source / "environment_events.jsonl", {
                                **event, "group_index": group_index,
                                "sample_index": sample_index,
                                "policy_version": policy_version})

                        row = run_task(
                            task_id, actor,
                            lambda event_sink: ShopSession(args.base_url, event_sink=event_sink),
                            on_turn, on_event, max_turns=args.max_turns,
                            prompt_profile="teacher", restrict_tools_to_observation=False)
                        row.update(group_index=group_index, batch_index=batch_index,
                                   sample_index=sample_index, sampling_seed=seed,
                                   policy_version=policy_version)
                        append_jsonl(source / "tasks.jsonl", row)
                        errors = []
                        for turn in turns:
                            old = turn["old_logprobs"]
                            actual = actor.logprobs(turn["prompt_ids"], turn["generated_ids"])
                            if (len(old) != len(actual) or
                                    len(old) != len(turn["behavior_logprobs"]) or
                                    len(old) != len(turn["generated_ids"]) or
                                    not all(math.isfinite(value) for value in
                                            (*old, *turn["behavior_logprobs"]))):
                                raise RuntimeError("Generated tokens and policy probabilities differ")
                            differences = [abs(a - b) for a, b in zip(old, actual, strict=True)]
                            errors.extend(differences)
                            append_jsonl(source / "probability_audit.jsonl", {
                                "task_id": task_id, "group_index": group_index,
                                "sample_index": sample_index,
                                "turn_index": turn["turn_index"],
                                "audited_tokens": len(differences),
                                "max_abs_error": max(differences)})
                        mean_error = sum(errors) / len(errors) if errors else None
                        max_error = max(errors) if errors else None
                        audited = bool(errors) and all(math.isfinite(x) for x in errors)
                        audit_passed = (audited and mean_error <= AUDIT_MEAN_TOLERANCE
                                        and max_error <= AUDIT_MAX_TOLERANCE)
                        valid = (row["status"] == "environment_terminal" and
                                 row.get("outcome") is not None and
                                 row["outcome"].get("reward_valid") is True)
                        sample = {"task_id": task_id, "sample_index": sample_index,
                                  "seed": seed, "status": row["status"],
                                  "reward_valid": valid,
                                  "reward": row["outcome"]["reward"] if valid else None,
                                  "strict_success": row["strict_success"],
                                  "turns": row["turns"], "generated_tokens": row["generated_tokens"],
                                  "audited_tokens": len(errors),
                                  "logprob_mean_abs_error": mean_error,
                                  "logprob_max_abs_error": max_error,
                                  "probability_audit_passed": audit_passed}
                        samples.append(sample)
                        print(json.dumps(sample, ensure_ascii=False), flush=True)
                    group = {"task_id": task_id, "group_index": group_index,
                             "batch_index": batch_index, "policy_version": policy_version,
                             "samples": samples, **group_signal(samples)}
                    groups.append(group)
                    append_jsonl(source / "groups.jsonl", group)
                    print(json.dumps({"task_id": task_id,
                                      "eligible": group["eligible"],
                                      "drop_reason": group["drop_reason"]}), flush=True)
                    torch.cuda.empty_cache()
                if sum(group["eligible"] for group in groups) >= 2:
                    break
            del actor
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        write_json(source / "summary.json", {
            "completed": False, "training_started": False,
            "attempted_groups": len(groups),
            "error": f"{type(exc).__name__}: {exc}"})
        raise
    eligible = [group for group in groups if group["eligible"]]
    all_samples = [sample for group in groups for sample in group["samples"]]
    audited = [sample for sample in all_samples if sample["audited_tokens"]]
    summary = {"completed": True, "training_started": False,
               "optimizer_updates": 0, "policy_version": policy_version,
               "group_size": GROUP_SIZE, "attempted_batches": 1 + groups[-1]["batch_index"],
               "attempted_groups": len(groups), "attempted_trajectories": len(all_samples),
               "valid_reward_groups": sum(g["all_rewards_valid"] for g in groups),
               "nonzero_advantage_groups": sum(g["nonzero_advantage"] for g in groups),
               "eligible_groups": len(eligible),
               "selected_task_ids_for_update": [g["task_id"] for g in eligible[:2]],
               "ready_for_update": len(eligible) >= 2,
               "drop_reason_counts": {reason: sum(g["drop_reason"] == reason for g in groups)
                                      for reason in ("invalid_reward", "same_reward",
                                                     "probability_audit_failed")},
               "next_task_offset": args.task_offset + len(groups),
               "audited_tokens": sum(s["audited_tokens"] for s in audited),
               "max_logprob_abs_error": max((s["logprob_max_abs_error"] for s in audited),
                                            default=None),
               "probability_audit_passed_for_selected": bool(eligible) and all(
                   s["probability_audit_passed"] for g in eligible[:2] for s in g["samples"]),
               "scope": "Stage C G=4 current-policy collection only; no optimizer update"}
    write_json(source / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("dry-run", "init", "collect"),
                        default="dry-run")
    parser.add_argument("--start-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    args = parser.parse_args()
    if (args.task_offset < 0 or args.max_turns < 1 or args.max_new_tokens < 1 or
            args.max_prompt_tokens < args.max_new_tokens + 512):
        parser.error("Invalid task offset or prompt/generation budget")
    if args.action == "init" and args.parent_checkpoint:
        parser.error("Initial LoRA creation cannot use a parent checkpoint")
    start_path = args.start_manifest.resolve()
    start, train = validate_start(start_path)
    run = args.output_dir.resolve()
    plan = {"kind": "stage-c-grpo-g4-collection-v1",
            "start_manifest": str(start_path),
            "start_manifest_sha256": sha256(start_path),
            "model_weights_sha256": start["model_weights_sha256"],
            "train_pool_sha256": start["grpo_train_sha256"],
            "seed": args.seed, "group_size": GROUP_SIZE,
            "groups_per_batch": GROUPS_PER_BATCH, "max_batches": MAX_BATCHES,
            "temperature": 0.7, "top_p": 0.9,
            "max_turns": args.max_turns,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens, "base_url": args.base_url,
            "model_dtype": "fp32", "training_started": False}
    if args.action == "dry-run":
        print(json.dumps({"dry_run": True, "action": "collect",
                          "output_dir": str(run), **plan}, ensure_ascii=False, indent=2))
        return
    if run.exists():
        if read_json(run / "plan.json") != plan:
            raise RuntimeError("Existing output has a different collection plan")
    else:
        run.mkdir(parents=True)
        write_json(run / "plan.json", plan)
    if args.action == "init":
        result = initialize_adapter(run, start, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    run_collection(args, run, start, train)


if __name__ == "__main__":
    main()
