"""One audited Stage C GRPO update per invocation, with resumable checkpoints.

The normal path requires two eligible G=4 groups. --technical-single-group
explicitly permits one group for a training-system check, not a formal run.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

from filelock import FileLock
from stage_c_grpo_entry import group_signal, read_json, task_window, validate_start

from shopmini.artifacts import sha256, versions, write_json
from shopmini.objective import advantages, clipped_loss
from shopmini.upstream import MINI_ROOT, require_upstream

MEAN_TOL = 0.03
MAX_TOL = 0.2
RELOAD_TOL = 1e-5


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def checked_checkpoint(path: Path, base_sha: str, version: int) -> dict:
    complete = read_json(path / "complete.json")
    if (complete.get("optimizer_updates") != version or
            complete.get("base_model_sha256") != base_sha or
            sha256(path / "adapter/adapter_model.safetensors") != complete.get("adapter_sha256") or
            sha256(path / "state.pt") != complete.get("state_sha256")):
        raise RuntimeError("Checkpoint is incomplete or changed")
    return complete


def select_source(source: Path, start_path: Path, parent: Path | None,
                  technical: bool, prior_sources: list[dict], cursor: int):
    start, train_ids = validate_start(start_path)
    manifest, summary = read_json(source / "manifest.json"), read_json(source / "summary.json")
    settings = manifest.get("settings", {})
    kind = manifest.get("kind")
    if kind == "stage-c-grpo-g4-technical-prefix-v1":
        if not technical:
            raise RuntimeError("Interrupted prefix is only usable for explicit technical checks")
        origin = manifest["technical_prefix_origin"]
        original = Path(origin["path"])
        if any(sha256(original / name) != digest for name, digest in origin["sha256"].items()):
            raise RuntimeError("Interrupted source changed after prefix extraction")
        for name in ("groups.jsonl", "tasks.jsonl", "turns.jsonl",
                     "probability_audit.jsonl", "environment_events.jsonl"):
            expected = [row for row in rows(original / name) if row["group_index"] == 0]
            if rows(source / name) != expected:
                raise RuntimeError("Technical prefix differs from completed source group")
    if (kind not in ("stage-c-grpo-g4-current-policy-rollouts-v1",
                     "stage-c-grpo-g4-technical-prefix-v1") or
            manifest.get("training_started") is not False or
            summary.get("completed") is not True or summary.get("training_started") is not False or
            summary.get("optimizer_updates") != 0 or
            manifest.get("start_manifest_sha256") != sha256(start_path) or
            manifest.get("model_weights_sha256") != start["model_weights_sha256"] or
            manifest.get("train_pool_sha256") != start["grpo_train_sha256"] or
            settings.get("group_size") != 4 or settings.get("groups_per_batch") != 2 or
            settings.get("required_eligible_groups") != 2 or
            settings.get("model_dtype") != "fp32" or
            settings.get("action_constraint") != "none" or
            settings.get("sampling") != {"do_sample": True, "temperature": 0.7,
                                         "top_p": 0.9, "top_k": 0}):
        raise RuntimeError("Source is not the frozen Stage C G=4 policy collection")
    offset = settings["task_offset"]
    candidates = settings["candidate_task_ids"]
    if (offset < cursor or task_window(train_ids, offset, len(candidates)) != candidates or
            len(candidates) != 6):
        raise RuntimeError("Source task cursor or frozen train task order differs")
    if parent is None:
        if manifest.get("policy_version") != 0 or manifest.get("parent_checkpoint") is not None:
            raise RuntimeError("First update must use policy version zero")
        initial = Path(manifest["adapter_path"])
        state = read_json(initial.parent / "initial_complete.json")
        if (initial.name != "initial_adapter" or state.get("policy_version") != 0 or
                state.get("base_model_sha256") != start["model_weights_sha256"] or
                state.get("adapter_sha256") != manifest["adapter_sha256"]):
            raise RuntimeError("Initial LoRA provenance differs")
        version = 0
    else:
        version = read_json(parent / "complete.json")["optimizer_updates"]
        complete = checked_checkpoint(parent, start["model_weights_sha256"], version)
        if (manifest.get("policy_version") != version or
                Path(manifest.get("parent_checkpoint", "")).resolve() != parent.resolve() or
                Path(manifest["adapter_path"]).resolve() != (parent / "adapter").resolve() or
                manifest["adapter_sha256"] != complete["adapter_sha256"]):
            raise RuntimeError("Collected policy differs from parent checkpoint")
    if sha256(Path(manifest["adapter_path"]) / "adapter_model.safetensors") != manifest["adapter_sha256"]:
        raise RuntimeError("Rollout adapter weights changed")
    groups, tasks, turns = (rows(source / name) for name in
                            ("groups.jsonl", "tasks.jsonl", "turns.jsonl"))
    if (not groups or len(groups) != summary["attempted_groups"] or
            len(tasks) != 4 * len(groups) or
            summary["next_task_offset"] != offset + len(groups) or
            [g["task_id"] for g in groups] != candidates[:len(groups)] or
            [g["group_index"] for g in groups] != list(range(len(groups)))):
        raise RuntimeError("Group count, task order, or next cursor changed")
    if any(offset < old["next_task_offset"] and old["task_offset"] < offset + len(groups)
           for old in prior_sources):
        raise RuntimeError("Source reuses task offsets from a previous update")
    eligible = []
    for group in groups:
        task_id, group_index = group["task_id"], group["group_index"]
        if (group["policy_version"] != version or
                [s["sample_index"] for s in group["samples"]] != list(range(4)) or
                any(s["task_id"] != task_id for s in group["samples"])):
            raise RuntimeError("Group policy or sample identity changed")
        calculated = group_signal(group["samples"])
        if any(group[key] != value for key, value in calculated.items()):
            raise RuntimeError("Group eligibility differs from rewards and audit")
        samples = []
        for index, sample in enumerate(group["samples"]):
            matched = [t for t in tasks if t["group_index"] == group_index and
                       t["sample_index"] == index]
            if len(matched) != 1:
                raise RuntimeError("Missing or duplicate trajectory")
            task = matched[0]
            sample_turns = sorted([t for t in turns if t["group_index"] == group_index and
                                   t["sample_index"] == index], key=lambda t: t["turn_index"])
            if (task["task_id"] != task_id or task["policy_version"] != version or
                    task["sampling_seed"] != sample["seed"] or
                    task["status"] != sample["status"] or
                    task["turns"] != len(sample_turns) or
                    [t["turn_index"] for t in sample_turns] != list(range(len(sample_turns))) or
                    sum(len(t["generated_ids"]) for t in sample_turns) != task["generated_tokens"]):
                raise RuntimeError("Trajectory metadata differs from group")
            valid = (task["status"] == "environment_terminal" and
                     task.get("outcome") is not None and task["outcome"].get("reward_valid") is True)
            if valid != sample["reward_valid"] or (valid and not math.isclose(
                    task["outcome"]["reward"], sample["reward"], abs_tol=1e-9)):
                raise RuntimeError("Saved reward differs from task outcome")
            for turn in sample_turns:
                n = len(turn["generated_ids"])
                if (turn["task_id"] != task_id or turn["policy_version"] != version or
                        turn["sampling_seed"] != sample["seed"] or
                        not turn["prompt_ids"] or not n or
                        n != turn["generated_tokens"] or
                        len(turn["prompt_ids"]) != turn["prompt_tokens"] or
                        n != len(turn["old_logprobs"]) or
                        n != len(turn["behavior_logprobs"]) or
                        n > settings["max_new_tokens"] or
                        len(turn["prompt_ids"]) > settings["max_prompt_tokens"] or
                        not all(math.isfinite(v) for v in
                                (*turn["old_logprobs"], *turn["behavior_logprobs"]))):
                    raise RuntimeError("Incomplete token-level policy record")
            if group["eligible"]:
                if not valid or not sample_turns:
                    raise RuntimeError("Eligible group has invalid or empty trajectory")
                samples.append({"reward": sample["reward"], "turns": sample_turns})
        if group["eligible"]:
            eligible.append((group, samples))
    if len(eligible) != summary["eligible_groups"] or not eligible:
        raise RuntimeError("Eligible-group count differs or no valid signal")
    if not technical and (len(eligible) < 2 or summary["ready_for_update"] is not True):
        raise RuntimeError("Formal update requires two eligible G=4 groups")
    selected = eligible[:1 if technical else 2]
    hashes = {name: sha256(source / name) for name in
              ("manifest.json", "summary.json", "groups.jsonl", "tasks.jsonl", "turns.jsonl")}
    provenance = {"source": str(source), "source_hashes": hashes,
                  "attempted_task_ids": [g["task_id"] for g in groups],
                  "selected_task_ids": [g["task_id"] for g, _ in selected],
                  "policy_version": version, "task_offset": offset,
                  "next_task_offset": summary["next_task_offset"]}
    return start, manifest, summary, selected, provenance


def load_model(base_path: Path, adapter: Path, trainable: bool):
    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_path, dtype=torch.float32, local_files_only=True,
        attn_implementation="sdpa", device_map={"": "cuda:0"})
    model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    model.config.use_cache = False
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.train()
    else:
        model.eval()
    return model


def update(args):
    import torch

    from shopmini.policy import completion_logprobs
    require_upstream()
    root, source, start_path = (p.resolve() for p in
                                (args.output_dir, args.source, args.start_manifest))
    if args.resume:
        plan = read_json(root / "plan.json")
        latest = read_json(root / "latest.json")
        if (plan["start_manifest_sha256"] != sha256(start_path) or
                plan["learning_rate"] != args.learning_rate or
                plan["max_peak_gib"] != args.max_peak_gib or
                plan["technical_single_group"] != args.technical_single_group or
                latest["optimizer_updates"] < 1):
            raise RuntimeError("Resume plan differs or no completed checkpoint exists")
        parent = Path(latest["checkpoint"]).resolve()
        prior_metrics = parent.parent / "metrics.json"
        if (sha256(parent / "complete.json") != latest["complete_sha256"] or
                sha256(prior_metrics) != latest["metrics_sha256"] or
                len(plan["sources"]) != latest["optimizer_updates"]):
            raise RuntimeError("Latest checkpoint marker changed")
        prior_sources = plan["sources"]
        for previous in prior_sources:
            old_source = Path(previous["source"])
            if any(sha256(old_source / name) != digest
                   for name, digest in previous["source_hashes"].items()):
                raise RuntimeError("A prior rollout source changed after its update")
        cursor = latest["next_task_offset"]
    else:
        if root.exists():
            raise RuntimeError("Output directory already exists; use --resume or a fresh path")
        parent, prior_sources, cursor = None, [], 0
    start, manifest, summary, selected, provenance = select_source(
        source, start_path, parent, args.technical_single_group, prior_sources, cursor)
    version = provenance["policy_version"]
    if args.resume and (version != latest["optimizer_updates"] or
                        sha256(Path(manifest["adapter_path"]) / "adapter_model.safetensors") !=
                        checked_checkpoint(parent, start["model_weights_sha256"], version)["adapter_sha256"]):
        raise RuntimeError("Resume would use stale policy rollouts")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] / 2**30 < 18:
        raise RuntimeError("FP32 update requires CUDA and 18 GiB free GPU memory")
    if not args.resume:
        root.mkdir(parents=True)
        plan = {"kind": "stage-c-grpo-audited-update-v1", "start_manifest": str(start_path),
                "start_manifest_sha256": sha256(start_path),
                "base_model_sha256": start["model_weights_sha256"],
                "learning_rate": args.learning_rate, "max_peak_gib": args.max_peak_gib,
                "technical_single_group": args.technical_single_group,
                "sources": [], "versions": versions()}
        write_json(root / "plan.json", plan)
    step = version + 1
    step_dir = root / f"step-{step:04d}"
    step_dir.mkdir(exist_ok=False)
    print(f"Results: {step_dir}", flush=True)
    t0 = time.monotonic()
    with FileLock(str(MINI_ROOT / ".gpu.lock"), timeout=0):
        torch.cuda.reset_peak_memory_stats()
        model = load_model(Path(start["model_path"]), Path(manifest["adapter_path"]), True)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not named or any("lora_" not in n for n, _ in named):
            raise RuntimeError("Only LoRA parameters may be trainable")
        params = [p for _, p in named]
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0)
        if parent:
            state = torch.load(parent / "state.pt", map_location="cpu", weights_only=False)
            if (state["optimizer_updates"] != version or
                    state["next_task_offset"] != cursor):
                raise RuntimeError("Optimizer version or saved task cursor differs")
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        errors = []
        with torch.no_grad():
            for group, samples in selected:
                for sample in samples:
                    for turn in sample["turns"]:
                        actual = completion_logprobs(model, turn["prompt_ids"],
                                                      turn["generated_ids"], "cuda")
                        old = torch.tensor(turn["old_logprobs"], device="cuda")
                        errors.extend((actual.float() - old).abs().cpu().tolist())
                        del actual, old
                print(f"Audited task {group['task_id']} from policy {version}", flush=True)
        if (not errors or not all(math.isfinite(e) for e in errors) or
                sum(errors) / len(errors) > MEAN_TOL or max(errors) > MAX_TOL):
            raise RuntimeError("Pre-update token probability audit failed")
        before = [p.detach().clone() for p in params]
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        reward_details = []
        for group, samples in selected:
            rewards = [s["reward"] for s in samples]
            adv = advantages(rewards).tolist()
            reward_details.append({"task_id": group["task_id"], "rewards": rewards,
                                   "advantages": adv})
            for sample, advantage in zip(samples, adv, strict=True):
                sample_tokens = sum(len(t["generated_ids"]) for t in sample["turns"])
                for turn in sample["turns"]:
                    new = completion_logprobs(model, turn["prompt_ids"],
                                              turn["generated_ids"], "cuda")
                    old = torch.tensor(turn["old_logprobs"], device="cuda")
                    weight = len(turn["generated_ids"]) / sample_tokens / 4 / len(selected)
                    loss = clipped_loss(new, old, advantage, epsilon=0.2) * weight
                    loss_total += float(loss.detach())
                    loss.backward()
                    del new, old, loss
            print(f"Backward complete for task {group['task_id']}", flush=True)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
        peak_before = torch.cuda.max_memory_allocated() / 2**30
        if not math.isfinite(grad_norm) or grad_norm <= 0 or peak_before > args.max_peak_gib:
            raise RuntimeError("Gradient or allocated GPU memory limit failed before step")
        optimizer.step()
        delta = math.sqrt(sum(float((p.detach() - old).float().square().sum())
                              for p, old in zip(params, before, strict=True)))
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError("LoRA parameters did not change")
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        probe = selected[0][1][0]["turns"][-1]
        with torch.no_grad():
            expected = completion_logprobs(model, probe["prompt_ids"],
                                           probe["generated_ids"], "cuda").float().cpu()
        staging = step_dir / "checkpoint.incomplete"
        staging.mkdir()
        model.save_pretrained(staging / "adapter", safe_serialization=True)
        torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "optimizer_updates": step,
                    "next_task_offset": provenance["next_task_offset"],
                    "source_hashes": provenance["source_hashes"]}, staging / "state.pt")
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        del model, optimizer, params, before
        gc.collect()
        torch.cuda.empty_cache()
        fresh = load_model(Path(start["model_path"]), staging / "adapter", False)
        with torch.inference_mode():
            actual = completion_logprobs(fresh, probe["prompt_ids"],
                                         probe["generated_ids"], "cuda").float().cpu()
        reload_error = float((actual - expected).abs().max())
        del fresh
        gc.collect()
        torch.cuda.empty_cache()
        if reload_error > RELOAD_TOL:
            raise RuntimeError(f"Independent adapter reload mismatch: {reload_error}")
        write_json(staging / "complete.json", {
            "optimizer_updates": step, "base_model_sha256": start["model_weights_sha256"],
            "adapter_sha256": sha256(staging / "adapter/adapter_model.safetensors"),
            "state_sha256": sha256(staging / "state.pt"),
            "next_task_offset": provenance["next_task_offset"],
            "parent_checkpoint": str(parent) if parent else None})
        checkpoint = step_dir / "checkpoint"
        staging.rename(checkpoint)
        checked_checkpoint(checkpoint, start["model_weights_sha256"], step)
    metrics = {"passed": True, "scope": ("single-group technical GRPO check" if
              args.technical_single_group else "two-group Stage C GRPO update"),
               "policy_version_before": version, "policy_version_after": step,
               "attempted_groups": summary["attempted_groups"],
               "eligible_groups": summary["eligible_groups"],
               "selected_groups": len(selected), "rewards": reward_details,
               "audited_tokens": len(errors), "preupdate_mean_abs_error": sum(errors) / len(errors),
               "preupdate_max_abs_error": max(errors), "loss": loss_total,
               "grad_norm": grad_norm, "parameter_delta_norm": delta,
               "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_reserved,
               "memory_limit_gib": args.max_peak_gib, "reload_max_abs_error": reload_error,
               "next_task_offset": provenance["next_task_offset"],
               "checkpoint": str(checkpoint), "wall_seconds": time.monotonic() - t0}
    write_json(step_dir / "metrics.json", metrics)
    plan["sources"].append(provenance)
    write_json(root / "plan.json", plan)
    write_json(root / "latest.json", {"optimizer_updates": step,
               "checkpoint": str(checkpoint),
               "complete_sha256": sha256(checkpoint / "complete.json"),
               "next_task_offset": provenance["next_task_offset"],
               "metrics_sha256": sha256(step_dir / "metrics.json")})
    all_tasks = [task for entry in plan["sources"] for task in entry["attempted_task_ids"]]
    write_json(root / "summary.json", {
        "completed": True, "optimizer_updates": step,
        "resume_verified": bool(args.resume and step >= 2),
        "unique_attempted_task_ids": len(set(all_tasks)),
        "attempted_task_ids": all_tasks,
        "repeated_task_ids_allowed_with_fresh_rollouts": True,
        "source_offsets_nonoverlap": all(
            earlier["next_task_offset"] <= later["task_offset"]
            for earlier, later in zip(plan["sources"], plan["sources"][1:])),
        "policy_versions_consumed": [entry["policy_version"] for entry in plan["sources"]],
        "latest_checkpoint": str(checkpoint),
        "technical_single_group": args.technical_single_group,
        "formal_grpo_result": not args.technical_single_group})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technical-single-group", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-peak-gib", type=float, default=22)
    args = parser.parse_args()
    if args.learning_rate <= 0 or not 10 <= args.max_peak_gib <= 23.5:
        parser.error("Learning rate must be positive; GPU limit must be 10..23.5 GiB")
    update(args)


if __name__ == "__main__":
    main()
