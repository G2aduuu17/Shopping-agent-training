from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from filelock import FileLock

from .artifacts import manifest, sha256, write_json
from .catalog import load_tasks, score
from .objective import advantages, clipped_loss


def evaluate(policy, tasks, seed):
    rows = []
    for index, task in enumerate(tasks):
        row = policy.sample(task, seed + index, greedy=True)
        result = score(task, row["text"], truncated=row["truncated"])
        rows.append({"task_id": task["task_id"], "text": row["text"], **result})
    return {"tasks": len(rows), "successes": sum(r["success"] for r in rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "mean_reward": sum(r["reward"] for r in rows) / len(rows), "rows": rows}


def run(cfg, resume=False, stop_after_iteration=None):
    from .policy import Policy

    if stop_after_iteration is not None and not 1 <= stop_after_iteration < cfg.iterations:
        raise ValueError("stop_after_iteration must be >= 1 and below total iterations")
    out = Path(cfg.output_dir)
    train = load_tasks(cfg.task_file, "train")
    dev = load_tasks(cfg.task_file, "dev")
    if cfg.tasks_per_iteration > len(train):
        raise ValueError("tasks_per_iteration exceeds distinct train tasks")
    # Refuse collisions. A lock is exclusive and never silently stolen after a crash.
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("Output already exists; use --resume or a fresh output_dir")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".run.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        # All runs of this project share a process-released GPU lease, even with different outputs.
        gpu_lock = Path(__file__).resolve().parents[2] / ".gpu.lock"
        with FileLock(str(gpu_lock), timeout=0):
            return _run(cfg, out, train, dev, resume, Policy, stop_after_iteration)
    finally:
        lock.unlink(missing_ok=True)


def _run(cfg, out, train, dev, resume, policy_class, stop_after_iteration=None):
    start_time = time.monotonic()
    torch.manual_seed(cfg.seed)
    provenance = manifest(cfg)
    checkpoint = None
    state = {"next_iteration": 0, "updates": 0, "generated_tokens": 0, "wall_seconds": 0.0}
    if resume:
        old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if old != provenance:
            raise ValueError("Resume requires identical config, data, model hashes and dependencies")
        pointer = json.loads((out / "latest.json").read_text())
        checkpoint = out / pointer["checkpoint"]
        if sha256(checkpoint / "adapter_model.safetensors") != pointer["adapter_sha256"]:
            raise ValueError("Checkpoint adapter checksum mismatch")
        state = json.loads((checkpoint / "state.json").read_text())
    else:
        write_json(out / "manifest.json", provenance)
    policy = policy_class(cfg, checkpoint)
    optimizer = torch.optim.AdamW(policy.params, lr=cfg.learning_rate, weight_decay=0.0)
    if checkpoint:
        saved = torch.load(checkpoint / "optimizer.pt", map_location="cuda", weights_only=True)
        optimizer.load_state_dict(saved)
    else:
        write_json(out / "trainable_parameters.json", {"names": policy.trainable_names,
                   "count": sum(p.numel() for p in policy.params)})
        print("Evaluating initial policy on dev", flush=True)
        write_json(out / "eval_before.json", evaluate(policy, dev, cfg.seed))
        _commit(out, "initial", policy, optimizer, state)
    elapsed_before = state["wall_seconds"]
    for iteration in range(state["next_iteration"], cfg.iterations):
        if elapsed_before + time.monotonic() - start_time >= cfg.max_wall_seconds:
            break
        maximum = cfg.tasks_per_iteration * cfg.group_size * cfg.max_new_tokens
        if state["generated_tokens"] + maximum > cfg.max_generated_tokens:
            break  # Do not begin a batch that cannot fit the token budget.
        groups = []
        batch_path = out / f"rollouts-{iteration:04d}.json"
        for task_index in range(cfg.tasks_per_iteration):
            task = train[(iteration * cfg.tasks_per_iteration + task_index) % len(train)]
            group = []
            for sample_index in range(cfg.group_size):
                if elapsed_before + time.monotonic() - start_time >= cfg.max_wall_seconds:
                    write_json(out / "interrupted_batch.json", groups + [group])
                    raise TimeoutError("Wall budget reached; incomplete batch excluded")
                seed = cfg.seed + iteration * 10000 + task_index * cfg.group_size + sample_index
                row = policy.sample(task, seed)
                row.update(task_id=task["task_id"], iteration=iteration, sample_index=sample_index)
                row["outcome"] = score(task, row["text"], truncated=row["truncated"])
                state["generated_tokens"] += len(row["generated_ids"])
                group.append(row)
                print(f"i={iteration} task={task['task_id']} sample={sample_index} "
                      f"reward={row['outcome']['reward']} text={row['text']!r}", flush=True)
            groups.append(group)
        write_json(batch_path, groups)
        policy.release_generation_memory()
        eligible = []
        for group in groups:
            if not all(r["outcome"]["reward_valid"] for r in group):
                continue
            adv = advantages([r["outcome"]["reward"] for r in group])
            if torch.all(adv == 0):
                continue
            eligible.extend(zip(group, adv.tolist(), strict=True))
        audit = []
        policy.model.train()
        # Audit EVERY eligible trajectory before any gradient or update.
        with torch.no_grad():
            for row, _ in eligible:
                new = policy.logprobs(row)
                old = torch.tensor(row["old_logprobs"], device=new.device)
                delta = (new - old).abs()
                stats = {"mean": float(delta.mean()), "max": float(delta.max())}
                if (not torch.isfinite(delta).all() or stats["mean"] > cfg.audit_mean_tolerance
                        or stats["max"] > cfg.audit_max_tolerance):
                    write_json(out / "audit_failed.json", {"iteration": iteration, **stats})
                    raise ValueError(f"Old/new probability audit failed: {stats}")
                audit.append(stats)
        before = [p.detach().clone() for p in policy.params] if eligible else []
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for row, adv in eligible:
            new = policy.logprobs(row)
            old = torch.tensor(row["old_logprobs"], device=new.device)
            loss = clipped_loss(new, old, adv, cfg.clip_epsilon) / len(eligible)
            loss.backward()
            total_loss += float(loss.detach())
            del loss, new, old
        delta_norm = grad_norm = 0.0
        if eligible:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                policy.params, cfg.max_grad_norm, error_if_nonfinite=True))
            optimizer.step()
            delta_norm = math.sqrt(sum(float((p.detach().float() - previous.float()).square().sum())
                                       for p, previous in zip(policy.params, before, strict=True)))
            if not math.isfinite(delta_norm) or delta_norm == 0:
                raise RuntimeError("Update did not produce finite, changed LoRA parameters")
            state["updates"] += 1
        del before
        optimizer.zero_grad(set_to_none=True)
        state["next_iteration"] = iteration + 1
        state["wall_seconds"] = elapsed_before + time.monotonic() - start_time
        metrics = {"iteration": iteration, "eligible_groups": len(eligible) // cfg.group_size,
                   "attempted_groups": len(groups), "loss": total_loss, "grad_norm": grad_norm,
                   "parameter_delta_norm": delta_norm, "probability_audit": audit,
                   "peak_allocated_gib": policy.peak_gib(), **state}
        write_json(out / f"metrics-{iteration:04d}.json", metrics)
        _commit(out, f"iteration-{iteration:04d}", policy, optimizer, state)
        print(json.dumps(metrics), flush=True)
        if stop_after_iteration == state["next_iteration"]:
            result = {**state, "status": "paused_after_commit"}
            write_json(out / "pause.json", result)
            return result
    print("Evaluating final policy on the same dev tasks", flush=True)
    write_json(out / "eval_after.json", evaluate(policy, dev, cfg.seed))
    if cfg.synthetic_prompt_tokens is not None and state["next_iteration"]:
        # Fixed inputs evaluated on the FINAL policy, not sampling-time old probabilities.
        last = json.loads((out / f"rollouts-{state['next_iteration']-1:04d}.json").read_text())
        policy.model.eval()
        probes = []
        with torch.no_grad():
            for row in last[0]:
                probes.append({"prompt_ids": row["prompt_ids"],
                               "generated_ids": row["generated_ids"],
                               "final_logprobs": policy.logprobs(row).cpu().tolist()})
        write_json(out / "final_probability_probe.json", probes)
    committed_peaks = [json.loads((out / f"metrics-{i:04d}.json").read_text())["peak_allocated_gib"]
                       for i in range(state["next_iteration"])]
    summary = {**state, "status": "updated" if state["updates"] else "no_reward_signal",
               "experiment": "synthetic-single-decision-shopping", "formal_result": False,
               "peak_allocated_gib": max([policy.peak_gib(), *committed_peaks])}
    write_json(out / "summary.json", summary)
    return summary


def _commit(out, name, policy, optimizer, state):
    final = out / name
    staging = out / (name + ".incomplete")
    if final.exists() or staging.exists():
        raise ValueError(f"Checkpoint collision: {name}; inspect incomplete/orphaned outputs")
    staging.mkdir()
    policy.save(staging)
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    write_json(staging / "state.json", state)
    digest = sha256(staging / "adapter_model.safetensors")
    staging.rename(final)
    write_json(out / "latest.json", {"checkpoint": name, "adapter_sha256": digest})
