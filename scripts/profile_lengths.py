"""Profile real prompt lengths in isolated GPU processes; not an accuracy benchmark."""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from shopmini.artifacts import versions, write_json
from shopmini.config import load_config
from shopmini.context import padded_prompt

ROOT = Path(__file__).resolve().parents[1]


def worker(config_path, length, out):
    import math

    import psutil
    import torch
    from filelock import FileLock

    from shopmini.catalog import load_tasks, score
    from shopmini.objective import advantages, clipped_loss
    from shopmini.policy import Policy

    cfg = replace(load_config(config_path), max_prompt_tokens=length)
    with FileLock(str(ROOT / ".gpu.lock"), timeout=0):
        torch.manual_seed(cfg.seed)
        policy = Policy(cfg)
        task = load_tasks(cfg.task_file, "train")[2]
        prompt = padded_prompt(policy.tokenizer, task, length)
        stages = {}

        def begin():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return time.perf_counter()

        def end(name, start):
            torch.cuda.synchronize()
            stages[name] = {
                "seconds": time.perf_counter() - start,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "process_rss_gib_at_end": psutil.Process().memory_info().rss / 2**30,
            }
            policy.peak_gib()

        start = begin()
        rows = [policy.sample(task, cfg.seed + i, prompt_ids=prompt) for i in range(cfg.group_size)]
        end("generation", start)
        for row in rows:
            row["outcome"] = score(task, row["text"], truncated=row["truncated"])
        write_json(out / "rollouts.json", rows)
        policy.release_generation_memory()
        policy.model.train()
        differences = []
        start = begin()
        with torch.no_grad():
            for row in rows:
                lp = policy.logprobs(row)
                differences.append((lp - torch.tensor(row["old_logprobs"], device="cuda")).abs().cpu())
        end("probability_recompute", start)
        delta = torch.cat(differences)
        audit = {"mean": float(delta.mean()), "p99": float(delta.quantile(0.99)),
                 "max": float(delta.max())}
        audit_ok = (bool(torch.isfinite(delta).all()) and audit["mean"] <= cfg.audit_mean_tolerance
                    and audit["max"] <= cfg.audit_max_tolerance)
        if not audit_ok:
            write_json(out / "result.json", {"status": "audit_failed", "prompt_tokens": length,
                                             "stages": stages, "audit": audit})
            return False
        adv = advantages([row["outcome"]["reward"] for row in rows])
        optimizer = torch.optim.AdamW(policy.params, lr=cfg.learning_rate, weight_decay=0)
        before = [p.detach().clone() for p in policy.params]
        start = begin()
        optimizer.zero_grad(set_to_none=True)
        # Even zero-advantage groups exercise the full backward graph for memory profiling.
        # They never count as an actual RL update.
        for row, advantage in zip(rows, adv, strict=True):
            lp = policy.logprobs(row)
            loss = clipped_loss(lp, torch.tensor(row["old_logprobs"], device="cuda"),
                                advantage, cfg.clip_epsilon) / len(rows)
            loss.backward()
            del loss, lp
        norm = float(torch.nn.utils.clip_grad_norm_(policy.params, cfg.max_grad_norm,
                                                  error_if_nonfinite=True))
        updated = bool(torch.any(adv != 0))
        if updated:
            optimizer.step()
        end("backward_and_step", start)
        change = math.sqrt(sum(float((p.detach().float() - old.float()).square().sum())
                               for p, old in zip(policy.params, before, strict=True)))
        if not math.isfinite(change) or (updated and change == 0):
            raise RuntimeError("Non-finite or ineffective parameter update")
        generated = sum(len(row["generated_ids"]) for row in rows)
        result = {"status": "passed", "purpose": "synthetic-context-capacity-probe",
                  "formal_result": False, "prompt_tokens": len(prompt),
                  "generated_tokens": generated, "group_size": len(rows),
                  "generation_tokens_per_second": generated / stages["generation"]["seconds"],
                  "audit": audit, "grad_norm": norm, "optimizer_updated": updated,
                  "parameter_delta_norm": change, "stages": stages, "versions": versions()}
        write_json(out / "result.json", result)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_smoke.json"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(n < 1024 or n > 8192 for n in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        parser.error("Use distinct lengths between 1024 and 8192")
    if args.worker:
        out = Path(args.output)
        try:
            ok = worker(args.config, args.lengths[0], out)
        except Exception as exc:
            write_json(out / "result.json", {"status": "failed", "prompt_tokens": args.lengths[0],
                                             "error_type": type(exc).__name__, "error": str(exc)})
            raise
        raise SystemExit(0 if ok else 1)
    out = ROOT / "outputs" / ("capacity-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                             + "-" + uuid4().hex[:6])
    out.mkdir(parents=True)
    cfg = load_config(args.config)
    write_json(out / "config.json", cfg.to_dict())
    results = []
    print(f"Results: {out}", flush=True)
    for length in args.lengths:
        case = out / str(length)
        case.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config",
                   str(Path(args.config).resolve()), "--lengths", str(length), "--output", str(case)]
        print(f"Testing {length} actual prompt tokens", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        with (case / "console.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        result_path = case / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            "status": "process_failed", "prompt_tokens": length, "exit_code": completed.returncode}
        results.append(result)
        write_json(out / "summary.json", results)
        print(json.dumps(result), flush=True)
        if completed.returncode:
            raise SystemExit(f"Stopped at {length}; inspect {case}")


if __name__ == "__main__":
    main()
