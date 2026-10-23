from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from .artifacts import versions, write_json
from .config import load_config


def doctor(output):
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "versions": versions(), "disk_free_gib": shutil.disk_usage(Path.cwd()).free / 2**30}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                                 "--format=csv"], capture_output=True, text=True, timeout=15,
                                check=False)
        report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = str(exc)
    try:
        import psutil
        report["ram_total_gib"] = psutil.virtual_memory().total / 2**30
        report["ram_available_gib"] = psutil.virtual_memory().available / 2**30
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if report["cuda_available"]:
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            x.square().mean().backward()
            report["cuda_backward_passed"] = bool(torch.isfinite(x.grad).all())
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
    except (ImportError, RuntimeError, OSError) as exc:
        report["runtime_error"] = str(exc)
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="ShopMini single-GPU GRPO laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    doc = sub.add_parser("doctor", help="Check installed dependencies and actual CUDA backward")
    doc.add_argument("--output", default="outputs/preflight.json")
    smoke = sub.add_parser("cpu-smoke", help="Tiny categorical-policy correctness test, no LLM")
    smoke.add_argument("--output", default="outputs/cpu-smoke")
    train = sub.add_parser("train", help="Real Qwen LoRA GRPO on synthetic shopping tasks")
    train.add_argument("--config", required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after-iteration", type=int,
                       help="Exit after committing this many iterations; total config stays unchanged")
    train.add_argument("--output-dir", help="Override output directory, resolved from current directory")
    ev = sub.add_parser("evaluate", help="Load a committed adapter independently and evaluate")
    ev.add_argument("--config", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--split", choices=["dev", "final"], default="dev")
    ev.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        result = doctor(args.output)
        if not result.get("cuda_backward_passed"):
            raise SystemExit(1)
    elif args.command == "cpu-smoke":
        from .smoke import cpu_smoke
        cpu_smoke(args.output)
    else:
        cfg = load_config(args.config)
        if args.command == "train" and args.output_dir:
            cfg = replace(cfg, output_dir=str(Path(args.output_dir).resolve()))
        from .catalog import load_tasks
        if args.command == "train" and args.dry_run:
            print(json.dumps({"config": cfg.to_dict(),
                  "train_tasks": len(load_tasks(cfg.task_file, "train")),
                  "dev_tasks": len(load_tasks(cfg.task_file, "dev")),
                  "maximum_rollout_tokens_per_iteration": cfg.tasks_per_iteration * cfg.group_size * cfg.max_new_tokens,
                  "experiment": "synthetic-single-decision-shopping; NOT ShopSimulator"}, indent=2))
            return
        if args.command == "train":
            from .runner import run
            print(json.dumps(run(cfg, args.resume, args.stop_after_iteration), indent=2))
        else:
            from filelock import FileLock

            from .policy import Policy
            from .runner import evaluate
            gpu_lock = Path(__file__).resolve().parents[2] / ".gpu.lock"
            with FileLock(str(gpu_lock), timeout=0):
                result = evaluate(Policy(cfg, Path(args.checkpoint).resolve(), trainable=False),
                                  load_tasks(cfg.task_file, args.split), cfg.seed)
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
