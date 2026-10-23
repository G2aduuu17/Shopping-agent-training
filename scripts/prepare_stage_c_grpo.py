"""Freeze the completed Stage C model and disjoint GRPO task split.

This is a read-only prerequisite check except for its new output manifest. It
does not collect rollouts or update model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MINI = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_ids(path: Path) -> list[int]:
    return [int(json.loads(line)["task_id"]) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-run", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=MINI / "data/shopsim_splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model, sft, split, output = (path.resolve() for path in (
        args.model, args.sft_run, args.split_dir, args.output_dir))
    merge = read_json(model / "merge_complete.json")
    sft_summary = read_json(sft / "summary.json")
    sft_manifest = read_json(sft / "manifest.json")
    split_manifest = read_json(split / "manifest.json")
    require(merge.get("completed") is True and sft_summary.get("completed") is True,
            "Stage C merge or SFT run is incomplete")
    require(Path(merge["merged_model"]).resolve() == model,
            "Merge report points at another model")
    require(Path(merge["stage_c_adapter"]).resolve() == (sft / "adapter"),
            "Stage C SFT adapter lineage differs from merge report")
    require(sft_summary["adapter_sha256"] == merge["stage_c_adapter_sha256"],
            "Stage C SFT adapter hash differs from merge report")
    require(sha256(sft / "adapter/adapter_model.safetensors") ==
            merge["stage_c_adapter_sha256"], "Stage C adapter bytes changed")
    require(sha256(model / "model.safetensors") == merge["merged_weights_sha256"],
            "Merged model bytes changed")
    require(sha256(model / "tokenizer.json") == merge["tokenizer_sha256"],
            "Merged tokenizer bytes changed")

    train_file = split / "grpo_train.jsonl"
    from shopmini.upstream import require_upstream

    upstream = require_upstream()
    validation_file = upstream / "data/grpo/validation.jsonl"
    final_file = upstream / "data/evaluation/tasks.jsonl"
    train, validation, final = map(task_ids,
                                    (train_file, validation_file, final_file))
    frozen = split_manifest["grpo"]
    require(sha256(train_file) == frozen["filtered_train_sha256"],
            "Frozen GRPO train file hash changed")
    require(sha256(validation_file) ==
            split_manifest["sources"]["grpo_validation"]["sha256"],
            "GRPO validation file hash changed")
    require(sha256(final_file) == split_manifest["sources"]["final_200"]["sha256"],
            "Final-200 task list hash changed")
    require(len(train) == frozen["filtered_train_count"] == 994 and
            len(validation) == frozen["validation_count"] == 50 and
            len(final) == 200, "Task counts differ from frozen split")
    require(validation == frozen["validation_task_ids"],
            "GRPO validation order differs from frozen split")
    require(all(len(ids) == len(set(ids)) for ids in (train, validation, final)),
            "A task list contains duplicate IDs")
    sft_train = {int(value) for value in split_manifest["sft"]["train_task_ids"]}
    sft_validation = {int(value) for value in
                      split_manifest["sft"]["validation_task_ids"]}
    require(set(sft_manifest["train_task_ids"]) == sft_train and
            set(sft_manifest["validation_task_ids"]) == sft_validation,
            "Completed Stage C task lists differ from frozen SFT split")
    require(sft_train.isdisjoint(sft_validation), "SFT train and validation overlap")
    pools = {"sft_train": sft_train, "sft_validation": sft_validation,
             "grpo_train": set(train), "grpo_validation": set(validation),
             "final_200": set(final)}
    require(all(first.isdisjoint(second) for index, first in
                enumerate(pools.values()) for second in
                list(pools.values())[index + 1:]), "Task split overlap detected")

    manifest = {
        "kind": "stage-c-grpo-start-freeze-v1",
        "passed": True,
        "training_started": False,
        "model_path": str(model),
        "model_weights_sha256": merge["merged_weights_sha256"],
        "tokenizer_sha256": merge["tokenizer_sha256"],
        "merge_report_sha256": sha256(model / "merge_complete.json"),
        "stage_c_sft_manifest_sha256": sha256(sft / "manifest.json"),
        "stage_c_sft_summary_sha256": sha256(sft / "summary.json"),
        "split_manifest_sha256": sha256(split / "manifest.json"),
        "grpo_train_path": str(train_file),
        "grpo_train_sha256": sha256(train_file),
        "grpo_train_tasks": len(train),
        "grpo_validation_path": str(validation_file),
        "grpo_validation_sha256": sha256(validation_file),
        "grpo_validation_tasks": len(validation),
        "grpo_validation_task_ids": validation,
        "final_200_path": str(final_file),
        "final_200_sha256": sha256(final_file),
        "final_200_tasks": len(final),
        "task_splits_disjoint": True,
    }
    if output.exists():
        existing = output / "manifest.json"
        require(existing.is_file() and read_json(existing) == manifest,
                "Output exists with a different or incomplete manifest")
    else:
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    print(json.dumps({"passed": True, "training_started": False,
                      "output": str(output), "model_weights_sha256":
                      manifest["model_weights_sha256"], "grpo_train_tasks": len(train),
                      "grpo_validation_tasks": len(validation),
                      "final_200_tasks": len(final),
                      "task_splits_disjoint": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
