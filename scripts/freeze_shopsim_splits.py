"""Freeze the local SFT/GRPO task split without changing upstream data."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from shopmini.upstream import MINI_ROOT, UPSTREAM_ROOT, require_upstream

ROOT = MINI_ROOT
UPSTREAM = UPSTREAM_ROOT
OUTPUT = ROOT / "data" / "shopsim_splits"
SOURCES = {
    "sft_curriculum": "upstream/data/sft_curriculum/manifest.json",
    "sft_trajectories": "upstream/data/sft_pure_v4/all.jsonl",
    "grpo_train": "upstream/data/grpo/train.jsonl",
    "grpo_validation": "upstream/data/grpo/validation.jsonl",
    "final_200": "upstream/data/evaluation/tasks.jsonl",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_path(name: str):
    return UPSTREAM / SOURCES[name].removeprefix("upstream/")


def canonical_lf_sha256(data: bytes) -> str:
    """Match Git's LF blob hash convention on Windows CRLF checkouts."""
    return sha256(data.replace(b"\r\n", b"\n"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ids(rows: list[dict], name: str) -> set[int]:
    result = [row["task_id"] for row in rows]
    if any(type(task_id) is not int for task_id in result) or len(result) != len(set(result)):
        raise ValueError(f"{name}: task IDs must be unique integers")
    return set(result)


def disjoint(left: set[int], right: set[int], name: str) -> None:
    overlap = left & right
    if overlap:
        raise ValueError(f"{name}: overlapping task IDs: {sorted(overlap)}")


def build() -> dict[str, bytes]:
    require_upstream()
    curriculum = json.loads((UPSTREAM / "data/sft_curriculum/manifest.json").read_text(encoding="utf-8"))
    for name, pin in (("sft_trajectories", curriculum["source"]),
                      ("final_200", curriculum["evaluation"])):
        actual = canonical_lf_sha256(source_path(name).read_bytes())
        if actual != pin["sha256"]:
            raise ValueError(f"{name}: source hash differs from upstream curriculum pin")
    sft_rows = read_jsonl(source_path("sft_trajectories"))
    grpo_rows = read_jsonl(source_path("grpo_train"))
    grpo_val_rows = read_jsonl(source_path("grpo_validation"))
    final_rows = read_jsonl(source_path("final_200"))

    sft_train: set[int] = set()
    sft_val: set[int] = set()
    for bucket_name, bucket in curriculum["buckets"].items():
        train = bucket["train_task_ids"]
        val = bucket["validation_task_ids"]
        if len(train) != len(set(train)) or len(val) != len(set(val)):
            raise ValueError(f"{bucket_name}: duplicate curriculum IDs")
        if len(train) + len(val) != bucket["rows"]:
            raise ValueError(f"{bucket_name}: row count mismatch")
        if sft_train & set(train) or sft_val & set(val):
            raise ValueError(f"{bucket_name}: duplicated IDs across buckets")
        sft_train.update(train)
        sft_val.update(val)

    sft_source = ids(sft_rows, "SFT source")
    grpo_source = ids(grpo_rows, "GRPO train")
    grpo_val = ids(grpo_val_rows, "GRPO validation")
    final = ids(final_rows, "Final-200")
    disjoint(sft_train, sft_val, "SFT train/validation")
    if sft_source != sft_train | sft_val:
        raise ValueError("SFT curriculum does not exactly cover the trajectory source")
    if (len(sft_source), len(sft_train), len(sft_val)) != (
        curriculum["counts"]["rows"], curriculum["counts"]["train"],
        curriculum["counts"]["validation"],
    ):
        raise ValueError("SFT curriculum counts mismatch")
    disjoint(grpo_source, grpo_val, "GRPO train/validation")
    disjoint(sft_source, grpo_val, "SFT/GRPO validation")
    disjoint(sft_source, final, "SFT/Final-200")
    disjoint(grpo_source, final, "GRPO train/Final-200")
    disjoint(grpo_val, final, "GRPO validation/Final-200")

    excluded_train = sorted(sft_train & grpo_source)
    excluded_val = sorted(sft_val & grpo_source)
    excluded = set(excluded_train + excluded_val)
    filtered_rows = [row for row in grpo_rows if row["task_id"] not in excluded]
    filtered_ids = ids(filtered_rows, "filtered GRPO train")
    disjoint(filtered_ids, sft_source, "filtered GRPO/SFT")
    if len(filtered_ids) != len(grpo_source) - len(excluded):
        raise ValueError("filtered GRPO count mismatch")

    filtered_bytes = ("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                              + "\n" for row in filtered_rows)).encode("utf-8")
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={UPSTREAM.as_posix()}", "rev-parse", "HEAD"],
        cwd=UPSTREAM, text=True,
    ).strip()
    manifest = {
        "schema_version": "shopmini-shopsim-splits-v1",
        "upstream_commit": commit,
        "sources": {name: {"path": path, "sha256": sha256(source_path(name).read_bytes())}
                    for name, path in SOURCES.items()},
        "sft": {
            "train_task_ids": sorted(sft_train),
            "validation_task_ids": sorted(sft_val),
            "train_count": len(sft_train),
            "validation_count": len(sft_val),
        },
        "grpo": {
            "source_train_count": len(grpo_source),
            "excluded_sft_train_task_ids": excluded_train,
            "excluded_sft_validation_task_ids": excluded_val,
            "filtered_train_path": "data/shopsim_splits/grpo_train.jsonl",
            "filtered_train_sha256": sha256(filtered_bytes),
            "filtered_train_count": len(filtered_ids),
            "validation_task_ids": sorted(grpo_val),
            "validation_count": len(grpo_val),
        },
        "final_200": {"task_count": len(final), "use": "final_evaluation_only"},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return {"manifest.json": manifest_bytes, "grpo_train.jsonl": filtered_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write", action="store_true", help="create or refresh frozen files")
    actions.add_argument("--check", action="store_true", help="verify sources and frozen files")
    args = parser.parse_args()
    expected = build()
    if args.write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, content in expected.items():
            (OUTPUT / name).write_bytes(content)
    else:
        for name, content in expected.items():
            path = OUTPUT / name
            if not path.is_file() or path.read_bytes() != content:
                raise SystemExit(f"FAIL: {path} missing or differs from source data")
    manifest = json.loads(expected["manifest.json"])
    print(json.dumps({
        "passed": True,
        "mode": "write" if args.write else "check",
        "output": str(OUTPUT),
        "sft_train": manifest["sft"]["train_count"],
        "sft_validation": manifest["sft"]["validation_count"],
        "grpo_train": manifest["grpo"]["filtered_train_count"],
        "grpo_validation": manifest["grpo"]["validation_count"],
        "excluded_grpo_task_ids": sorted(
            manifest["grpo"]["excluded_sft_train_task_ids"]
            + manifest["grpo"]["excluded_sft_validation_task_ids"]
        ),
        "final_200": manifest["final_200"]["task_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
