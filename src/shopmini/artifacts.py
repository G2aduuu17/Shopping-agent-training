from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "psutil"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def manifest(config):
    model = Path(config.model_path)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    files = ["config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]
    files += sorted(set(index["weight_map"].values()))
    hashes = {}
    for name in files:
        print(f"Hashing {name}", flush=True)
        hashes[name] = sha256(model / name)
        metadata = model / ".cache" / "huggingface" / "download" / (name + ".metadata")
        if metadata.is_file():
            lines = metadata.read_text().splitlines()
            if not lines or lines[0] != config.model_revision:
                raise ValueError(f"Download revision mismatch: {name}")
            if name.endswith(".safetensors") and len(lines) > 1 and lines[1] != hashes[name]:
                raise ValueError(f"Downloaded weight checksum mismatch: {name}")
    return {"schema_version": 1, "experiment": "synthetic-single-decision-shopping",
            "upstream_reference": "https://github.com/YYHDBL/shopping-grpo-longhorizon",
            "upstream_code_used": False, "config": config.to_dict(),
            "model_file_hashes": hashes, "task_sha256": sha256(config.task_file),
            "versions": versions(), "reward_version": "mini-catalog-v1",
            "objective": "trajectory_mean; group_centered; beta=0; temperature=1; full_softmax"}
