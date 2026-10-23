from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_path: str
    model_revision: str
    output_dir: str
    task_file: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    iterations: int = 2
    tasks_per_iteration: int = 2
    group_size: int = 2
    max_prompt_tokens: int = 1024
    max_new_tokens: int = 24
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    lora_rank: int = 8
    lora_alpha: int = 16
    max_grad_norm: float = 1.0
    audit_mean_tolerance: float = 0.03
    audit_max_tolerance: float = 0.2
    max_wall_seconds: int = 1800
    max_generated_tokens: int = 2048
    min_free_gpu_gib: float = 10
    max_peak_gpu_gib: float = 22
    synthetic_prompt_tokens: int | None = None

    def validate(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("model_revision must be a resolved 40-character commit")
        for name in ("iterations", "tasks_per_iteration", "group_size", "max_prompt_tokens",
                     "max_new_tokens", "lora_rank", "lora_alpha", "max_wall_seconds",
                     "max_generated_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        if self.synthetic_prompt_tokens is not None and (
                type(self.synthetic_prompt_tokens) is not int
                or not 1 <= self.synthetic_prompt_tokens <= self.max_prompt_tokens):
            raise ValueError("synthetic_prompt_tokens must be a positive integer within prompt budget")
        if self.device != "cuda" or self.dtype != "bfloat16":
            raise ValueError("Real-model v0 supports CUDA + BF16; use cpu-smoke for CPU tests")
        for name in ("learning_rate", "clip_epsilon", "max_grad_norm", "audit_mean_tolerance",
                     "audit_max_tolerance", "min_free_gpu_gib", "max_peak_gpu_gib"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between zero and one")
        if self.max_new_tokens > 128:
            raise ValueError("v0 retains generation scores temporarily; max_new_tokens <= 128")
        for name in ("model_path", "task_file"):
            if not Path(getattr(self, name)).exists():
                raise ValueError(f"Missing {name}: {getattr(self, name)}")
        return self

    def to_dict(self):
        result = asdict(self)
        if self.synthetic_prompt_tokens is None:
            result.pop("synthetic_prompt_tokens")  # Keep existing unpadded manifests compatible.
        return result


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("model_path", "output_dir", "task_file"):
        raw[key] = str((path.parent / raw[key]).resolve())
    return Config(**raw).validate()
