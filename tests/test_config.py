import json
from pathlib import Path

import pytest

from shopmini.config import load_config


def test_relative_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / "tasks.jsonl").write_text("{}")
    path = config_dir / "smoke.json"
    raw = {"model_path": "../model", "task_file": "../tasks.jsonl", "output_dir": "../out",
           "model_revision": "a" * 40}
    path.write_text(json.dumps(raw))
    monkeypatch.chdir(Path(__file__).parent)
    cfg = load_config(path)
    assert cfg.model_path == str(tmp_path / "model")
    assert "synthetic_prompt_tokens" not in cfg.to_dict()
    raw["synthetic_prompt_tokens"] = 1024
    path.write_text(json.dumps(raw))
    assert load_config(path).to_dict()["synthetic_prompt_tokens"] == 1024
    raw["synthetic_prompt_tokens"] = 2048
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="synthetic_prompt_tokens"):
        load_config(path)
    raw.pop("synthetic_prompt_tokens")
    raw["group_size"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="group_size"):
        load_config(path)
