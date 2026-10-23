"""Checkpoint lineage and signal-selection checks for the Stage C updater."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stage_c_grpo_entry import task_window
from update_stage_c_grpo import checked_checkpoint

from shopmini.artifacts import sha256


def test_checkpoint_rejects_changed_optimizer_state(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "state.pt").write_bytes(b"state")
    (checkpoint / "complete.json").write_text(json.dumps({
        "optimizer_updates": 1, "base_model_sha256": "base",
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "state_sha256": sha256(checkpoint / "state.pt")}), encoding="utf-8")
    assert checked_checkpoint(checkpoint, "base", 1)["optimizer_updates"] == 1
    (checkpoint / "state.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="incomplete or changed"):
        checked_checkpoint(checkpoint, "base", 1)


def test_task_window_cycles_without_resetting_global_cursor():
    train = [11, 22, 33, 44]
    assert task_window(train, 3, 6) == [44, 11, 22, 33, 44, 11]
