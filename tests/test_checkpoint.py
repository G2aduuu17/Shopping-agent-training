import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from shopmini.artifacts import sha256
from shopmini.runner import _commit


class TinySavedPolicy:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([0.3]))

    def save(self, directory):
        save_file({"weight": self.weight.detach()}, directory / "adapter_model.safetensors")


def test_checkpoint_commit_pointer_and_collision(tmp_path):
    policy = TinySavedPolicy()
    optimizer = torch.optim.AdamW([policy.weight])
    policy.weight.square().backward()
    optimizer.step()
    _commit(tmp_path, "initial", policy, optimizer, {"next_iteration": 0})
    latest = json.loads((tmp_path / "latest.json").read_text())
    checkpoint = tmp_path / latest["checkpoint"]
    assert latest["adapter_sha256"] == sha256(checkpoint / "adapter_model.safetensors")
    assert torch.equal(load_file(checkpoint / "adapter_model.safetensors")["weight"], policy.weight)
    saved = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    restored = torch.optim.AdamW([policy.weight])
    restored.load_state_dict(saved)
    assert len(restored.state) == 1
    with pytest.raises(ValueError, match="collision"):
        _commit(tmp_path, "initial", policy, optimizer, {})


def test_failed_checkpoint_does_not_advance_latest(tmp_path):
    policy = TinySavedPolicy()
    optimizer = torch.optim.AdamW([policy.weight])
    _commit(tmp_path, "initial", policy, optimizer, {"next_iteration": 0})
    old_pointer = (tmp_path / "latest.json").read_bytes()

    class BrokenPolicy:
        def save(self, directory):
            raise OSError("simulated disk failure")

    with pytest.raises(OSError):
        _commit(tmp_path, "next", BrokenPolicy(), optimizer, {"next_iteration": 1})
    assert (tmp_path / "latest.json").read_bytes() == old_pointer
    assert (tmp_path / "next.incomplete").exists()
