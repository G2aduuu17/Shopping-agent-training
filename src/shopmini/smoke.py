"""Offline CPU test of real sampling/autograd; a tiny policy, NOT a language model result."""
import json
from pathlib import Path

import torch

from .artifacts import write_json
from .objective import advantages, clipped_loss


def cpu_smoke(output):
    torch.manual_seed(42)
    # An actual learned categorical policy over three catalog choices.
    logits = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW([logits], lr=0.1, weight_decay=0)
    initial = logits.softmax(-1).detach().clone()
    trace = []
    updates = 0
    for iteration in range(20):
        with torch.no_grad():
            actions = torch.multinomial(logits.softmax(-1), 8, replacement=True)
            old = logits.log_softmax(-1)[actions].clone()
            rewards = (actions == 0).float()  # Test fixture's correct product is action 0.
            adv = advantages(rewards)
        new = logits.log_softmax(-1)[actions]
        assert torch.allclose(new.detach(), old)
        loss = torch.stack([clipped_loss(n[None], o[None], a)
                            for n, o, a in zip(new, old, adv, strict=True)]).mean()
        optimizer.zero_grad()
        if torch.any(adv != 0):
            loss.backward()
            optimizer.step()
            updates += 1
        trace.append({"iteration": iteration, "actions": actions.tolist(),
                      "rewards": rewards.tolist(), "loss": float(loss.detach())})
    final = logits.softmax(-1).detach()
    if not (updates > 0 and final[0] > initial[0]):
        raise RuntimeError("CPU GRPO directionality smoke failed")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits.detach(), "optimizer": optimizer.state_dict()}, out / "tiny.pt")
    restored = torch.load(out / "tiny.pt", weights_only=True)
    assert torch.equal(restored["logits"], logits.detach())
    result = {"experiment": "cpu-categorical-policy-math-smoke", "real_llm": False,
              "updates": updates, "initial_probabilities": initial.tolist(),
              "final_probabilities": final.tolist(), "checkpoint_reload": True, "trace": trace}
    write_json(out / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
    return result
