import runpy
from pathlib import Path

import pytest


class TokenizerFixture:
    def apply_chat_template(self, chat, **kwargs):
        assert any(m["role"] == "user" for m in chat)
        return "SYSTEM:" + chat[0]["content"] + "\nUSER:" + chat[1]["content"] + "\nASSISTANT:"

    def encode(self, text, **kwargs):
        return list(text.encode())


def test_exact_padding_preserves_user_task_and_generation_suffix():
    path = Path(__file__).parents[1] / "scripts/profile_lengths.py"
    pad = runpy.run_path(str(path))["padded_prompt"]
    task = {"requirements": {"category": "mouse"}, "products": [{"id": "A"}]}
    ids = pad(TokenizerFixture(), task, 1024)
    assert len(ids) == 1024
    decoded = bytes(ids).decode()
    assert decoded.endswith('"products": [{"id": "A"}]}\nASSISTANT:')
    assert "USER: Reference catalog note." in decoded
    with pytest.raises(ValueError, match="requested length"):
        pad(TokenizerFixture(), task, 1)
