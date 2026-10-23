import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from shopmini.objective import token_logprobs
from shopmini.policy import completion_logprobs


def test_suffix_logits_match_full_causal_forward_and_gradients():
    torch.manual_seed(4)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=31, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, attention_dropout=0.0,
    )).eval()
    prompt, generated = [1, 7, 3], [9, 4, 2]
    actual = completion_logprobs(model, prompt, generated, "cpu")
    full = model(torch.tensor([prompt + generated]), use_cache=False).logits
    expected = token_logprobs(full[0, len(prompt)-1:-1], torch.tensor(generated))
    assert torch.allclose(actual, expected, atol=1e-6)
    param = model.get_input_embeddings().weight
    grad_a = torch.autograd.grad(actual.sum(), param, retain_graph=True)[0]
    grad_b = torch.autograd.grad(expected.sum(), param)[0]
    assert torch.allclose(grad_a, grad_b, atol=1e-5)
