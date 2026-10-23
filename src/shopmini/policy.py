from __future__ import annotations

import gc

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForConditionalGeneration

from .catalog import messages
from .context import padded_prompt
from .objective import token_logprobs

TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}


def completion_logprobs(model, prompt, generated, device):
    if not prompt or not generated:
        raise ValueError("Empty prompt or generation")
    inputs = torch.tensor([prompt + generated], device=device)
    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                   use_cache=False, logits_to_keep=len(generated) + 1)
    return token_logprobs(output.logits[0, :-1], inputs[0, len(prompt):])


class Policy:
    def __init__(self, cfg, adapter=None, trainable=True):
        self.cfg = cfg
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install the CUDA PyTorch wheel")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This preset requires BF16 support")
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 < cfg.min_free_gpu_gib:
            raise RuntimeError("Insufficient free GPU memory; close other GPU workloads")
        torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.model_path, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa", device_map={"": "cuda:0"},
        )
        # Match full module names, not suffixes: vision modules must not receive adapters.
        targets = [name for name, module in base.named_modules()
                   if isinstance(module, torch.nn.Linear) and "visual" not in name
                   and "vision" not in name and name.rsplit(".", 1)[-1] in TARGETS]
        if not targets:
            raise RuntimeError("No compatible text LoRA targets found")
        if adapter:
            self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
        else:
            self.model = get_peft_model(base, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0, target_modules=targets, bias="none",
            ))
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        if trainable:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if trainable and (not self.params or any("lora_" not in n for n in self.trainable_names)):
            raise RuntimeError("Only LoRA parameters may be trainable")
        self.eos_ids = self.model.generation_config.eos_token_id
        self.eos_ids = [self.eos_ids] if isinstance(self.eos_ids, int) else self.eos_ids
        if not self.eos_ids:
            self.eos_ids = [self.tokenizer.eos_token_id]

    def encode(self, task):
        if self.cfg.synthetic_prompt_tokens is not None:
            return padded_prompt(self.tokenizer, task, self.cfg.synthetic_prompt_tokens)
        ids = self.tokenizer.apply_chat_template(
            messages(task), tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=False,
        )
        if len(ids) > self.cfg.max_prompt_tokens:
            raise ValueError(f"Task {task['task_id']} exceeds prompt budget; no silent truncation")
        return ids

    @torch.no_grad()
    def sample(self, task, seed, greedy=False, prompt_ids=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.eval()
        prompt = self.encode(task) if prompt_ids is None else list(prompt_ids)
        if not prompt or len(prompt) > self.cfg.max_prompt_tokens:
            raise ValueError("Empty or over-budget prompt")
        inputs = torch.tensor([prompt], device="cuda")
        # Fresh GenerationConfig prevents inherited top-k, penalties or suppression.
        gen = GenerationConfig(
            max_new_tokens=self.cfg.max_new_tokens, do_sample=not greedy,
            temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            eos_token_id=self.eos_ids, pad_token_id=self.tokenizer.pad_token_id or self.eos_ids[0],
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True,
            return_dict_in_generate=True, output_scores=not greedy,
        )
        output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     generation_config=gen, logits_to_keep=1)
        generated = output.sequences[0, len(prompt):]
        old = []
        if not greedy:
            for token, scores in zip(generated, output.scores, strict=True):
                old.append(float(token_logprobs(scores, token.reshape(1))[0].cpu()))
        ids = generated.cpu().tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        result = {"prompt_ids": prompt, "generated_ids": ids, "old_logprobs": old,
                  "text": text, "seed": seed, "truncated": not ids or ids[-1] not in self.eos_ids}
        del output, inputs, generated
        return result

    def logprobs(self, row):
        prompt, generated = row["prompt_ids"], row["generated_ids"]
        # Suffix includes the predecessor of the first generated token and a final unused logit.
        return completion_logprobs(self.model, prompt, generated, "cuda")

    def peak_gib(self):
        peak = torch.cuda.max_memory_allocated() / 2**30
        if peak > self.cfg.max_peak_gpu_gib:
            raise RuntimeError(f"Peak allocation {peak:.2f} GiB exceeds configured budget")
        return peak

    def save(self, directory):
        self.model.save_pretrained(str(directory), safe_serialization=True)

    def release_generation_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
