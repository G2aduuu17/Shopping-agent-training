def padded_prompt(tokenizer, task, length):
    from shopmini.catalog import messages

    chat = messages(task)
    rendered = tokenizer.apply_chat_template(chat, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    # Render a complete conversation: Qwen rejects system-only template calls.
    # Insert synthetic text inside the user message, preserving task and assistant suffix.
    boundary = rendered.index(chat[1]["content"])
    prefix = tokenizer.encode(rendered[:boundary], add_special_tokens=False)
    suffix = tokenizer.encode(rendered[boundary:], add_special_tokens=False)
    if length < len(prefix) + len(suffix):
        raise ValueError("Cannot pad this template to requested length")
    filler = tokenizer.encode(" Reference catalog note.", add_special_tokens=False)
    if not filler:
        raise ValueError("Empty filler encoding")
    count = length - len(prefix) - len(suffix)
    return prefix + (filler * (count // len(filler) + 1))[:count] + suffix
