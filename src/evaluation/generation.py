import gc
from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.inference_mode()
def generate(
    model_name: str,
    prompt_builder: Callable,
    prompts: list[str],
    stop: list[str] | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 512,
) -> list[dict]:
    """
    Generates a greedy completion for every prompt with one model.

    Args:
        model_name (str): Hugging Face model id or local path.
        prompt_builder (Callable): Turns (prompt, tokenizer) into the full prompt string.
        prompts (list[str]): Raw prompts (e.g. GSM8K questions).
        stop (list[str] | None): Stop strings; output is also cut at the first one.
        batch_size (int): Prompts per generate call.
        max_new_tokens (int): Generation limit per prompt.
    Returns:
        list[dict]: One dict per prompt: {"output": str, "n_tokens": int, "hit_limit": bool}.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    results = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(
                [prompt_builder(p, tokenizer) for p in batch],
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0,
                stop_strings=stop,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
            )

            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            n_tokens = (new_tokens != tokenizer.pad_token_id).sum(dim=1).tolist()
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for text, n in zip(texts, n_tokens):
                for s in stop or []:
                    text = text.split(s)[0]
                results.append({"output": text, "n_tokens": n, "hit_limit": n >= max_new_tokens})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results
