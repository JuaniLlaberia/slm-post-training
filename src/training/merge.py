import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_adapter(base_model: str, adapter_dir: str, out_dir: str) -> str:
    """
    Merges a saved LoRA adapter (e.g. a training checkpoint) into its base model and saves a standalone model,
    so it can be scored with evaluate() like any Hugging Face model.

    Args:
        base_model (str): Hugging Face model id or local path the adapter was trained on.
        adapter_dir (str): Folder with adapter_config.json and the adapter weights.
        out_dir (str): Folder for the merged model and its tokenizer.
    Returns:
        str: out_dir, to pass straight to evaluate().
    """
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)

    del model, merged
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir
