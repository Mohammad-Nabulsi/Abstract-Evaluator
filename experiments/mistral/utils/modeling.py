from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _get_fast_language_model():
    from unsloth import FastLanguageModel

    return FastLanguageModel


def infer_num_layers(model) -> int:
    cfg = model.config
    for attr in ["num_hidden_layers", "n_layers", "num_layers"]:
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError("Could not infer transformer layer count.")


def freeze_lower_lora_layers(model, freeze_ratio: float = 0.0):
    if freeze_ratio <= 0:
        return {"num_layers": infer_num_layers(model), "cutoff": 0}

    num_layers = infer_num_layers(model)
    cutoff = int(num_layers * freeze_ratio)

    layer_pat = re.compile(r"\.layers\.(\d+)\.")
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        if "lora_" not in name.lower():
            param.requires_grad = False
            frozen += param.numel()
            continue

        match = layer_pat.search(name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx < cutoff:
                param.requires_grad = False
                frozen += param.numel()
            else:
                param.requires_grad = True
                trainable += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()

    return {
        "num_layers": num_layers,
        "cutoff": cutoff,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }


def print_trainable_parameters(model) -> Dict[str, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    print(f"Trainable: {trainable:,} / Total: {total:,} = {pct:.4f}%")
    return {"trainable_params": float(trainable), "total_params": float(total), "trainable_pct": float(pct)}


def load_mistral_base_model(
    model_name: str,
    max_seq_length: int,
    dtype=torch.bfloat16,
    load_in_4bit: bool = False,
):
    FastLanguageModel = _get_fast_language_model()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def apply_lora_to_model(
    model,
    lora_cfg: Dict[str, Any],
    seed: int,
):
    FastLanguageModel = _get_fast_language_model()
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    freeze_info = freeze_lower_lora_layers(model, float(lora_cfg.get("freeze_ratio", 0.0)))
    trainable_info = print_trainable_parameters(model)
    freeze_info.update(trainable_info)
    return model, freeze_info


def load_mistral_model_for_training(
    model_name: str,
    max_seq_length: int,
    lora_cfg: Dict[str, Any],
    seed: int,
    dtype=torch.bfloat16,
    load_in_4bit: bool = False,
) -> Tuple[Any, Any, Dict[str, Any]]:
    model, tokenizer = load_mistral_base_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model, freeze_info = apply_lora_to_model(model, lora_cfg=lora_cfg, seed=seed)
    return model, tokenizer, freeze_info


def load_mistral_model_for_inference(
    model_name: str,
    adapter_dir: Path | None,
    max_seq_length: int,
    dtype=torch.bfloat16,
    load_in_4bit: bool = False,
):
    FastLanguageModel = _get_fast_language_model()
    model, tokenizer = load_mistral_base_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
