from __future__ import annotations

from typing import Any, Dict, List, Tuple

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


def build_label_maps(num_labels: int) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {str(i): i for i in range(num_labels)}
    return id2label, label2id


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        msg = str(e).lower()
        if "tiktoken" in msg or "protobuf" in msg or "sentencepiece" in msg:
            raise RuntimeError(
                "Tokenizer initialization failed. Install missing tokenizer deps with:\n"
                "pip install protobuf tiktoken sentencepiece\n"
                "Then restart the Jupyter kernel and run again."
            ) from e
        raise


def tokenize_batch(batch, tokenizer, max_length: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
    )


def freeze_lower_encoder_layers(model, freeze_lower_n: int) -> None:
    if freeze_lower_n <= 0:
        return
    deberta = getattr(model, "deberta", None)
    if deberta is None:
        return
    encoder = getattr(deberta, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        return

    for i, layer in enumerate(layers):
        if i < freeze_lower_n:
            for param in layer.parameters():
                param.requires_grad = False


def load_deberta_score_model(
    model_name: str,
    num_labels: int = 5,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: List[str] | None = None,
    lora_bias: str = "none",
    freeze_lower_n_when_no_lora: int = 8,
    prefer_safetensors: bool = True,
):
    id2label, label2id = build_label_maps(num_labels)
    common_kwargs = dict(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    if prefer_safetensors:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                use_safetensors=True,
                **common_kwargs,
            )
        except Exception as e:
            # Fallback only when the repository truly lacks safetensors files.
            msg = str(e).lower()
            missing_safe = "safetensors" in msg and ("no file named" in msg or "not found" in msg)
            if not missing_safe:
                raise
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                **common_kwargs,
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **common_kwargs,
        )

    model.config.id2label = id2label
    model.config.label2id = label2id

    if use_lora:
        target_modules = lora_target_modules or ["query_proj", "value_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=lora_bias,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        freeze_lower_encoder_layers(model, freeze_lower_n_when_no_lora)

    return model


def build_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)
