from __future__ import annotations

import logging
from typing import Callable, List, Optional

import pandas as pd
import torch


def prepare_tokenizer_for_generation(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


@torch.no_grad()
def batched_generate_texts(
    model,
    tokenizer,
    prompts: List[str],
    max_seq_length: int,
    max_new_tokens: int = 180,
    batch_size: int = 8,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prepare_tokenizer_for_generation(tokenizer)
    predictions: List[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        tokenizer.padding_side = "left"

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[:, prompt_len:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend([str(t).strip() for t in texts])

        if logger is not None:
            logger.info(
                "Generated batch %d-%d/%d",
                start,
                min(start + batch_size, len(prompts)),
                len(prompts),
            )

    return predictions


def generate_predictions_from_messages(
    eval_df: pd.DataFrame,
    model,
    tokenizer,
    make_inference_prompt_fn: Callable,
    parse_score_fn: Callable,
    parse_rationale_fn: Callable,
    max_seq_length: int,
    max_new_tokens: int = 180,
    batch_size: int = 8,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    prompts = [make_inference_prompt_fn(tokenizer, msgs) for msgs in eval_df["messages"]]
    predictions = batched_generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        device=device,
        logger=logger,
    )

    keep_cols = [c for c in ["id", "paper_id", "score", "rationale", "target_json"] if c in eval_df.columns]
    out = eval_df[keep_cols].copy().reset_index(drop=True)
    out["prediction_text"] = predictions
    out["pred_score"] = out["prediction_text"].apply(parse_score_fn)
    out["pred_rationale"] = out["prediction_text"].apply(parse_rationale_fn)
    return out

