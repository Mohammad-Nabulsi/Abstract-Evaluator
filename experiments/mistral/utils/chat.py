from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_SYSTEM_PROMPT = "You are a strict research abstract evaluator. You return only valid JSON."


def format_rubric(rubric: Dict[str, Any]) -> str:
    if isinstance(rubric, dict) and "score_scale" in rubric and "criteria" in rubric:
        score_scale = rubric.get("score_scale", {})
        criteria = rubric.get("criteria", {})

        score_lines = ["Score scale:"]
        for k, v in sorted(score_scale.items(), key=lambda x: int(str(x[0]))):
            score_lines.append(f"{k} = {v}")

        criteria_lines = ["Criteria:"]
        for k, v in sorted(criteria.items(), key=lambda x: str(x[0])):
            criteria_lines.append(f"{k}: {v}")

        return "\n".join(score_lines + [""] + criteria_lines)

    if isinstance(rubric, dict):
        return "\n".join(
            [f"{k} = {v}" for k, v in sorted(rubric.items(), key=lambda x: str(x[0]))]
        )

    return str(rubric)


def make_user_prompt(row: pd.Series) -> str:
    title_block = ""
    if "title" in row and pd.notna(row["title"]) and str(row["title"]).strip():
        title_block = f"\nTitle:\n{str(row['title']).strip()}\n"

    return (
        f"Task:\n{row['task']}\n\n"
        f"Reference:\n{row['reference']}\n\n"
        f"Rubric:\n{format_rubric(row['rubric'])}\n"
        f"{title_block}\n"
        f"Submission:\n{row['submission']}\n\n"
        "Return only valid JSON with exactly these keys: score, rationale.\n"
        "Do not include markdown, analysis, or extra text."
    )


def make_assistant_answer(row: pd.Series) -> str:
    return json.dumps(
        {"score": int(row["score"]), "rationale": str(row["rationale"]).strip()},
        ensure_ascii=False,
    )


def make_messages(
    row: pd.Series,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": make_user_prompt(row)},
        {"role": "assistant", "content": make_assistant_answer(row)},
    ]


def add_messages_and_targets(
    df: pd.DataFrame,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> pd.DataFrame:
    out = df.copy()
    out["messages"] = out.apply(lambda row: make_messages(row, system_prompt=system_prompt), axis=1)
    out["target_json"] = out.apply(make_assistant_answer, axis=1)
    return out


def normalize_messages_for_template(messages):
    # Handle HF/Arrow edge-case where a list-of-dicts becomes dict-of-lists.
    if isinstance(messages, dict):
        roles = messages.get("role", [])
        contents = messages.get("content", [])
        if isinstance(roles, list) and isinstance(contents, list):
            return [{"role": str(r), "content": str(c)} for r, c in zip(roles, contents)]

    if isinstance(messages, list):
        out = []
        for m in messages:
            if isinstance(m, dict):
                out.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
        return out

    # Fallback to a minimal valid user message.
    return [{"role": "user", "content": str(messages)}]


def apply_mistral_chat_template(tokenizer, messages, add_generation_prompt: bool = False):
    messages = normalize_messages_for_template(messages)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_formatting_func(tokenizer):
    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = apply_mistral_chat_template(
                tokenizer,
                messages,
                add_generation_prompt=False,
            )
            if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)
        return texts

    return formatting_prompts_func


def make_inference_prompt(tokenizer, messages):
    system_user_messages = [m for m in normalize_messages_for_template(messages) if m["role"] in ["system", "user"]]
    return apply_mistral_chat_template(
        tokenizer,
        system_user_messages,
        add_generation_prompt=True,
    )


def estimate_token_lengths(df_part: pd.DataFrame, tokenizer, messages_col: str = "messages") -> np.ndarray:
    lens = []
    for msgs in df_part[messages_col].tolist():
        txt = apply_mistral_chat_template(tokenizer, msgs, add_generation_prompt=False)
        lens.append(len(tokenizer.encode(txt, add_special_tokens=False)))
    return np.array(lens)

