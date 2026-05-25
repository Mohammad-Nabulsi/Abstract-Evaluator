from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import Dataset, DatasetDict


DEFAULT_KEEP_COLS = ["id", "paper_id", "messages", "score", "rationale", "target_json"]


def export_messages_jsonl(df_part: pd.DataFrame, path: Path, messages_col: str = "messages") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df_part.iterrows():
            f.write(json.dumps({"messages": row[messages_col]}, ensure_ascii=False) + "\n")


def export_split_jsonl(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    messages_col: str = "messages",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"
    export_messages_jsonl(train_df, train_path, messages_col=messages_col)
    export_messages_jsonl(val_df, val_path, messages_col=messages_col)
    export_messages_jsonl(test_df, test_path, messages_col=messages_col)
    return {
        "train_jsonl": str(train_path),
        "validation_jsonl": str(val_path),
        "test_jsonl": str(test_path),
    }


def to_hf_dataset(
    df_part: pd.DataFrame,
    keep_cols: Optional[Iterable[str]] = None,
) -> Dataset:
    cols = list(keep_cols) if keep_cols is not None else list(DEFAULT_KEEP_COLS)
    cols = [c for c in cols if c in df_part.columns]
    return Dataset.from_pandas(df_part[cols], preserve_index=False)


def to_hf_dataset_dict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keep_cols: Optional[Iterable[str]] = None,
) -> DatasetDict:
    return DatasetDict(
        {
            "train": to_hf_dataset(train_df, keep_cols=keep_cols),
            "validation": to_hf_dataset(val_df, keep_cols=keep_cols),
            "test": to_hf_dataset(test_df, keep_cols=keep_cols),
        }
    )

