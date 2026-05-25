from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_REFERENCE, DEFAULT_RUBRIC, DEFAULT_TASK, REQUIRED_COLUMNS


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_train_val_test_dfs(
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_dataframe(train_path), load_dataframe(val_path), load_dataframe(test_path)


def _stringify_nested_dict(d: Dict) -> Dict:
    out: Dict[str, object] = {}
    for k, v in d.items():
        key = str(k)
        if isinstance(v, dict):
            out[key] = {str(sk): str(sv) for sk, sv in v.items()}
        else:
            out[key] = str(v)
    return out


def normalize_rubric_value(x: object) -> Dict:
    if isinstance(x, dict):
        return _stringify_nested_dict(x)

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return copy.deepcopy(DEFAULT_RUBRIC)

    if isinstance(x, str):
        raw = x.strip()
        if not raw:
            return copy.deepcopy(DEFAULT_RUBRIC)
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return _stringify_nested_dict(parsed)
        except Exception:
            pass

    return copy.deepcopy(DEFAULT_RUBRIC)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_dataset(
    df: pd.DataFrame,
    min_submission_chars: int = 50,
    min_rationale_chars: int = 20,
) -> pd.DataFrame:
    validate_required_columns(df)
    out = df.copy()

    if "validation_status" in out.columns:
        valid_values = {"valid", "ok", "pass", "passed", "true", "1", "nan", "none"}
        status = out["validation_status"].astype(str).str.lower()
        out = out[status.isin(valid_values) | out["validation_status"].isna()].copy()

    out["paper_id"] = out["paper_id"].astype(str)
    out["submission"] = out["submission"].astype(str).str.strip()
    out["rationale"] = out["rationale"].astype(str).str.strip()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")

    out = out.dropna(subset=["paper_id", "submission", "score", "rationale"])
    out = out[out["submission"].str.len() >= min_submission_chars]
    out = out[out["rationale"].str.len() >= min_rationale_chars]
    out = out[out["score"].between(0, 4)]
    out["score"] = out["score"].astype(int)

    if "task" not in out.columns:
        out["task"] = DEFAULT_TASK
    else:
        out["task"] = out["task"].fillna(DEFAULT_TASK).astype(str)

    if "reference" not in out.columns:
        out["reference"] = DEFAULT_REFERENCE
    else:
        out["reference"] = out["reference"].fillna(DEFAULT_REFERENCE).astype(str)

    if "rubric" not in out.columns:
        out["rubric"] = None

    out["rubric"] = out["rubric"].apply(normalize_rubric_value)

    if "id" not in out.columns:
        out["id"] = [f"{pid}_row_{i:06d}" for i, pid in enumerate(out["paper_id"])]

    out = out.drop_duplicates(subset=["paper_id", "submission", "score", "rationale"])
    out = out.reset_index(drop=True)
    return out


def clean_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return clean_dataset(train_df), clean_dataset(val_df), clean_dataset(test_df)


def split_train_val_test_from_combined(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 3407,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    clean_df = clean_dataset(df)
    stratify = clean_df["score"] if clean_df["score"].nunique() > 1 else None
    train_df, remaining = train_test_split(
        clean_df,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )

    remaining_ratio = val_size + test_size
    val_ratio_within_remaining = val_size / remaining_ratio
    stratify_remaining = remaining["score"] if remaining["score"].nunique() > 1 else None
    val_df, test_df = train_test_split(
        remaining,
        train_size=val_ratio_within_remaining,
        random_state=seed,
        stratify=stratify_remaining,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def score_distribution(df: pd.DataFrame) -> Dict[int, float]:
    counts = df["score"].value_counts(normalize=True).sort_index()
    return {int(k): float(v) for k, v in counts.items()}


def sample_n_rows(df: pd.DataFrame, n: int, seed: int = 3407) -> pd.DataFrame:
    n = min(n, len(df))
    if n <= 0:
        return df.head(0).copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

