from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .modeling import tokenize_batch


def tokenize_dataset_dict(ds, tokenizer, max_length: int):
    return ds.map(
        lambda batch: tokenize_batch(batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
    )


def evaluate_score_predictions(pred_df: pd.DataFrame) -> Dict[str, float]:
    if "score" not in pred_df.columns:
        return {}
    if pred_df["score"].isna().all():
        return {}

    y_true = pred_df["score"].astype(int).values
    y_pred = pred_df["pred_score"].astype(int).values
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "within_1_accuracy": float((np.abs(y_true - y_pred) <= 1).mean()),
        "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict_split(
    trainer,
    split_ds,
    source_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pred = trainer.predict(split_ds)
    logits = pred.predictions
    pred_scores = np.argmax(logits, axis=-1).astype(int)

    keep_cols = [c for c in ["id", "paper_id", "score", "text"] if c in source_df.columns]
    out = source_df[keep_cols].copy().reset_index(drop=True)
    out["pred_score"] = pred_scores
    metrics = evaluate_score_predictions(out)
    return out, metrics


def save_predictions_and_metrics(
    output_root: Path,
    run_name: str,
    split_name: str,
    pred_df: pd.DataFrame,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    eval_dir = output_root / "eval" / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    pred_path = eval_dir / f"{split_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    payload = {"split": split_name, **(metrics or {})}
    metrics_path = eval_dir / f"{split_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "predictions_csv": str(pred_path),
        "metrics_json": str(metrics_path),
    }
