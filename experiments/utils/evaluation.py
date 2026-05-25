from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import evaluate  # type: ignore
except Exception:  # pragma: no cover
    evaluate = None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def parse_score(text: str) -> Optional[int]:
    obj = extract_json_object(text)
    if obj is not None and "score" in obj:
        try:
            score = int(obj["score"])
            if 0 <= score <= 4:
                return score
        except Exception:
            pass

    match = re.search(r'"?score"?\s*[:=]\s*([0-4])', str(text))
    if match:
        return int(match.group(1))

    return None


def parse_rationale(text: str) -> str:
    obj = extract_json_object(text)
    if obj is not None and "rationale" in obj:
        rationale = str(obj["rationale"]).strip()
        if rationale:
            return rationale

    fallback = str(text).strip()
    return fallback if fallback else "__EMPTY_OUTPUT__"


_ROUGE = None
_BLEU = None
_BERTSCORE = None


def _load_metric_once(name: str):
    if evaluate is None:
        raise RuntimeError("The `evaluate` package is not available in this environment.")
    return evaluate.load(name)


def _get_rouge():
    global _ROUGE
    if _ROUGE is None:
        _ROUGE = _load_metric_once("rouge")
    return _ROUGE


def _get_bleu():
    global _BLEU
    if _BLEU is None:
        _BLEU = _load_metric_once("bleu")
    return _BLEU


def _get_bertscore():
    global _BERTSCORE
    if _BERTSCORE is None:
        _BERTSCORE = _load_metric_once("bertscore")
    return _BERTSCORE


def compute_eval_metrics(
    pred_df: pd.DataFrame,
    include_bertscore: bool = False,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    valid_score_mask = pred_df["pred_score"].notna()
    out["json_parse_rate"] = float(valid_score_mask.mean())

    if valid_score_mask.any():
        y_true = pred_df.loc[valid_score_mask, "score"].astype(int).values
        y_pred = pred_df.loc[valid_score_mask, "pred_score"].astype(int).values
        out["score_accuracy"] = float((y_true == y_pred).mean())
        out["score_mae"] = float(np.abs(y_true - y_pred).mean())
        out["score_within_1_accuracy"] = float((np.abs(y_true - y_pred) <= 1).mean())
    else:
        out["score_accuracy"] = 0.0
        out["score_mae"] = 999.0
        out["score_within_1_accuracy"] = 0.0

    preds = pred_df["pred_rationale"].fillna("").astype(str).tolist()
    refs = pred_df["rationale"].fillna("").astype(str).tolist()

    try:
        rouge = _get_rouge().compute(predictions=preds, references=refs)
        out.update({f"rouge_{k}": float(v) for k, v in rouge.items()})
    except Exception:
        out["rouge_1"] = float("nan")
        out["rouge_2"] = float("nan")
        out["rouge_l"] = float("nan")
        out["rouge_lsum"] = float("nan")

    try:
        bleu = _get_bleu().compute(predictions=preds, references=[[r] for r in refs])
        out["bleu"] = float(bleu["bleu"])
    except Exception:
        out["bleu"] = float("nan")

    if include_bertscore:
        try:
            bert = _get_bertscore().compute(predictions=preds, references=refs, lang="en")
            out["bertscore_precision"] = float(np.mean(bert["precision"]))
            out["bertscore_recall"] = float(np.mean(bert["recall"]))
            out["bertscore_f1"] = float(np.mean(bert["f1"]))
        except Exception:
            out["bertscore_precision"] = float("nan")
            out["bertscore_recall"] = float("nan")
            out["bertscore_f1"] = float("nan")

    return out


def save_predictions_and_metrics(
    pred_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: Path,
    split_name: str,
    tag: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{split_name}_predictions_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    metrics_path = output_dir / f"metrics_{tag}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return {"predictions_csv": str(pred_path), "metrics_json": str(metrics_path)}


def add_error_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["score_error"] = out["pred_score"] - out["score"]
    out["abs_score_error"] = out["score_error"].abs()
    return out

