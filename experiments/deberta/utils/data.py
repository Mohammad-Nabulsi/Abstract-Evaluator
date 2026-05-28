from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Dict

import pandas as pd

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


def _import_datasets_types():
    # Lazy import avoids loading pyarrow during notebook bootstrap imports.
    from datasets import Dataset, DatasetDict

    return Dataset, DatasetDict


def _normalize_rubric(rubric: Any) -> Dict[str, Any]:
    if isinstance(rubric, dict):
        return rubric

    if isinstance(rubric, str):
        raw = rubric.strip()
        if not raw:
            return {}
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"rubric": raw}

    return {}


def format_rubric(rubric: Any) -> str:
    rubric_dict = _normalize_rubric(rubric)
    if not rubric_dict:
        return ""

    if "score_scale" in rubric_dict and "criteria" in rubric_dict:
        score_scale = rubric_dict.get("score_scale", {})
        criteria = rubric_dict.get("criteria", {})

        score_lines = ["Score scale:"]
        for k, v in sorted(score_scale.items(), key=lambda x: int(str(x[0]))):
            score_lines.append(f"{k} = {v}")

        criteria_lines = ["Criteria:"]
        for k, v in sorted(criteria.items(), key=lambda x: str(x[0])):
            criteria_lines.append(f"{k}: {v}")

        return "\n".join(score_lines + [""] + criteria_lines)

    lines = [f"{k}: {v}" for k, v in sorted(rubric_dict.items(), key=lambda x: str(x[0]))]
    return "\n".join(lines)


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_input_text(
    row: pd.Series,
    max_submission_chars: int = 3000,
    max_task_chars: int = 300,
    max_reference_chars: int = 600,
    max_rubric_chars: int = 500,
    max_title_chars: int = 200,
) -> str:
    # Put submission first so truncation at 512 tokens preserves the answer.
    submission = _clip_text(row.get("submission", ""), max_submission_chars)
    task = _clip_text(row.get("task", ""), max_task_chars)
    reference = _clip_text(row.get("reference", ""), max_reference_chars)
    rubric_text = _clip_text(format_rubric(row.get("rubric")), max_rubric_chars)

    title_block = ""
    if "title" in row and pd.notna(row["title"]) and str(row["title"]).strip():
        title = _clip_text(row["title"], max_title_chars)
        title_block = f"Title:\n{title}\n\n"

    rubric_block = f"Rubric:\n{rubric_text}\n\n" if rubric_text else ""

    return (
        f"Submission:\n{submission}\n\n"
        f"Task:\n{task}\n\n"
        f"{title_block}"
        f"Reference:\n{reference}\n\n"
        f"{rubric_block}"
    ).strip()


def add_score_only_text(
    df: pd.DataFrame,
    max_submission_chars: int = 3000,
    max_task_chars: int = 300,
    max_reference_chars: int = 600,
    max_rubric_chars: int = 500,
    max_title_chars: int = 200,
) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out.apply(
        lambda r: build_input_text(
            r,
            max_submission_chars=max_submission_chars,
            max_task_chars=max_task_chars,
            max_reference_chars=max_reference_chars,
            max_rubric_chars=max_rubric_chars,
            max_title_chars=max_title_chars,
        ),
        axis=1,
    )
    out["label"] = out["score"].astype(int)
    return out


def to_hf_dataset(df: pd.DataFrame) -> "Dataset":
    Dataset, _ = _import_datasets_types()
    keep_cols = [c for c in ["id", "paper_id", "text", "label", "score"] if c in df.columns]
    ds = Dataset.from_pandas(df[keep_cols], preserve_index=False)
    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    return ds


def to_hf_dataset_dict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> "DatasetDict":
    _, DatasetDict = _import_datasets_types()
    return DatasetDict(
        {
            "train": to_hf_dataset(train_df),
            "validation": to_hf_dataset(val_df),
            "test": to_hf_dataset(test_df),
        }
    )
