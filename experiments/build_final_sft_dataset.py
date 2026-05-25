#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/test_scored_1000_run3_async.jsonl",
            "data/processed/batches/sample100_run1_scored.jsonl",
            "data/processed/sample10_scored_v2.jsonl",
            "data/processed/test_scored_100_async_run2.jsonl",
            "data/processed/test_scored_10_async.jsonl",
        ],
    )
    ap.add_argument("--output", default="data/processed/final_sft_dataset_1220.jsonl")
    args = ap.parse_args()

    dfs = []
    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        df = load_jsonl(p)
        dfs.append(df)
        print(f"[ok] {p} rows={len(df)}")

    if not dfs:
        raise RuntimeError("No input files found.")

    df = pd.concat(dfs, ignore_index=True)

    keep_cols = [
        "paper_id",
        "title",
        "submission",
        "score",
        "rationale",
        "degradation_types",
        "target_score",
        "validation_status",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["paper_id"] = df["paper_id"].astype(str)
    df["submission"] = df["submission"].astype(str).str.strip()
    df["rationale"] = df["rationale"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.dropna(subset=["paper_id", "submission", "score", "rationale"])
    df = df[df["submission"].str.len() >= 50]
    df = df[df["rationale"].str.len() >= 20]
    df = df[df["score"].between(0, 4)]
    df["score"] = df["score"].astype(int)

    if "validation_status" in df.columns:
        df["validation_status"] = df["validation_status"].fillna("PASS")

    # keep last seen record per paper_id
    df = df.drop_duplicates(subset=["paper_id"], keep="last").reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", lines=True, force_ascii=False)

    print(f"\nSaved: {out}")
    print(f"Rows: {len(df)}")
    print(df["score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
