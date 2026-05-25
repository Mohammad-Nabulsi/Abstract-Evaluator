#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import Counter

import pandas as pd

RANDOM_SEED = 42

SCORE_DISTRIBUTION = {
    4: 0.15,
    3: 0.225,
    2: 0.35,
    1: 0.175,
    0: 0.10,
}

STRONG_DEGRADATIONS = [
    "missing_methodology",
    "missing_results",
    "vague_contribution",
    "unclear_objective",
    "method_result_mismatch",
    "generic_abstract",
]

MEDIUM_DEGRADATIONS = [
    "missing_problem",
    "poor_logical_flow",
    "weak_significance",
    "no_baseline_or_comparison",
]

MILD_DEGRADATIONS = [
    "overclaiming",
    "too_much_background",
    "weak_conclusion",
    "readability_degradation",
]

COMPATIBLE_PAIRS = {
    "missing_methodology": [
        "missing_results",
        "unclear_objective",
        "vague_contribution",
    ],
    "missing_results": [
        "missing_methodology",
        "overclaiming",
        "weak_significance",
        "no_baseline_or_comparison",
    ],
    "vague_contribution": [
        "generic_abstract",
        "weak_significance",
        "unclear_objective",
    ],
    "unclear_objective": [
        "poor_logical_flow",
        "missing_problem",
        "missing_methodology",
    ],
    "generic_abstract": [
        "vague_contribution",
        "weak_significance",
        "missing_results",
    ],
    "weak_significance": [
        "vague_contribution",
        "no_baseline_or_comparison",
        "missing_problem",
    ],
    "overclaiming": [
        "missing_results",
        "weak_significance",
    ],
    "poor_logical_flow": [
        "unclear_objective",
        "missing_problem",
    ],
    "no_baseline_or_comparison": [
        "weak_significance",
        "missing_results",
    ],
    "missing_problem": [
        "unclear_objective",
        "poor_logical_flow",
        "weak_significance",
    ],
}

ABSTRACT_STYLES = [
    "concise_research_style",
    "student_submission_style",
    "standard_academic_style",
    "slightly_verbose_style",
]

BANNED_PAIRS = {
    frozenset(("too_much_background", "no_baseline_or_comparison")),
    frozenset(("poor_logical_flow", "readability_degradation")),
    frozenset(("missing_problem", "readability_degradation")),
    frozenset(("too_much_background", "generic_abstract")),
}

ALL_DEGRADATIONS = STRONG_DEGRADATIONS + MEDIUM_DEGRADATIONS + MILD_DEGRADATIONS


def assign_scores(n_rows, rng):
    exact = {s: n_rows * p for s, p in SCORE_DISTRIBUTION.items()}
    base = {s: int(math.floor(v)) for s, v in exact.items()}
    remainder = n_rows - sum(base.values())
    fractional = sorted(
        ((s, exact[s] - base[s]) for s in SCORE_DISTRIBUTION.keys()),
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(remainder):
        base[fractional[i % len(fractional)][0]] += 1

    scores = []
    for s in sorted(base.keys(), reverse=True):
        scores.extend([s] * base[s])
    rng.shuffle(scores)
    return scores


def weighted_balanced_pick(candidates, counts, rng):
    if not candidates:
        raise ValueError("No candidates available for balanced pick.")
    vals = [counts.get(c, 0) for c in candidates]
    max_count = max(vals) if vals else 0
    weights = [(max_count - counts.get(c, 0) + 1) for c in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def is_banned_pair(a, b):
    return frozenset((a, b)) in BANNED_PAIRS


def pick_second(first, allowed_pool, counts, rng):
    compatible = COMPATIBLE_PAIRS.get(first, [])
    compatible_candidates = [
        d
        for d in compatible
        if d in allowed_pool and d != first and not is_banned_pair(first, d)
    ]
    if compatible_candidates:
        return weighted_balanced_pick(compatible_candidates, counts, rng)

    fallback = [d for d in allowed_pool if d != first and not is_banned_pair(first, d)]
    if not fallback:
        fallback = [d for d in allowed_pool if d != first]
    if not fallback:
        raise ValueError(f"No valid second degradation for first={first}")
    return weighted_balanced_pick(fallback, counts, rng)


def pick_style(score, rng):
    if score == 4:
        return "original"
    if score in (0, 1):
        return rng.choices(
            ["student_submission_style", "standard_academic_style"],
            weights=[0.6, 0.4],
            k=1,
        )[0]
    return rng.choice(ABSTRACT_STYLES)


def plan_for_score(score, counts, rng):
    if score == 4:
        return {
            "degradation_count": 0,
            "degradation_severity": "none",
            "degradation_types": [],
            "is_original": True,
        }

    if score == 3:
        pool = MILD_DEGRADATIONS + MEDIUM_DEGRADATIONS
        d1 = weighted_balanced_pick(pool, counts, rng)
        counts[d1] += 1
        return {
            "degradation_count": 1,
            "degradation_severity": "mild_or_medium",
            "degradation_types": [d1],
            "is_original": False,
        }

    if score == 2:
        if rng.random() < 0.5:
            d1 = weighted_balanced_pick(STRONG_DEGRADATIONS, counts, rng)
            counts[d1] += 1
            return {
                "degradation_count": 1,
                "degradation_severity": "medium_or_strong",
                "degradation_types": [d1],
                "is_original": False,
            }
        d1 = weighted_balanced_pick(MEDIUM_DEGRADATIONS, counts, rng)
        d2 = pick_second(d1, MEDIUM_DEGRADATIONS, counts, rng)
        counts[d1] += 1
        counts[d2] += 1
        return {
            "degradation_count": 2,
            "degradation_severity": "medium_or_strong",
            "degradation_types": [d1, d2],
            "is_original": False,
        }

    if score == 1:
        if rng.random() < 0.5:
            d1 = weighted_balanced_pick(STRONG_DEGRADATIONS, counts, rng)
            d2 = pick_second(d1, STRONG_DEGRADATIONS + MEDIUM_DEGRADATIONS, counts, rng)
        else:
            d1 = weighted_balanced_pick(STRONG_DEGRADATIONS, counts, rng)
            d2 = pick_second(d1, STRONG_DEGRADATIONS, counts, rng)
        counts[d1] += 1
        counts[d2] += 1
        return {
            "degradation_count": 2,
            "degradation_severity": "strong",
            "degradation_types": [d1, d2],
            "is_original": False,
        }

    if score == 0:
        d1 = weighted_balanced_pick(STRONG_DEGRADATIONS, counts, rng)
        d2 = pick_second(d1, STRONG_DEGRADATIONS, counts, rng)
        counts[d1] += 1
        counts[d2] += 1
        return {
            "degradation_count": 2,
            "degradation_severity": "severe",
            "degradation_types": [d1, d2],
            "is_original": False,
        }

    raise ValueError(f"Unsupported score: {score}")


def ensure_valid_row(row):
    if row["target_score"] == 4:
        assert row["is_original"] is True
        assert row["degradation_count"] == 0
        assert row["degradation_severity"] == "none"
        assert row["degradation_types"] == []
        assert row["abstract_style"] == "original"
    else:
        assert row["is_original"] is False
        assert isinstance(row["degradation_types"], list)
        assert row["degradation_count"] == len(row["degradation_types"])


def load_input(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Input must be .json or .csv")


def main():
    parser = argparse.ArgumentParser(description="Assign degradation planning metadata.")
    parser.add_argument(
        "--input",
        default="data/openreview/iclr2024_openreview_10000_merged.json",
        help="Input dataset path (.json or .csv)",
    )
    parser.add_argument(
        "--output_csv",
        default="data/processed/abstracts_with_degradation_plan.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output_jsonl",
        default="data/processed/abstracts_with_degradation_plan.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = load_input(args.input).copy()
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Input dataset is empty.")

    scores = assign_scores(n_rows, rng)
    counts = Counter({d: 0 for d in ALL_DEGRADATIONS})

    planned = {
        "target_score": [],
        "degradation_count": [],
        "degradation_severity": [],
        "degradation_types": [],
        "abstract_style": [],
        "is_original": [],
    }

    for score in scores:
        row_plan = plan_for_score(score, counts, rng)
        style = pick_style(score, rng)
        row = {
            "target_score": score,
            "degradation_count": row_plan["degradation_count"],
            "degradation_severity": row_plan["degradation_severity"],
            "degradation_types": row_plan["degradation_types"],
            "abstract_style": style,
            "is_original": row_plan["is_original"],
        }
        ensure_valid_row(row)
        for k in planned:
            planned[k].append(row[k])

    for k, v in planned.items():
        df[k] = v

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    score_counts = df["target_score"].value_counts().sort_index(ascending=False)
    severity_counts = df["degradation_severity"].value_counts()
    type_counts = Counter()
    for types in df["degradation_types"]:
        for t in types:
            type_counts[t] += 1

    print("=== Score Distribution ===")
    for s in [4, 3, 2, 1, 0]:
        c = int(score_counts.get(s, 0))
        pct = (c / n_rows) * 100.0
        print(f"score={s}: count={c}, pct={pct:.2f}%")

    print("\n=== Degradation Type Counts ===")
    for d in ALL_DEGRADATIONS:
        print(f"{d}: {type_counts.get(d, 0)}")

    print("\n=== Degradation Severity Counts ===")
    for sev, c in severity_counts.items():
        print(f"{sev}: {int(c)}")

    n_original = int(df["is_original"].sum())
    print(f"\nNumber of original rows: {n_original}")

    print("\n=== Sample Rows (5) ===")
    sample_cols = [
        "paper_id",
        "target_score",
        "degradation_count",
        "degradation_severity",
        "degradation_types",
        "abstract_style",
    ]
    existing_sample_cols = [c for c in sample_cols if c in df.columns]
    print(df[existing_sample_cols].head(5).to_string(index=False))

    csv_df = df.copy()
    csv_df["degradation_types"] = csv_df["degradation_types"].apply(json.dumps)
    csv_df.to_csv(args.output_csv, index=False)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
