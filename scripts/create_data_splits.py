#!/usr/bin/env python3
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SEED = 3407
FINAL_TRAIN = 910
FINAL_VAL = 100
FINAL_TEST = 100
TARGET_TOTAL_VAL = 200
TARGET_TOTAL_TEST = 200


def read_json_objects(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    out = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        # Some files contain literal separator tokens between objects, e.g. "\n".
        while i + 1 < n and text[i] == "\\" and text[i + 1] == "n":
            i += 2
            while i < n and text[i].isspace():
                i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(text, i)
        out.append(obj)
        i = j
    return out


def pick_degradation_col(rows: List[dict]) -> Optional[str]:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for c in ["degradation_type", "degradation", "degrade_type", "perturbation_type", "degradation_types"]:
        if c in keys:
            return c
    return None


def build_joint_strata(rows: List[dict], deg_col: Optional[str]) -> List[str]:
    scores = [str(r["score"]) for r in rows]
    if deg_col is None:
        return scores

    degs = [str(r.get(deg_col, "missing")) if r.get(deg_col, None) is not None else "missing" for r in rows]
    joint = [f"{s}||{d}" for s, d in zip(scores, degs)]

    counts = Counter(joint)
    joint = [j if counts[j] >= 2 else f"{s}||__other__" for s, j in zip(scores, joint)]
    if min(Counter(joint).values()) < 2:
        return scores
    return joint


def stratified_pick_indices(n_rows: int, labels: List[str], pick_n: int, seed: int) -> List[int]:
    if pick_n < 0 or pick_n > n_rows:
        raise ValueError(f"pick_n must be in [0, {n_rows}], got {pick_n}")
    if pick_n == 0:
        return []

    by_class: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)

    total = n_rows
    alloc = {}
    remainders = []
    taken = 0
    for c, idxs in by_class.items():
        exact = pick_n * (len(idxs) / total)
        base = min(len(idxs), int(math.floor(exact)))
        alloc[c] = base
        taken += base
        remainders.append((exact - base, c))

    need = pick_n - taken
    remainders.sort(reverse=True, key=lambda x: x[0])
    k = 0
    while need > 0:
        _, c = remainders[k % len(remainders)]
        if alloc[c] < len(by_class[c]):
            alloc[c] += 1
            need -= 1
        k += 1

    rng = random.Random(seed)
    picked = []
    for c, idxs in by_class.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        picked.extend(shuffled[: alloc[c]])
    rng.shuffle(picked)
    return sorted(picked)


def split_fixed_counts_stratified(
    rows: List[dict], train_rows: int, val_rows: int, test_rows: int, seed: int
) -> Tuple[List[dict], List[dict], List[dict], Optional[str]]:
    total_needed = train_rows + val_rows + test_rows
    if len(rows) < total_needed:
        raise ValueError(f"Need at least {total_needed} rows, found {len(rows)}")

    rng = random.Random(seed)
    base = rows[:]
    rng.shuffle(base)
    base = base[:total_needed]

    deg_col = pick_degradation_col(base)
    strata_all = build_joint_strata(base, deg_col=deg_col)

    try:
        test_idx = set(stratified_pick_indices(len(base), strata_all, test_rows, seed))
    except Exception:
        score_labels = [str(r["score"]) for r in base]
        test_idx = set(stratified_pick_indices(len(base), score_labels, test_rows, seed))

    rest = [r for i, r in enumerate(base) if i not in test_idx]
    test = [r for i, r in enumerate(base) if i in test_idx]

    strata_rest = build_joint_strata(rest, deg_col=deg_col)
    try:
        val_idx = set(stratified_pick_indices(len(rest), strata_rest, val_rows, seed))
    except Exception:
        score_labels_rest = [str(r["score"]) for r in rest]
        val_idx = set(stratified_pick_indices(len(rest), score_labels_rest, val_rows, seed))

    val = [r for i, r in enumerate(rest) if i in val_idx]
    train = [r for i, r in enumerate(rest) if i not in val_idx]
    return train, val, test, deg_col


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    final_path = root / "data/processed/final_sft_dataset_1220.jsonl"
    gen_path = root / "data/processed/genaration_run_2.jsonl"
    out_root = root / "data/data"

    final_rows = read_json_objects(final_path)
    gen_rows = read_json_objects(gen_path)

    final_train, final_val, final_test, _ = split_fixed_counts_stratified(
        final_rows, FINAL_TRAIN, FINAL_VAL, FINAL_TEST, SEED
    )

    needed_val_from_gen = TARGET_TOTAL_VAL - len(final_val)
    needed_test_from_gen = TARGET_TOTAL_TEST - len(final_test)
    if needed_val_from_gen < 0 or needed_test_from_gen < 0:
        raise ValueError("final split already exceeds target val/test totals")

    gen_train_rows = len(gen_rows) - needed_val_from_gen - needed_test_from_gen
    if gen_train_rows < 0:
        raise ValueError(
            f"Not enough rows in genaration_run_2.jsonl to fill missing val/test: "
            f"need {needed_val_from_gen + needed_test_from_gen}, found {len(gen_rows)}"
        )

    gen_train, gen_val, gen_test, _ = split_fixed_counts_stratified(
        gen_rows, gen_train_rows, needed_val_from_gen, needed_test_from_gen, SEED
    )

    train_all = final_train + gen_train
    val_all = final_val + gen_val
    test_all = final_test + gen_test

    write_jsonl(out_root / "train/all.jsonl", train_all)
    write_jsonl(out_root / "val/all.jsonl", val_all)
    write_jsonl(out_root / "test/all.jsonl", test_all)

    write_jsonl(out_root / "train/final_sft_dataset_1220_train.jsonl", final_train)
    write_jsonl(out_root / "val/final_sft_dataset_1220_val.jsonl", final_val)
    write_jsonl(out_root / "test/final_sft_dataset_1220_test.jsonl", final_test)

    write_jsonl(out_root / "train/genaration_run_2_train.jsonl", gen_train)
    write_jsonl(out_root / "val/genaration_run_2_val.jsonl", gen_val)
    write_jsonl(out_root / "test/genaration_run_2_test.jsonl", gen_test)

    summary = {
        "seed": SEED,
        "final_dataset_rows": len(final_rows),
        "generation_dataset_rows": len(gen_rows),
        "final_split": {"train": len(final_train), "val": len(final_val), "test": len(final_test)},
        "generation_split": {"train": len(gen_train), "val": len(gen_val), "test": len(gen_test)},
        "combined_split": {"train": len(train_all), "val": len(val_all), "test": len(test_all)},
    }
    (out_root / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
