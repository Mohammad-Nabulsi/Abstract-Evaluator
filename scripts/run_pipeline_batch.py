#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{i}: {exc}") from exc
    return rows


def load_used_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    used = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                used.add(s)
    return used


def append_used_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for pid in ids:
            f.write(pid + "\n")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def summarize_verified(path: Path) -> None:
    rows = load_jsonl(path)
    c = Counter(str(r.get("validation_status", "")) for r in rows)
    cs = Counter((int(r.get("target_score", -1)), str(r.get("validation_status", ""))) for r in rows)
    print("\n=== Verified Summary ===")
    for k, v in c.items():
        print(f"{k}: {v}")
    print("\nBy target_score:")
    for (score, st), n in sorted(cs.items()):
        print(f"target_score={score}, validation_status={st}: {n}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run apply->score->verify on next unused batch.")
    p.add_argument("--source", default="data/processed/abstracts_with_degradation_plan.jsonl")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--state-file", default="data/processed/state/used_paper_ids.txt")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--output-dir", default="data/processed/batches")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    source = Path(args.source)
    state_file = Path(args.state_file)
    out_dir = Path(args.output_dir)

    if not source.exists():
        raise FileNotFoundError(source)

    rows = load_jsonl(source)
    used = load_used_ids(state_file)

    selected = []
    selected_ids = []
    for r in rows:
        pid = str(r.get("paper_id", "")).strip()
        if not pid or pid in used:
            continue
        selected.append(r)
        selected_ids.append(pid)
        if len(selected) >= args.batch_size:
            break

    if len(selected) < args.batch_size:
        raise RuntimeError(f"Only found {len(selected)} unused rows; requested {args.batch_size}")

    # Reserve IDs immediately so future sampling won't reuse them even on interruption.
    append_used_ids(state_file, selected_ids)

    tag = args.tag.strip() or datetime.utcnow().strftime("batch_%Y%m%dT%H%M%SZ")
    plan_path = out_dir / f"{tag}_plan.jsonl"
    gen_path = out_dir / f"{tag}_generated.jsonl"
    score_path = out_dir / f"{tag}_scored.jsonl"
    verify_path = out_dir / f"{tag}_verified.jsonl"

    write_jsonl(plan_path, selected)
    print(f"Selected {len(selected)} rows and reserved IDs in {state_file}")
    print(f"Batch tag: {tag}")

    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in environment")

    py = sys.executable

    run_cmd([
        py, "scripts/apply_degradation.py",
        "--input", str(plan_path),
        "--output", str(gen_path),
        "--model", args.model,
        "--sleep", "0",
    ], env)

    run_cmd([
        py, "scripts/scoring.py",
        "--input", str(gen_path),
        "--output", str(score_path),
        "--model", args.model,
        "--sleep", "0",
    ], env)

    run_cmd([
        py, "scripts/verify_generated_submissions.py",
        "--input", str(score_path),
        "--output", str(verify_path),
        "--model", args.model,
        "--sleep", "0",
    ], env)

    summarize_verified(verify_path)
    print("\nOutputs:")
    print(plan_path)
    print(gen_path)
    print(score_path)
    print(verify_path)


if __name__ == "__main__":
    main()
