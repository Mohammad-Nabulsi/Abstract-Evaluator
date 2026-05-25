#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

SOURCE = Path("data/processed/abstracts_with_degradation_plan.jsonl")
STATE = Path("data/processed/state/used_paper_ids.txt")
SAMPLE_PLAN = Path("data/processed/test_plan_10_async.jsonl")
OUTPUT = Path("data/processed/test_generated_10_async.jsonl")
TIMING = Path("logs/test_generation_10_timing.json")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def load_used(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.open("r", encoding="utf-8") if line.strip()}


def append_used(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for pid in ids:
            f.write(pid + "\n")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    rows = load_jsonl(SOURCE)
    used = load_used(STATE)

    selected: List[Dict] = []
    selected_ids: List[str] = []
    for r in rows:
        pid = str(r.get("paper_id", "")).strip()
        if not pid or pid in used:
            continue
        selected.append(r)
        selected_ids.append(pid)
        if len(selected) == 10:
            break

    if len(selected) < 10:
        raise RuntimeError(f"Could not find 10 NEW samples; found {len(selected)}")

    # reserve immediately so future sampling cannot reuse these IDs
    append_used(STATE, selected_ids)

    write_jsonl(SAMPLE_PLAN, selected)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TIMING.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    cmd = [
        sys.executable,
        "scripts/generate_submissions_async.py",
        "--input", str(SAMPLE_PLAN),
        "--output", str(OUTPUT),
        "--model", "gpt-5-mini",
        "--limit", "10",
        "--batch-size", "5",
        "--concurrency", "2",
        "--max-retries", "3",
        "--timeout", "120",
        "--timing-log", str(TIMING),
    ]

    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Generation test failed with code {proc.returncode}")

    out_rows = load_jsonl(OUTPUT)
    generated = sum(1 for r in out_rows if r.get("generation_status") == "generated")
    failed = sum(1 for r in out_rows if r.get("generation_status") == "failed")
    kept = sum(1 for r in out_rows if r.get("generation_status") == "original_kept")

    print("\n=== Test Summary ===")
    print(f"rows={len(out_rows)} generated={generated} failed={failed} original_kept={kept}")
    print(f"sample_plan={SAMPLE_PLAN}")
    print(f"output={OUTPUT}")
    print(f"timing={TIMING}")


if __name__ == "__main__":
    main()
