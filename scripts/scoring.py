#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set

from openai import OpenAI


SCORING_PROMPT_TEMPLATE = """You are a conference reviewer evaluating only the abstract, not the full paper.

Task:
Evaluate the quality of the following research abstract for conference acceptance.

Reference:
A strong research abstract clearly presents the problem, methodology, contribution, and experimental evidence.

Rubric:
0 = Strong reject
1 = Reject
2 = Borderline
3 = Accept
4 = Strong accept

Title:
{title}

Submitted abstract:
{submission}

Known abstract-level flaw profile:
{degradation_types}

Target score:
{target_score}

Reviewer style:
{reviewer_style}

Write a realistic rationale.

Rules:
1. The rationale must match the target score.
2. Discuss only the abstract quality.
3. Mention the actual flaws visible in the submitted abstract.
4. Do not mention the original abstract.
5. Do not mention synthetic data or degradation.
6. Keep it 2-5 sentences.
7. Use the reviewer style, but stay professional.
8. The rationale should also add the positive side of the abstract.
9. Also tell what to fix about it.

Return valid JSON only:
{{
  "score": {target_score},
  "rationale": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score generated submissions and add rationale.")
    p.add_argument(
        "--input",
        default="data/processed/abstracts_with_generated_submissions.jsonl",
        help="Input JSONL path with generated submissions.",
    )
    p.add_argument(
        "--output",
        default="data/processed/abstracts_with_scored_submissions.jsonl",
        help="Output JSONL path with score and rationale appended.",
    )
    p.add_argument("--model", default="gpt-5-mini", help="OpenAI model")
    p.add_argument("--limit", type=int, default=None, help="Optional row limit")
    p.add_argument("--resume", action="store_true", help="Resume from existing output by paper_id")
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls")
    return p.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}: {exc}") from exc
    return rows


def load_processed_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            pid = str(rec.get("paper_id", "")).strip()
            if pid:
                ids.add(pid)
    return ids


def parse_degradation_types(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        t = value.strip()
        if not t:
            return []
        try:
            parsed = json.loads(t)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [t]
    return []


def call_scorer(client: OpenAI, model: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)

    title = str(out.get("title", ""))
    submission = str(out.get("submission", ""))
    degradation_types = parse_degradation_types(out.get("degradation_types"))
    target_score = int(out.get("target_score", 2))
    reviewer_style = str(out.get("reviewer_style", "balanced and concise"))

    prompt = SCORING_PROMPT_TEMPLATE.format(
        title=title,
        submission=submission,
        degradation_types=json.dumps(degradation_types, ensure_ascii=False),
        target_score=target_score,
        reviewer_style=reviewer_style,
    )

    max_retries = 3
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty scorer response")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Scorer output must be JSON object")

            score = parsed.get("score", target_score)
            rationale = parsed.get("rationale", "")

            score = int(score)
            score = max(0, min(4, score))
            if not isinstance(rationale, str):
                rationale = str(rationale)

            out["score"] = score
            out["rationale"] = rationale.strip()
            out["scoring_model"] = model
            out["scoring_status"] = "scored"
            out["scoring_error"] = ""
            return out
        except Exception as exc:  # noqa: BLE001
            last_error = f"attempt {attempt}/3: {exc}"
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    out["score"] = int(out.get("target_score", 2)) if str(out.get("target_score", "")).isdigit() else 2
    out["rationale"] = ""
    out["scoring_model"] = model
    out["scoring_status"] = "failed"
    out["scoring_error"] = last_error
    return out


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    processed_ids: Set[str] = set()
    mode = "w"
    if args.resume and os.path.exists(args.output):
        processed_ids = load_processed_ids(args.output)
        mode = "a"

    client = OpenAI(api_key=api_key)

    processed = 0
    scored = 0
    failed = 0
    status_counts: Counter = Counter()
    score_counts: Counter = Counter()

    with open(args.output, mode, encoding="utf-8") as fout:
        for rec in rows:
            pid = str(rec.get("paper_id", "")).strip()
            if args.resume and pid and pid in processed_ids:
                continue

            out = call_scorer(client, args.model, rec)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            status = out.get("scoring_status", "failed")
            status_counts[status] += 1
            score_counts[out.get("score", "")] += 1

            if status == "scored":
                scored += 1
            else:
                failed += 1

            if processed % 50 == 0:
                print(f"processed={processed} scored={scored} failed={failed}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("\n=== Final scoring_status counts ===")
    for k, v in status_counts.items():
        print(f"{k}: {v}")

    print("\n=== Final generated score counts ===")
    for k in sorted(score_counts.keys()):
        print(f"score={k}: {score_counts[k]}")


if __name__ == "__main__":
    main()
