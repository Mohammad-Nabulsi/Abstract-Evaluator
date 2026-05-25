#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set

from openai import OpenAI


VERIFIER_PROMPT_TEMPLATE = """You are validating a synthetic training example for abstract evaluation.

Original title:
{title}

Original abstract:
{abstract}

Generated submitted abstract:
{submission}

Target score:
{target_score}

Requested degradation types:
{degradation_types}

Generated rationale:
{rationale}

Check:
1. Does the submitted abstract preserve the original topic?
2. Are the requested abstract-level flaws actually present?
3. Does the score match the severity of the submitted abstract?
4. Does the rationale correctly describe the submitted abstract?
5. Does the rationale avoid discussing full-paper flaws not visible from the abstract?
6. Does the example avoid mentioning synthetic data, degradation, or the original abstract?

Return valid JSON:
{{
  "validation_status": "PASS" or "FAIL",
  "validation_reason": "...",
  "detected_flaws": [...]
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify generated abstract submissions using an LLM verifier.")
    p.add_argument(
        "--input",
        default="data/processed/abstracts_with_generated_submissions.jsonl",
        help="Input JSONL containing generated submissions.",
    )
    p.add_argument(
        "--output",
        default="data/processed/abstracts_with_verified_submissions.jsonl",
        help="Output JSONL with validation fields appended.",
    )
    p.add_argument("--model", default="gpt-5-mini", help="Verifier model")
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls")
    p.add_argument("--limit", type=int, default=None, help="Optional max rows to process")
    p.add_argument("--resume", action="store_true", help="Skip rows already present in output by paper_id")
    p.add_argument("--drop-fail-output", default="", help="Optional output path with FAIL rows removed")
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


def coerce_rationale(rec: Dict[str, Any]) -> str:
    val = rec.get("rationale", "")
    if val is None:
        return ""
    return str(val)


def validate_verifier_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Verifier output is not a JSON object")

    status = str(obj.get("validation_status", "")).strip().upper()
    reason = obj.get("validation_reason", "")
    flaws = obj.get("detected_flaws", [])

    if status not in {"PASS", "FAIL"}:
        raise ValueError("validation_status must be PASS or FAIL")
    if not isinstance(reason, str):
        raise ValueError("validation_reason must be a string")
    if not isinstance(flaws, list):
        raise ValueError("detected_flaws must be a list")

    return {
        "validation_status": status,
        "validation_reason": reason,
        "detected_flaws": flaws,
    }


def verify_record(client: OpenAI, model: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)

    title = str(out.get("title", ""))
    abstract = str(out.get("abstract", ""))
    submission = str(out.get("submission", ""))
    target_score = out.get("target_score", "")
    degradation_types = parse_degradation_types(out.get("degradation_types"))
    rationale = coerce_rationale(out)

    prompt = VERIFIER_PROMPT_TEMPLATE.format(
        title=title,
        abstract=abstract,
        submission=submission,
        target_score=target_score,
        degradation_types=json.dumps(degradation_types, ensure_ascii=False),
        rationale=rationale,
    )

    max_retries = 3
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty verifier response")
            parsed = json.loads(content)
            verdict = validate_verifier_json(parsed)

            out["validation_status"] = verdict["validation_status"]
            out["validation_reason"] = verdict["validation_reason"]
            out["detected_flaws"] = verdict["detected_flaws"]
            out["validation_model"] = model
            out["validation_error"] = ""
            return out
        except Exception as exc:  # noqa: BLE001
            last_error = f"attempt {attempt}/{max_retries}: {exc}"
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    out["validation_status"] = "FAIL"
    out["validation_reason"] = "Verifier call failed"
    out["detected_flaws"] = []
    out["validation_model"] = model
    out["validation_error"] = last_error
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

    status_counts: Counter = Counter()
    score_status_counts: Counter = Counter()

    processed = 0
    pass_count = 0
    fail_count = 0

    with open(args.output, mode, encoding="utf-8") as fout:
        for rec in rows:
            pid = str(rec.get("paper_id", "")).strip()
            if args.resume and pid and pid in processed_ids:
                continue

            verified = verify_record(client, args.model, rec)
            fout.write(json.dumps(verified, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            status = verified.get("validation_status", "FAIL")
            score = verified.get("target_score", "")
            status_counts[status] += 1
            score_status_counts[(score, status)] += 1

            if status == "PASS":
                pass_count += 1
            else:
                fail_count += 1

            if processed % 50 == 0:
                print(f"processed={processed} pass={pass_count} fail={fail_count}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("\n=== Final validation_status counts ===")
    for k, v in status_counts.items():
        print(f"{k}: {v}")

    print("\n=== Final counts by target_score and validation_status ===")
    items = sorted(score_status_counts.items(), key=lambda x: (int(x[0][0]) if str(x[0][0]).isdigit() else 999, x[0][1]))
    for (score, status), count in items:
        print(f"target_score={score}, validation_status={status}: {count}")

    if args.drop_fail_output:
        verified_rows = load_jsonl(args.output)
        kept = [r for r in verified_rows if str(r.get("validation_status", "")).upper() == "PASS"]
        os.makedirs(os.path.dirname(args.drop_fail_output) or ".", exist_ok=True)
        with open(args.drop_fail_output, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved PASS-only file: {args.drop_fail_output} ({len(kept)} rows)")


if __name__ == "__main__":
    main()
