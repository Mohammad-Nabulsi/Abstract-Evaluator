#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Set

from openai import OpenAI


PROMPT_TEMPLATE = """You are generating synthetic training data for a research abstract evaluation model.

Original abstract:
{abstract}


Target degradation plan:
- Target score: {target_score}
- Severity: {degradation_severity}
- Degradation types: {degradation_types}
- Writing style: {abstract_style}

Rewrite the original abstract into a realistic submitted abstract.

Rules:
1. Preserve the same research topic.
2. Preserve the same general field and terminology.
3. Preserve the paper identity.
4. Preserve approximately 60–90% of the original wording and sentence structure unless the requested flaw requires restructuring.
5. Apply only the requested abstract-level flaws.
6. Do not critique the abstract.
7. Do not mention degradation, synthetic data, score, rubric, or evaluation.
8. Do not invent fake numerical results unless the flaw is overclaiming.
9. Keep it between 80 and 220 words.
10. Make it sound naturally written by a real researcher or student.
11. Do not make score 0 text nonsensical; make it scientifically empty, vague, or incomplete but still realistic.
12. Return valid JSON only.

Return exactly:
{{
  "submission": "...",
  "applied_degradations": [...]
}}"""

TEMPERATURE_BY_SCORE = {
    3: 0.5,
    2: 0.5,
    1: 0.6,
    0: 0.6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate degraded abstract submissions from a planning JSONL.")
    parser.add_argument(
        "--input",
        default="data/processed/abstracts_with_degradation_plan.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/abstracts_with_generated_submissions.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation by skipping rows already in output.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep duration between successful API calls.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def load_processed_ids(path: str) -> Set[str]:
    processed: Set[str] = set()
    if not os.path.exists(path):
        return processed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = str(rec.get("paper_id", "")).strip()
            if pid:
                processed.add(pid)
    return processed


def parse_degradation_types(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [text]
    return []


def build_keywords_text(keywords: Any) -> str:
    if keywords is None:
        return ""
    if isinstance(keywords, list):
        return ", ".join(str(k) for k in keywords if k is not None)
    if isinstance(keywords, str):
        return keywords
    return str(keywords)


def validate_generation_output(raw: str) -> Dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object.")

    submission = parsed.get("submission")
    applied = parsed.get("applied_degradations")

    if not isinstance(submission, str):
        raise ValueError("`submission` must be a string.")
    if not isinstance(applied, list):
        raise ValueError("`applied_degradations` must be a list.")

    words = len(submission.split())
    if words < 60 or words > 260:
        # Soft validation only.
        pass

    return {
        "submission": submission,
        "applied_degradations": applied,
    }


def model_temperature_for_score(score: Any) -> float:
    try:
        score_int = int(score)
    except (TypeError, ValueError):
        score_int = 2
    return TEMPERATURE_BY_SCORE.get(score_int, 0.6)


def call_model(
    client: OpenAI,
    model: str,
    title: str,
    abstract: str,
    keywords_text: str,
    target_score: int,
    degradation_severity: str,
    degradation_types: List[str],
    abstract_style: str,
    temperature: float,
) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        title=title,
        abstract=abstract,
        keywords_text=keywords_text,
        target_score=target_score,
        degradation_severity=degradation_severity,
        degradation_types=json.dumps(degradation_types, ensure_ascii=False),
        abstract_style=abstract_style,
    )

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Empty response content from model.")
    return validate_generation_output(content)


def process_record(client: OpenAI, rec: Dict[str, Any], model: str, sleep_s: float) -> Dict[str, Any]:
    out = dict(rec)

    target_score = int(out.get("target_score", 2))
    title = str(out.get("title", ""))
    abstract = str(out.get("abstract", ""))
    keywords_text = build_keywords_text(out.get("keywords"))
    degradation_types = parse_degradation_types(out.get("degradation_types"))
    degradation_severity = str(out.get("degradation_severity", ""))
    abstract_style = str(out.get("abstract_style", "standard_academic_style"))

    if target_score == 4:
        out["submission"] = abstract
        out["applied_degradations"] = []
        out["generation_model"] = "none"
        out["generation_status"] = "original_kept"
        out["generation_error"] = ""
        return out

    max_retries = 3
    backoff_base = 1.0
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            generated = call_model(
                client=client,
                model=model,
                title=title,
                abstract=abstract,
                keywords_text=keywords_text,
                target_score=target_score,
                degradation_severity=degradation_severity,
                degradation_types=degradation_types,
                abstract_style=abstract_style,
                temperature=model_temperature_for_score(target_score),
            )
            out["submission"] = generated["submission"]
            out["applied_degradations"] = generated["applied_degradations"]
            out["generation_model"] = model
            out["generation_status"] = "generated"
            out["generation_error"] = ""
            if sleep_s > 0:
                time.sleep(sleep_s)
            return out
        except Exception as exc:  # noqa: BLE001
            last_error = f"attempt {attempt}/{max_retries}: {exc}"
            if attempt < max_retries:
                time.sleep(backoff_base * (2 ** (attempt - 1)))

    out["submission"] = ""
    out["applied_degradations"] = []
    out["generation_model"] = model
    out["generation_status"] = "failed"
    out["generation_error"] = last_error
    return out


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    records = load_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    processed_ids: Set[str] = set()
    mode = "w"
    if args.resume and os.path.exists(args.output):
        processed_ids = load_processed_ids(args.output)
        mode = "a"

    client = OpenAI(api_key=api_key)

    processed_count = 0
    generated_count = 0
    failed_count = 0
    skipped_originals_count = 0

    status_counter: Counter = Counter()
    score_status_counter: Counter = Counter()

    with open(args.output, mode, encoding="utf-8") as fout:
        for rec in records:
            paper_id = str(rec.get("paper_id", "")).strip()
            if args.resume and paper_id and paper_id in processed_ids:
                continue

            out = process_record(client, rec, args.model, args.sleep)

            status = out.get("generation_status", "")
            score = out.get("target_score", "")

            processed_count += 1
            status_counter[status] += 1
            score_status_counter[(score, status)] += 1

            if status == "generated":
                generated_count += 1
            elif status == "failed":
                failed_count += 1
            elif status == "original_kept":
                skipped_originals_count += 1

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if processed_count % 50 == 0:
                print(
                    f"processed={processed_count} generated={generated_count} "
                    f"failed={failed_count} skipped_originals={skipped_originals_count}"
                )

    print("\n=== Final Generation Status Counts ===")
    for k, v in status_counter.items():
        print(f"{k}: {v}")

    print("\n=== Final Counts by target_score and generation_status ===")
    sorted_items = sorted(score_status_counter.items(), key=lambda x: (int(x[0][0]), x[0][1]))
    for (score, status), cnt in sorted_items:
        print(f"target_score={score}, status={status}: {cnt}")


if __name__ == "__main__":
    main()
