#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from openai import AsyncOpenAI

PROMPT_TEMPLATE = """You are generating synthetic training data for a research abstract evaluation model.

You must process multiple rows and return one JSON object only.

Rows to process:
{rows_json}

Instructions for each row:
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
  "items": [
    {{
      "row_id": "...",
      "submission": "...",
      "applied_degradations": [...]
    }}
  ]
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async batched synthetic abstract generation.")
    p.add_argument("--input", default="data/processed/abstracts_with_degradation_plan.jsonl")
    p.add_argument("--output", default="data/processed/abstracts_with_generated_submissions.jsonl")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--test-sample-output", default="")
    p.add_argument("--timing-log", default="")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{i}: {exc}") from exc
    return rows


def append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_processed_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("row_id", "")).strip()
            if rid:
                ids.add(rid)
    return ids


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


def make_row_id(rec: Dict[str, Any], idx: int) -> str:
    pid = str(rec.get("paper_id", "")).strip()
    return pid if pid else f"row_{idx:08d}"


def mark_failed(rec: Dict[str, Any], model: str, err: str) -> Dict[str, Any]:
    out = dict(rec)
    out["submission"] = ""
    out["applied_degradations"] = []
    out["generation_model"] = model
    out["generation_status"] = "failed"
    out["generation_error"] = err
    return out


def mark_original(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out["submission"] = str(rec.get("abstract", ""))
    out["applied_degradations"] = []
    out["generation_model"] = "none"
    out["generation_status"] = "original_kept"
    out["generation_error"] = ""
    return out


def validate_batch_response(raw: str, requested_ids: Set[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    errs: List[str] = []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, [f"invalid_json: {exc}"]

    if not isinstance(parsed, dict):
        return {}, ["response_not_object"]
    items = parsed.get("items")
    if not isinstance(items, list):
        return {}, ["missing_items_list"]

    mapped: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            errs.append("item_not_object")
            continue
        rid = str(item.get("row_id", "")).strip()
        if rid not in requested_ids:
            errs.append(f"unknown_row_id:{rid}")
            continue

        submission = item.get("submission")
        applied = item.get("applied_degradations")
        if not isinstance(submission, str) or not submission.strip():
            errs.append(f"invalid_submission:{rid}")
            continue
        if not isinstance(applied, list):
            errs.append(f"invalid_applied_degradations:{rid}")
            continue

        mapped[rid] = {
            "submission": submission,
            "applied_degradations": applied,
        }

    missing = sorted(requested_ids - set(mapped.keys()))
    for rid in missing:
        errs.append(f"missing_row_id:{rid}")

    return mapped, errs


async def call_batch(
    client: AsyncOpenAI,
    model: str,
    rows_payload: List[Dict[str, Any]],
    timeout_s: float,
    max_retries: int,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    prompt = PROMPT_TEMPLATE.format(rows_json=json.dumps(rows_payload, ensure_ascii=False))

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=timeout_s,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty_response_content")
            requested = {str(r["row_id"]) for r in rows_payload}
            mapped, errs = validate_batch_response(content, requested)
            if errs:
                last_error = f"validation_error: {';'.join(errs)}"
                if attempt < max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
                return mapped, last_error
            return mapped, ""
        except Exception as exc:  # noqa: BLE001
            last_error = f"attempt {attempt}/{max_retries}: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(2 ** (attempt - 1))

    return {}, last_error


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    batch: List[Dict[str, Any]],
    timeout_s: float,
    max_retries: int,
    sem: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    async with sem:
        rows_payload: List[Dict[str, Any]] = []
        for rec in batch:
            rows_payload.append(
                {
                    "row_id": rec["row_id"],
                    "title": str(rec.get("title", "")),
                    "abstract": str(rec.get("abstract", "")),
                    "keywords": build_keywords_text(rec.get("keywords")),
                    "target_score": int(rec.get("target_score", 2)),
                    "degradation_severity": str(rec.get("degradation_severity", "")),
                    "degradation_types": parse_degradation_types(rec.get("degradation_types")),
                    "abstract_style": str(rec.get("abstract_style", "standard_academic_style")),
                }
            )

        mapped, batch_err = await call_batch(
            client=client,
            model=model,
            rows_payload=rows_payload,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

        out_rows: List[Dict[str, Any]] = []
        for rec in batch:
            rid = rec["row_id"]
            if rid in mapped:
                out = dict(rec)
                out["submission"] = mapped[rid]["submission"]
                out["applied_degradations"] = mapped[rid]["applied_degradations"]
                out["generation_model"] = model
                out["generation_status"] = "generated"
                out["generation_error"] = ""
                out_rows.append(out)
            else:
                err = batch_err or f"missing_output_for_row_id:{rid}"
                out_rows.append(mark_failed(rec, model, err))

        return out_rows


def batched(items: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    records = load_jsonl(input_path)

    # attach stable row_id
    for i, rec in enumerate(records):
        rec["row_id"] = make_row_id(rec, i)

    if args.limit is not None:
        records = records[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids: Set[str] = set()
    if args.resume:
        processed_ids = load_processed_ids(output_path)

    selected = [r for r in records if r["row_id"] not in processed_ids]

    if args.test_sample_output:
        p = Path(args.test_sample_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in selected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    start = time.perf_counter()

    status_counter: Counter = Counter()
    generated_count = 0
    failed_count = 0
    original_kept_count = 0
    processed_count = 0

    # First, handle score==4 rows immediately
    model_rows: List[Dict[str, Any]] = []
    for rec in selected:
        score = int(rec.get("target_score", 2))
        if score == 4:
            out = mark_original(rec)
            append_jsonl(output_path, out)
            processed_count += 1
            original_kept_count += 1
            status_counter[out["generation_status"]] += 1
        else:
            model_rows.append(rec)

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    batches = batched(model_rows, max(1, args.batch_size))
    tasks = [
        asyncio.create_task(
            process_batch(
                client=client,
                model=args.model,
                batch=b,
                timeout_s=args.timeout,
                max_retries=args.max_retries,
                sem=sem,
            )
        )
        for b in batches
    ]

    for coro in asyncio.as_completed(tasks):
        out_rows = await coro
        for out in out_rows:
            append_jsonl(output_path, out)
            processed_count += 1
            st = out.get("generation_status", "failed")
            status_counter[st] += 1
            if st == "generated":
                generated_count += 1
            elif st == "failed":
                failed_count += 1

    elapsed = time.perf_counter() - start
    rows_per_sec = (processed_count / elapsed) if elapsed > 0 else 0.0
    avg_sec_per_row = (elapsed / processed_count) if processed_count > 0 else 0.0

    print("\n=== Final Generation Status Counts ===")
    for k, v in status_counter.items():
        print(f"{k}: {v}")

    print("\n=== Summary ===")
    print(f"processed_rows={processed_count}")
    print(f"generated={generated_count}")
    print(f"failed={failed_count}")
    print(f"original_kept={original_kept_count}")
    print(f"runtime_seconds={elapsed:.3f}")
    print(f"rows_per_second={rows_per_sec:.3f}")

    if args.timing_log:
        timing_path = Path(args.timing_log)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "batch_size": args.batch_size,
            "concurrency": args.concurrency,
            "total_runtime_seconds": elapsed,
            "rows_processed": processed_count,
            "generated": generated_count,
            "failed": failed_count,
            "original_kept": original_kept_count,
            "rows_per_second": rows_per_sec,
            "average_seconds_per_row": avg_sec_per_row,
            "input": str(input_path),
            "output": str(output_path),
        }
        with timing_path.open("w", encoding="utf-8") as f:
            json.dump(timing, f, ensure_ascii=False, indent=2)
        print(f"timing_log_saved={timing_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
