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

PROMPT_TEMPLATE = """You are a conference reviewer evaluating only the abstract, not the full paper.

Rows to score:
{rows_json}

Task for each row:
- Evaluate the submitted abstract quality for conference acceptance.
- Strong abstracts clearly present problem, methodology, contribution, and evidence.

Rubric:
0 = Strong reject
1 = Reject
2 = Borderline
3 = Accept
4 = Strong accept

Rules:
1. Rationale must match target score.
2. Discuss only abstract quality.
3. Mention visible flaws in submitted abstract.
4. Do not mention original abstract.
5. Do not mention synthetic data or degradation.
6. Keep rationale 2-5 sentences.
7. Keep tone professional.

Return valid JSON only:
{{
  "items": [
    {{
      "row_id": "...",
      "score": 0,
      "rationale": "..."
    }}
  ]
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/test_generated_1000_run3.jsonl")
    p.add_argument("--output", default="data/processed/test_scored_1000_run3.jsonl")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--timing-log", default="")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for i,l in enumerate(f,1):
            s=l.strip()
            if not s: continue
            rows.append(json.loads(s))
    return rows


def append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        f.flush(); os.fsync(f.fileno())


def load_processed_ids(path: Path) -> Set[str]:
    ids=set()
    if not path.exists(): return ids
    with path.open("r",encoding="utf-8") as f:
        for l in f:
            s=l.strip()
            if not s: continue
            try:r=json.loads(s)
            except Exception: continue
            rid=str(r.get("row_id","")).strip()
            if rid: ids.add(rid)
    return ids


def parse_degradation_types(value: Any) -> List[str]:
    if value is None: return []
    if isinstance(value,list): return [str(v) for v in value]
    if isinstance(value,str):
        t=value.strip()
        if not t: return []
        try:
            p=json.loads(t)
            if isinstance(p,list): return [str(v) for v in p]
        except Exception:
            pass
        return [t]
    return []


def batched(items: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i+n] for i in range(0,len(items),n)]


def validate(raw: str, requested: Set[str]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    try:
        obj=json.loads(raw)
    except Exception as e:
        return {}, f"invalid_json:{e}"
    if not isinstance(obj,dict) or not isinstance(obj.get("items"),list):
        return {}, "missing_items"
    out={}
    for it in obj["items"]:
        if not isinstance(it,dict):
            continue
        rid=str(it.get("row_id","")).strip()
        if rid not in requested:
            continue
        score=it.get("score")
        rationale=it.get("rationale")
        if not isinstance(rationale,str) or not rationale.strip():
            continue
        try:
            score=int(score)
        except Exception:
            continue
        score=max(0,min(4,score))
        out[rid]={"score":score,"rationale":rationale.strip()}
    missing=requested-set(out.keys())
    if missing:
        return out, "missing_row_ids:"+",".join(sorted(missing))
    return out, ""


async def call_batch(client: AsyncOpenAI, model: str, rows_payload: List[Dict[str, Any]], timeout: float, retries: int):
    prompt=PROMPT_TEMPLATE.format(rows_json=json.dumps(rows_payload,ensure_ascii=False))
    last=""
    req={r["row_id"] for r in rows_payload}
    for a in range(1,retries+1):
        try:
            resp=await asyncio.wait_for(client.chat.completions.create(
                model=model,
                response_format={"type":"json_object"},
                messages=[{"role":"user","content":prompt}],
            ), timeout=timeout)
            content=resp.choices[0].message.content or ""
            out,err=validate(content,req)
            if err:
                last=f"validation_error:{err}"
                if a<retries: await asyncio.sleep(2**(a-1)); continue
            return out,last
        except Exception as e:
            last=f"attempt {a}/{retries}: {e}"
            if a<retries: await asyncio.sleep(2**(a-1))
    return {}, last


async def process_batch(client, model, batch, sem, timeout, retries):
    async with sem:
        payload=[]
        for r in batch:
            payload.append({
                "row_id":r["row_id"],
                "title":str(r.get("title","")),
                "submission":str(r.get("submission","")),
                "target_score":int(r.get("target_score",2)),
                "degradation_types":parse_degradation_types(r.get("degradation_types")),
            })
        out,err=await call_batch(client,model,payload,timeout,retries)
        rows=[]
        for r in batch:
            rid=r["row_id"]
            rec=dict(r)
            if rid in out:
                rec["score"]=out[rid]["score"]
                rec["rationale"]=out[rid]["rationale"]
                rec["scoring_model"]=model
                rec["scoring_status"]="scored"
                rec["scoring_error"]=""
            else:
                rec["score"]=int(r.get("target_score",2))
                rec["rationale"]=""
                rec["scoring_model"]=model
                rec["scoring_status"]="failed"
                rec["scoring_error"]=err or f"missing_output_for_row_id:{rid}"
            rows.append(rec)
        return rows


async def main_async(args):
    inp=Path(args.input); outp=Path(args.output)
    rows=load_jsonl(inp)
    for i,r in enumerate(rows):
        rid=str(r.get("row_id","")).strip()
        if not rid:
            pid=str(r.get("paper_id","")).strip()
            r["row_id"]=pid if pid else f"row_{i:08d}"
    if args.limit is not None:
        rows=rows[:args.limit]
    outp.parent.mkdir(parents=True,exist_ok=True)
    processed=load_processed_ids(outp) if args.resume else set()
    rows=[r for r in rows if r["row_id"] not in processed]

    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key: raise EnvironmentError("OPENAI_API_KEY is not set")
    client=AsyncOpenAI(api_key=api_key)
    sem=asyncio.Semaphore(max(1,args.concurrency))

    start=time.perf_counter()
    c=Counter(); done=0
    tasks=[asyncio.create_task(process_batch(client,args.model,b,sem,args.timeout,args.max_retries)) for b in batched(rows,max(1,args.batch_size))]
    for t in asyncio.as_completed(tasks):
        for rec in await t:
            append_jsonl(outp,rec)
            st=rec.get("scoring_status","failed")
            c[st]+=1; done+=1
    elapsed=time.perf_counter()-start

    print("\n=== Final scoring_status counts ===")
    for k,v in c.items(): print(f"{k}: {v}")
    print(f"processed={done} runtime_seconds={elapsed:.3f}")

    if args.timing_log:
        tp=Path(args.timing_log); tp.parent.mkdir(parents=True,exist_ok=True)
        with tp.open('w',encoding='utf-8') as f:
            json.dump({
                "timestamp":datetime.now(timezone.utc).isoformat(),
                "model":args.model,
                "batch_size":args.batch_size,
                "concurrency":args.concurrency,
                "rows_processed":done,
                "scored":c.get("scored",0),
                "failed":c.get("failed",0),
                "total_runtime_seconds":elapsed,
                "rows_per_second":(done/elapsed if elapsed>0 else 0),
                "average_seconds_per_row":(elapsed/done if done>0 else 0),
                "input":str(inp),
                "output":str(outp),
            },f,ensure_ascii=False,indent=2)


def main():
    args=parse_args()
    asyncio.run(main_async(args))

if __name__=="__main__":
    main()
