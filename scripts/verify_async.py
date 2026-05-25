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

PROMPT_TEMPLATE = """You are validating synthetic training examples for abstract evaluation.

Rows to validate:
{rows_json}

For each row, check:
1. submitted abstract preserves original topic
2. requested abstract-level flaws are present
3. score matches severity
4. rationale correctly describes submitted abstract
5. rationale avoids full-paper-only claims
6. no mention of synthetic data/degradation/original abstract

Return valid JSON only:
{{
  "items": [
    {{
      "row_id": "...",
      "validation_status": "PASS" or "FAIL",
      "validation_reason": "...",
      "detected_flaws": [...]
    }}
  ]
}}
"""


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/processed/test_scored_1000_run3.jsonl")
    p.add_argument("--output",default="data/processed/test_verified_1000_run3.jsonl")
    p.add_argument("--model",default="gpt-5-mini")
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--batch-size",type=int,default=5)
    p.add_argument("--concurrency",type=int,default=10)
    p.add_argument("--max-retries",type=int,default=3)
    p.add_argument("--timeout",type=float,default=120.0)
    p.add_argument("--timing-log",default="")
    return p.parse_args()


def load_jsonl(path: Path):
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for l in f:
            s=l.strip()
            if s: rows.append(json.loads(s))
    return rows


def append_jsonl(path: Path, rec: Dict[str,Any]):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def load_processed_ids(path: Path)->Set[str]:
    ids=set()
    if not path.exists(): return ids
    for l in path.open('r',encoding='utf-8'):
        s=l.strip()
        if not s: continue
        try:r=json.loads(s)
        except Exception: continue
        rid=str(r.get('row_id','')).strip()
        if rid: ids.add(rid)
    return ids


def batched(items,n):
    return [items[i:i+n] for i in range(0,len(items),n)]


def validate(raw: str, req:Set[str])->Tuple[Dict[str,Dict[str,Any]],str]:
    try: obj=json.loads(raw)
    except Exception as e: return {}, f"invalid_json:{e}"
    if not isinstance(obj,dict) or not isinstance(obj.get('items'),list): return {}, 'missing_items'
    out={}
    for it in obj['items']:
        if not isinstance(it,dict): continue
        rid=str(it.get('row_id','')).strip()
        if rid not in req: continue
        st=str(it.get('validation_status','')).strip().upper()
        rs=it.get('validation_reason')
        fl=it.get('detected_flaws')
        if st not in {'PASS','FAIL'}: continue
        if not isinstance(rs,str): continue
        if not isinstance(fl,list): continue
        out[rid]={"validation_status":st,"validation_reason":rs,"detected_flaws":fl}
    miss=req-set(out.keys())
    if miss: return out, 'missing_row_ids:'+','.join(sorted(miss))
    return out,''


async def call_batch(client,model,payload,timeout,retries):
    prompt=PROMPT_TEMPLATE.format(rows_json=json.dumps(payload,ensure_ascii=False))
    req={p['row_id'] for p in payload}
    last=''
    for a in range(1,retries+1):
        try:
            resp=await asyncio.wait_for(client.chat.completions.create(
                model=model,
                response_format={"type":"json_object"},
                messages=[{"role":"user","content":prompt}],
            ), timeout=timeout)
            content=resp.choices[0].message.content or ''
            out,err=validate(content,req)
            if err:
                last='validation_error:'+err
                if a<retries: await asyncio.sleep(2**(a-1)); continue
            return out,last
        except Exception as e:
            last=f"attempt {a}/{retries}: {e}"
            if a<retries: await asyncio.sleep(2**(a-1))
    return {}, last


async def process_batch(client,model,batch,sem,timeout,retries):
    async with sem:
        payload=[]
        for r in batch:
            payload.append({
                "row_id":r['row_id'],
                "title":str(r.get('title','')),
                "abstract":str(r.get('abstract','')),
                "submission":str(r.get('submission','')),
                "target_score":int(r.get('target_score',2)),
                "degradation_types":r.get('degradation_types',[]),
                "rationale":str(r.get('rationale','')),
            })
        out,err=await call_batch(client,model,payload,timeout,retries)
        rows=[]
        for r in batch:
            rid=r['row_id']; rec=dict(r)
            if rid in out:
                rec.update(out[rid])
                rec['validation_model']=model
                rec['validation_error']=''
            else:
                rec['validation_status']='FAIL'
                rec['validation_reason']='Verifier call failed or missing row output'
                rec['detected_flaws']=[]
                rec['validation_model']=model
                rec['validation_error']=err or f"missing_output_for_row_id:{rid}"
            rows.append(rec)
        return rows


async def main_async(args):
    inp=Path(args.input); outp=Path(args.output)
    rows=load_jsonl(inp)
    for i,r in enumerate(rows):
        rid=str(r.get('row_id','')).strip()
        if not rid:
            pid=str(r.get('paper_id','')).strip(); r['row_id']=pid if pid else f"row_{i:08d}"
    if args.limit is not None: rows=rows[:args.limit]
    outp.parent.mkdir(parents=True,exist_ok=True)
    done_ids=load_processed_ids(outp) if args.resume else set()
    rows=[r for r in rows if r['row_id'] not in done_ids]

    api_key=os.getenv('OPENAI_API_KEY')
    if not api_key: raise EnvironmentError('OPENAI_API_KEY is not set')
    client=AsyncOpenAI(api_key=api_key)
    sem=asyncio.Semaphore(max(1,args.concurrency))

    start=time.perf_counter(); c=Counter(); done=0
    tasks=[asyncio.create_task(process_batch(client,args.model,b,sem,args.timeout,args.max_retries)) for b in batched(rows,max(1,args.batch_size))]
    for t in asyncio.as_completed(tasks):
        for rec in await t:
            append_jsonl(outp,rec)
            c[rec.get('validation_status','FAIL')]+=1
            done+=1
    elapsed=time.perf_counter()-start

    print("\n=== Final validation_status counts ===")
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
                "pass":c.get('PASS',0),
                "fail":c.get('FAIL',0),
                "total_runtime_seconds":elapsed,
                "rows_per_second":(done/elapsed if elapsed>0 else 0),
                "average_seconds_per_row":(elapsed/done if done>0 else 0),
                "input":str(inp),
                "output":str(outp),
            },f,ensure_ascii=False,indent=2)


def main():
    args=parse_args(); asyncio.run(main_async(args))

if __name__=='__main__':
    main()
