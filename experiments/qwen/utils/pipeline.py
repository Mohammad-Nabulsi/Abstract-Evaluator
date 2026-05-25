from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import wandb

from experiments.utils.evaluation import compute_eval_metrics, parse_rationale, parse_score
from experiments.utils.generation import generate_predictions_from_messages
from .chat import estimate_token_lengths, make_inference_prompt
from .modeling import load_qwen3_base_model, load_qwen3_model_for_inference


def estimate_qwen_token_percentiles(
    model_name: str,
    max_seq_length: int,
    df: pd.DataFrame,
    percentiles: Optional[List[float]] = None,
) -> Dict[str, int]:
    percentiles = percentiles or [0.5, 0.75, 0.9, 0.95, 1.0]
    model, tokenizer = load_qwen3_base_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    lengths = estimate_token_lengths(df, tokenizer)
    out = {f"p{int(p * 100):02d}": int(float(pd.Series(lengths).quantile(p))) for p in percentiles}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def evaluate_single_adapter(
    cfg,
    adapter_dir: Optional[Path],
    tag: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_bertscore: bool = False,
    use_wandb: Optional[bool] = None,
    logger=None,
) -> Dict[str, Any]:
    use_wandb = cfg.wandb.enabled if use_wandb is None else use_wandb
    eval_out_dir = cfg.output_root / "eval" / cfg.run_name
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_qwen3_model_for_inference(
        model_name=cfg.model_name,
        adapter_dir=adapter_dir,
        max_seq_length=cfg.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    if logger is not None:
        logger.info("Generating validation predictions for %s", tag)
    val_pred = generate_predictions_from_messages(
        eval_df=val_df,
        model=model,
        tokenizer=tokenizer,
        make_inference_prompt_fn=make_inference_prompt,
        parse_score_fn=parse_score,
        parse_rationale_fn=parse_rationale,
        max_seq_length=cfg.max_seq_length,
        max_new_tokens=cfg.generation.max_new_tokens,
        batch_size=cfg.generation.batch_size,
        do_sample=cfg.generation.do_sample,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        logger=logger,
    )
    val_pred.to_csv(eval_out_dir / f"validation_predictions_{tag}.csv", index=False)
    val_metrics = compute_eval_metrics(val_pred, include_bertscore=include_bertscore)
    val_metrics = {f"validation/{k}": v for k, v in val_metrics.items()}

    if logger is not None:
        logger.info("Generating test predictions for %s", tag)
    test_pred = generate_predictions_from_messages(
        eval_df=test_df,
        model=model,
        tokenizer=tokenizer,
        make_inference_prompt_fn=make_inference_prompt,
        parse_score_fn=parse_score,
        parse_rationale_fn=parse_rationale,
        max_seq_length=cfg.max_seq_length,
        max_new_tokens=cfg.generation.max_new_tokens,
        batch_size=cfg.generation.batch_size,
        do_sample=cfg.generation.do_sample,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        logger=logger,
    )
    test_pred.to_csv(eval_out_dir / f"test_predictions_{tag}.csv", index=False)
    test_metrics = compute_eval_metrics(test_pred, include_bertscore=include_bertscore)
    test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}

    metrics = {
        "run_name": cfg.run_name,
        "model_name": cfg.model_name,
        "checkpoint_tag": tag,
        **val_metrics,
        **test_metrics,
    }

    metrics_path = eval_out_dir / f"metrics_{tag}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.run_name}_{tag}_eval",
            reinit=True,
            dir=str(cfg.wandb.dir) if cfg.wandb.dir else None,
        )
        wandb.log(metrics)
        wandb.finish()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def evaluate_saved_adapters(
    cfg,
    run_info: Dict[str, Any],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_bertscore: bool = False,
    include_best_adapter: bool = True,
    logger=None,
) -> pd.DataFrame:
    all_metrics = []

    if include_best_adapter:
        best_adapter = Path(run_info["adapter_dir"])
        if best_adapter.exists():
            all_metrics.append(
                evaluate_single_adapter(
                    cfg=cfg,
                    adapter_dir=best_adapter,
                    tag="best",
                    val_df=val_df,
                    test_df=test_df,
                    include_bertscore=include_bertscore,
                    logger=logger,
                )
            )
        elif logger is not None:
            logger.warning("Best adapter not found: %s", best_adapter)

    epoch_root = Path(run_info["epoch_adapter_dir"])
    epoch_dirs = sorted(
        [p for p in epoch_root.glob("epoch_*") if p.is_dir()],
        key=lambda p: int(re.search(r"epoch_(\d+)$", p.name).group(1))
        if re.search(r"epoch_(\d+)$", p.name)
        else 10**9,
    )

    for ep_dir in epoch_dirs:
        all_metrics.append(
            evaluate_single_adapter(
                cfg=cfg,
                adapter_dir=ep_dir,
                tag=ep_dir.name,
                val_df=val_df,
                test_df=test_df,
                include_bertscore=include_bertscore,
                logger=logger,
            )
        )

    metrics_df = pd.DataFrame(all_metrics)
    eval_out_dir = cfg.output_root / "eval" / cfg.run_name
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(eval_out_dir / "all_checkpoint_metrics.csv", index=False)
    return metrics_df


def evaluate_base_model(
    cfg,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_bertscore: bool = False,
    logger=None,
) -> Dict[str, Any]:
    base_eval_dir = cfg.output_root / "eval" / f"{cfg.run_name}_base_no_finetune"
    base_eval_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_qwen3_model_for_inference(
        model_name=cfg.model_name,
        adapter_dir=None,
        max_seq_length=cfg.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    if logger is not None:
        logger.info("Generating validation predictions for base model")
    val_pred = generate_predictions_from_messages(
        eval_df=val_df,
        model=model,
        tokenizer=tokenizer,
        make_inference_prompt_fn=make_inference_prompt,
        parse_score_fn=parse_score,
        parse_rationale_fn=parse_rationale,
        max_seq_length=cfg.max_seq_length,
        max_new_tokens=cfg.generation.max_new_tokens,
        batch_size=cfg.generation.batch_size,
        do_sample=cfg.generation.do_sample,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        logger=logger,
    )
    val_pred.to_csv(base_eval_dir / "validation_predictions_base_no_finetune.csv", index=False)
    val_metrics = compute_eval_metrics(val_pred, include_bertscore=include_bertscore)
    val_metrics = {f"validation/{k}": v for k, v in val_metrics.items()}

    if logger is not None:
        logger.info("Generating test predictions for base model")
    test_pred = generate_predictions_from_messages(
        eval_df=test_df,
        model=model,
        tokenizer=tokenizer,
        make_inference_prompt_fn=make_inference_prompt,
        parse_score_fn=parse_score,
        parse_rationale_fn=parse_rationale,
        max_seq_length=cfg.max_seq_length,
        max_new_tokens=cfg.generation.max_new_tokens,
        batch_size=cfg.generation.batch_size,
        do_sample=cfg.generation.do_sample,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        logger=logger,
    )
    test_pred.to_csv(base_eval_dir / "test_predictions_base_no_finetune.csv", index=False)
    test_metrics = compute_eval_metrics(test_pred, include_bertscore=include_bertscore)
    test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}

    metrics = {
        "run_name": f"{cfg.run_name}_base_no_finetune",
        "model_name": cfg.model_name,
        "checkpoint_tag": "base_no_finetune",
        **val_metrics,
        **test_metrics,
    }

    with (base_eval_dir / "metrics_base_no_finetune.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.run_name}_base_no_finetune_eval",
            reinit=True,
            dir=str(cfg.wandb.dir) if cfg.wandb.dir else None,
        )
        wandb.log(metrics)
        wandb.finish()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics

