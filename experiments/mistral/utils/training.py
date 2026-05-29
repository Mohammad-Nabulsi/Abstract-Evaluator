from __future__ import annotations

import gc
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
import wandb
from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
from trl import SFTTrainer
from trl.trainer.base_trainer import BaseTrainer

from experiments.utils.evaluation import compute_eval_metrics, parse_rationale, parse_score
from experiments.utils.generation import generate_predictions_from_messages
from .chat import build_formatting_func, make_inference_prompt


class SafeSFTTrainer(SFTTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        # Keep standard training loss behavior but skip TRL's extra logits-based
        # metrics hooks that can break with some Unsloth logits wrappers.
        mode = "train" if self.model.training else "eval"
        inputs["use_cache"] = False
        loss, outputs = BaseTrainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
                self._total_train_tokens += num_tokens
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(
                    inputs["position_ids"].size(1),
                    device=inputs["position_ids"].device,
                )
                num_tokens = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
                self._total_train_tokens += num_tokens
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        return (loss, outputs) if return_outputs else loss


def _pretokenize_messages_dataset(dataset, tokenizer, max_seq_length: int) -> Dataset:
    rows = []
    eos = tokenizer.eos_token or ""
    for ex in dataset:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        if eos and not text.endswith(eos):
            text += eos
        toks = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        rows.append(
            {
                "input_ids": toks["input_ids"],
                "attention_mask": toks.get("attention_mask", [1] * len(toks["input_ids"])),
            }
        )
    return Dataset.from_list(rows)


class EpochCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        save_root: Path,
        save_epochs: Optional[set] = None,
        save_every_n_epochs: int = 1,
    ):
        self.save_root = save_root
        self.save_epochs = save_epochs
        self.save_every_n_epochs = max(1, int(save_every_n_epochs))

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        epoch_num = int(round(state.epoch or 0))
        if epoch_num <= 0:
            return

        if self.save_epochs is not None:
            if epoch_num not in self.save_epochs:
                return
        elif self.save_every_n_epochs > 1 and (epoch_num % self.save_every_n_epochs) != 0:
            return

        ckpt_dir = self.save_root / f"epoch_{epoch_num}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if model is not None:
            model.save_pretrained(str(ckpt_dir))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(ckpt_dir))


class EpochEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        eval_out_dir: Path,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        max_seq_length: int,
        max_new_tokens: int,
        batch_size: int,
        include_bertscore: bool,
        use_wandb: bool,
        eval_every_n_epochs: int = 1,
        logger=None,
    ):
        self.eval_out_dir = eval_out_dir
        self.eval_out_dir.mkdir(parents=True, exist_ok=True)
        self.val_df = val_df
        self.test_df = test_df
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.include_bertscore = include_bertscore
        self.use_wandb = use_wandb
        self.eval_every_n_epochs = max(1, int(eval_every_n_epochs))
        self.logger = logger
        self.epoch_metrics = []

    def _evaluate_split(self, split_df: pd.DataFrame, split_name: str, tag: str, model, tokenizer):
        pred_df = generate_predictions_from_messages(
            eval_df=split_df,
            model=model,
            tokenizer=tokenizer,
            make_inference_prompt_fn=make_inference_prompt,
            parse_score_fn=parse_score,
            parse_rationale_fn=parse_rationale,
            max_seq_length=self.max_seq_length,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            logger=self.logger,
        )
        pred_df.to_csv(self.eval_out_dir / f"{split_name}_predictions_{tag}.csv", index=False)
        metrics = compute_eval_metrics(pred_df, include_bertscore=self.include_bertscore)
        return metrics

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        epoch_num = int(round(state.epoch or 0))
        if epoch_num <= 0:
            return
        if (epoch_num % self.eval_every_n_epochs) != 0:
            return

        tag = f"epoch_{epoch_num}"
        if model is None or tokenizer is None:
            return

        model.eval()
        val_metrics = self._evaluate_split(self.val_df, "validation", tag, model, tokenizer)

        metrics = {
            "checkpoint_tag": tag,
            "epoch": epoch_num,
            **{f"validation/{k}": v for k, v in val_metrics.items()},
        }

        if self.test_df is not None and len(self.test_df) > 0:
            test_metrics = self._evaluate_split(self.test_df, "test", tag, model, tokenizer)
            metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

        metrics_path = self.eval_out_dir / f"metrics_{tag}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        self.epoch_metrics.append(metrics)
        pd.DataFrame(self.epoch_metrics).to_csv(
            self.eval_out_dir / "all_checkpoint_metrics.csv",
            index=False,
        )

        if self.use_wandb:
            wandb.log({f"epoch_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

        if self.logger is not None:
            self.logger.info("Saved per-epoch eval metrics for %s to %s", tag, metrics_path)


def train_mistral7(
    cfg,
    ds,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_bertscore_for_epoch_eval: bool = False,
    run_epoch_generation_eval: bool = False,
    run_epoch_test_eval: bool = False,
    checkpoint_every_n_epochs: int = 1,
    generation_eval_every_n_epochs: int = 1,
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
    logger=None,
) -> Dict[str, Any]:
    from .modeling import load_mistral_model_for_training

    output_dir = cfg.output_root / "models" / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unsloth + TRL SFT can require raw logits during training metrics/loss hooks.
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    if cfg.wandb.enabled:
        if cfg.wandb.dir is not None:
            cfg.wandb.dir.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            tags=cfg.wandb.tags,
            group=cfg.wandb.run_group,
            config={
                "model_name": cfg.model_name,
                "max_seq_length": cfg.max_seq_length,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                **cfg.train.__dict__,
                **{f"lora_{k}": v for k, v in cfg.lora_cfg.items() if k != "target_modules"},
                "target_modules": cfg.lora_cfg["target_modules"],
                "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
                "generation_eval_every_n_epochs": generation_eval_every_n_epochs,
                "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
            },
            reinit=True,
            dir=str(cfg.wandb.dir) if cfg.wandb.dir else None,
        )

    model, tokenizer, freeze_info = load_mistral_model_for_training(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        lora_cfg=cfg.lora_cfg,
        seed=cfg.seed,
        dtype=torch.bfloat16,
        load_in_4bit=getattr(cfg, "use_4bit", False),
    )
    formatting_func = build_formatting_func(tokenizer)

    epoch_ckpt_root = output_dir / "epoch_adapters"
    epoch_ckpt_root.mkdir(parents=True, exist_ok=True)
    eval_out_dir = cfg.output_root / "eval" / cfg.run_name
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    overwrite_output_dir = resume_from_checkpoint is None

    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        logging_steps=cfg.train.logging_steps,
        eval_strategy=cfg.train.eval_strategy,
        eval_steps=cfg.train.eval_steps,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        report_to="wandb" if cfg.wandb.enabled else "none",
        run_name=cfg.run_name,
        seed=cfg.seed,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        dataloader_pin_memory=getattr(cfg.train, "dataloader_pin_memory", True),
        auto_find_batch_size=getattr(cfg.train, "auto_find_batch_size", True),
        tf32=getattr(cfg.train, "tf32", True),
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        group_by_length=cfg.train.group_by_length,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.train.early_stopping_patience,
            early_stopping_threshold=cfg.train.early_stopping_threshold,
        ),
        EpochCheckpointCallback(
            epoch_ckpt_root,
            save_every_n_epochs=checkpoint_every_n_epochs,
        ),
    ]

    if run_epoch_generation_eval:
        callbacks.append(
            EpochEvaluationCallback(
                eval_out_dir=eval_out_dir,
                val_df=val_df,
                test_df=test_df if run_epoch_test_eval else None,
                max_seq_length=cfg.max_seq_length,
                max_new_tokens=cfg.generation.max_new_tokens,
                batch_size=cfg.train.per_device_eval_batch_size,
                include_bertscore=include_bertscore_for_epoch_eval,
                use_wandb=cfg.wandb.enabled,
                eval_every_n_epochs=generation_eval_every_n_epochs,
                logger=logger,
            )
        )

    train_dataset_for_trainer = ds["train"]
    eval_dataset_for_trainer = ds["validation"]
    sft_kwargs = {
        "model": model,
        "args": args,
        "callbacks": callbacks,
    }
    sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_sig:
        # Work around TRL/Unsloth pickle failures during dataset preprocessing by
        # pre-tokenizing ourselves and skipping TRL's internal prepare step.
        from trl import SFTConfig

        train_dataset_for_trainer = _pretokenize_messages_dataset(ds["train"], tokenizer, cfg.max_seq_length)
        eval_dataset_for_trainer = _pretokenize_messages_dataset(ds["validation"], tokenizer, cfg.max_seq_length)

        if not isinstance(args, SFTConfig):
            args = SFTConfig(**args.to_dict())
        args.dataset_kwargs = {"skip_prepare_dataset": True}
        args.max_length = None
        args.dataset_num_proc = None
        sft_kwargs["args"] = args
        sft_kwargs["train_dataset"] = train_dataset_for_trainer
        sft_kwargs["eval_dataset"] = eval_dataset_for_trainer
        sft_kwargs["processing_class"] = tokenizer
    else:
        sft_kwargs["train_dataset"] = train_dataset_for_trainer
        sft_kwargs["eval_dataset"] = eval_dataset_for_trainer
        sft_kwargs["formatting_func"] = formatting_func
        if "tokenizer" in sft_sig:
            sft_kwargs["tokenizer"] = tokenizer
        if "max_seq_length" in sft_sig:
            sft_kwargs["max_seq_length"] = cfg.max_seq_length

    trainer = SafeSFTTrainer(**sft_kwargs)

    start = time.time()
    resume_arg = str(resume_from_checkpoint) if resume_from_checkpoint else None
    if resume_arg and logger is not None:
        logger.info("Resuming trainer from checkpoint: %s", resume_arg)
    train_result = trainer.train(resume_from_checkpoint=resume_arg)
    elapsed_min = (time.time() - start) / 60

    adapter_dir = output_dir / "best_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = train_result.metrics
    metrics["train_minutes"] = elapsed_min
    metrics.update(
        {
            f"freeze_{k}": v
            for k, v in freeze_info.items()
            if isinstance(v, (int, float))
        }
    )

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    freeze_info_path = output_dir / "freeze_info.json"
    with freeze_info_path.open("w", encoding="utf-8") as f:
        json.dump(freeze_info, f, indent=2, ensure_ascii=False)

    if cfg.wandb.enabled:
        wandb.log({"train_minutes": elapsed_min})
        wandb.finish()

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_info = {
        "model_name": cfg.model_name,
        "run_name": cfg.run_name,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "epoch_adapter_dir": str(epoch_ckpt_root),
        "eval_dir": str(eval_out_dir),
        "resumed_from": str(resume_arg) if resume_arg else None,
    }

    run_info_path = output_dir / "run_info.json"
    with run_info_path.open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    if logger is not None:
        logger.info("Training completed. Best adapter at %s", adapter_dir)

    return run_info

