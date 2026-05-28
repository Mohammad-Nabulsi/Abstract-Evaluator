from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments


def compute_score_only_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = labels.astype(int)

    accuracy = float((preds == labels).mean())
    mae = float(np.abs(preds - labels).mean())
    within_1 = float((np.abs(preds - labels) <= 1).mean())
    rmse = float(np.sqrt(((preds - labels) ** 2).mean()))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    return {
        "accuracy": accuracy,
        "mae": mae,
        "within_1_accuracy": within_1,
        "rmse": rmse,
        "macro_f1": macro_f1,
    }


def _has_usable_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.device_count() > 0
    except Exception:
        return False


def _upcast_trainable_fp16_params_to_fp32(model) -> int:
    upcasted = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
            upcasted += 1
    return upcasted


def _compute_balanced_class_weights(train_labels: np.ndarray, num_labels: int) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=num_labels).astype(np.float32)
    total = float(counts.sum())
    nonzero = counts > 0

    weights = np.zeros_like(counts, dtype=np.float32)
    if nonzero.any() and total > 0:
        weights[nonzero] = total / (float(nonzero.sum()) * counts[nonzero])
        weights[nonzero] /= float(weights[nonzero].mean())

    return torch.tensor(weights, dtype=torch.float32)


class WeightedClassificationTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is None or self.class_weights is None:
            loss = outputs.get("loss")
            if loss is None:
                raise RuntimeError("Model output did not include loss and labels/class_weights were unavailable.")
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device, dtype=logits.dtype))
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def _build_training_arguments(cfg, output_dir: Path) -> TrainingArguments:
    use_cuda_amp = _has_usable_cuda()
    args_dict: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        max_grad_norm=cfg.train.max_grad_norm,
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        fp16=bool(cfg.train.fp16 and use_cuda_amp),
        bf16=bool(cfg.train.bf16 and use_cuda_amp),
        report_to="wandb" if cfg.wandb_enabled else "none",
        run_name=cfg.run_name,
        seed=cfg.seed,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        args_dict["eval_strategy"] = cfg.train.eval_strategy
    else:
        args_dict["evaluation_strategy"] = cfg.train.eval_strategy

    if cfg.train.eval_strategy == "steps":
        args_dict["eval_steps"] = cfg.train.eval_steps
    else:
        args_dict["eval_steps"] = None

    return TrainingArguments(**args_dict)


def train_deberta(
    cfg,
    ds,
    model,
    tokenizer,
) -> Tuple[Trainer, Dict[str, Any]]:
    output_dir = cfg.output_root / "models" / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # AMP unscale expects trainable params to remain fp32.
    if bool(cfg.train.fp16):
        upcasted = _upcast_trainable_fp16_params_to_fp32(model)
        if upcasted > 0:
            warnings.warn(
                f"Upcasted {upcasted} trainable fp16 parameter tensors to fp32 for stable AMP training.",
                RuntimeWarning,
                stacklevel=2,
            )

    training_args = _build_training_arguments(cfg, output_dir=output_dir)
    class_weights = None
    if bool(getattr(cfg.train, "use_class_weights", False)):
        train_labels = np.asarray(ds["train"]["labels"], dtype=np.int64)
        class_weights = _compute_balanced_class_weights(train_labels, num_labels=int(cfg.num_labels))

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.train.early_stopping_patience,
            early_stopping_threshold=cfg.train.early_stopping_threshold,
        )
    ]
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=compute_score_only_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedClassificationTrainer(**trainer_kwargs)

    trainer.train()
    eval_metrics = trainer.evaluate(ds["validation"])

    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    run_info = {
        "model_name": cfg.model_name,
        "run_name": cfg.run_name,
        "output_dir": str(output_dir),
        "best_model_dir": str(best_model_dir),
        "validation_metrics": eval_metrics,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    return trainer, run_info
