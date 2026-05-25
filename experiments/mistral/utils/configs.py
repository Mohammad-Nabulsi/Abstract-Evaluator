from __future__ import annotations

from pathlib import Path

from experiments.utils.config import DataPaths, PipelineConfig


def build_mistral7_default_config(project_root: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    cfg.run_name = "mistral7b_abstract_evaluator_lora_l4_no_quant"
    cfg.data_paths = DataPaths(
        train_path=project_root / "data/data/train/all.jsonl",
        val_path=project_root / "data/data/val/all.jsonl",
        test_path=project_root / "data/data/test/all.jsonl",
    )
    cfg.output_root = project_root / "experiments/artifacts/abstract_evaluator_mistral7b_sft"
    cfg.wandb.project = "abstract-evaluator-mistral7b-sft"
    cfg.wandb.dir = project_root / "experiments/artifacts/wandb"
    cfg.wandb.tags = ["mistral7b", "lora", "sft", "abstract-evaluator", "l4-24gb", "no-quant"]

    # L4 (24GB VRAM) friendly defaults for full LoRA (no quantization).
    cfg.use_4bit = False
    cfg.max_seq_length = 2048
    cfg.train.per_device_train_batch_size = 2
    cfg.train.per_device_eval_batch_size = 4
    cfg.train.gradient_accumulation_steps = 4
    cfg.train.gradient_checkpointing = True
    cfg.generation.batch_size = cfg.train.per_device_eval_batch_size

    cfg.ensure_dirs()
    return cfg


def build_data_paths(
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> DataPaths:
    return DataPaths(train_path=train_path, val_path=val_path, test_path=test_path)

