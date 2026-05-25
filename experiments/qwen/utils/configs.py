from __future__ import annotations

from pathlib import Path

from experiments.utils.config import DataPaths, PipelineConfig, default_qwen_paths


def build_qwen3_default_config(project_root: Path) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.data_paths = default_qwen_paths(project_root)
    cfg.output_root = project_root / "experiments/artifacts/abstract_evaluator_qwen3_sft"
    cfg.wandb.dir = project_root / "experiments/artifacts/wandb"
    cfg.wandb.tags = ["qwen3", "lora", "sft", "abstract-evaluator"]
    cfg.ensure_dirs()
    return cfg


def build_data_paths(
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> DataPaths:
    return DataPaths(train_path=train_path, val_path=val_path, test_path=test_path)

