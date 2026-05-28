from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from experiments.utils.config import DataPaths, default_qwen_paths


@dataclass
class DebertaTrainConfig:
    num_train_epochs: int = 6
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_mae"
    greater_is_better: bool = False
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    dataloader_num_workers: int = 2
    fp16: bool = True
    bf16: bool = False
    use_class_weights: bool = True


@dataclass
class DebertaLoraConfig:
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: list[str] = field(default_factory=lambda: ["query_proj", "value_proj"])
    freeze_lower_n_when_no_lora: int = 8


@dataclass
class DebertaPipelineConfig:
    seed: int = 3407
    model_name: str = "microsoft/deberta-v3-base"
    run_name: str = "deberta_v3_base_abstract_evaluator_lora_score_only_3e-5"
    max_length: int = 512
    num_labels: int = 5
    output_root: Path = Path("../artifacts/abstract_evaluator_deberta_score")
    data_paths: DataPaths = field(default_factory=DataPaths)
    train: DebertaTrainConfig = field(default_factory=DebertaTrainConfig)
    lora: DebertaLoraConfig = field(default_factory=DebertaLoraConfig)
    wandb_enabled: bool = False
    wandb_project: str = "abstract-evaluator-deberta-score"
    wandb_entity: Optional[str] = None
    wandb_dir: Optional[Path] = None
    train_rows: Optional[int] = None
    val_rows: Optional[int] = None
    test_rows: Optional[int] = None

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.wandb_dir:
            self.wandb_dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["data_paths"] = self.data_paths.as_dict()
        payload["wandb_dir"] = str(self.wandb_dir) if self.wandb_dir else None
        return payload


def build_deberta_default_config(project_root: Path) -> DebertaPipelineConfig:
    cfg = DebertaPipelineConfig()
    cfg.data_paths = default_qwen_paths(project_root)
    cfg.output_root = project_root / "experiments/artifacts/abstract_evaluator_deberta_score"
    cfg.wandb_dir = project_root / "experiments/artifacts/wandb"
    cfg.ensure_dirs()
    return cfg


def build_data_paths(
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> DataPaths:
    return DataPaths(train_path=train_path, val_path=val_path, test_path=test_path)
