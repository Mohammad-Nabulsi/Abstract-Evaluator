from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_COLUMNS = ["paper_id", "submission", "score", "rationale"]

DEFAULT_TASK = (
    "Evaluate the quality of the following research abstract for conference acceptance."
)
DEFAULT_REFERENCE = (
    "A strong research abstract clearly presents the problem, methodology, "
    "contribution, and experimental evidence."
)
DEFAULT_RUBRIC = {
    "score_scale": {
        "0": "Very poor abstract: missing most core components, unclear, generic, or unusable.",
        "1": "Weak abstract: contains a few useful elements but major components are missing or vague.",
        "2": "Borderline abstract: understandable but incomplete; some important components are weak or missing.",
        "3": "Good abstract: mostly complete, clear, and logically structured, with minor weaknesses.",
        "4": "Excellent abstract: complete, clear, concise, well-structured, and strongly communicates the paper's contribution and evidence.",
    },
    "criteria": {
        "1_background_or_context": "Provides enough background or introduction to understand the research area and motivation.",
        "2_problem_statement": "Clearly identifies the research problem, gap, or limitation being addressed.",
        "3_objective_or_purpose": "States the main objective, research question, or purpose of the work.",
        "4_methodology": "Explains the methods, approach, model, experiment, dataset, or procedure used.",
        "5_results_or_findings": "Reports concrete results, findings, observations, or evidence rather than only intentions.",
        "6_contribution": "Clarifies what is new, useful, or significant about the work.",
        "7_conclusion_or_implication": "Provides a conclusion, implication, impact, or takeaway from the work.",
        "8_clarity_and_conciseness": "Uses clear, precise, and concise language without unnecessary vagueness or filler.",
        "9_logical_flow": "Presents the abstract in a coherent order: context/problem -> objective -> method -> results -> contribution.",
        "10_specificity_and_evidence": "Avoids generic claims and supports statements with specific details, comparisons, numbers, or evidence when possible.",
    },
}


@dataclass
class DataPaths:
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None
    test_path: Optional[Path] = None
    combined_path: Optional[Path] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "train_path": str(self.train_path) if self.train_path else None,
            "val_path": str(self.val_path) if self.val_path else None,
            "test_path": str(self.test_path) if self.test_path else None,
            "combined_path": str(self.combined_path) if self.combined_path else None,
        }


@dataclass
class TrainConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 8e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_total_limit: int = 4
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    auto_find_batch_size: bool = True
    tf32: bool = True


@dataclass
class GenerationConfig:
    max_new_tokens: int = 180
    batch_size: int = 8
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "abstract-evaluator-qwen3-sft"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    run_group: Optional[str] = None
    dir: Optional[Path] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["dir"] = str(self.dir) if self.dir else None
        return payload


@dataclass
class PipelineConfig:
    seed: int = 3407
    model_name: str = "Qwen/Qwen3-8B"
    run_name: str = "qwen3_8b_abstract_evaluator_lora_no_quant"
    max_seq_length: int = 2048
    use_4bit: bool = False
    output_root: Path = Path("../artifacts/abstract_evaluator_qwen3_sft")
    data_paths: DataPaths = field(default_factory=DataPaths)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    lora_cfg: Dict[str, Any] = field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "bias": "none",
            "freeze_ratio": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }
    )
    train_rows: Optional[int] = None
    val_rows: Optional[int] = None
    test_rows: Optional[int] = None

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.wandb.dir:
            self.wandb.dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["data_paths"] = self.data_paths.as_dict()
        payload["wandb"] = self.wandb.as_dict()
        return payload


def default_qwen_paths(project_root: Path) -> DataPaths:
    return DataPaths(
        train_path=project_root / "data/data/train/all.jsonl",
        val_path=project_root / "data/data/val/all.jsonl",
        test_path=project_root / "data/data/test/all.jsonl",
    )

