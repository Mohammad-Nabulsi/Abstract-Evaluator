from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.mistral.utils.chat import (
    add_messages_and_targets,
    estimate_token_lengths,
    make_inference_prompt,
)
from experiments.mistral.utils.configs import build_mistral7_default_config
from experiments.mistral.utils.training import EpochEvaluationCallback
from experiments.utils.data import clean_train_val_test, load_train_val_test_dfs, sample_n_rows
from experiments.utils.datasets_io import export_split_jsonl, to_hf_dataset_dict
from experiments.utils.evaluation import compute_eval_metrics, parse_rationale, parse_score
from experiments.utils.generation import generate_predictions_from_messages
from experiments.utils.logging_utils import setup_logger


class _FakeBatch(dict):
    def to(self, *_args, **_kwargs):
        return self


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def encode(self, text, add_special_tokens=False):
        return [2] * max(1, len(str(text).split()))

    def __call__(self, batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048):
        batch_size = len(batch_prompts)
        seq_len = 8
        ids = torch.full((batch_size, seq_len), fill_value=5, dtype=torch.long)
        return _FakeBatch({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def batch_decode(self, token_rows, skip_special_tokens=True):
        outputs = []
        for row in token_rows.tolist():
            score = int(row[1]) if len(row) > 1 else 0
            outputs.append(
                json.dumps(
                    {
                        "score": score % 5,
                        "rationale": f"Mock rationale for score {score % 5}",
                    },
                    ensure_ascii=False,
                )
            )
        return outputs


class FakeModel:
    def __init__(self):
        self.counter = 0

    def eval(self):
        return self

    def generate(self, input_ids=None, **kwargs):
        batch_size = input_ids.shape[0]
        gen = []
        for _ in range(batch_size):
            score = self.counter % 5
            gen.append([101, score, 102, 1])
            self.counter += 1
        gen_t = torch.tensor(gen, dtype=torch.long)
        return torch.cat([input_ids, gen_t], dim=1)


def main():
    project_root = PROJECT_ROOT
    cfg = build_mistral7_default_config(project_root)
    cfg.run_name = "mistral7_modular_smoke_10rows"
    cfg.wandb.enabled = False

    smoke_root = cfg.output_root / "smoke_modular_10rows"
    logs_dir = smoke_root / "logs"
    logger = setup_logger(
        name="mistral7_modular_smoke",
        log_dir=logs_dir,
        log_file="smoke_test_10rows.log",
    )

    train_df, val_df, test_df = load_train_val_test_dfs(
        train_path=cfg.data_paths.train_path,
        val_path=cfg.data_paths.val_path,
        test_path=cfg.data_paths.test_path,
    )
    train_df, val_df, test_df = clean_train_val_test(train_df, val_df, test_df)

    train_df = sample_n_rows(train_df, n=10, seed=cfg.seed)
    val_df = sample_n_rows(val_df, n=10, seed=cfg.seed + 1)
    test_df = sample_n_rows(test_df, n=10, seed=cfg.seed + 2)

    train_df = add_messages_and_targets(train_df)
    val_df = add_messages_and_targets(val_df)
    test_df = add_messages_and_targets(test_df)

    jsonl_paths = export_split_jsonl(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=smoke_root / "jsonl",
    )
    logger.info("JSONL exports: %s", jsonl_paths)

    ds = to_hf_dataset_dict(train_df, val_df, test_df)
    logger.info("HF dataset sizes: train=%d val=%d test=%d", len(ds["train"]), len(ds["validation"]), len(ds["test"]))

    fake_tokenizer = FakeTokenizer()
    lens = estimate_token_lengths(pd.concat([train_df, val_df, test_df], ignore_index=True), fake_tokenizer)
    logger.info(
        "Token length percentiles (fake tokenizer): p50=%d p95=%d p100=%d",
        int(pd.Series(lens).quantile(0.5)),
        int(pd.Series(lens).quantile(0.95)),
        int(lens.max()),
    )

    fake_model = FakeModel()
    pred_df = generate_predictions_from_messages(
        eval_df=val_df,
        model=fake_model,
        tokenizer=fake_tokenizer,
        make_inference_prompt_fn=make_inference_prompt,
        parse_score_fn=parse_score,
        parse_rationale_fn=parse_rationale,
        max_seq_length=cfg.max_seq_length,
        max_new_tokens=64,
        batch_size=4,
    )
    metrics = compute_eval_metrics(pred_df, include_bertscore=False)
    (smoke_root / "eval").mkdir(parents=True, exist_ok=True)
    pred_path = smoke_root / "eval" / "validation_predictions_smoke.csv"
    metrics_path = smoke_root / "eval" / "metrics_smoke.json"
    pred_df.to_csv(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved smoke predictions + metrics.")

    callback = EpochEvaluationCallback(
        eval_out_dir=smoke_root / "epoch_eval",
        val_df=val_df,
        test_df=test_df,
        max_seq_length=cfg.max_seq_length,
        max_new_tokens=64,
        batch_size=4,
        include_bertscore=False,
        use_wandb=False,
        logger=logger,
    )
    callback.on_epoch_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(epoch=1.0),
        control=SimpleNamespace(),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
    )
    callback.on_epoch_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(epoch=2.0),
        control=SimpleNamespace(),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
    )

    summary = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "jsonl_paths": jsonl_paths,
        "predictions_csv": str(pred_path),
        "metrics_json": str(metrics_path),
        "epoch_eval_dir": str(smoke_root / "epoch_eval"),
        "log_file": str(logs_dir / "smoke_test_10rows.log"),
    }
    summary_path = logs_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
