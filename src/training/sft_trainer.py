"""src/training/sft_trainer.py — Phase 1: Supervised Fine-Tuning.

Fine-tunes GPT-2 on the cover letter dataset using TRL's SFTTrainer.
Supports LoRA for memory efficiency.
All metrics logged to MLflow.

Compatible with TRL 1.1.0.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

from src.data.dataset import build_synthetic_dataset


class MLflowLoggerCallback(TrainerCallback):
    """Custom callback to log training metrics to MLflow."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and mlflow.active_run():
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


def run_sft_training(config_path: str = "configs/sft_config.yaml") -> str:
    """
    Run SFT fine-tuning pipeline.

    Returns:
        Path to the saved SFT model checkpoint.
    """
    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name    = cfg["model_name"]
    output_dir    = cfg["output_dir"]
    experiment    = cfg.get("mlflow_experiment", "coverbuild-sft")
    use_lora      = cfg.get("use_lora", True)
    max_length    = cfg.get("max_seq_length", 512)
    dataset_path  = cfg.get("dataset_path", "data/processed/cover_letters.jsonl")
    fp16          = cfg.get("fp16", True) and torch.cuda.is_available()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CoverBuild RL — Phase 1: SFT Training")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  fp16: {fp16}")
    print(f"{'='*60}\n")

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=f"sft-{int(time.time())}")
    mlflow.log_params({
        "model_name": model_name,
        "use_lora":   use_lora,
        "max_length": max_length,
        "fp16":       fp16,
    })

    try:
        # ── Tokenizer ────────────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ── Dataset ──────────────────────────────────────────────────────────
        print("📦 Building dataset...")
        raw_data = build_synthetic_dataset(500, dataset_path)
        hf_dataset = HFDataset.from_list(raw_data)
        split = hf_dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
        print(f"   Train: {len(train_ds)} | Eval: {len(eval_ds)}")
        mlflow.log_params({"n_train": len(train_ds), "n_eval": len(eval_ds)})

        # ── LoRA config ───────────────────────────────────────────────────────
        peft_config = None
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.1),
                target_modules=cfg.get("lora_target_modules", ["c_attn"]),
                bias="none",
            )

        # ── SFT Config ────────────────────────────────────────────────────────
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
            learning_rate=float(cfg.get("learning_rate", 2e-5)),
            warmup_steps=cfg.get("warmup_steps", 50),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
            max_length=max_length,               # TRL 1.1.0 uses max_length
            fp16=fp16,
            save_steps=cfg.get("save_steps", 100),
            logging_steps=cfg.get("logging_steps", 10),
            eval_steps=cfg.get("eval_steps", 50),
            eval_strategy="steps",
            load_best_model_at_end=cfg.get("load_best_model_at_end", True),
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            report_to="none",
            dataset_text_field="text",           # column to train on
        )

        # ── SFT Trainer ───────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model_name,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
            processing_class=tokenizer,          # TRL 1.1.0 uses processing_class
            callbacks=[MLflowLoggerCallback()],
        )

        print("🚀 Starting SFT training...")
        t0 = time.time()
        train_result = trainer.train()
        elapsed = time.time() - t0

        mlflow.log_metrics({
            "final_train_loss": train_result.training_loss,
            "training_time_sec": elapsed,
        })

        # ── Save model ────────────────────────────────────────────────────────
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_artifact(output_dir, artifact_path="sft_model")

        print(f"\n✅ SFT training complete in {elapsed:.1f}s")
        print(f"   Final train loss: {train_result.training_loss:.4f}")
        print(f"   Saved to: {output_dir}")

    finally:
        mlflow.end_run()

    return output_dir
