"""src/training/reward_trainer.py — Phase 2: Reward Model Training.

Trains a DistilBERT classifier on pairwise preference data.
Uses a margin ranking loss: reward(chosen) > reward(rejected).
All metrics logged to MLflow.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.synthetic_prefs import build_preference_dataset
from src.models.reward_model import RewardModelNetwork


# ---------------------------------------------------------------------------
# PyTorch Dataset for preference pairs
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) triples for reward model training."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str = "data/preferences/prefs.jsonl",
        n_synthetic: int = 300,
        max_length: int = 512,
    ):
        import json

        path = Path(data_path)
        if path.exists():
            with open(path) as f:
                self.pairs = [json.loads(line) for line in f]
            print(f"📂 Loaded {len(self.pairs)} preference pairs from {data_path}")
        else:
            print(f"⚙️  Generating {n_synthetic} preference pairs...")
            self.pairs = build_preference_dataset(n_synthetic, data_path)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        prompt   = pair["prompt"]
        chosen   = pair["chosen"]
        rejected = pair["rejected"]

        def encode(text: str) -> dict:
            return self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

        return {
            "chosen_input_ids":      encode(prompt + " [SEP] " + chosen)["input_ids"].squeeze(),
            "chosen_attention_mask": encode(prompt + " [SEP] " + chosen)["attention_mask"].squeeze(),
            "rejected_input_ids":      encode(prompt + " [SEP] " + rejected)["input_ids"].squeeze(),
            "rejected_attention_mask": encode(prompt + " [SEP] " + rejected)["attention_mask"].squeeze(),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_reward_training(config_path: str = "configs/reward_config.yaml") -> str:
    """
    Train the reward model on pairwise preference data.

    Loss: Margin Ranking Loss — encourage reward(chosen) > reward(rejected) + margin

    Returns:
        Path to saved reward model checkpoint.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_model = cfg.get("base_model", "distilbert-base-uncased")
    output_dir = cfg.get("output_dir", "./models/reward")
    experiment = cfg.get("mlflow_experiment", "coverbuild-reward")
    data_path  = cfg.get("preference_data_path", "data/preferences/prefs.jsonl")
    n_epochs   = cfg.get("num_train_epochs", 3)
    batch_size = cfg.get("per_device_train_batch_size", 8)
    lr         = cfg.get("learning_rate", 2e-5)
    max_length = cfg.get("max_length", 512)
    fp16       = cfg.get("fp16", True) and torch.cuda.is_available()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CoverBuild RL — Phase 2: Reward Model Training")
    print(f"  Backbone: {base_model}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device.upper()}")
    print(f"{'='*60}\n")

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=f"reward-{int(time.time())}")
    mlflow.log_params({
        "base_model": base_model,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "fp16": fp16,
    })

    try:
        # ── Tokenizer & Dataset ───────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        dataset = PreferenceDataset(tokenizer, data_path=data_path, max_length=max_length)
        # 90/10 split
        n_train = int(0.9 * len(dataset))
        n_eval  = len(dataset) - n_train
        train_ds, eval_ds = torch.utils.data.random_split(
            dataset, [n_train, n_eval], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False)
        mlflow.log_params({"n_train": n_train, "n_eval": n_eval})

        # ── Model ─────────────────────────────────────────────────────────────
        model = RewardModelNetwork(base_model).to(device)
        scaler = torch.cuda.amp.GradScaler() if fp16 else None

        # ── Optimizer & Scheduler ─────────────────────────────────────────────
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = n_epochs * len(train_loader)
        warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Margin ranking loss: pushes reward(chosen) > reward(rejected) + margin
        margin_loss = nn.MarginRankingLoss(margin=0.5)
        target_ones = torch.ones(batch_size).to(device)  # "chosen is better"

        # ── Training loop ─────────────────────────────────────────────────────
        global_step = 0
        best_eval_loss = float("inf")
        t0 = time.time()

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                chosen_ids   = batch["chosen_input_ids"].to(device)
                chosen_mask  = batch["chosen_attention_mask"].to(device)
                reject_ids   = batch["rejected_input_ids"].to(device)
                reject_mask  = batch["rejected_attention_mask"].to(device)

                actual_batch = chosen_ids.size(0)
                tgt = target_ones[:actual_batch]

                optimizer.zero_grad()

                if fp16:
                    with torch.cuda.amp.autocast():
                        chosen_reward  = model(chosen_ids, chosen_mask)
                        reject_reward  = model(reject_ids, reject_mask)
                        loss = margin_loss(chosen_reward, reject_reward, tgt)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    chosen_reward  = model(chosen_ids, chosen_mask)
                    reject_reward  = model(reject_ids, reject_mask)
                    loss = margin_loss(chosen_reward, reject_reward, tgt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                global_step += 1

                if global_step % 10 == 0:
                    avg_chosen  = chosen_reward.mean().item()
                    avg_reject  = reject_reward.mean().item()
                    accuracy    = (chosen_reward > reject_reward).float().mean().item()
                    print(
                        f"  Step {global_step:4d} | Loss: {loss.item():.4f} | "
                        f"Acc: {accuracy:.3f} | "
                        f"R(chosen)={avg_chosen:.3f} R(rejected)={avg_reject:.3f}"
                    )
                    mlflow.log_metrics({
                        "train_loss":   loss.item(),
                        "reward_acc":   accuracy,
                        "avg_chosen_r":  avg_chosen,
                        "avg_reject_r":  avg_reject,
                    }, step=global_step)

            # ── Eval ──────────────────────────────────────────────────────────
            model.eval()
            eval_loss = 0.0
            eval_acc  = 0.0
            with torch.no_grad():
                for batch in eval_loader:
                    chosen_ids  = batch["chosen_input_ids"].to(device)
                    chosen_mask = batch["chosen_attention_mask"].to(device)
                    reject_ids  = batch["rejected_input_ids"].to(device)
                    reject_mask = batch["rejected_attention_mask"].to(device)
                    actual_batch = chosen_ids.size(0)
                    tgt = target_ones[:actual_batch]

                    cr = model(chosen_ids, chosen_mask)
                    rr = model(reject_ids, reject_mask)
                    eval_loss += margin_loss(cr, rr, tgt).item()
                    eval_acc  += (cr > rr).float().mean().item()

            eval_loss /= len(eval_loader)
            eval_acc  /= len(eval_loader)
            print(f"\n📊 Epoch {epoch+1}/{n_epochs} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.3f}\n")
            mlflow.log_metrics({"eval_loss": eval_loss, "eval_acc": eval_acc}, step=global_step)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), Path(output_dir) / "reward_model.pt")
                print(f"   💾 Best model checkpoint saved (eval_loss={eval_loss:.4f})")

        elapsed = time.time() - t0
        mlflow.log_metrics({
            "best_eval_loss": best_eval_loss,
            "training_time_sec": elapsed,
        })
        tokenizer.save_pretrained(output_dir)
        mlflow.log_artifact(output_dir, artifact_path="reward_model")

        print(f"\n✅ Reward model training complete in {elapsed:.1f}s")
        print(f"   Best eval loss: {best_eval_loss:.4f}")
        print(f"   Saved to: {output_dir}")

    finally:
        mlflow.end_run()

    return output_dir
