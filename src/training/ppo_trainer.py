"""src/training/ppo_trainer.py — Phase 3: REINFORCE + KL-penalty RL Training.

TRL 1.1.0 removed PPOTrainer. We implement our own REINFORCE-with-baseline
RL loop which is conceptually identical to the core PPO idea:

  1. Generate completions with policy model
  2. Score with reward model  
  3. Compute per-token log-probs under policy AND reference (frozen SFT)
  4. KL penalty: reward -= kl_coef * KL(policy || reference)
  5. REINFORCE gradient update: loss = -mean(reward * log_prob)

This is equivalent to the PPO objective without the clipping (simpler, often
works just as well for token-level text generation).

All metrics logged to MLflow. Best model registered to MLflow Model Registry.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import (
    JOB_DESCRIPTIONS,
    APPLICANT_PROFILES,
    build_prompt,
)
from src.models.reward_model import RewardModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompts(n_prompts: int = 200) -> list[str]:
    """Build a pool of cover letter prompts."""
    prompts = []
    for i in range(n_prompts):
        job     = JOB_DESCRIPTIONS[i % len(JOB_DESCRIPTIONS)]
        profile = APPLICANT_PROFILES[i % len(APPLICANT_PROFILES)]
        prompts.append(build_prompt(job, profile))
    return prompts


def compute_sequence_log_prob(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """
    Compute sum of log-probs of the response tokens under `model`.

    Args:
        model:          Causal LM (policy or reference).
        input_ids:      Full sequence tensor (prompt + response), shape (1, L).
        response_start: Index where response tokens begin.

    Returns:
        Scalar tensor — sum of log P(token | context) over response tokens.
    """
    with torch.no_grad():
        outputs = model(input_ids)
        logits  = outputs.logits  # (1, L, V)

    # Shift: logits[t] predicts token at position t+1
    shift_logits = logits[:, :-1, :]         # (1, L-1, V)
    shift_labels = input_ids[:, 1:]           # (1, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (1, L-1, V)
    token_lp  = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)                              # (1, L-1)

    # Sum only over the response portion
    resp_lp = token_lp[:, response_start - 1:].sum(dim=1)  # (1,)
    return resp_lp.squeeze()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_ppo_training(
    config_path: str = "configs/ppo_config.yaml",
    sft_path: Optional[str] = None,
    reward_model_path: Optional[str] = None,
) -> str:
    """
    Run the REINFORCE + KL-penalty RLHF loop.

    Returns:
        Path to the saved final model checkpoint.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sft_model_path = sft_path or cfg.get("sft_model_path", "./models/sft")
    rm_path        = reward_model_path or cfg.get("reward_model_path", "./models/reward")
    output_dir     = cfg.get("output_dir", "./models/ppo")
    experiment     = cfg.get("mlflow_experiment", "coverbuild-ppo")
    total_steps    = cfg.get("total_steps", 500)
    batch_size     = cfg.get("batch_size", 4)
    lr             = float(cfg.get("learning_rate", 1.4e-5))
    max_new_tokens = cfg.get("max_new_tokens", 256)
    kl_coef        = float(cfg.get("init_kl_coef", 0.2))
    log_every      = cfg.get("log_every_n_steps", 10)
    save_every     = cfg.get("save_every_n_steps", 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CoverBuild RL — Phase 3: REINFORCE+KL RLHF Training")
    print(f"  SFT Model:    {sft_model_path}")
    print(f"  Reward Model: {rm_path}")
    print(f"  Device: {device.upper()} | Steps: {total_steps} | Batch: {batch_size}")
    print(f"  KL coef: {kl_coef}")
    print(f"{'='*60}\n")

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=f"rl-{int(time.time())}")
    mlflow.log_params({
        "sft_model_path": sft_model_path,
        "reward_model":   rm_path,
        "total_steps":    total_steps,
        "batch_size":     batch_size,
        "lr":             lr,
        "kl_coef":        kl_coef,
        "max_new_tokens": max_new_tokens,
        "algorithm":      "REINFORCE+KL",
    })

    try:
        # ── Tokenizer ────────────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dtype = torch.float16 if device == "cuda" else torch.float32

        # ── Policy model (trainable) ──────────────────────────────────────────
        print("🤖 Loading policy model...")
        policy = AutoModelForCausalLM.from_pretrained(
            sft_model_path, torch_dtype=dtype
        ).to(device)
        policy.train()

        # ── Reference model (frozen SFT — KL baseline) ───────────────────────
        print("🔒 Creating frozen reference model...")
        reference = copy.deepcopy(policy)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)

        # ── Reward model ──────────────────────────────────────────────────────
        print("🏆 Loading reward model...")
        reward_model = RewardModel.from_pretrained(rm_path, device=device)

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, policy.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

        # Generation settings
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=float(cfg.get("temperature", 0.9)),
            top_k=cfg.get("top_k", 50),
            top_p=float(cfg.get("top_p", 0.95)),
            do_sample=True,
            repetition_penalty=float(cfg.get("repetition_penalty", 1.3)),  # prevents GPT-2 repetition loops
            no_repeat_ngram_size=cfg.get("no_repeat_ngram_size", 4),       # blocks repeated 4-grams
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompts = build_prompts(200)
        best_reward = -float("inf")
        t0 = time.time()

        print(f"\n🚀 Starting RL loop ({total_steps} steps × batch {batch_size})...\n")

        for step in range(total_steps):
            # ── Sample batch ──────────────────────────────────────────────────
            batch_prompts = [
                prompts[(step * batch_size + i) % len(prompts)]
                for i in range(batch_size)
            ]

            rewards_list    = []
            log_probs_list  = []
            kl_list         = []
            completions     = []

            policy.eval()
            for prompt in batch_prompts:
                # Tokenise prompt
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)
                prompt_ids = enc["input_ids"]
                prompt_len = prompt_ids.shape[1]

                # Generate response
                with torch.no_grad():
                    out = policy.generate(**enc, **gen_kwargs)

                full_ids = out  # shape (1, prompt_len + resp_len)
                completion = tokenizer.decode(
                    full_ids[0][prompt_len:], skip_special_tokens=True
                )
                completions.append(completion)

                # Reward (scalar)
                r = reward_model.score(prompt, completion)
                rewards_list.append(r)

                # Log-prob under policy & reference for KL
                policy_lp  = compute_sequence_log_prob(policy,    full_ids, prompt_len)
                ref_lp     = compute_sequence_log_prob(reference,  full_ids, prompt_len)
                kl         = (policy_lp - ref_lp).clamp(min=-10, max=10)
                kl_list.append(kl.item())
                log_probs_list.append(policy_lp)

            # ── Compute KL-adjusted rewards ────────────────────────────────────
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            kl_tensor      = torch.tensor(kl_list,      dtype=torch.float32, device=device)
            adj_rewards    = rewards_tensor - kl_coef * kl_tensor

            # Baseline (mean reward subtraction — variance reduction)
            baseline = adj_rewards.mean()
            advantages = adj_rewards - baseline

            # ── REINFORCE update ──────────────────────────────────────────────
            policy.train()
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for lp, adv in zip(log_probs_list, advantages):
                total_loss = total_loss + (-lp * adv.detach())

            loss = total_loss / batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            # ── Metrics ────────────────────────────────────────────────────────
            reward_mean = rewards_tensor.mean().item()
            kl_mean     = kl_tensor.mean().item()

            if step % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  Step {step+1:4d}/{total_steps} | "
                    f"Reward: {reward_mean:+.4f} | "
                    f"KL: {kl_mean:.4f} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
                mlflow.log_metrics({
                    "reward_mean":   reward_mean,
                    "reward_std":    rewards_tensor.std().item(),
                    "kl_divergence": kl_mean,
                    "rl_loss":       loss.item(),
                    "advantage_mean": advantages.mean().item(),
                }, step=step)

                # Log a sample
                if step % (log_every * 5) == 0:
                    sample = (
                        f"STEP {step+1}\n"
                        f"Prompt: {batch_prompts[0][:200]}...\n"
                        f"Completion: {completions[0][:400]}..."
                    )
                    mlflow.log_text(sample, artifact_file=f"samples/step_{step+1}.txt")

            # ── Save best ──────────────────────────────────────────────────────
            if reward_mean > best_reward:
                best_reward = reward_mean
                best_path = Path(output_dir) / "best"
                policy.save_pretrained(str(best_path))
                tokenizer.save_pretrained(str(best_path))
                mlflow.log_metric("best_reward", best_reward, step=step)

            # ── Periodic checkpoint ────────────────────────────────────────────
            if (step + 1) % save_every == 0:
                ckpt = Path(output_dir) / f"checkpoint-{step+1}"
                policy.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                print(f"   💾 Checkpoint at step {step+1}")

        # ── Final save ────────────────────────────────────────────────────────
        final_path = Path(output_dir) / "final"
        policy.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        elapsed = time.time() - t0
        mlflow.log_metrics({
            "best_reward":         best_reward,
            "total_training_time": elapsed,
        })

        # ── MLflow Model Registry ──────────────────────────────────────────────
        mlflow.log_artifact(str(final_path), artifact_path="rl_policy_model")
        try:
            model_uri = f"runs:/{run.info.run_id}/rl_policy_model"
            mlflow.register_model(model_uri, "CoverBuild-RL-Policy")
            print("📦 Model registered in MLflow Registry as 'CoverBuild-RL-Policy'")
        except Exception as e:
            print(f"   ⚠️ Registry skipped: {e}")

        print(f"\n✅ RL training complete in {elapsed:.1f}s")
        print(f"   Best reward: {best_reward:.4f}")
        print(f"   Final model: {final_path}")

    finally:
        mlflow.end_run()

    return str(final_path)
