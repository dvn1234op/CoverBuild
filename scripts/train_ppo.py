"""scripts/train_ppo.py — Entry point for Phase 3: PPO RLHF training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.ppo_trainer import run_ppo_training


def main():
    parser = argparse.ArgumentParser(description="CoverBuild RL — PPO RLHF Training")
    parser.add_argument(
        "--config", default="configs/ppo_config.yaml",
        help="Path to PPO config YAML"
    )
    parser.add_argument("--sft-model", default=None, help="Override SFT model path")
    parser.add_argument("--reward-model", default=None, help="Override reward model path")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if args.steps:
            cfg["total_steps"] = args.steps
        print("✅ PPO config loaded:")
        for k, v in cfg.items():
            print(f"   {k}: {v}")
        return

    import yaml
    if args.steps:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["total_steps"] = args.steps
        import tempfile, os
        tmp = "configs/_ppo_override.yaml"
        with open(tmp, "w") as f:
            yaml.dump(cfg, f)
        config_path = tmp
    else:
        config_path = args.config

    print("🚀 CoverBuild RL — PPO RLHF Training Phase")
    output_path = run_ppo_training(
        config_path,
        sft_path=args.sft_model,
        reward_model_path=args.reward_model,
    )
    print(f"\n✅ PPO training complete. Model at: {output_path}")
    print("   Next step: python scripts/serve.py")


if __name__ == "__main__":
    main()
