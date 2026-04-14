"""scripts/train_reward.py — Entry point for Phase 2: Reward Model training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.reward_trainer import run_reward_training


def main():
    parser = argparse.ArgumentParser(description="CoverBuild RL — Reward Model Training")
    parser.add_argument(
        "--config", default="configs/reward_config.yaml",
        help="Path to reward config YAML"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print("✅ Reward config loaded:")
        for k, v in cfg.items():
            print(f"   {k}: {v}")
        return

    print("🏆 CoverBuild RL — Reward Model Training Phase")
    output_path = run_reward_training(args.config)
    print(f"\n✅ Reward model complete. Model at: {output_path}")
    print("   Next step: python scripts/train_ppo.py")


if __name__ == "__main__":
    main()
