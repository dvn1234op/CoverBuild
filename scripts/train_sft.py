"""scripts/train_sft.py — Entry point for Phase 1: SFT training."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.sft_trainer import run_sft_training


def main():
    parser = argparse.ArgumentParser(description="CoverBuild RL — SFT Training")
    parser.add_argument(
        "--config", default="configs/sft_config.yaml",
        help="Path to SFT config YAML"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config without training"
    )
    args = parser.parse_args()

    if args.dry_run:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print("✅ Config loaded successfully:")
        for k, v in cfg.items():
            print(f"   {k}: {v}")
        return

    print("🎯 CoverBuild RL — Supervised Fine-Tuning Phase")
    output_path = run_sft_training(args.config)
    print(f"\n✅ SFT complete. Model at: {output_path}")
    print("   Next step: python scripts/train_reward.py")


if __name__ == "__main__":
    main()
