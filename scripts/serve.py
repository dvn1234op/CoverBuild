"""scripts/serve.py — Start the CoverBuild RL FastAPI server."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="CoverBuild RL — API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Hot reload (dev mode)")
    parser.add_argument("--policy-model", default=None, help="Override policy model path")
    parser.add_argument("--reward-model", default=None, help="Override reward model path")
    args = parser.parse_args()

    import os
    if args.policy_model:
        os.environ["POLICY_MODEL_PATH"] = args.policy_model
    if args.reward_model:
        os.environ["REWARD_MODEL_PATH"] = args.reward_model

    print(f"\n🌐 Starting CoverBuild RL API on http://{args.host}:{args.port}")
    print(f"   Docs:     http://localhost:{args.port}/docs")
    print(f"   MLflow:   http://localhost:5000 (run: mlflow ui --port 5000)\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
