.PHONY: install train-sft train-reward train-ppo train-all serve test clean mlflow-ui

# ─── Setup ───────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True)"

# ─── Training Pipeline ───────────────────────────────────────────────────────
train-sft:
	python scripts/train_sft.py

train-reward:
	python scripts/train_reward.py

train-ppo:
	python scripts/train_ppo.py

train-all: train-sft train-reward train-ppo
	@echo "✅ Full RLHF pipeline complete."

# ─── Serving ─────────────────────────────────────────────────────────────────
serve:
	python scripts/serve.py

# ─── MLflow UI ───────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --port 5000

# ─── Tests ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

# ─── Quick demo ──────────────────────────────────────────────────────────────
demo:
	python -c "\
from src.api.app import app; \
import uvicorn; \
uvicorn.run(app, host='0.0.0.0', port=8000)"
