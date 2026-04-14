# ⚡ CoverBuild RL — Cover Letter Generator with RLHF Feedback Loop

> **RLHF in miniature** — a GPT-2 policy model optimised with PPO using a learned reward model trained on human preference data. End-to-end pipeline from synthetic data generation to REST API inference.

---

## 🏗️ Architecture

```
Synthetic Data  →  SFT Fine-Tuning  →  Reward Model Training  →  PPO RLHF Loop  →  FastAPI REST API
     │                   │                      │                       │
 cover_letters.jsonl   models/sft/           models/reward/         models/ppo/
 prefs.jsonl          (GPT-2 + LoRA)       (DistilBERT)           (final policy)
```

### Three-Phase Training Pipeline

| Phase | Method | Model | Library |
|---|---|---|---|
| **1. SFT** | Supervised Fine-Tuning on cover letter corpus | GPT-2 + LoRA | `trl.SFTTrainer` |
| **2. Reward** | Margin ranking loss on preference pairs | DistilBERT | Custom PyTorch |
| **3. PPO** | Policy optimisation with KL penalty | GPT-2 + Value Head | `trl.PPOTrainer` |

---

## 📁 Project Structure

```
CoverBuild/
├── configs/            # YAML hyperparameter configs (SFT, Reward, PPO)
├── data/               # Generated datasets (auto-created on first run)
├── models/             # Saved model checkpoints (auto-created)
├── src/
│   ├── data/           # Dataset classes & synthetic data generators
│   ├── env/            # Custom Gymnasium text-generation environment
│   ├── models/         # Policy model & reward model wrappers
│   ├── training/       # SFT, Reward, and PPO trainers
│   ├── evaluation/     # ROUGE, BLEU, reward scoring
│   └── api/            # FastAPI application + Pydantic schemas
├── scripts/            # Entry-point scripts
├── tests/              # pytest test suite
├── frontend/           # Dark glassmorphism web UI
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full training pipeline

```bash
# Phase 1: Supervised Fine-Tuning (GPT-2 on cover letters)
python scripts/train_sft.py

# Phase 2: Reward Model (DistilBERT on preference pairs)
python scripts/train_reward.py

# Phase 3: PPO RLHF Loop
python scripts/train_ppo.py
```

Or run all three in sequence:

```bash
make train-all
```

### 3. Start the API

```bash
python scripts/serve.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. View training metrics

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

### 5. Open the frontend

Open `frontend/index.html` directly in your browser — the UI auto-connects to the local API.

---

## 🔧 Configuration

All hyperparameters live in `configs/`:

| File | Controls |
|---|---|
| `sft_config.yaml` | Learning rate, batch size, LoRA settings, epochs |
| `reward_config.yaml` | Reward model backbone, preference data path |
| `ppo_config.yaml` | KL coefficient, PPO epochs, generation settings |

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System health + model status |
| `GET` | `/model/info` | Loaded model metadata |
| `POST` | `/generate` | Generate a cover letter |
| `POST` | `/reward` | Score an existing cover letter |
| `GET` | `/samples` | Random generated samples |

### Example: Generate

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Software Engineer, requires Python and REST APIs",
    "applicant_profile": "3 years Python experience, BSc Computer Science",
    "max_length": 256,
    "temperature": 0.9
  }'
```

### Example: Score

```bash
curl -X POST http://localhost:8000/reward \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Software Engineer",
    "cover_letter": "Dear Hiring Manager, I am excited to apply..."
  }'
```

---

## 🏋️ Gymnasium Environment

The `CoverLetterEnv` wraps text generation as a proper RL environment:

```python
from transformers import AutoTokenizer
from src.env import CoverLetterEnv
from src.models.reward_model import RewardModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
reward_fn = RewardModel.fresh().score
env = CoverLetterEnv(tokenizer=tokenizer, reward_fn=reward_fn, max_length=256)

obs, info = env.reset()
print(info["prompt"])

# Token-level actions
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

- **Observation space**: `Box(0, vocab_size, shape=(max_length,), dtype=int64)` — current token sequence
- **Action space**: `Discrete(vocab_size)` — next token to generate
- **Reward**: 0 at each step; final scalar from reward model on episode end
- **Terminal**: EOS token generated or `max_length` reached

---

## 🧪 Tests

```bash
pytest tests/ -v

# Individual test files:
pytest tests/test_env.py     -v   # Gymnasium environment
pytest tests/test_reward.py  -v   # Reward model
pytest tests/test_api.py     -v   # FastAPI endpoints
```

---

## 📊 MLflow Tracking

Three separate MLflow experiments are created:

| Experiment | Metrics Tracked |
|---|---|
| `coverbuild-sft` | `train_loss`, `eval_loss`, samples/sec |
| `coverbuild-reward` | `train_loss`, `eval_loss`, `reward_acc`, `avg_chosen_r` |
| `coverbuild-ppo` | `reward_mean`, `kl_divergence`, `policy_loss`, `entropy` |

The best PPO checkpoint is registered in the MLflow Model Registry as **`CoverBuild-PPO-Policy`**.

---

## 🗂️ Reward Model Dimensions

Every generated letter is scored along 4 axes:

| Dimension | Weight | Measures |
|---|---|---|
| **Structure** | 30% | Proper salutation, closing, paragraph organisation |
| **Relevance** | 35% | Keyword overlap with job description |
| **Tone** | 20% | Professionalism, enthusiasm, lack of vague language |
| **Conciseness** | 15% | Appropriate length (100–400 words) |

Aggregate score is mapped to `[-1, 1]`.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `POLICY_MODEL_PATH` | `./models/ppo/final` | Path to PPO policy checkpoint |
| `SFT_MODEL_PATH` | `./models/sft` | Fallback to SFT model |
| `REWARD_MODEL_PATH` | `./models/reward` | Reward model checkpoint |

---

## 🛠️ Makefile Commands

```bash
make install      # pip install + NLTK punkt download
make train-sft    # Phase 1
make train-reward # Phase 2
make train-ppo    # Phase 3
make train-all    # All three phases sequentially
make serve        # Start FastAPI server
make test         # Run pytest
make mlflow-ui    # Open MLflow dashboard
make clean        # Remove __pycache__ etc.
```

---

## 📦 Key Dependencies

- **`torch`** — Core deep learning
- **`transformers`** — GPT-2 policy, DistilBERT reward backbone
- **`trl`** — `SFTTrainer`, `PPOTrainer`, `AutoModelForCausalLMWithValueHead`
- **`peft`** — LoRA for parameter-efficient fine-tuning
- **`gymnasium`** — Custom text generation RL environment
- **`mlflow`** — Experiment tracking + model registry
- **`fastapi`** + **`uvicorn`** — REST API server
- **`evaluate`** + **`rouge-score`** — ROUGE/BLEU evaluation

---

## 📈 Expected Training Results

| Metric | Before PPO (SFT baseline) | After PPO |
|---|---|---|
| Reward Mean | ~0.1 | ~0.3–0.5 |
| KL Divergence | 0.0 | 3–8 (adaptive) |
| Structure Score | ~0.4 | ~0.6 |
| Tone Score | ~0.3 | ~0.5 |

> Results vary by GPU, training steps, and random seed.