"""src/api/app.py — CoverBuild RL REST API.

Endpoints:
  GET  /health           System health + model status
  GET  /model/info       Loaded model metadata
  POST /generate         Generate a cover letter
  POST /reward           Score an existing cover letter
  GET  /samples          Random generated samples (for demo)

Model fallback chain:  PPO checkpoint > SFT checkpoint > gemma-3-1b-it (base)
"""

from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import mlflow
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Authenticate with HuggingFace for gated models (Gemma requires license acceptance)
_hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass  # Non-fatal — cached models still work

from src.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    RewardRequest,
    RewardResponse,
    ScoreBreakdown,
)
from src.data.dataset import build_prompt
from src.models.reward_model import RewardModel

# ── Global model state ───────────────────────────────────────────────────────

_policy_model  = None
_tokenizer     = None
_reward_model: Optional[RewardModel] = None
_device        = "cuda" if torch.cuda.is_available() else "cpu"
_model_path    = os.environ.get("POLICY_MODEL_PATH", "./models/ppo/final")
_sft_fallback  = os.environ.get("SFT_MODEL_PATH",    "./models/sft")
# Gemma 3 1B IT (best) requires HF token; Phi-3.5-mini needs no token and works immediately
_base_fallback = "Qwen/Qwen2.5-0.5B-Instruct"
_reward_path   = os.environ.get("REWARD_MODEL_PATH", "./models/reward")
_loaded_from   = None
_version       = "1.0.0"

# Strip chat-template tokens from model output
def _clean_output(text: str) -> str:
    for tok in ['<end_of_turn>', '<eos>', '<bos>', '<pad>']:
        text = text.replace(tok, '')
    import re as _re
    text = _re.sub(r'<[^>]+>', '', text)
    return text.strip()


def _load_policy():
    """Load the best available policy model (PPO > SFT > Gemma base)."""
    global _policy_model, _tokenizer, _loaded_from

    for path in [_model_path, _sft_fallback, _base_fallback]:
        try:
            print(f"Loading policy from: {path}")
            tok = AutoTokenizer.from_pretrained(path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                tok.pad_token_id = tok.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
            ).to(_device)
            model.eval()

            _tokenizer    = tok
            _policy_model = model
            _loaded_from  = path
            print(f"   Policy loaded OK from: {path}")
            return path
        except Exception as e:
            print(f"   Could not load from {path}: {e}")
            continue

    raise RuntimeError("Could not load any policy model — check your model paths.")


def _load_reward():
    """Load the reward model (heuristic fallback if untrained)."""
    global _reward_model
    try:
        _reward_model = RewardModel.from_pretrained(_reward_path, device=_device)
        print("   Reward model loaded OK")
    except Exception as e:
        print(f"   Reward model load failed ({e}), using heuristic scorer")
        _reward_model = RewardModel.fresh(device=_device)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n=== Starting CoverBuild RL API ===")
    _load_policy()
    _load_reward()
    mlflow.set_experiment("coverbuild-inference")
    print(f"MLflow: {mlflow.get_tracking_uri()}")
    print(f"Device: {_device.upper()}")
    print("API ready.\n")
    yield
    print("CoverBuild API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CoverBuild RL API",
    description=(
        "RLHF-powered cover letter generator. "
        "Generates professional cover letters and scores them with a trained reward model."
    ),
    version=_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Return system health and model load status."""
    return HealthResponse(
        status="healthy",
        model_loaded=_policy_model is not None,
        reward_model_loaded=_reward_model is not None,
        device=_device,
        model_name=_loaded_from or "none",
        version=_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about the loaded models."""
    if _policy_model is None:
        raise HTTPException(status_code=503, detail="Policy model not loaded")

    cfg = _policy_model.config
    return ModelInfoResponse(
        model_name=getattr(cfg, "model_type", "unknown"),
        model_path=_loaded_from or _model_path,
        device=_device,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=getattr(
            cfg, "max_position_embeddings",
            getattr(cfg, "n_positions", 0)
        ),
        use_lora=False,
        reward_model_name="distilbert-base-uncased",
        reward_model_path=_reward_path,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_cover_letter(request: GenerateRequest):
    """
    Generate a cover letter given a job description and applicant profile.

    Returns the generated text, reward score, and a multi-dimensional quality breakdown.
    """
    if _policy_model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Policy model not loaded")

    prompt = build_prompt(request.job_description, request.applicant_profile)

    try:
        t_start = time.perf_counter()

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(_device)

        prompt_token_count = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = _policy_model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.15,    # lighter penalty for Gemma (it's already good)
                no_repeat_ngram_size=4,
                pad_token_id=_tokenizer.pad_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

        new_token_ids = output_ids[0][prompt_token_count:]
        raw_text      = _tokenizer.decode(new_token_ids, skip_special_tokens=True)
        cover_letter  = _clean_output(raw_text)
        gen_time_ms   = (time.perf_counter() - t_start) * 1000

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Score with reward model
    reward_score   = _reward_model.score(prompt, cover_letter)
    breakdown_dict = _reward_model.score_breakdown(prompt, cover_letter)

    # Log to MLflow
    try:
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_metrics({
                "inference_reward": reward_score,
                "gen_time_ms":      gen_time_ms,
                "generated_tokens": len(new_token_ids),
            })
    except Exception:
        pass  # Don't fail the request if MLflow logging fails

    return GenerateResponse(
        cover_letter=cover_letter,
        reward_score=round(reward_score, 4),
        score_breakdown=ScoreBreakdown(
            structure=breakdown_dict["structure"],
            relevance=breakdown_dict["relevance"],
            tone=breakdown_dict["tone"],
            conciseness=breakdown_dict["conciseness"],
            aggregate=breakdown_dict["aggregate"],
        ),
        generation_time_ms=round(gen_time_ms, 2),
        model_version=_version,
        prompt_tokens=prompt_token_count,
        generated_tokens=len(new_token_ids),
    )


@app.post("/reward", response_model=RewardResponse, tags=["Evaluation"])
async def score_cover_letter(request: RewardRequest):
    """Score an existing cover letter using the reward model."""
    if _reward_model is None:
        raise HTTPException(status_code=503, detail="Reward model not loaded")

    prompt = f"Job Description:\n{request.job_description}\n\nCover Letter:\n"
    score     = _reward_model.score(prompt, request.cover_letter)
    breakdown = _reward_model.score_breakdown(prompt, request.cover_letter)

    if score >= 0.6:
        interpretation = "Excellent — professional, relevant, and well-structured."
    elif score >= 0.2:
        interpretation = "Good — solid cover letter with room for improvement."
    elif score >= -0.2:
        interpretation = "Average — meets basic requirements but lacks impact."
    else:
        interpretation = "Poor — significant improvements needed."

    return RewardResponse(
        score=round(score, 4),
        breakdown=ScoreBreakdown(**breakdown),
        interpretation=interpretation,
    )


@app.get("/samples", tags=["Generation"])
async def get_samples(n: int = 3):
    """Generate n sample cover letters using built-in prompts."""
    from src.data.dataset import JOB_DESCRIPTIONS, APPLICANT_PROFILES
    import random

    if _policy_model is None:
        raise HTTPException(status_code=503, detail="Policy model not loaded")

    results = []
    for _ in range(min(n, 5)):
        job     = random.choice(JOB_DESCRIPTIONS)
        profile = random.choice(APPLICANT_PROFILES)
        req = GenerateRequest(
            job_description=job,
            applicant_profile=profile,
            max_length=200,
        )
        result = await generate_cover_letter(req)
        results.append({
            "job_description":  job,
            "applicant_profile": profile,
            "cover_letter":     result.cover_letter,
            "reward_score":     result.reward_score,
        })

    return JSONResponse(content={"samples": results})
