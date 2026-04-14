"""src/api/schemas.py — Pydantic request/response schemas for CoverBuild API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The job posting / requirements description",
        json_schema_extra={"example": "Software Engineer at a fintech startup. Requirements: Python, REST APIs, SQL."},
    )
    applicant_profile: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Brief description of the applicant's background",
        json_schema_extra={"example": "3 years Python experience, built REST APIs, B.Sc. Computer Science."},
    )
    max_length: int = Field(
        default=256,
        ge=50,
        le=512,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.9,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more creative)",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.1,
        le=1.0,
        description="Nucleus sampling probability",
    )


class RewardRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The job description context",
    )
    cover_letter: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="The cover letter text to score",
    )


# ── Response schemas ─────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    structure: float   = Field(..., description="Structural quality [0-1]")
    relevance: float   = Field(..., description="Job relevance [0-1]")
    tone: float        = Field(..., description="Professionalism of tone [0-1]")
    conciseness: float = Field(..., description="Length & clarity [0-1]")
    aggregate: float   = Field(..., description="Weighted aggregate score [-1 to 1]")


class GenerateResponse(BaseModel):
    cover_letter: str         = Field(..., description="Generated cover letter text")
    reward_score: float       = Field(..., description="Reward model score [-1 to 1]")
    score_breakdown: ScoreBreakdown
    generation_time_ms: float = Field(..., description="Generation latency in ms")
    model_version: str        = Field(..., description="Model version used")
    prompt_tokens: int        = Field(..., description="Number of tokens in the prompt")
    generated_tokens: int     = Field(..., description="Number of tokens generated")


class RewardResponse(BaseModel):
    score: float              = Field(..., description="Aggregate reward [-1 to 1]")
    breakdown: ScoreBreakdown
    interpretation: str       = Field(..., description="Human-readable score interpretation")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    reward_model_loaded: bool
    device: str
    model_name: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    device: str
    vocab_size: int
    max_position_embeddings: int
    use_lora: bool
    reward_model_name: str
    reward_model_path: str
