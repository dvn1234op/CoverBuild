"""src/env/cover_letter_env.py — Custom Gymnasium environment for cover letter generation.

Architecture:
  - State:   Current token sequence being generated (int array)
  - Action:  Next token ID (Discrete(vocab_size)) — the policy chooses one token per step
  - Reward:  0 at each intermediate step. On episode end (EOS or max_length):
               → calls the reward model to score the full decoded text
               → returns scalar reward in [-1, 1]
  - Reset:   Samples a new job description prompt, resets token buffer
  - Render:  Decodes current token buffer to text

This env wraps nicely around TRL's PPOTrainer for the RLHF loop.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from transformers import AutoTokenizer

from src.data.dataset import JOB_DESCRIPTIONS, APPLICANT_PROFILES, build_prompt


class CoverLetterEnv(gym.Env):
    """
    Text generation environment for cover letter writing.

    The agent (policy model) generates a cover letter token-by-token.
    Episode ends when EOS token is produced or max_length is reached.
    Final reward is computed by the reward model on the complete text.
    """

    metadata = {"render_modes": ["human", "text"], "render_fps": 1}

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        reward_fn: Optional[callable] = None,
        max_length: int = 256,
        vocab_size: Optional[int] = None,
        render_mode: Optional[str] = "text",
    ):
        """
        Args:
            tokenizer:    HuggingFace tokenizer for the policy model.
            reward_fn:    Callable(prompt: str, completion: str) -> float.
                          If None, uses a simple heuristic scorer.
            max_length:   Maximum generation length (tokens).
            vocab_size:   Override tokenizer vocab size if needed.
            render_mode:  'human' (print) or 'text' (return str).
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.reward_fn = reward_fn or self._heuristic_reward
        self.max_length = max_length
        self.render_mode = render_mode

        # Vocabulary size
        _vocab_size = vocab_size or tokenizer.vocab_size
        self.vocab_size = _vocab_size

        # ── Spaces ──────────────────────────────────────────────────────────
        # Observation: current generated token sequence (padded to max_length)
        self.observation_space = spaces.Box(
            low=0,
            high=_vocab_size - 1,
            shape=(max_length,),
            dtype=np.int64,
        )

        # Action: choose the next token ID
        self.action_space = spaces.Discrete(_vocab_size)

        # ── Internal state ───────────────────────────────────────────────────
        self._current_prompt: str = ""
        self._prompt_ids: list[int] = []
        self._generated_ids: list[int] = []
        self._step_count: int = 0
        self._episode_reward: float = 0.0

        # EOS token id
        self._eos_token_id: int = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else _vocab_size - 1
        )

    # ── Core Gymnasium API ───────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Sample a new prompt and reset the generation state."""
        super().reset(seed=seed)

        # Sample job desc + applicant profile
        job = random.choice(JOB_DESCRIPTIONS)
        profile = random.choice(APPLICANT_PROFILES)
        self._current_prompt = build_prompt(job, profile)

        # Tokenize prompt
        encoded = self.tokenizer(
            self._current_prompt,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length // 2,  # leave room for generation
        )
        self._prompt_ids = encoded["input_ids"][0].tolist()
        self._generated_ids = []
        self._step_count = 0
        self._episode_reward = 0.0

        obs = self._get_obs()
        info = {
            "prompt": self._current_prompt,
            "prompt_length": len(self._prompt_ids),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Append one token (action) to the generated sequence.

        Returns:
            observation: updated token sequence
            reward:      0 at each step; final reward on terminal step
            terminated:  True when EOS or max_length reached
            truncated:   True if we hit hard env timeout
            info:        metadata dict
        """
        self._generated_ids.append(int(action))
        self._step_count += 1

        terminated = (
            int(action) == self._eos_token_id
            or len(self._generated_ids) >= self.max_length - len(self._prompt_ids)
        )
        truncated = self._step_count >= self.max_length * 2  # safety hard-stop

        reward = 0.0
        info: dict[str, Any] = {}

        if terminated or truncated:
            # Decode the generated completion (without the prompt)
            completion = self.tokenizer.decode(
                self._generated_ids, skip_special_tokens=True
            )
            # Compute final episodic reward
            reward = float(self.reward_fn(self._current_prompt, completion))
            self._episode_reward = reward
            info["generated_text"] = completion
            info["episode_reward"] = reward
            info["n_tokens"] = len(self._generated_ids)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Decode and display the current generated tokens."""
        text = (
            self._current_prompt
            + self.tokenizer.decode(self._generated_ids, skip_special_tokens=True)
        )
        if self.render_mode == "human":
            print("\n" + "─" * 60)
            print(text)
            print("─" * 60)
        return text

    def close(self):
        pass

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return the full token sequence as a padded numpy array."""
        full_ids = self._prompt_ids + self._generated_ids
        # Pad or truncate to max_length
        if len(full_ids) < self.max_length:
            padded = full_ids + [0] * (self.max_length - len(full_ids))
        else:
            padded = full_ids[-self.max_length:]
        return np.array(padded, dtype=np.int64)

    @staticmethod
    def _heuristic_reward(prompt: str, completion: str) -> float:
        """
        Fallback heuristic reward (used when no reward model is available).
        Scores based on length, keyword presence, and structure.
        Returns a value in [-1.0, 1.0].
        """
        if not completion.strip():
            return -1.0

        score = 0.0
        words = completion.split()
        n_words = len(words)

        # Length (100-350 words ideal)
        if 100 <= n_words <= 350:
            score += 0.3
        elif n_words < 30:
            score -= 0.5

        # Professionalism keywords
        good_kw = ["experience", "skills", "contribute", "team", "excited",
                   "passionate", "results", "expertise", "sincerely", "dear"]
        kw_hits = sum(1 for k in good_kw if k in completion.lower())
        score += min(kw_hits * 0.05, 0.3)

        # Structure check
        if "Dear" in completion:
            score += 0.1
        if any(w in completion for w in ["Sincerely", "Best regards", "Thank you"]):
            score += 0.1

        # Repetition penalty
        unique_ratio = len(set(words)) / max(n_words, 1)
        if unique_ratio < 0.5:
            score -= 0.2

        return round(max(-1.0, min(1.0, score)), 4)

    # ── Properties for external access ───────────────────────────────────────

    @property
    def current_prompt(self) -> str:
        return self._current_prompt

    @property
    def current_completion(self) -> str:
        return self.tokenizer.decode(self._generated_ids, skip_special_tokens=True)

    @property
    def full_sequence_ids(self) -> list[int]:
        return self._prompt_ids + self._generated_ids
