"""tests/test_env.py — Tests for the CoverLetterEnv Gymnasium environment."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from transformers import AutoTokenizer

from src.env.cover_letter_env import CoverLetterEnv


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def env(tokenizer):
    return CoverLetterEnv(tokenizer=tokenizer, max_length=64)


class TestCoverLetterEnv:

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (64,)
        assert obs.dtype == np.int64

    def test_reset_info_has_prompt(self, env):
        _, info = env.reset(seed=42)
        assert "prompt" in info
        assert len(info["prompt"]) > 0
        assert "prompt_length" in info

    def test_observation_space(self, env):
        assert env.observation_space.shape == (64,)
        assert env.action_space.n == env.vocab_size

    def test_step_returns_tuple(self, env):
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_reward_zero_on_nonterminal(self, env):
        env.reset(seed=1)
        # Take a step with a non-EOS, non-terminating action
        action = 50  # arbitrary non-EOS token
        obs, reward, terminated, truncated, info = env.step(action)
        if not terminated and not truncated:
            assert reward == 0.0

    def test_terminal_on_eos(self, env):
        env.reset(seed=2)
        eos_action = env._eos_token_id
        _, reward, terminated, _, info = env.step(eos_action)
        assert terminated is True
        assert "episode_reward" in info
        assert "generated_text" in info

    def test_heuristic_reward_empty_string(self, env):
        r = CoverLetterEnv._heuristic_reward("some prompt", "")
        assert r == -1.0

    def test_heuristic_reward_good_text(self, env):
        good_text = (
            "Dear Hiring Manager, I am excited to apply for this position. "
            "With three years of experience in software engineering, I have demonstrated "
            "the ability to deliver high-quality results. My skills in Python and REST APIs "
            "make me well-suited for your team. I look forward to contributing. Sincerely, Applicant."
        )
        r = CoverLetterEnv._heuristic_reward("Python engineer job", good_text)
        assert r > 0.0

    def test_render_returns_string(self, env):
        env.reset(seed=3)
        text = env.render()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_current_prompt_property(self, env):
        env.reset(seed=4)
        assert isinstance(env.current_prompt, str)
        assert len(env.current_prompt) > 0

    def test_gymnasium_check(self, tokenizer):
        """Verify the env passes basic gymnasium API checks."""
        env = CoverLetterEnv(tokenizer=tokenizer, max_length=32)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
