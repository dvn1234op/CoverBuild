"""tests/test_reward.py — Tests for the reward model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.models.reward_model import RewardModel, heuristic_breakdown


GOOD_LETTER = """Dear Hiring Manager,

I am excited to apply for the Software Engineer position at your company. 
With three years of Python and API development experience, I have demonstrated 
the ability to deliver scalable backend systems on time.

My expertise in FastAPI, PostgreSQL, and AWS aligns well with your requirements. 
I am passionate about clean code and system design, and I thrive in collaborative teams.

I would welcome the opportunity to contribute to your mission. Thank you for your consideration.

Sincerely,
Jane Doe"""

BAD_LETTER = "I want this job. hire me please. I know stuff. thanks."

SAMPLE_PROMPT = "### Job Description:\nSoftware Engineer\n\n### Cover Letter:\n"


class TestHeuristicBreakdown:

    def test_good_letter_scores_high(self):
        scores = heuristic_breakdown(SAMPLE_PROMPT, GOOD_LETTER)
        assert scores["structure"] > 0.3
        assert scores["relevance"] > 0.1
        assert "aggregate" not in scores  # breakdown doesn't include aggregate

    def test_bad_letter_scores_lower(self):
        good_scores = heuristic_breakdown(SAMPLE_PROMPT, GOOD_LETTER)
        bad_scores  = heuristic_breakdown(SAMPLE_PROMPT, BAD_LETTER)
        assert good_scores["structure"] >= bad_scores["structure"]

    def test_all_dimensions_present(self):
        scores = heuristic_breakdown(SAMPLE_PROMPT, GOOD_LETTER)
        for dim in ["structure", "relevance", "tone", "conciseness"]:
            assert dim in scores
            assert 0.0 <= scores[dim] <= 1.0

    def test_empty_completion(self):
        scores = heuristic_breakdown(SAMPLE_PROMPT, "")
        assert scores["conciseness"] < 0.5


class TestRewardModel:

    @pytest.fixture(scope="class")
    def reward_model(self):
        return RewardModel.fresh(base_model_name="distilbert-base-uncased", device="cpu")

    def test_score_returns_float(self, reward_model):
        score = reward_model.score(SAMPLE_PROMPT, GOOD_LETTER)
        assert isinstance(score, float)

    def test_score_in_range(self, reward_model):
        score = reward_model.score(SAMPLE_PROMPT, GOOD_LETTER)
        assert -1.5 <= score <= 1.5  # heuristic range

    def test_breakdown_has_all_keys(self, reward_model):
        bd = reward_model.score_breakdown(SAMPLE_PROMPT, GOOD_LETTER)
        for key in ["structure", "relevance", "tone", "conciseness", "aggregate"]:
            assert key in bd

    def test_good_scores_higher_than_bad(self, reward_model):
        good_s = reward_model.score(SAMPLE_PROMPT, GOOD_LETTER)
        bad_s  = reward_model.score(SAMPLE_PROMPT, BAD_LETTER)
        assert good_s >= bad_s

    def test_heuristic_fallback_when_untrained(self, reward_model):
        """Fresh model uses heuristic fallback — should still return a score."""
        score = reward_model.score(SAMPLE_PROMPT, GOOD_LETTER)
        assert score is not None
