"""src/data/synthetic_prefs.py — Generate pairwise preference data for reward model training.

Strategy:
  For each prompt, generate multiple completions with different quality heuristics,
  then rank them to create (prompt, chosen, rejected) triples used by RewardTrainer.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .dataset import JOB_DESCRIPTIONS, APPLICANT_PROFILES, build_prompt


# ---------------------------------------------------------------------------
# Quality Heuristics (simulate human preference scoring)
# ---------------------------------------------------------------------------

GOOD_KEYWORDS = [
    "excited", "passionate", "experienced", "demonstrated", "delivered",
    "contribute", "collaborate", "results", "expertise", "achieve",
    "proven", "skilled", "dedicated", "motivated", "innovative",
]

BAD_PATTERNS = [
    r"\bi\s+am\s+a\s+person\b",   # overly generic
    r"\bto\s+whom\s+it\s+may\s+concern\b",  # outdated opener
    r"\bblah\b", r"\bstuff\b", r"\bthing\b",  # vague language
    r"(.)\1{4,}",                 # repeated characters
]

REQUIRED_SECTIONS = ["Dear", "Sincerely", "experience", "skills"]


def score_completion_heuristic(completion: str) -> float:
    """
    Rule-based heuristic scorer (0.0 → 1.0).
    Models what a human might find in a good cover letter.
    """
    score = 0.5  # baseline

    # Length check (150–400 words is ideal)
    word_count = len(completion.split())
    if 150 <= word_count <= 400:
        score += 0.15
    elif word_count < 80 or word_count > 600:
        score -= 0.2

    # Good keyword density
    text_lower = completion.lower()
    good_hits = sum(1 for kw in GOOD_KEYWORDS if kw in text_lower)
    score += min(good_hits * 0.02, 0.15)

    # Required structural sections
    section_hits = sum(1 for s in REQUIRED_SECTIONS if s.lower() in text_lower)
    score += section_hits * 0.05

    # Penalise bad patterns
    for pat in BAD_PATTERNS:
        if re.search(pat, text_lower, re.IGNORECASE):
            score -= 0.1

    # Paragraph count (2–4 paragraphs preferred)
    paragraphs = [p.strip() for p in completion.split("\n\n") if p.strip()]
    if 2 <= len(paragraphs) <= 4:
        score += 0.05

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Completion variants — deliberately varying quality
# ---------------------------------------------------------------------------

GOOD_COMPLETIONS = [
    """\
Dear Hiring Manager,

I am excited to apply for the {role} position at your esteemed organization. With {experience}, I have cultivated a strong command of the technical and collaborative skills your team requires.

In my previous roles, I have demonstrated the ability to deliver {deliverable} while working closely with cross-functional teams. I am particularly passionate about {passion}, which makes this opportunity especially compelling.

My expertise in {skills} positions me to make an immediate and meaningful contribution. I am eager to bring my dedication and problem-solving mindset to your team.

Thank you sincerely for your time and consideration. I look forward to discussing this opportunity further.

Sincerely,
[Applicant Name]""",

    """\
Dear Recruiting Team,

Having closely reviewed the {role} opportunity, I believe my background in {experience} makes me an excellent fit for your team's needs.

Throughout my professional journey, I have consistently delivered {deliverable}, demonstrating both technical excellence and strong teamwork. I thrive in fast-paced environments where {passion} is valued.

I bring hands-on expertise in {skills}, and I am confident these strengths will allow me to contribute meaningfully from day one. I am also highly adaptable and committed to continuous learning.

I would be grateful for the opportunity to discuss how my profile aligns with your goals. Thank you for considering my application.

Best regards,
[Applicant Name]""",
]

BAD_COMPLETIONS = [
    "I want this job. I know stuff about {skills}. Hire me please. I am a person who does things.",
    "To Whom It May Concern, I am applying for the job. I have experience. I can do the stuff you need. Thanks.",
    "Hello, I am very good developer. I have many skill like {skills}. Please give job. I am hardworking person.",
    "{role} job looks good. I think i am fit for this role because I have experience and skills. Looking forward.",
]

FILL_VARS = [
    {
        "role": "Software Engineer",
        "experience": "3 years of Python and API development",
        "deliverable": "robust backend systems",
        "passion": "clean architecture and scalable code",
        "skills": "Python, FastAPI, PostgreSQL, and AWS",
    },
    {
        "role": "Data Scientist",
        "experience": "2 years in predictive analytics",
        "deliverable": "ML models that improved business KPIs by 30%",
        "passion": "turning data into strategic decisions",
        "skills": "Python, scikit-learn, SQL, and data visualization",
    },
    {
        "role": "Machine Learning Engineer",
        "experience": "PhD-level research in LLM alignment",
        "deliverable": "state-of-the-art NLP models",
        "passion": "advancing AI capabilities responsibly",
        "skills": "PyTorch, HuggingFace Transformers, and CUDA",
    },
]


def generate_pref_pair(idx: int) -> dict:
    """Generate one (prompt, chosen, rejected) preference triple."""
    fill = FILL_VARS[idx % len(FILL_VARS)]
    job = JOB_DESCRIPTIONS[idx % len(JOB_DESCRIPTIONS)]
    profile = APPLICANT_PROFILES[idx % len(APPLICANT_PROFILES)]

    prompt = build_prompt(job, profile)

    chosen_template = random.choice(GOOD_COMPLETIONS)
    rejected_template = random.choice(BAD_COMPLETIONS)

    chosen = chosen_template.format(**fill)
    rejected = rejected_template.format(**fill)

    chosen_score = score_completion_heuristic(chosen)
    rejected_score = score_completion_heuristic(rejected)

    # Ensure chosen > rejected (swap if needed — shouldn't happen but safety check)
    if chosen_score < rejected_score:
        chosen, rejected = rejected, chosen
        chosen_score, rejected_score = rejected_score, chosen_score

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
    }


def build_preference_dataset(
    n_samples: int = 300, save_path: str = "data/preferences/prefs.jsonl"
) -> list[dict]:
    """Generate and save pairwise preference dataset."""
    data = [generate_pref_pair(i) for i in range(n_samples)]

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    avg_chosen = sum(d["chosen_score"] for d in data) / len(data)
    avg_rejected = sum(d["rejected_score"] for d in data) / len(data)
    print(f"✅ Saved {n_samples} preference pairs to {save_path}")
    print(f"   Avg chosen score:   {avg_chosen:.3f}")
    print(f"   Avg rejected score: {avg_rejected:.3f}")
    return data


if __name__ == "__main__":
    build_preference_dataset()
