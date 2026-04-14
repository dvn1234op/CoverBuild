"""src/data/dataset.py — CoverLetterDataset for SFT training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Synthetic cover letter corpus (used when no external dataset is provided)
# ---------------------------------------------------------------------------

JOB_DESCRIPTIONS = [
    "Software Engineer at a fintech startup. Requirements: Python, REST APIs, SQL, team player.",
    "Data Scientist at a healthcare company. Requirements: ML, scikit-learn, statistics, Python.",
    "Backend Developer at an e-commerce firm. Requirements: Node.js, PostgreSQL, AWS, microservices.",
    "Machine Learning Engineer at an AI lab. Requirements: PyTorch, model training, CUDA, research.",
    "DevOps Engineer at a SaaS company. Requirements: Kubernetes, Docker, CI/CD, Linux, Terraform.",
    "Product Manager at a consumer app. Requirements: roadmaps, Agile, stakeholder management, analytics.",
    "Frontend Engineer at a design agency. Requirements: React, TypeScript, CSS, Figma, accessibility.",
    "NLP Engineer at a language AI company. Requirements: transformers, HuggingFace, fine-tuning, BERT.",
    "Cloud Architect at a consulting firm. Requirements: AWS/GCP/Azure, solution design, IaC, security.",
    "Research Scientist at a robotics company. Requirements: reinforcement learning, C++, ROS, simulation.",
]

APPLICANT_PROFILES = [
    "3 years Python experience, built REST APIs, B.Sc. Computer Science, GitHub portfolio.",
    "MSc Data Science, 2 years at analytics firm, published paper on NLP, Kaggle top 10%.",
    "5 years backend experience, led migration to microservices, AWS certified.",
    "PhD candidate in ML, research on LLM alignment, multiple conference publications.",
    "4 years DevOps, reduced deployment time 60%, certified Kubernetes administrator.",
    "2 years PM at SaaS company, launched 3 features, strong analytics background.",
    "3 years frontend, React expert, accessibility advocate, design systems contributor.",
    "2 years NLP research, fine-tuned BERT for 5 NLP tasks, HuggingFace contributor.",
    "7 years cloud, designed multi-region AWS architectures, AWS Solutions Architect Pro.",
    "PhD Robotics, 4 years RL research, 3 published papers, open-source ROS packages.",
]

COVER_LETTER_TEMPLATES = [
    """\
Dear Hiring Manager,

I am excited to apply for the {role} position. With {experience}, I bring a strong foundation in the skills you seek.

Throughout my career, I have demonstrated the ability to {achievement}. I am passionate about {passion} and believe this aligns perfectly with your company's mission.

My technical expertise includes {skills}, which I have applied in real-world projects to deliver measurable results. I thrive in collaborative environments and consistently bring dedication to every challenge.

I would welcome the opportunity to discuss how my background can contribute to your team's success.

Sincerely,
[Applicant Name]""",

    """\
Dear Recruiting Team,

Having reviewed the {role} opportunity, I am confident that my background makes me an exceptional candidate.

{experience_expanded}. My hands-on experience with {skills} has allowed me to deliver high-quality results consistently.

I am particularly drawn to this role because {reason}. I am eager to bring my {strength} to your organization and grow alongside a talented team.

Thank you for considering my application. I look forward to the possibility of contributing to your success.

Best regards,
[Applicant Name]""",
]

FILL_VARS = [
    {
        "role": "Software Engineer",
        "experience": "3 years of Python development and REST API design",
        "achievement": "deliver scalable backend systems under tight deadlines",
        "passion": "clean, maintainable code and system design",
        "skills": "Python, FastAPI, PostgreSQL, and AWS",
        "experience_expanded": "I have spent 3 years building production-grade APIs",
        "reason": "I admire your focus on fintech innovation",
        "strength": "systems thinking and Python expertise",
    },
    {
        "role": "Data Scientist",
        "experience": "2 years in analytics with an MSc in Data Science",
        "achievement": "develop ML models that drive business intelligence",
        "passion": "transforming raw data into actionable insights",
        "skills": "Python, scikit-learn, SQL, and Tableau",
        "experience_expanded": "I have built predictive models deployed in production environments",
        "reason": "healthcare data science represents the intersection of impact and innovation",
        "strength": "statistical rigor and NLP research background",
    },
    {
        "role": "Machine Learning Engineer",
        "experience": "PhD-level expertise in LLM alignment and model training",
        "achievement": "push the boundaries of AI capabilities",
        "passion": "advancing the state of artificial intelligence",
        "skills": "PyTorch, HuggingFace Transformers, CUDA, and distributed training",
        "experience_expanded": "My research background spans reinforcement learning and large language models",
        "reason": "your lab is at the frontier of AI research",
        "strength": "deep technical research and implementation skills",
    },
]


def build_prompt(job_description: str, applicant_profile: str) -> str:
    """
    Build a prompt in Qwen2.5-Instruct chat format.
    Works with raw instruction-tuned model AND after SFT/RL fine-tuning.
    """
    return (
        "<|im_start|>system\n"
        "You are a professional cover letter writer. Write a concise, compelling cover letter "
        "based on the job description and applicant profile. Use a formal, confident tone. "
        "Include: greeting, 2-3 body paragraphs, and a closing.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Job Description:\n{job_description}\n\n"
        f"Applicant Profile:\n{applicant_profile}\n\n"
        "Write a professional cover letter for this position.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_synthetic_cover_letter(idx: int) -> dict:
    """Generate one synthetic (prompt, completion) pair."""
    template = random.choice(COVER_LETTER_TEMPLATES)
    fill = FILL_VARS[idx % len(FILL_VARS)]
    letter = template.format(**fill)
    job = JOB_DESCRIPTIONS[idx % len(JOB_DESCRIPTIONS)]
    profile = APPLICANT_PROFILES[idx % len(APPLICANT_PROFILES)]
    prompt = build_prompt(job, profile)
    return {"prompt": prompt, "completion": letter, "text": prompt + letter}


def build_synthetic_dataset(
    n_samples: int = 500, save_path: Optional[str] = None
) -> list[dict]:
    """Generate a synthetic cover letter dataset."""
    data = [generate_synthetic_cover_letter(i) for i in range(n_samples)]

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"✅ Saved {n_samples} samples to {save_path}")

    return data


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CoverLetterDataset(Dataset):
    """
    Dataset for SFT training on cover letters.

    Each item is a tokenized (prompt + completion) pair.
    The loss is computed only on the completion tokens (prompt tokens are masked).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: Optional[str] = None,
        n_synthetic: int = 500,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []

        if data_path and Path(data_path).exists():
            with open(data_path) as f:
                self.samples = [json.loads(line) for line in f]
            print(f"📂 Loaded {len(self.samples)} samples from {data_path}")
        else:
            print(f"⚙️  Generating {n_synthetic} synthetic cover letter samples...")
            self.samples = build_synthetic_dataset(n_synthetic)
            if data_path:
                build_synthetic_dataset(n_synthetic, data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample.get("text", sample["prompt"] + sample["completion"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Mask the prompt tokens in labels so loss is on completion only
        prompt_encoding = self.tokenizer(
            sample["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_prompt_list(self) -> list[str]:
        """Return just the prompts — used for PPO rollouts."""
        return [s["prompt"] for s in self.samples]
