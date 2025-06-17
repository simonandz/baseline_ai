# conscious/processor.py
from transformers import pipeline
import torch
from .config import REFINEMENT_MODEL

class ThoughtRefiner:
    def __init__(self):
        self.model = pipeline(
            "summarization",
            model=REFINEMENT_MODEL,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def refine(self, thought: str, category: str | None = None) -> str:
        token_len = len(thought.split())
        # If already concise (<12 words) or ends with '?', keep as‑is.
        if token_len < 12 or thought.strip().endswith("?"):
            return thought

        if category in {"problem_solving", "insight"}:
            prompt = f"Clarify this insight: {thought}"
        else:
            prompt = f"Summarize briefly: {thought}"

        return self.model(prompt, max_length=60, temperature=0.7)[0]["summary_text"]