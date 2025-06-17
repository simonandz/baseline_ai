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
        # keep thoughts that already include at least one question and ≤3 sentences
        if "?" in thought and len(re.split(r"(?<=[.!?])\s+", thought)) <= 3:
            return thought
        token_len = len(thought.split())
        if token_len < 12 or thought.strip().endswith("?"):
            return thought
        prompt = "Explain this clearly in one sentence: " + thought
        return self.model(prompt, max_length=60, temperature=0.3)[0]["summary_text"]