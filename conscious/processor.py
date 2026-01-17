# conscious/processor.py
"""
Thought refinement using BART summarization.
"""
from transformers import pipeline
import re
import torch
import logging
from .config import REFINEMENT_MODEL, REFINEMENT_MAX_TOKENS

logger = logging.getLogger(__name__)


class ThoughtRefiner:
    def __init__(self):
        self.model = pipeline(
            "summarization",
            model=REFINEMENT_MODEL,
            device=0 if torch.cuda.is_available() else -1
        )

    def refine(self, thought: str) -> str:
        """
        Refine a thought by summarizing if too long.
        Short thoughts and questions are passed through unchanged.
        """
        # Keep thoughts that already include a question and are concise
        if "?" in thought and len(re.split(r"(?<=[.!?])\s+", thought)) <= 3:
            return thought

        token_len = len(thought.split())

        # Short thoughts or questions pass through
        if token_len < 12 or thought.strip().endswith("?"):
            return thought

        # Long thoughts get summarized
        try:
            # Summarization pipeline expects the text directly
            result = self.model(
                thought,
                max_length=REFINEMENT_MAX_TOKENS,
                min_length=10,
                do_sample=False
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.warning(f"Refinement failed, returning original: {e}")
            return thought