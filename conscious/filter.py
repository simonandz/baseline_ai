# conscious/filter.py
"""
Thought filtering with salience and novelty scoring.
Uses centralized thresholds from config.yaml via config.py.
"""
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict
import logging

from .config import SALIENCE_THRESHOLD, NOVELTY_THRESHOLD

logger = logging.getLogger(__name__)


class ThoughtFilter:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.recent_embeddings = []

        # Use centralized thresholds from config
        self.salience_thresh = SALIENCE_THRESHOLD
        self.novelty_thresh = NOVELTY_THRESHOLD

        # Keywords for salience detection
        self.keywords = ["why", "how", "important", "remember", "idea",
                         "solution", "problem", "question", "insight"]

    def _calculate_salience(self, thought: str) -> float:
        """Calculate salience score based on curiosity and self-reference."""
        score = 0.0
        # Curiosity: proportion of sentences ending with '?'
        sentences = re.split(r"(?<=[.!?])\s+", thought)
        q_count = sum(1 for s in sentences if s.strip().endswith("?"))
        score += 0.25 * min(1, q_count)  # max +0.25
        # Self-reference
        if re.search(r"\b(I|me|my|mine)\b", thought, re.I):
            score += 0.10
        # Length bonus (avoid trivial questions)
        score += min(0.2, len(thought.split()) / 60)
        return min(1.0, score)


    def _calculate_novelty(self, embedding: np.ndarray) -> float:
        """Embedding-based novelty check"""
        if not self.recent_embeddings:
            return 1.0
            
        similarities = cosine_similarity([embedding], self.recent_embeddings)[0]
        return 1.0 - np.max(similarities)

    def evaluate(self, thought: str) -> Dict:
        """Full evaluation pipeline"""
        try:
            embedding = self.embedder.encode([thought])[0]
            
            # Update tracking
            self.recent_embeddings.append(embedding)
            if len(self.recent_embeddings) > 50:
                self.recent_embeddings.pop(0)
            
            # Calculate scores
            salience = self._calculate_salience(thought)
            novelty = self._calculate_novelty(embedding)
            
            return {
                "raw": thought,
                "salience": salience,
                "novelty": novelty,
                "passes": (salience >= self.salience_thresh) 
                        or (novelty >= self.novelty_thresh)
            }
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {
                "raw": thought,
                "salience": 0.5,
                "novelty": 1.0,
                "passes": True  # Default to passing on error
            }