# conscious/filter.py
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ThoughtFilter:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.recent_embeddings = []
        
        # Thresholds
        self.salience_thresh = 0.40
        self.novelty_thresh = 0.35
        
        # FIXED KEYWORDS - removed problematic regex
        self.keywords = ["why", "how", "important", "remember", "idea", 
                         "solution", "problem", "question", "insight"]

# Raise novelty threshold & add explicit bonus for interrogatives.
# (Replace _calculate_salience with this version.)

    def _calculate_salience(self, thought: str) -> float:
        score = 0.0
        # curiosity: proportion of sentences ending with '?'
        sentences = re.split(r"(?<=[.!?])\s+", thought)
        q_count = sum(1 for s in sentences if s.strip().endswith("?"))
        score += 0.25 * min(1, q_count)          # max +0.25
        # self‑reference
        if re.search(r"\b(I|me|my|mine)\b", thought, re.I):
            score += 0.10
        # length bonus (avoid trivial questions)
        score += min(0.2, len(thought.split()) / 60)
        return min(1.0, score)

    # thresholds inside __init__
        self.salience_thresh = 0.45
        self.novelty_thresh  = 0.38


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