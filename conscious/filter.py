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
        self.salience_thresh = 0.35
        self.novelty_thresh = 0.25
        
        # FIXED KEYWORDS - removed problematic regex
        self.keywords = ["why", "how", "important", "remember", "idea", 
                         "solution", "problem", "question", "insight"]

    def _calculate_salience(self, thought: str) -> float:
        """Improved salience scoring without problematic regex"""
        score = 0.0
        
        # Length bonus
        word_count = len(thought.split())
        score += min(0.2, word_count / 50)
        
        # Question bonus
        if "?" in thought:
            score += 0.2
            
        # Self-reference bonus - FIXED REGEX
        if re.search(r"\b(I|me|my|mine)\b", thought, re.IGNORECASE):
            score += 0.1
            
        # Keyword bonus - SIMPLIFIED CHECK
        lower_thought = thought.lower()
        for keyword in self.keywords:
            if keyword in lower_thought:
                score += 0.1
                break  # Only add once
        
        # Complete sentence bonus
        if thought and thought[0].isupper() and thought.rstrip()[-1] in {'.', '?', '!'}:
            score += 0.05
            
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