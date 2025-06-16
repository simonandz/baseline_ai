# conscious/filter.py
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict

class ThoughtFilter:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.recent_embeddings = []
        
        # Thresholds (from config.py)
        self.salience_thresh = 0.35
        self.novelty_thresh = 0.25
        self.relevance_thresh = 0.3

    def _calculate_salience(self, thought: str) -> float:
        """Improved salience scoring"""
        score = 0.0
        
        # Length bonus
        word_count = len(thought.split())
        score += min(0.2, word_count / 50)
        
        # Keyword bonuses
        keywords = {
            "why": 0.15, "how": 0.15, "?": 0.2, 
            r"\b(I|me|my)\b": 0.1
        }
        
        for pattern, bonus in keywords.items():
            if re.search(pattern, thought, re.IGNORECASE):
                score += bonus
                
        # Complete sentence bonus
        if thought[0].isupper() and thought.endswith(('.','?','!')):
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