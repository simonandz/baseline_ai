import numpy as np
import re
from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

class ConsciousProcessor:
    def __init__(self):
        # Load models
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.refiner = pipeline(
            "summarization",
            model="facebook/bart-large",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # State
        self.recent_embeddings = []
        self.recent_thoughts = []
        
        # Categories
        self.categories = {
            "problem_solving": ["solve", "fix", "debug"],
            "insight": ["realize", "understand", "meaning"],
            "planning": ["plan", "schedule", "next"],
            "curiosity": ["why", "how", "wonder"],
            "memory": ["remember", "recall", "nostalgia"]
        }

    def _calculate_salience(self, thought: str) -> float:
        """Dynamic salience scoring"""
        score = min(0.2, len(thought.split()) / 50)  # Length bonus
        
        # Question bonus
        if "?" in thought:
            score += 0.2
            
        # Self-reference bonus
        if re.search(r"\b(I|me|my)\b", thought, re.IGNORECASE):
            score += 0.15
            
        # Complete sentence bonus
        if thought[0].isupper() and thought.rstrip()[-1] in {'.','?','!'}:
            score += 0.1
            
        return min(1.0, score)

    def _calculate_novelty(self, embedding: np.ndarray) -> float:
        """Embedding-based novelty"""
        if not self.recent_embeddings:
            return 1.0
        similarities = np.dot(self.recent_embeddings, embedding)
        return 1.0 - np.max(similarities) if similarities.size > 0 else 1.0

    def process_thought(self, thought: str) -> Dict:
        """Full processing pipeline"""
        # Generate embedding
        embedding = self.embedder.encode([thought])[0]
        
        # Calculate metrics
        salience = self._calculate_salience(thought)
        novelty = self._calculate_novelty(embedding)
        
        # Update memory
        self.recent_embeddings.append(embedding)
        if len(self.recent_embeddings) > 50:
            self.recent_embeddings.pop(0)
            
        # Determine if thought passes
        passes = (salience >= 0.35 or novelty >= 0.7)
        
        # Refine if passing
        refined = self._refine_thought(thought) if passes else None
        
        return {
            "raw": thought,
            "refined": refined,
            "salience": salience,
            "novelty": novelty,
            "passed": passes
        }

    def _refine_thought(self, thought: str) -> str:
        """Quality refinement"""
        result = self.refiner(
            thought,
            max_length=60,
            temperature=0.7,
            num_beams=2,
            early_stopping=False
        )
        return result[0]['summary_text']