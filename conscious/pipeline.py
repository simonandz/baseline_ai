# conscious/pipeline.py
from typing import Dict  # IMPORT ADDED HERE
from .filter import ThoughtFilter
from .processor import ThoughtRefiner

class ConsciousProcessor:
    def __init__(self):
        self.filter = ThoughtFilter()
        self.refiner = ThoughtRefiner()
        self.recent_embeddings = []
        
    def process(self, thought: str) -> Dict:  # Now Dict is defined
        """Full conscious processing pipeline"""
        # Generate evaluation
        evaluation = self.filter.evaluate(thought)
        
        # Update embeddings for novelty tracking
        if hasattr(self.filter, 'embedder'):
            embedding = self.filter.embedder.encode([thought])[0]
            self.recent_embeddings.append(embedding)
            if len(self.recent_embeddings) > 50:
                self.recent_embeddings.pop(0)
        
        # Refine and return if passes
        if evaluation["passes"]:
            refined = self.refiner.refine(thought)
            return {**evaluation, "refined": refined}
        
        return evaluation