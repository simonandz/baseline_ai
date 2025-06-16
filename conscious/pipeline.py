# conscious/pipeline.py
from .filter import ThoughtFilter
from .processor import ThoughtRefiner

class ConsciousPipeline:
    def __init__(self):
        self.filter = ThoughtFilter()
        self.refiner = ThoughtRefiner()
    
    def process(self, thought: str) -> Dict:
        """Full conscious processing"""
        evaluation = self.filter.evaluate(thought)
        
        if evaluation["passes"]:
            refined = self.refiner.refine(evaluation["raw"])
            return {**evaluation, "refined": refined}
        
        return evaluation