import torch
from transformers import pipeline
from .config import REFINEMENT_MODEL, REFINEMENT_MAX_TOKENS, REFINEMENT_TEMPERATURE

class ThoughtRefiner:
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model=REFINEMENT_MODEL,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def refine_thought(self, thought: str, category: str) -> str:
        """Rephrase thought for clarity and coherence"""
        # Category-specific prompts
        prompts = {
            "problem_solving": "Clearly state the problem and potential solution: ",
            "planning": "Convert this into an actionable plan: ",
            "insight": "Articulate this insight clearly: ",
            "curiosity": "Formulate this as a precise question: ",
            "memory": "Describe this memory concisely: ",
            "emotion": "Express this feeling precisely: ",
            "misc": "Rephrase this thought clearly: "
        }
        
        prompt = prompts.get(category, prompts["misc"]) + thought
        
        # Generate refined version
        refined = self.summarizer(
            prompt,
            max_length=REFINEMENT_MAX_TOKENS,
            temperature=REFINEMENT_TEMPERATURE,
            num_beams=4,
            early_stopping=True
        )[0]['summary_text']
        
        # Post-processing
        refined = refined.replace("n't", " not").replace("'s", " is")
        return refined.strip()