import threading
import time
import numpy as np
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Subconscious:
    def __init__(self, output_queue, memory=None, model_name="EleutherAI/gpt-neo-1.3B"):
        self.queue = output_queue
        self.memory = memory
        self.model_name = model_name
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        logger.info(f"Loading generator: {model_name}")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        self.recent_embeddings = []
        logger.info("Subconscious initialized")

    def _is_duplicate(self, thought: str) -> bool:
        """Improved semantic duplicate detection"""
        if not thought.strip():
            return True
            
        embedding = self.embedder.encode([thought])[0]
        
        if self.recent_embeddings:
            similarities = np.dot(self.recent_embeddings, embedding) / (
                np.linalg.norm(self.recent_embeddings, axis=1) * np.linalg.norm(embedding)
            )
            if np.any(similarities > 0.85):
                return True
                
        self.recent_embeddings.append(embedding)
        if len(self.recent_embeddings) > 100:
            self.recent_embeddings.pop(0)
        return False

    def _generate_thought(self) -> str:
        """Generate with context awareness"""
        context = ""
        if self.memory:
            try:
                context = "\n".join(self.memory.get_recent(2))
            except Exception as e:
                logger.error(f"Memory error: {e}")
        
        prompt = f"""
        Context: {context}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Generate a novel thought about current context or a random observation:
        """
        
        for _ in range(3):  # Retry on failure
            try:
                result = self.generator(
                    prompt,
                    max_new_tokens=40,
                    temperature=0.8,
                    do_sample=True
                )
                thought = result[0]['generated_text'].replace(prompt, "").strip()
                
                # Extract first complete sentence
                if '.' in thought:
                    thought = thought.split('.')[0] + '.'
                elif '?' in thought:
                    thought = thought.split('?')[0] + '?'
                
                if thought and not self._is_duplicate(thought):
                    return thought
                    
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM - reducing model precision")
                self.generator.model = self.generator.model.half()
            except Exception as e:
                logger.error(f"Generation error: {e}")
        
        return ""

    def _run(self):
        logger.info("Subconscious thread started")
        while not self._stop_event.is_set():
            try:
                thought = self._generate_thought()
                if thought:
                    self.queue.put(thought)
                    logger.debug(f"Generated: {thought}")
                time.sleep(3)  # Interval
            except Exception as e:
                logger.error(f"Runtime error: {e}")
        logger.info("Subconscious thread stopped")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        logger.info("Subconscious stopped")