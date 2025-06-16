import threading
import time
import numpy as np
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging
import queue
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Subconscious:
    def __init__(
        self,
        output_queue: queue.Queue,
        memory=None,
        model_name: str = "EleutherAI/gpt-neo-1.3B",
        interval: float = 3.0,
        device: torch.device = None  # Add device parameter
    ):
        """
        Continuous thought generator
        
        Args:
            output_queue: Queue to send thoughts to conscious module
            memory: MemoryManager instance for context
            model_name: HF model identifier
            interval: Seconds between thought generation
        """
        self.queue = output_queue
        self.memory = memory
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        
        # Models
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"Loading generator: {model_name}")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=device if device else 0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # State
        self.recent_embeddings = []
        self.thought_count = 0
        logger.info("Subconscious initialized")

    def _get_context(self) -> str:
        """Get relevant context from memory"""
        if not self.memory:
            return ""
        
        try:
            recent = self.memory.get_recent_memories(2)
            return "Recent memories:\n" + "\n".join(recent) if recent else ""
        except Exception as e:
            logger.error(f"Context error: {e}")
            return ""

    def _is_duplicate(self, thought: str) -> bool:
        """Check if thought is semantically similar to recent ones"""
        if not thought.strip():
            return True
            
        embedding = self.embedder.encode([thought])[0]
        
        if self.recent_embeddings:
            # Convert to numpy array for sklearn
            embeddings_array = np.array(self.recent_embeddings)
            
            # Calculate similarity - note: cosine_similarity expects 2D arrays
            similarities = cosine_similarity(
                [embedding],  # Query embedding (1 x D)
                embeddings_array  # Recent embeddings (N x D)
            )
            
            if np.max(similarities) > 0.85:  # Similarity threshold
                return True
                
        self.recent_embeddings.append(embedding)
        if len(self.recent_embeddings) > 100:  # Keep last 100 embeddings
            self.recent_embeddings.pop(0)
        return False

    def _generate_thought(self) -> Optional[str]:
        """Generate one thought with context awareness"""
        context = self._get_context()
        prompt = f"""
        {context}
        Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Generate a novel, concise thought (1 sentence) about:
        - Current context if relevant
        - Or an interesting observation
        Thought:"""
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                result = self.generator(
                    prompt,
                    max_new_tokens=40,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.generator.tokenizer.eos_token_id
                )
                raw_thought = result[0]['generated_text'].replace(prompt, "").strip()
                
                # Extract first complete sentence
                if '.' in raw_thought:
                    thought = raw_thought.split('.')[0] + '.'
                elif '?' in raw_thought:
                    thought = raw_thought.split('?')[0] + '?'
                else:
                    thought = raw_thought
                
                if thought and not self._is_duplicate(thought):
                    return thought
                    
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM - reducing batch size")
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Generation error (attempt {attempt+1}): {e}")
        
        return None

    def _run(self):
        """Main generation loop"""
        logger.info("Subconscious thread started")
        while not self._stop_event.is_set():
            try:
                thought = self._generate_thought()
                if thought:
                    self.queue.put(thought)
                    self.thought_count += 1
                    logger.debug(f"Generated thought #{self.thought_count}: {thought}")
            except Exception as e:
                logger.error(f"Runtime error: {e}")
            
            time.sleep(self.interval)
        logger.info("Subconscious thread stopped")

    def start(self):
        """Start the generation thread"""
        self._thread.start()

    def stop(self):
        """Stop the generation thread"""
        self._stop_event.set()
        self._thread.join()
        logger.info("Subconscious stopped")

# Test code must come AFTER the class definition
if __name__ == "__main__":
    import time
    import queue
    
    print("Running subconscious test...")
    test_q = queue.Queue()
    
    # Initialize with test queue
    sub = Subconscious(output_queue=test_q)
    
    # Start generation
    sub.start()
    print("Generation started (10 seconds)...")
    
    # Let it run for 10 seconds
    time.sleep(10)
    
    # Stop and show results
    sub.stop()
    print(f"\nTest complete. Generated {test_q.qsize()} thoughts:")
    
    # Print the generated thoughts
    while not test_q.empty():
        print(f" - {test_q.get()}")