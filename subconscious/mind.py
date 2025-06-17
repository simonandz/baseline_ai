import threading
import time
import numpy as np
import torch
import re
import gc
import logging
from typing import Optional
from datetime import datetime
from queue import Queue
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Subconscious:
    def __init__(
        self,
        output_queue: Queue,
        memory=None,
        model_name: str = "EleutherAI/gpt-neo-1.3B",
        interval: float = 3.0,
        device: Optional[torch.device] = None,
        max_embedding_history: int = 100,
        similarity_threshold: float = 0.85
    ):
        """
        Enhanced continuous thought generator with improved error handling
        
        Args:
            output_queue: Thread-safe queue for thought output
            memory: MemoryManager instance for context
            model_name: HF model identifier
            interval: Seconds between thought generation
            device: Explicit device specification
            max_embedding_history: Size of recent thought embedding cache
            similarity_threshold: Semantic similarity threshold for duplicates
        """
        self.queue = output_queue
        self.memory = memory
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        
        # Configure device
        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        logger.info(f"Loading generator: {model_name}")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=self.device,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32
        )
        self.generator.model.config.max_length = 128
        self.generator.model.config.pad_token_id = self.generator.tokenizer.eos_token_id
        
        # State management
        self.recent_embeddings = []
        self.thought_count = 0
        self.max_embedding_history = max_embedding_history
        self.similarity_threshold = similarity_threshold
        logger.info(f"Subconscious initialized on {self.device}")

    def _get_context(self) -> str:
        """Safely retrieves and formats context from memory"""
        if not self.memory:
            return ""
        
        try:
            recent = self.memory.get_recent_memories(2)
            if not recent:
                return ""
            
            # Handle both dict and string memory items
            context_lines = []
            for mem in recent:
                if isinstance(mem, dict):
                    if 'content' in mem:
                        context_lines.append(mem['content'])
                    elif 'text' in mem:
                        context_lines.append(mem['text'])
                    else:
                        context_lines.append(str(mem))
                else:
                    context_lines.append(str(mem))
            
            return "Recent memories:\n" + "\n".join(context_lines)
        except Exception as e:
            logger.error(f"Context error: {e}", exc_info=True)
            return ""

    def _is_duplicate(self, thought: str) -> bool:
        """True if new thought is semantically too close to recent history."""
        if not thought.strip():
            return True

        # ----- embed the candidate -----
        emb = self.embedder.encode(thought, convert_to_tensor=False)  # 1-D np.array
        emb = np.asarray(emb).reshape(1, -1)                          # 2-D row

        with self._lock:
            if self.recent_embeddings:
                # stack history into a clean (N, dim) matrix
                hist = np.vstack(self.recent_embeddings)              # 2-D

                # cosine similarity: returns (1, N) — flatten to 1-D
                sims = cosine_similarity(emb, hist)[0]

                if sims.max() >= self.similarity_threshold:
                    return True

            # store *flat* 1-D copy to avoid 3-D stacks next time
            self.recent_embeddings.append(emb.flatten())
            if len(self.recent_embeddings) > self.max_embedding_history:
                self.recent_embeddings.pop(0)

        return False


    def _generate_thought(self) -> Optional[str]:
        """Generates a novel thought with context awareness"""
        context = self._get_context()
        prompt = f"""
        {context}
        Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Generate a concise, single-sentence thought about:
        - Current context if relevant
        - Or an interesting observation
        Avoid repeating recent thoughts.
        Thought:"""
        
        for attempt in range(3):
            try:
                result = self.generator(
                    prompt,
                    max_new_tokens=40,
                    temperature=0.8,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.generator.tokenizer.eos_token_id
                )
                
                raw_thought = result[0]['generated_text'].replace(prompt, "").strip()
                sentences = re.split(r'(?<=[.!?])\s+', raw_thought)
                thought = sentences[0] if sentences else raw_thought
                
                if thought and not self._is_duplicate(thought):
                    return thought
                    
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM - reducing memory usage")
                torch.cuda.empty_cache()
                self.generator.model.config.max_length = max(
                    64, self.generator.model.config.max_length // 2
                )
            except Exception as e:
                logger.error(f"Generation error (attempt {attempt+1}): {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        
        return None

    def _run(self):
        """Main generation loop with enhanced resilience"""
        logger.info("Subconscious thread started")
        while not self._stop_event.is_set():
            try:
                thought = self._generate_thought()
                if thought:
                    with self._lock:
                        self.queue.put(thought)
                        self.thought_count += 1
                    logger.debug(f"Generated thought #{self.thought_count}: {thought}")
            except Exception as e:
                logger.error(f"Runtime error: {e}", exc_info=True)
                time.sleep(5)  # Backoff on error
            
            time.sleep(self.interval)
        logger.info("Subconscious thread stopped")

    def start(self):
        """Starts the generation thread"""
        if not self._thread.is_alive():
            self._thread.start()
        else:
            logger.warning("Subconscious thread already running")

    def stop(self):
        """Gracefully stops the generation thread"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Subconscious stopped")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
        torch.cuda.empty_cache()


# Test harness
if __name__ == "__main__":
    from queue import Queue
    import time
    
    class MockMemory:
        def get_recent_memories(self, n):
            return [
                {"content": "Remembered meeting with team yesterday"},
                "Scheduled project review for tomorrow"
            ]
    
    print("Running enhanced subconscious test...")
    test_q = Queue()
    mock_memory = MockMemory()
    
    sub = Subconscious(
        output_queue=test_q,
        memory=mock_memory,
        model_name="EleutherAI/gpt-neo-125M",  # Smaller model for testing
        interval=2.0
    )
    
    sub.start()
    print("Generation running for 10 seconds...")
    time.sleep(10)
    
    sub.stop()
    print(f"\nTest complete. Generated {test_q.qsize()} thoughts:")
    while not test_q.empty():
        print(f" - {test_q.get()}")