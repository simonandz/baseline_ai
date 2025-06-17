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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
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
        model_name: str = "EleutherAI/gpt-neo-2.7B",
        interval: float = 4.0,
        device: Optional[torch.device] = None,
        max_embedding_history: int = 80,
        similarity_threshold: float = 0.82
    ):
        """
        Optimized subconscious for GPT-Neo 2.7B with robust duplicate detection
        
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
        
        # Device configuration
        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Model loading with 8-bit quantization
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device
        )
        
        logger.info(f"Loading generator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        # Generation parameters
        self.generation_config = {
            "max_new_tokens": 50,
            "temperature": 0.85,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # State management
        self.recent_embeddings = []
        self.thought_count = 0
        self.max_embedding_history = max_embedding_history
        self.similarity_threshold = similarity_threshold
        logger.info(f"Subconscious initialized on {self.device}")

    def _get_context(self) -> str:
        """Retrieves and formats context with token limit"""
        if not self.memory:
            return ""
        
        try:
            recent = self.memory.get_recent_memories(2)
            if not recent:
                return ""
            
            context_lines = []
            token_count = 0
            max_tokens = 256  # Hard cap for 2.7B model
            
            for mem in recent:
                content = str(mem['content']) if isinstance(mem, dict) else str(mem)
                tokens = len(self.tokenizer.tokenize(content))
                
                if token_count + tokens > max_tokens:
                    break
                    
                context_lines.append(content)
                token_count += tokens
            
            return "Context:\n" + "\n".join(context_lines) if context_lines else ""
        except Exception as e:
            logger.error(f"Context error: {e}", exc_info=True)
            return ""

    def _is_duplicate(self, thought: str) -> bool:
        """True if new thought is semantically too close to recent history."""
        if not thought.strip():
            return True

        # Embed the candidate thought
        emb = self.embedder.encode(thought, convert_to_tensor=False)
        emb = np.asarray(emb).reshape(1, -1)

        with self._lock:
            if self.recent_embeddings:
                hist = np.vstack(self.recent_embeddings)
                sims = cosine_similarity(emb, hist)[0]
                if np.max(sims) >= self.similarity_threshold:
                    return True

            self.recent_embeddings.append(emb.flatten())
            if len(self.recent_embeddings) > self.max_embedding_history:
                self.recent_embeddings.pop(0)

        return False

    def _generate_thought(self) -> Optional[str]:
        """Generates thoughts with OOM protection"""
        context = self._get_context()
        prompt = f"""
        {context}
        Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Generate one concise thought (1 sentence max):
        - About current context if relevant
        - Or an original observation
        
        Thought:"""
        
        for attempt in range(3):
            try:
                with torch.inference_mode():
                    result = self.generator(
                        prompt,
                        **self.generation_config
                    )
                
                raw_thought = result[0]['generated_text'].replace(prompt, "").strip()
                sentences = re.split(r'(?<=[.!?])\s+', raw_thought)
                thought = sentences[0] if sentences else raw_thought
                
                if thought and not self._is_duplicate(thought):
                    return thought
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM - reducing parameters")
                torch.cuda.empty_cache()
                self.generation_config["max_new_tokens"] = max(
                    30, self.generation_config["max_new_tokens"] - 10
                )
            except Exception as e:
                logger.error(f"Generation error (attempt {attempt+1}): {e}")
            finally:
                gc.collect()
                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()
        
        return None

    def _run(self):
        """Main loop with adaptive timing"""
        logger.info("Subconscious thread started")
        base_interval = self.interval
        
        while not self._stop_event.is_set():
            start_time = time.time()
            
            try:
                thought = self._generate_thought()
                if thought:
                    with self._lock:
                        self.queue.put(thought)
                        self.thought_count += 1
                    logger.debug(f"Thought #{self.thought_count}: {thought[:60]}...")
                    
                elapsed = time.time() - start_time
                sleep_time = max(0.1, base_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Runtime error: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info("Subconscious thread stopped")

    def start(self):
        """Starts with memory checks"""
        if not torch.cuda.is_available():
            logger.warning("Running on CPU - performance will be limited")
            
        if not self._thread.is_alive():
            self._thread.start()
            logger.info("Generation thread started")
        else:
            logger.warning("Thread already running")

    def stop(self):
        """Ensures clean shutdown"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        torch.cuda.empty_cache()
        logger.info("Subconscious fully stopped")

    def __del__(self):
        self.stop()


# Test Harness
if __name__ == "__main__":
    from queue import Queue
    import time
    
    class MockMemory:
        def get_recent_memories(self, n):
            return [
                {"content": "Remembered meeting with team yesterday"},
                "Scheduled project review for tomorrow"
            ]
    
    print("Testing Subconscious with GPT-Neo 2.7B...")
    test_q = Queue()
    mock_memory = MockMemory()
    
    sub = Subconscious(
        output_queue=test_q,
        memory=mock_memory,
        model_name="EleutherAI/gpt-neo-2.7B",
        interval=2.0
    )
    
    sub.start()
    print("Generation running for 15 seconds...")
    time.sleep(15)
    
    sub.stop()
    print(f"\nGenerated {test_q.qsize()} thoughts:")
    while not test_q.empty():
        print(f" - {test_q.get()}")
