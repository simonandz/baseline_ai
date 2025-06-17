# subconscious/mind.py
"""
Background "subconscious" thought generator using Phi-2 (2.7B) with duplicate suppression and dynamic context.
"""
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

from .config import (
    DEFAULT_MODEL,
    USE_8BIT,
    DEVICE_MAP,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    CONTEXT_WINDOW,
    CONTEXT_UPDATE_INTERVAL,
    DUPLICATE_THRESHOLD,
    INTERVAL_SECONDS
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Subconscious:
    def __init__(
        self,
        output_queue: Queue,
        memory=None,
        model_name: str = DEFAULT_MODEL,
        interval: float = INTERVAL_SECONDS,
        device: Optional[torch.device] = None,
        max_embedding_history: int = 80,
        similarity_threshold: float = DUPLICATE_THRESHOLD
    ):
        """
        Subconscious thought generator with Phi-2, embedding-based duplicate filtering,
        and periodic context injection.

        Args:
            output_queue: queue for emitting generated thoughts
            memory: MemoryManager for context retrieval
            model_name: HuggingFace model identifier (default Phi-2)
            interval: seconds between generation cycles
            device: explicit torch device, else auto-detect
            max_embedding_history: cache size for recent embeddings
            similarity_threshold: max cosine similarity to allow new thought
        """
        self.queue = output_queue
        self.memory = memory
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

        # Device selection
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")

        # Embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device
        )

        # Generation model
        logger.info(f"Loading Phi-2 generator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=DEVICE_MAP,
            torch_dtype=torch.float16,
            load_in_8bit=USE_8BIT
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        # Sampling configuration
        self.generation_config = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # State for duplicate suppression
        self.recent_embeddings = []
        self.thought_count = 0
        self.max_embedding_history = max_embedding_history
        self.similarity_threshold = similarity_threshold

        logger.info(f"Subconscious initialized on {self.device}")

    def _get_context(self) -> str:
        """Fetch recent memories up to token limit for prompt context."""
        if not self.memory:
            return ""
        try:
            recent = self.memory.get_recent_memories(CONTEXT_WINDOW)
            if not recent:
                return ""

            lines, total = [], 0
            for mem in recent:
                text = mem['content'] if isinstance(mem, dict) else str(mem)
                count = len(self.tokenizer.tokenize(text))
                if total + count > MAX_NEW_TOKENS * 4:
                    break
                lines.append(text)
                total += count
            return "Context:\n" + "\n".join(lines) if lines else ""
        except Exception as e:
            logger.error(f"Context error: {e}", exc_info=True)
            return ""

    def _is_duplicate(self, thought: str) -> bool:
        """Check semantic similarity against recent embeddings."""
        if not thought.strip():
            return True
        emb = self.embedder.encode(thought, convert_to_tensor=False)
        emb = np.asarray(emb).reshape(1, -1)

        with self._lock:
            if self.recent_embeddings:
                sims = cosine_similarity(emb, np.vstack(self.recent_embeddings))[0]
                if sims.max() >= self.similarity_threshold:
                    return True
            self.recent_embeddings.append(emb.flatten())
            if len(self.recent_embeddings) > self.max_embedding_history:
                self.recent_embeddings.pop(0)
        return False

    def _generate_thought(self) -> Optional[str]:
        """Generate a concise new thought, retrying on failures or OOM."""
        ctx = self._get_context()
        prompt = (
            f"{ctx}\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            "Generate a concise thought (1 sentence):\nThought:"
        )
        for _ in range(3):
            try:
                with torch.inference_mode():
                    out = self.generator(prompt, **self.generation_config)
                text = out[0]['generated_text'].replace(prompt, '').strip()
                candidate = re.split(r'(?<=[.!?])\s+', text)[0]
                if candidate and not self._is_duplicate(candidate):
                    return candidate
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM: reducing token budget")
                torch.cuda.empty_cache()
                self.generation_config['max_new_tokens'] = max(10, self.generation_config['max_new_tokens'] - 10)
            except Exception as e:
                logger.error(f"Generation error: {e}")
            finally:
                gc.collect()
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()
        return None

    def _run(self):
        """Main loop: generate thoughts at fixed intervals."""
        logger.info("Subconscious thread starting")
        while not self._stop_event.is_set():
            start = time.time()
            thought = self._generate_thought()
            if thought:
                with self._lock:
                    self.queue.put(thought)
                    self.thought_count += 1
                logger.debug(f"Thought #{self.thought_count}: {thought}")
            elapsed = time.time() - start
            time.sleep(max(0.1, self.interval - elapsed))
        logger.info("Subconscious thread stopped")

    def start(self):
        """Launch generation thread."""
        if not self._thread.is_alive():
            self._thread.start()
            logger.info("Generation thread launched")
        else:
            logger.warning("Thread already active")

    def stop(self):
        """Signal shutdown and join thread."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        torch.cuda.empty_cache()
        logger.info("Subconscious stopped cleanly")

    def __del__(self):
        self.stop()