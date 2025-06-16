import threading
import time
import os
import sys
from datetime import datetime
from collections import deque
from typing import Deque, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
)
import torch

# Path setup for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from .config import (
        INTERVAL_SECONDS,
        DEFAULT_MODEL,
        PROMPT_PREFIX,
        MAX_NEW_TOKENS,
        USE_8BIT,
        DEVICE_MAP,
        MEMORY_CONTEXT_SIZE,
        TEMPERATURE,
        TOP_P,
        TOP_K,
        TFIDF_SIMILARITY_THRESHOLD,
        MAX_RECENT_THOUGHTS,
    )
except ImportError:
    from config import (
        INTERVAL_SECONDS,
        DEFAULT_MODEL,
        PROMPT_PREFIX,
        MAX_NEW_TOKENS,
        USE_8BIT,
        DEVICE_MAP,
        MEMORY_CONTEXT_SIZE,
        TEMPERATURE,
        TOP_P,
        TOP_K,
        TFIDF_SIMILARITY_THRESHOLD,
        MAX_RECENT_THOUGHTS,
    )


class Subconscious:
    """Continuous thought generator with temporal awareness and anti-repetition."""

    def __init__(
        self,
        interval: int = INTERVAL_SECONDS,
        model_name: str = DEFAULT_MODEL,
        memory_manager=None,
        context_size: int = MEMORY_CONTEXT_SIZE,
    ) -> None:
        self.interval = interval
        self.memory_manager = memory_manager
        self.context_size = context_size
        self.last_thought: Optional[str] = None
        self.recent: List[str] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

        # Model initialization
        self._pipeline = TextGenerationPipeline(
            model=AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=USE_8BIT,
                device_map=DEVICE_MAP,
                torch_dtype=torch.float16,
            ),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
        )

    def start(self) -> None:
        """Start the background thought generation."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the generation thread."""
        self._stop_event.set()
        self._thread.join()

    def _fetch_context(self) -> str:
        """Retrieve relevant memories for context."""
        if not self.memory_manager:
            return ""
        memories: List[str] = self.memory_manager.get_recent_memories(self.context_size)
        return "\n".join(memories) + ("\n" if memories else "")

    def _is_duplicate(self, text: str) -> bool:
        """Check if thought is too similar to recent ones using TF-IDF."""
        if not self.recent:
            return False
            
        # Update TF-IDF matrix
        docs = self.recent + [text]
        new_matrix = self.vectorizer.fit_transform(docs)
        
        # Compare against all recent thoughts
        similarities = cosine_similarity(new_matrix[-1:], new_matrix[:-1])
        return np.max(similarities) > TFIDF_SIMILARITY_THRESHOLD

    def _generate_once(self) -> str:
        """Generate a single thought with temporal context."""
        prompt_parts = []
        context = self._fetch_context()
        if context:
            prompt_parts.append(context.strip())
        
        # Temporal context
        now = datetime.now().isoformat(timespec="seconds")
        prompt_parts.append(f"Current time: {now}")
        prompt_parts.append(PROMPT_PREFIX)
        prompt = "\n".join(prompt_parts)

        out = self._pipeline(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            return_full_text=False,
        )[0]["generated_text"].strip()
        return out

    def _generate_thought(self) -> Optional[str]:
        """Generate a novel thought or skip if too similar."""
        for _ in range(3):
            text = self._generate_once()
            if not self._is_duplicate(text):
                self.recent.append(text)
                if len(self.recent) > MAX_RECENT_THOUGHTS:
                    self.recent.pop(0)
                self.last_thought = text
                if self.memory_manager:
                    self.memory_manager.add_episodic(text)
                return text
        return None  # Skip this interval

    def _run(self) -> None:
        """Main generation loop."""
        while not self._stop_event.is_set():
            try:
                thought = self._generate_thought()
                if thought:
                    ts = datetime.now().isoformat(timespec="seconds")
                    print(f"[{ts}] {thought}")
            except Exception as exc:
                print(f"Error: {exc}")
            time.sleep(self.interval)


if __name__ == "__main__":
    from memory.manager import MemoryManager

    mem = MemoryManager()
    thinker = Subconscious(memory_manager=mem)
    thinker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        thinker.stop()
        print("Subconscious loop stopped.")