# subconscious/mind.py

import threading
import time
import os
import sys
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from typing import Deque, List, Optional

import torch
from transformers import pipeline

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .config import (
    INTERVAL_SECONDS,
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
)

class Subconscious:
    """Continuously generates raw thoughts with memory-based context and no repeats."""

    def __init__(
        self,
        thought_queue: "queue.Queue",
        memory_manager=None,
        model_name: str = DEFAULT_MODEL,
        interval: int = INTERVAL_SECONDS,
    ):
        self.thought_queue = thought_queue
        self.memory_manager = memory_manager
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Internal counters and history
        self.thought_count = 0
        self.recent: Deque[str] = deque(maxlen=50)

        # Build generator pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            framework="pt",
            device_map=DEVICE_MAP,
            load_in_8bit=USE_8BIT,
            torch_dtype=torch.float16
        )
        # Cache eos_token_id for padding
        self.eos_token_id = self.generator.tokenizer.eos_token_id
        # Last context string
        self.last_context = ""

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _jaccard(self, a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _is_duplicate(self, text: str) -> bool:
        return any(self._jaccard(text, prev) >= DUPLICATE_THRESHOLD for prev in self.recent)

    def _fetch_context(self) -> str:
        """Retrieve the last CONTEXT_WINDOW episodic memories."""
        if not self.memory_manager:
            return ""
        mems = getattr(self.memory_manager, 'get_recent_memories', lambda n: [])(CONTEXT_WINDOW)
        return "\n".join(mems) + ("\n" if mems else "")

    def _build_prompt(self) -> str:
        parts: List[str] = []
        # Refresh context every CONTEXT_UPDATE_INTERVAL thoughts
        if self.thought_count % CONTEXT_UPDATE_INTERVAL == 0:
            self.last_context = self._fetch_context()
        if self.last_context:
            parts.append(self.last_context.strip())
        # Temporal marker
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"Time: {now}")
        # Instruct chain-of-thought
        parts.append("Subconscious thought (extend previous idea, no repeats):")
        return "\n".join(parts)

    def _generate_thought(self) -> Optional[str]:
        """Generate a single thought, retrying up to 3 times to avoid duplicates."""
        prompt = self._build_prompt()
        for _ in range(3):
            out = self.generator(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                pad_token_id=self.eos_token_id,
                return_full_text=False,
            )[0]["generated_text"].strip()
            # Strip prompt prefix
            thought = out.replace(prompt, "").strip().split("\n")[0]
            if thought and not self._is_duplicate(thought):
                break
        else:
            # fallback if all attempts duplicate
            thought = out.split("\n")[0].strip()
        self.recent.append(thought)
        return thought

    def _run(self) -> None:
        import queue
        while not self._stop_event.is_set():
            thought = self._generate_thought()
            if thought:
                timestamp = datetime.now().isoformat()
                self.thought_queue.put(f"{timestamp}|{thought}")
                # Store episodic with default salience
                if self.memory_manager:
                    self.memory_manager.add_episodic(thought)
                self.thought_count += 1
            time.sleep(self.interval)

if __name__ == "__main__":
    import queue
    from memory.manager import MemoryManager
    tq = queue.Queue()
    mem = MemoryManager()
    sub = Subconscious(thought_queue=tq, memory_manager=mem)
    sub.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sub.stop()