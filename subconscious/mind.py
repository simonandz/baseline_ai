# subconscious/mind.py

import threading
import time
import os
import sys
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

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
    """Continuously generates raw thoughts with memory context and duplicate suppression."""

    def __init__(
        self,
        thought_queue: "queue.Queue",
        memory_manager=None,
        model_name: str = DEFAULT_MODEL,
        interval: int = INTERVAL_SECONDS,
    ) -> None:
        self.thought_queue = thought_queue
        self.memory_manager = memory_manager
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # State
        self.thought_count = 0
        self.recent: Deque[str] = deque(maxlen=50)
        self.last_context: str = ""

        # Load 8‑bit model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=USE_8BIT,
            device_map=DEVICE_MAP,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.generator = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

    # ---------- helpers ---------- #
    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return 0.0 if not sa or not sb else len(sa & sb) / len(sa | sb)

    def _is_duplicate(self, text: str) -> bool:
        return any(self._jaccard(text, prev) >= DUPLICATE_THRESHOLD for prev in self.recent)

    def _fetch_context(self) -> str:
        if not self.memory_manager:
            return ""
        mems = getattr(self.memory_manager, "get_recent_memories", lambda n: [])(CONTEXT_WINDOW)
        return "\n".join(mems) + ("\n" if mems else "")

    def _build_prompt(self) -> str:
        if self.thought_count % CONTEXT_UPDATE_INTERVAL == 0:
            self.last_context = self._fetch_context()
        parts: List[str] = []
        if self.last_context:
            parts.append(self.last_context.strip())
        parts.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        parts.append("Subconscious thought (extend previous idea, avoid repetition):")
        return "\n".join(parts)

    def _generate_thought(self) -> Optional[str]:
        prompt = self._build_prompt()
        for _ in range(3):  # try up to 3 times to avoid duplicates
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
            thought = out.split("\n")[0].strip()
            if thought and not self._is_duplicate(thought):
                break
        else:
            thought = out.split("\n")[0].strip()
        self.recent.append(thought)
        return thought

    # ---------- background loop ---------- #
    def _run(self) -> None:
        import queue  # local import to avoid circular refs in type hints
        while not self._stop_event.is_set():
            try:
                thought = self._generate_thought()
                if thought:
                    ts = datetime.now().isoformat()
                    self.thought_queue.put(f"{ts}|{thought}")
                    if self.memory_manager:
                        self.memory_manager.add_episodic(thought)
                    self.thought_count += 1
            except Exception as exc:
                print(f"Subconscious error: {exc}")
            time.sleep(self.interval)

    # ---------- control ---------- #
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()


# Stand‑alone debug runner
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
