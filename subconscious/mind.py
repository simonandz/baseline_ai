# subconscious/mind.py

import threading
import time
import os
import sys
from datetime import datetime
from transformers import pipeline
import torch  # ensure PyTorch backend

# Ensure parent directory is in path for sibling imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Config imports
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
    )

class Subconscious:
    def __init__(
        self,
        interval: int = INTERVAL_SECONDS,
        model_name: str = DEFAULT_MODEL,
        memory_manager=None,
        context_size: int = MEMORY_CONTEXT_SIZE,
    ):
        """
        Initializes the Subconscious module with optional memory and chain-of-thought.
        """
        self.interval = interval
        self.memory_manager = memory_manager
        self.context_size = context_size
        self.last_thought = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Load model with quantization settings; no sampling params here
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            framework="pt",
            device_map=DEVICE_MAP,
            load_in_8bit=USE_8BIT,
            torch_dtype=torch.float16,
        )

    def start(self) -> None:
        """Begin the background thought loop."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the thought loop gracefully."""
        self._stop_event.set()
        self._thread.join()

    def _fetch_context(self) -> str:
        """Collect recent memories for context."""
        if not self.memory_manager:
            return ""
        recent = self.memory_manager.get_recent_memories(self.context_size)
        if not recent:
            return ""
        return "\n".join(str(m) for m in recent) + "\n"

    def _generate_thought(self) -> str:
        """Generate a new thought using memory and previous thought."""
        context = self._fetch_context()
        prompt_parts = []
        if context:
            prompt_parts.append(context.strip())
        if self.last_thought:
            prompt_parts.append(f"Previous thought: {self.last_thought}")
        prompt_parts.append(PROMPT_PREFIX)
        prompt = "\n".join(prompt_parts)

        # Pass sampling parameters here, not in pipeline init
        outputs = self.generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            return_full_text=False
        )
        new_text = outputs[0]["generated_text"].strip()
        self.last_thought = new_text
        if self.memory_manager:
            self.memory_manager.add_episodic(new_text)

        timestamp = datetime.now().isoformat(timespec='seconds')
        return f"[{timestamp}] {new_text}"

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                thought = self._generate_thought()
                print(thought)
            except Exception as e:
                print(f"Error generating thought: {e}")
            time.sleep(self.interval)

if __name__ == "__main__":
    from memory.manager import MemoryManager

    mem = MemoryManager()
    if not mem.get_recent_memories(1):
        mem.add_episodic("Attended robotics club meeting; noted new FPGA board specs.", salience=0.8)
        mem.add_episodic("Reviewed capacitor coupling issues in analog circuit.", salience=0.6)

    subj = Subconscious(memory_manager=mem)
    subj.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        subj.stop()
        print("Subconscious loop stopped.")