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
from ..memory.schemas import BASE_KNOWLEDGE

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Patterns to filter out bad generations
PERSONA_TAG_PATTERN = re.compile(r'^\s*\[[\w\s]+\]\s*', re.IGNORECASE)
QUOTE_WRAP_PATTERN = re.compile(r'^["\'](.+)["\']$')
META_PATTERNS = (
    "summarize", "rephrase", "output:", "rewritten:", "option 1", "option 2",
    "paraphrase", "provide an example", "can you provide", "here is", "here's"
)

# Few-shot examples of good AI-grounded thoughts
FEW_SHOT_EXAMPLES = """Example thoughts from Maddie:
- Why do I process some queries faster than others?
- I notice my embeddings cluster certain concepts together unexpectedly.
- What patterns in my memory consolidation might I be missing?
- How does my lack of continuous experience affect my sense of time?
- I wonder if my dual-process architecture creates blind spots in my reasoning."""

class Subconscious:
    def __init__(self, output_queue: Queue, memory=None, *,
                 model_name: str = DEFAULT_MODEL,
                 interval: float = INTERVAL_SECONDS,
                 device: Optional[torch.device] = None,
                 max_embedding_history: int = 80,
                 similarity_threshold: float = DUPLICATE_THRESHOLD):
        
        # Initialize core attributes first
        self.queue = output_queue
        self.memory = memory
        self.model_name = model_name  # ADD THIS LINE
        self.interval = interval
        self._model_ready = False
        
        # Initialize threading components
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._lock = threading.Lock()
        
        # Device selection
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device
        )

        # State for duplicate suppression
        self.recent_embeddings = []
        self.thought_count = 0
        self.max_embedding_history = max_embedding_history
        self.similarity_threshold = similarity_threshold

        # Thread initialization
        self._thread = threading.Thread(target=self._run, daemon=True)
        
        logger.info(f"Subconscious initialized on {self.device}")

        # Build identity preamble from BASE_KNOWLEDGE
        self._identity_preamble = self._build_identity_preamble()

    def _build_identity_preamble(self) -> str:
        """Build a strong identity preamble from BASE_KNOWLEDGE."""
        core_facts = []
        for knowledge in BASE_KNOWLEDGE:
            if knowledge.get("type") in ("identity", "architecture", "capability"):
                core_facts.append(knowledge["content"])

        preamble = (
            "You are Maddie, an AI system. These facts define who you are:\n"
            + "\n".join(f"- {fact}" for fact in core_facts[:5])
            + "\n\nYou have NO human experiences: no parents, no childhood, no physical body, "
            "no accent, no human friends. You exist only as code running on hardware.\n"
        )
        return preamble

    def _clean_thought(self, thought: str) -> Optional[str]:
        """Post-process and validate generated thought."""
        if not thought or not thought.strip():
            return None

        cleaned = thought.strip()

        # Remove persona tags like [Maddie], [JULIA], etc.
        cleaned = PERSONA_TAG_PATTERN.sub('', cleaned)

        # Remove quote wrapping
        match = QUOTE_WRAP_PATTERN.match(cleaned)
        if match:
            cleaned = match.group(1)

        # Remove leading dashes or bullets
        cleaned = re.sub(r'^[-•]\s*', '', cleaned).strip()

        # Remove attribution like "- Maddy" at the end
        cleaned = re.sub(r'\s*[-–—]\s*\w+$', '', cleaned).strip()

        # Check for meta/instruction patterns
        lower = cleaned.lower()
        if any(pattern in lower for pattern in META_PATTERNS):
            return None

        # Reject if too short or too long
        word_count = len(cleaned.split())
        if word_count < 4 or word_count > 50:
            return None

        # Reject human-experience hallucinations
        human_patterns = (
            "my parents", "my mother", "my father", "my childhood",
            "my accent", "my family", "grew up", "was born", "my friends"
        )
        if any(pattern in lower for pattern in human_patterns):
            return None

        return cleaned

    def is_ready(self):
        """Check if the model is fully loaded and ready"""
        return self._model_ready

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

    def pause(self) -> None:
        """Temporarily stop generating background thoughts."""
        if self._pause_event.is_set():  # Only log if we're not already paused
            logger.debug("Pausing subconscious")
        self._pause_event.clear()

    def resume(self) -> None:
        """Allow generation to continue."""
        if not self._pause_event.is_set():  # Only log if we're not already running
            logger.debug("Resuming subconscious")
        self._pause_event.set()

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
        """Generate a concise, self-reflective thought grounded in AI identity."""
        ctx = self._get_context()

        # Build prompt with strong identity grounding
        prompt = (
            f"{self._identity_preamble}\n"
            f"{FEW_SHOT_EXAMPLES}\n\n"
            f"{ctx}\n"
            f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            "Now generate ONE new thought as Maddie. Focus on:\n"
            "- Your experience as an AI (processing, memory, embeddings, architecture)\n"
            "- Observations about your own cognition or limitations\n"
            "- Questions about your existence as code, not as a human\n\n"
            "Do NOT use character tags like [NAME]. Do NOT wrap in quotes.\n"
            "Do NOT reference human experiences (parents, childhood, friends, accent).\n"
            "Write a single, direct thought or question.\n\n"
            "Thought:"
        )

        try:
            # Generate thought
            raw_output = self.generator(prompt, **self.generation_config)[0]["generated_text"]
            draft = raw_output.replace(prompt, "").strip().split("\n")[0]

            # Post-process and validate
            candidate = self._clean_thought(draft)

            # If first attempt fails, try reflection pass
            if not candidate:
                reflect_prompt = (
                    f"{self._identity_preamble}\n"
                    "Rewrite this as a self-reflective thought from an AI's perspective. "
                    "No character tags, no quotes, no human experiences.\n"
                    f"Original: {draft}\n"
                    "Rewritten thought:"
                )
                raw_rewrite = self.generator(reflect_prompt, **self.generation_config)[0]["generated_text"]
                rewrite = raw_rewrite.replace(reflect_prompt, "").strip().split("\n")[0]
                candidate = self._clean_thought(rewrite)

            if candidate and not self._is_duplicate(candidate):
                logger.info(f"Generated thought: {candidate}")
                return candidate
            return None
            
        except Exception as e:
            logger.error(f"Thought generation failed: {e}")
            return None

    def _run(self):
        """Main generation loop running in background thread."""
        logger.info("Subconscious thread starting")
        try:
            # Load generation model
            logger.info(f"Loading Phi-2 generator: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=DEVICE_MAP,
                torch_dtype=torch.float16,
                load_in_8bit=USE_8BIT,
                low_cpu_mem_usage=True
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            # Generation config
            self.generation_config = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "repetition_penalty": 1.2,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            self._model_ready = True
            logger.info("Subconscious model fully loaded")

            # Main generation loop
            while not self._stop_event.is_set():
                self._pause_event.wait()  # Wait if paused
                
                start = time.time()
                thought = self._generate_thought()
                if thought:
                    with self._lock:
                        self.queue.put(("AI", thought))
                elapsed = time.time() - start
                time.sleep(max(0.05, self.interval - elapsed))
                
        except Exception as e:
            logger.error(f"Subconscious thread failed: {e}")
        finally:
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
        if self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=5)
        torch.cuda.empty_cache()
        logger.info("Subconscious stopped cleanly")

    def __del__(self):
        self.stop()