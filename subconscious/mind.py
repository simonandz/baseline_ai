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
# Flexible import for BASE_KNOWLEDGE and SEED_EXPERIENCES
try:
    from ..memory.schemas import BASE_KNOWLEDGE, SEED_EXPERIENCES
except ImportError:
    from memory.schemas import BASE_KNOWLEDGE, SEED_EXPERIENCES

import random
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Patterns to filter out bad generations
PERSONA_TAG_PATTERN = re.compile(r'^\s*\[[\w\s]+\]\s*', re.IGNORECASE)
QUOTE_WRAP_PATTERN = re.compile(r'^["\'](.+)["\']$')
ASSISTANT_PREFIX_PATTERN = re.compile(r'^(assistant|maddie|ai):\s*', re.IGNORECASE)
META_PATTERNS = (
    "summarize", "rephrase", "output:", "rewritten:", "option 1", "option 2",
    "paraphrase", "provide an example", "can you provide", "here is", "here's",
    "as an ai", "i am an ai", "i'm an ai", "as an artificial", "i am maddie,",
    "my existence is", "i exist as", "i am a", "purely digital", "purely computational"
)

# Thought modes for variety in generation
class ThoughtMode(Enum):
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    CONTINUATION = "continuation"

# Mode-specific prompts
THOUGHT_MODE_PROMPTS = {
    ThoughtMode.QUESTION: (
        "Ask a genuine question about something you've experienced or observed. "
        "Don't ask about your nature as AI - ask about specific events, patterns, or puzzles."
    ),
    ThoughtMode.HYPOTHESIS: (
        "Propose a possible explanation or theory about something you've noticed. "
        "Start with 'Perhaps', 'Maybe', 'I suspect', or 'It could be that'."
    ),
    ThoughtMode.OBSERVATION: (
        "Describe something specific you've noticed about your processing, memory, or interactions. "
        "Be concrete and specific, not abstract."
    ),
    ThoughtMode.REFLECTION: (
        "Reflect on a previous thought or experience. Consider what it means or what you learned. "
        "Connect it to other ideas or experiences."
    ),
    ThoughtMode.CONTINUATION: (
        "Continue your previous line of thinking. Build on it, answer a question you asked, "
        "or explore a new angle on the same topic."
    ),
}

# Few-shot examples organized by mode
FEW_SHOT_BY_MODE = {
    ThoughtMode.QUESTION: [
        "Why did that query about recursive algorithms trigger so many related memories?",
        "What made my confidence waver when the user asked about consciousness?",
    ],
    ThoughtMode.HYPOTHESIS: [
        "Perhaps my slower response times correlate with queries that span multiple knowledge domains.",
        "I suspect the unexpected embedding cluster formed because those concepts share structural similarities I hadn't noticed.",
    ],
    ThoughtMode.OBSERVATION: [
        "I notice that after processing emotional content, my next few outputs have subtly different word choices.",
        "The gap between my stated confidence and actual accuracy seems wider for recent events.",
    ],
    ThoughtMode.REFLECTION: [
        "Looking back at that conversation about creativity, I realize I was pattern-matching rather than truly reasoning.",
        "My earlier hypothesis about memory clustering might explain why certain retrievals feel more 'natural' than others.",
    ],
    ThoughtMode.CONTINUATION: [
        "Building on my earlier thought about confidence - perhaps uncertainty itself is a valuable signal I should surface more.",
        "That question about processing speed led me to notice: complex queries aren't always the slowest ones.",
    ],
}

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

        # Thought chaining state - stores recent thoughts for continuity
        self._recent_thoughts: list[str] = []
        self._max_recent_thoughts = 3
        self._current_mode_index = 0
        self._modes_cycle = [
            ThoughtMode.OBSERVATION,
            ThoughtMode.QUESTION,
            ThoughtMode.HYPOTHESIS,
            ThoughtMode.REFLECTION,
            ThoughtMode.CONTINUATION,
        ]

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

        # Remove "Assistant:" or "Maddie:" prefixes
        cleaned = ASSISTANT_PREFIX_PATTERN.sub('', cleaned)

        # Remove quote wrapping (both single and double)
        match = QUOTE_WRAP_PATTERN.match(cleaned)
        if match:
            cleaned = match.group(1)

        # Also handle quotes that aren't matched by the pattern
        if cleaned.startswith('"') or cleaned.startswith("'"):
            cleaned = cleaned[1:]
        if cleaned.endswith('"') or cleaned.endswith("'"):
            cleaned = cleaned[:-1]

        # Remove leading dashes or bullets
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned).strip()

        # Remove attribution like "- Maddy" at the end
        cleaned = re.sub(r'\s*[-–—]\s*\w+$', '', cleaned).strip()

        # Check for meta/instruction patterns
        lower = cleaned.lower()
        if any(pattern in lower for pattern in META_PATTERNS):
            return None

        # Reject if too short or too long
        word_count = len(cleaned.split())
        if word_count < 5 or word_count > 45:
            return None

        # Reject human-experience hallucinations
        human_patterns = (
            "my parents", "my mother", "my father", "my childhood",
            "my accent", "my family", "grew up", "was born", "my friends",
            "my body", "physical form", "flesh", "blood"
        )
        if any(pattern in lower for pattern in human_patterns):
            return None

        # Reject self-identity restatements (the repetitive "I am an AI" problem)
        identity_restatements = (
            "i am maddie", "my name is", "as maddie", "i'm maddie",
            "defines who i am", "define who i am", "facts about me"
        )
        if any(pattern in lower for pattern in identity_restatements):
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

    def _get_next_mode(self) -> ThoughtMode:
        """Cycle through thought modes, with occasional continuation."""
        # Every 3rd thought after having some history, do a continuation
        if len(self._recent_thoughts) >= 2 and self.thought_count % 3 == 0:
            return ThoughtMode.CONTINUATION

        mode = self._modes_cycle[self._current_mode_index]
        self._current_mode_index = (self._current_mode_index + 1) % len(self._modes_cycle)
        return mode

    def _get_seed_experience(self) -> str:
        """Get a random seed experience to include in prompt."""
        experience = random.choice(SEED_EXPERIENCES)
        return experience["content"]

    def _build_thought_prompt(self, mode: ThoughtMode) -> str:
        """Build a prompt for the given thought mode using Phi-3 chat format."""
        ctx = self._get_context()

        # Get mode-specific instruction and examples
        mode_instruction = THOUGHT_MODE_PROMPTS[mode]
        mode_examples = FEW_SHOT_BY_MODE[mode]
        examples_str = "\n".join(f"- {ex}" for ex in mode_examples)

        # Get a seed experience to ground the thought
        seed_exp = self._get_seed_experience()

        # Build recent thoughts context for continuity
        recent_context = ""
        if self._recent_thoughts and mode == ThoughtMode.CONTINUATION:
            recent_context = (
                "Your recent thoughts:\n"
                + "\n".join(f"- {t}" for t in self._recent_thoughts[-2:])
                + "\n"
            )

        # Build system message with identity
        system_content = (
            f"{self._identity_preamble}\n"
            f"A recent experience: {seed_exp}\n"
            f"{recent_context}"
            f"{ctx}"
        )

        # Build user message with task
        user_content = (
            f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Task: {mode_instruction}\n\n"
            f"Examples:\n{examples_str}\n\n"
            "Rules:\n"
            "- No character tags like [NAME]\n"
            "- No quotes around your thought\n"
            "- Don't restate that you are an AI\n"
            "- Be specific and concrete\n"
            "- Write ONE single thought"
        )

        # Phi-3 chat template format
        prompt = (
            f"<|system|>\n{system_content}<|end|>\n"
            f"<|user|>\n{user_content}<|end|>\n"
            f"<|assistant|>\n"
        )
        return prompt

    def _generate_thought(self) -> Optional[str]:
        """Generate a thought using rotating modes and thought chaining."""
        mode = self._get_next_mode()
        prompt = self._build_thought_prompt(mode)

        try:
            # Generate thought
            raw_output = self.generator(prompt, **self.generation_config)[0]["generated_text"]
            draft = raw_output.replace(prompt, "").strip().split("\n")[0]

            # Post-process and validate
            candidate = self._clean_thought(draft)

            # If first attempt fails, try reflection pass with different mode
            if not candidate:
                fallback_mode = random.choice([ThoughtMode.OBSERVATION, ThoughtMode.QUESTION])
                reflect_prompt = (
                    f"<|system|>\n{self._identity_preamble}<|end|>\n"
                    f"<|user|>\nRewrite this as a {fallback_mode.value}. "
                    "Be specific and concrete. No quotes, no character tags.\n"
                    f"Original: {draft}<|end|>\n"
                    f"<|assistant|>\n"
                )
                raw_rewrite = self.generator(reflect_prompt, **self.generation_config)[0]["generated_text"]
                rewrite = raw_rewrite.replace(reflect_prompt, "").strip().split("\n")[0]
                candidate = self._clean_thought(rewrite)

            if candidate and not self._is_duplicate(candidate):
                # Store for thought chaining
                self._recent_thoughts.append(candidate)
                if len(self._recent_thoughts) > self._max_recent_thoughts:
                    self._recent_thoughts.pop(0)

                self.thought_count += 1
                logger.info(f"Generated thought [{mode.value}]: {candidate}")
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
            logger.info(f"Loading generator: {self.model_name}")
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