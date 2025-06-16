import threading
import time
import queue
from datetime import datetime
from typing import Optional
from transformers import pipeline
from .config import *

class Subconscious:
    def __init__(
        self,
        thought_queue: queue.Queue,
        memory_manager=None,
        model_name: str = DEFAULT_MODEL,
        interval: int = INTERVAL_SECONDS
    ):
        self.thought_queue = thought_queue
        self.memory_manager = memory_manager
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        
        # More efficient generation parameters
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Context tracking
        self.last_context = ""
        self.context_update_interval = 5  # Update context every 5 thoughts

    def _get_context(self) -> str:
        """Get relevant context from memory"""
        if not self.memory_manager or random.random() > 0.3:  # 30% chance to use context
            return ""
            
        # Get both episodic and semantic context
        episodic = self.memory_manager.get_recent_episodic(3)
        semantic = self.memory_manager.get_relevant_semantic(self.last_thought or "", 2)
        
        context = []
        if episodic:
            context.append("Recent memories:\n" + "\n".join(episodic))
        if semantic:
            context.append("Related concepts:\n" + "\n".join(semantic))
        
        return "\n\n".join(context)

    def _generate_thought(self) -> Optional[str]:
        """Generate a raw subconscious thought"""
        try:
            # Update context periodically
            if random.random() < (1/self.context_update_interval):
                self.last_context = self._get_context()
            
            prompt = self._build_prompt()
            
            output = self.generator(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                num_return_sequences=1,
                pad_token_id=50256  # Silence warning
            )[0]['generated_text']
            
            # Clean output
            thought = output.replace(prompt, "").strip()
            thought = thought.split("\n")[0]  # Take only first line
            return thought if thought else None
            
        except Exception as e:
            print(f"Subconscious generation error: {e}")
            return None

    def _build_prompt(self) -> str:
        """Construct the generation prompt"""
        prompt_parts = []
        
        # Add context if available
        if self.last_context:
            prompt_parts.append(self.last_context)
            prompt_parts.append("")  # Empty line separator
        
        # Temporal context
        now = datetime.now().strftime("%A, %B %d, %I:%M %p")
        prompt_parts.append(f"Current time: {now}")
        
        # Core prompt
        prompt_parts.append("Raw subconscious thought:")
        return "\n".join(prompt_parts)

    def _run(self):
        """Main generation loop"""
        thought_count = 0
        while not self._stop_event.is_set():
            try:
                thought = self._generate_thought()
                if thought:
                    # Add timestamp and queue
                    timestamp = datetime.now().isoformat()
                    self.thought_queue.put(f"{timestamp}|{thought}")
                    thought_count += 1
                    
                    # Update memory
                    if self.memory_manager:
                        self.memory_manager.add_episodic(thought)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Subconscious error: {e}")
                time.sleep(1)  # Prevent tight error loops

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()