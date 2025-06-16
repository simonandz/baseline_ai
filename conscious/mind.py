import threading
import time
import queue
from datetime import datetime
from .filter import ThoughtFilter
from .processor import ThoughtRefiner
from .config import CONSCIOUS_CHECK_INTERVAL

class Conscious:
    def __init__(self, thought_queue: queue.Queue, memory_manager=None):
        self.thought_queue = thought_queue
        self.memory_manager = memory_manager
        self.filter = ThoughtFilter()
        self.refiner = ThoughtRefiner()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        
    def start(self):
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        self._thread.join()
    
    def _process_thought(self, thought: str):
        """Full conscious processing pipeline"""
        # Step 1: Evaluate thought
        evaluation = self.filter.evaluate_thought(thought)
        
        # Step 2: Refine if valuable
        if evaluation["passes"]:
            refined = self.refiner.refine_thought(thought, evaluation["category"])
            evaluation["refined"] = refined
            
            # Step 3: Output and store
            self._output_thought(refined, evaluation)
            if self.memory_manager:
                self.memory_manager.add_conscious_memory(evaluation)
    
    def _output_thought(self, thought: str, evaluation: dict):
        """Output with context-aware formatting"""
        category = evaluation["category"]
        emojis = {
            "problem_solving": "🔧",
            "planning": "📅",
            "insight": "💡",
            "curiosity": "❓",
            "memory": "🔍",
            "emotion": "❤️",
            "misc": "💭"
        }
        emoji = emojis.get(category, "💭")
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"{emoji} [{ts}] {thought}")
    
    def _run(self):
        while not self._stop_event.is_set():
            try:
                # Check for new thoughts
                if not self.thought_queue.empty():
                    thought = self.thought_queue.get_nowait()
                    self._process_thought(thought)
                
                time.sleep(CONSCIOUS_CHECK_INTERVAL)
            except Exception as e:
                print(f"Conscious error: {e}")