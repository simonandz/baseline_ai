import queue
import time
from subconscious.mind import Subconscious
from conscious.pipeline import ConsciousProcessor
from memory.manager import MemoryManager

class Orchestrator:
    def __init__(self):
        self.thought_queue = queue.Queue(maxsize=100)
        self.memory = MemoryManager()
        self.subconscious = Subconscious(self.thought_queue, self.memory)
        self.conscious = ConsciousProcessor()

    def start(self):
        print("Starting mind...")
        self.subconscious.start()
        self._run_conscious()

    def _run_conscious(self):
        while True:
            try:
                if not self.thought_queue.empty():
                    thought = self.thought_queue.get()
                    result = self.conscious.process_thought(thought)
                    
                    if result['passed']:
                        print(f"\n💭 Conscious Thought: {result['refined']}")
                        self.memory.add_thought(
                            result['refined'], 
                            salience=result['salience']
                        )
                
                time.sleep(0.1)  # Frequent checks
                
            except KeyboardInterrupt:
                print("\nShutting down...")
                self.subconscious.stop()
                break

if __name__ == "__main__":
    mind = Orchestrator()
    mind.start()