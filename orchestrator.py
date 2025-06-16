import queue
import time
from subconscious.mind import Subconscious
from conscious.mind import Conscious
from memory.manager import MemoryManager

def main():
    # Shared thought queue
    thought_queue = queue.Queue(maxsize=50)
    
    # Initialize modules
    mem_manager = MemoryManager()
    subconscious = Subconscious(
        thought_queue=thought_queue,
        memory_manager=mem_manager
    )
    conscious = Conscious(
        thought_queue=thought_queue,
        memory_manager=mem_manager
    )
    
    # Start the mind
    subconscious.start()
    conscious.start()
    print("Mind started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping mind...")
        subconscious.stop()
        conscious.stop()
        print("Mind stopped.")

if __name__ == "__main__":
    main()