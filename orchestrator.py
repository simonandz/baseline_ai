import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better CUDA error messages

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import time
import queue
from subconscious.mind import Subconscious
from conscious import ConsciousProcessor
from memory.manager import MemoryManager

def main():
    # Shared thought queue
    thought_queue = queue.Queue(maxsize=100)
    
    # Initialize modules
    print("Initializing MemoryManager...")
    mem_manager = MemoryManager()
    
    print("Initializing ConsciousProcessor...")
    processor = ConsciousProcessor()
    
    print("Initializing Subconscious...")
    try:
        # Use explicit parameter names
        subconscious = Subconscious(
            output_queue=thought_queue,
            memory=mem_manager  # Parameter name changed
        )
    except TypeError as e:
        print(f"Subconscious init error: {e}")
        # Fallback to simplified initialization
        subconscious = Subconscious(thought_queue)
    
    # Start the mind
    print("Starting subconscious thread...")
    subconscious.start()
    
    print("Mind started. Press Ctrl+C to stop.")
    last_status = time.time()
    thought_count = 0
    
    try:
        while True:
            if not thought_queue.empty():
                thought = thought_queue.get()
                thought_count += 1
                result = processor.process(thought)
                
                if result.get("passed", False):
                    refined = result.get("refined", "")
                    print(f"💭 [{thought_count}] {refined}")
                    mem_manager.add_thought(
                        refined, 
                        salience=result.get("salience", 0.5)
                    )
            
            # Status update every 30 seconds
            if time.time() - last_status > 30:
                print(f"Status: {thought_count} thoughts processed | Queue: {thought_queue.qsize()}")
                last_status = time.time()
            
            time.sleep(0.1)  # Reduced sleep for better responsiveness
    
    except KeyboardInterrupt:
        print("\nStopping mind...")
        subconscious.stop()
        print("Mind stopped. Final thought count:", thought_count)

if __name__ == "__main__":
    main()