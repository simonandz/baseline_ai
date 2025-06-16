import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better CUDA error messages

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)

import time
import queue
import traceback
from subconscious.mind import Subconscious
from conscious.pipeline import ConsciousProcessor
from memory.manager import MemoryManager

def main():
    # Initialize components
    print("Initializing MemoryManager...")
    mem_manager = MemoryManager()
    
    print("Initializing ConsciousProcessor...")
    processor = ConsciousProcessor()
    
    # Shared thought queue
    thought_queue = queue.Queue(maxsize=100)
    
    print("Initializing Subconscious...")
    try:
        subconscious = Subconscious(
            output_queue=thought_queue,
            memory=mem_manager,
            model_name="EleutherAI/gpt-neo-1.3B",
            interval=3.0
        )
    except Exception as e:
        print(f"Subconscious init failed: {e}")
        traceback.print_exc()
        return

    # Start the mind
    print("Starting subconscious thread...")
    subconscious.start()
    
    print("""
    AI Mind Started
    ===============
    Press Ctrl+C to stop
    """)
    
    thought_count = 0
    last_status = time.time()
    
    try:
        while True:
            # Process thoughts from queue
            if not thought_queue.empty():
                thought = thought_queue.get()
                thought_count += 1
                
                try:
                    result = processor.process(thought)
                    
                    if result.get("passes", False):
                        refined = result.get("refined", thought)
                        print(f"\n💭 Thought #{thought_count}: {refined}")
                        
                        # Store in memory
                        mem_manager.add_thought(
                            content=refined,
                            salience=result.get("salience", 0.5)
                        )
                except Exception as e:
                    print(f"Conscious processing error: {e}")
                    traceback.print_exc()
            
            # Status update every 30 seconds
            if time.time() - last_status > 30:
                status = f"""
                System Status
                -------------
                Thoughts generated: {subconscious.thought_count}
                Queue size: {thought_queue.qsize()}
                Memory entries: {len(mem_manager.get_recent_memories(1000))}
                """
                print(status)
                last_status = time.time()
            
            time.sleep(0.1)  # Reduced sleep for responsiveness

    except KeyboardInterrupt:
        print("\nShutting down...")
        subconscious.stop()
        print(f"""
        Final Stats
        -----------
        Total thoughts processed: {thought_count}
        Final queue size: {thought_queue.qsize()}
        """)

if __name__ == "__main__":
    main()