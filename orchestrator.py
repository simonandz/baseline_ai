# orchestrator.py - Updated Version
import torch
import time
import os
import queue
from conscious.pipeline import ConsciousProcessor
from subconscious import Subconscious
from memory.manager import MemoryManager
from memory.preload import initialize_base_knowledge

# Resolve CUDA initialization conflicts
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

def main():
    # Initialize memory manager
    mem_manager = MemoryManager()
    
    # Load base knowledge
    initialize_base_knowledge(mem_manager)
    
    # Print core identity
    print(f"\n{'='*40}")
    print(f"AI Identity: Maddie")
    
    # Get identity from base knowledge instead of memory manager
    # (Assuming initialize_base_knowledge prints identity)
    identity = [
        "I am Maddie, an artificial intelligence system designed to simulate human thought processes."
    ]
    for fact in identity:
        print(f"- {fact}")
    print(f"{'='*40}\n")
    
    # Initialize components
    thought_queue = queue.Queue(maxsize=100)
    processor = ConsciousProcessor()
    
    # Initialize Subconscious
    subconscious = Subconscious(
        output_queue=thought_queue,
        memory=mem_manager,
        model_name="EleutherAI/gpt-neo-125M",
        device=device
    )
    
    # Start system
    subconscious.start()
    print("Subconscious started")

    # In orchestrator.py after subconscious.start()
    def monitor_subconscious():
        """Real-time display of subconscious thoughts"""
        print("\nSubconscious Monitor Active - Press Ctrl+C to stop")
        try:
            while True:
                if not thought_queue.empty():
                    thought = thought_queue.get()
                    print(f"🧠 [{time.strftime('%H:%M:%S')}] {thought}")
                    thought_queue.task_done()  # Important to mark as processed
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Monitoring stopped")

    # Start in a separate thread
    monitor_thread = threading.Thread(target=monitor_subconscious, daemon=True)
    monitor_thread.start()

    # Main loop
    last_consolidation = time.time()
    while True:
        # Process thoughts
        if not thought_queue.empty():
            thought = thought_queue.get()
            result = processor.process(thought)
            
            if result.get("passes", False):
                refined = result.get("refined", thought)
                print(f"💭 {refined}")
                
                # Store as observation
                mem_manager.add_memory(
                    content=refined,
                    salient_score=result.get("salience", 0.5)  # Changed parameter name
                )
        
        # Periodically consolidate memories
        if time.time() - last_consolidation > 3600:  # Every hour
            mem_manager.consolidate_memory()  # Changed method name
            last_consolidation = time.time()
            print("Memory consolidation completed")
            
        # Small sleep to prevent busy waiting
        time.sleep(0.1)

if __name__ == "__main__":
    main()