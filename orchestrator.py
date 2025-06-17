# Add to the top of orchestrator.py
import torch
import time  # Add this with other imports
import os
import queue  # <-- Add this line at the top
from conscious.pipeline import ConsciousProcessor
from subconscious import Subconscious  # Note: capital S

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

# orchestrator.py (partial)
from memory.manager import MemoryManager
from memory.preload import initialize_base_knowledge

def main():
    # Initialize memory with base knowledge
    mem_manager = MemoryManager()
    initialize_base_knowledge(mem_manager)
    
    # Print core identity
    identity = mem_manager.get_core_identity()
    print(f"\n{'='*40}")
    print(f"AI Identity: Maddie")
    for fact in identity['core']:
        print(f"- {fact}")
    print(f"{'='*40}\n")
    
    # Initialize other components
    thought_queue = queue.Queue(maxsize=100)  # Now works with the import
    processor = ConsciousProcessor()
    
    # In orchestrator.py, modify the Subconscious initialization
    subconscious = Subconscious(
        output_queue=thought_queue,
        memory=mem_manager,
        model_name="EleutherAI/gpt-neo-125M",  # Smaller model
        device=device  # Pass explicit device
    )
    
    # Start system
    subconscious.start()
    
    # Main loop
    last_consolidation = time.time()  # Initialize with current time
    while True:
        if not thought_queue.empty():
            thought = thought_queue.get()
            result = processor.process(thought)
            
            if result.get("passes", False):
                refined = result.get("refined", thought)
                print(f"💭 {refined}")
                
                # Store as observation
                mem_manager.add_memory(
                    content=refined,
                    memory_type="observation",
                    salience=result.get("salience", 0.5)
                )
        
        # Periodically consolidate memories
        if time.time() - last_consolidation > 3600:  # Every hour
            mem_manager.consolidate_insights()
            last_consolidation = time.time()