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
    thought_queue = queue.Queue(maxsize=100)
    processor = ConsciousProcessor()
    
    subconscious = Subconscious(
        output_queue=thought_queue,
        memory=mem_manager
    )
    
    # Start system
    subconscious.start()
    
    # Main loop
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