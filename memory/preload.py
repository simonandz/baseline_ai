# memory/preload.py
from .schemas import BASE_KNOWLEDGE
from .manager import MemoryManager

def initialize_base_knowledge(memory_manager: MemoryManager):
    """Load immutable core knowledge about Maddie"""
    for knowledge in BASE_KNOWLEDGE:
        # Extract parameters with defaults
        content = knowledge["content"]
        salient_score = knowledge.get("salience", 1.0)  # Default to highest importance
        memory_type = knowledge.get("type", "core")
        
        # Add to memory using new API
        memory_manager.add_memory(
            content=content,
            salient_score=salient_score
        )
    print("Base knowledge loaded: Identity, Purpose, Architecture")