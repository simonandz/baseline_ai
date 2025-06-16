# memory/preload.py
from .schemas import BASE_KNOWLEDGE
from .manager import MemoryManager

def initialize_base_knowledge(memory_manager: MemoryManager):
    """Load immutable core knowledge about Maddie"""
    for knowledge in BASE_KNOWLEDGE:
        memory_manager.add_memory(
            content=knowledge["content"],
            memory_type=knowledge["type"],
            salience=knowledge["salience"],
            immutable=knowledge.get("immutable", False)
        )
    print("Base knowledge loaded: Identity, Purpose, Architecture")