"""
Optimized configuration for continuous thought generation
with reduced repetition and temporal awareness.
"""

# Generation interval (seconds)
INTERVAL_SECONDS = 5

# Model configuration
DEFAULT_MODEL = "EleutherAI/gpt-j-6B"
USE_8BIT = True
DEVICE_MAP = "auto"

# Generation parameters
TEMPERATURE = 0.85          # Increased creativity
TOP_P = 0.92                # Broader sampling
TOP_K = 100                 # Wider token selection
MAX_NEW_TOKENS = 60         # Response length

# Context management
MEMORY_CONTEXT_SIZE = 8     # Recent memories to include
PROMPT_PREFIX = "Subconscious thought:"

# Similarity detection
TFIDF_SIMILARITY_THRESHOLD = 0.65  # Strict duplicate prevention
MAX_RECENT_THOUGHTS = 50    # Size of thought buffer