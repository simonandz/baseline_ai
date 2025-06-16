# subconscious/config.py
"""
Configuration settings for the Subconscious module.
"""

# Interval between thought generations (in seconds)
INTERVAL_SECONDS = 5

# Default Hugging Face model optimized for CPU
# The largest model recommended for an 8th-gen Surface Pro is "gpt2-medium" (~345M parameters).
DEFAULT_MODEL = "gpt2-medium"

# Sampling parameters for creative, coherent outputs
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50

# Prompt prefix for thought generation
PROMPT_PREFIX = "Subconscious thought:"

# Maximum number of new tokens to generate per thought
MAX_NEW_TOKENS = 40

# Device: -1 for CPU only (no GPU)
DEVICE = -1

# Number of recent memories to fetch for context
MEMORY_CONTEXT_SIZE = 5
