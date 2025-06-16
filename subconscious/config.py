
# subconscious/config.py
"""
Configuration settings for the Subconscious module.
Optimized for Google Colab GPU (12GB+ VRAM).
"""

# Interval between thought generations (in seconds)
INTERVAL_SECONDS = 5

# Default Hugging Face model for Colab GPU
# - "EleutherAI/gpt-j-6B" (~6B parameters)
#   loaded in 8-bit to fit in ~8GB VRAM
DEFAULT_MODEL = "EleutherAI/gpt-j-6B"

# Quantization settings
USE_8BIT = True           # requires bitsandbytes
DEVICE_MAP = "auto"      # let accelerate pick GPU device

# Sampling parameters for creative, coherent outputs
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 50

# Prompt prefix for thought generation
PROMPT_PREFIX = "Subconscious thought:"

# Maximum number of new tokens to generate per thought
MAX_NEW_TOKENS = 40

# Number of recent memories to fetch for context
MEMORY_CONTEXT_SIZE = 5
