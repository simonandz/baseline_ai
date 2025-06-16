# subconscious/config.py
"""
Configuration settings for the Subconscious module.
Optimized for Google Colab GPU and dynamic context.
"""

# Generation frequency (seconds)
INTERVAL_SECONDS = 3

# Model and quantization
DEFAULT_MODEL = "EleutherAI/gpt-j-6B"
USE_8BIT = True
DEVICE_MAP = "auto"

# Generation length
MAX_NEW_TOKENS = 40

# Sampling parameters
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 100

# How many past memories to include in context
CONTEXT_WINDOW = 3

# How often to refresh context (in number of thoughts)
CONTEXT_UPDATE_INTERVAL = 5

# Duplicate suppression threshold (Jaccard similarity)
DUPLICATE_THRESHOLD = 0.85
