# subconscious/config.py
"""
Configuration settings for the Subconscious module.
Optimized for Google Colab GPU and dynamic context.
"""

# Generation frequency (seconds)
INTERVAL_SECONDS = 3

# Model and quantization
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
USE_8BIT = True
DEVICE_MAP = "auto"

# Generation length - increased for Phi-3's better coherence
MAX_NEW_TOKENS = 60

# Sampling parameters
TEMPERATURE = 0.6
TOP_P = 0.88
TOP_K = 60

# How many past memories to include in context
CONTEXT_WINDOW = 3

# How often to refresh context (in number of thoughts)
CONTEXT_UPDATE_INTERVAL = 5

# Duplicate suppression threshold (Jaccard similarity)
DUPLICATE_THRESHOLD = 0.85