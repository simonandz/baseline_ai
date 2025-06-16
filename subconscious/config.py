# Generation parameters
INTERVAL_SECONDS = 3  # More frequent raw thoughts
DEFAULT_MODEL = "EleutherAI/gpt-j-6B"
USE_8BIT = True
MAX_NEW_TOKENS = 40  # Shorter raw thoughts

# Creative generation parameters
TEMPERATURE = 0.9  # More creative
TOP_P = 0.95
TOP_K = 100

# Context parameters
CONTEXT_WINDOW = 3  # Last 3 memories for context