"""
Enhanced configuration for Subconscious module
Optimized for GPT-Neo 2.7B on Colab GPU with fail-safes
"""

import torch
from typing import Literal

# Hardware Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto"  # Lets accelerate handle layer distribution
TORCH_DTYPE = torch.float16  # Critical for 2.7B model

# Model Selection (Prioritized list)
MODEL_CHOICES = {
    "2.7B": "EleutherAI/gpt-neo-2.7B",
    "1.3B": "EleutherAI/gpt-neo-1.3B",
    "fallback": "EleutherAI/gpt-neo-125M"
}
DEFAULT_MODEL = MODEL_CHOICES["2.7B"]

# Quantization (8-bit saves ~4GB VRAM)
USE_8BIT = True
LOAD_IN_4BIT = False  # More aggressive but unstable

# Memory Management
MAX_MEMORY_MAP = {
    0: "20GiB",  # GPU0
    "cpu": "30GiB"  # Offload to CPU if needed
}

# Generation Parameters
MAX_NEW_TOKENS = 60  # Increased for richer thoughts
TEMPERATURE = 0.85  # Lower than 125M for coherence
TOP_P = 0.9
TOP_K = 50  # Reduced for larger model
REPETITION_PENALTY = 1.2  # Prevent looping

# Context Management
CONTEXT_WINDOW = 2  # Reduced for memory constraints
CONTEXT_UPDATE_INTERVAL = 3
CONTEXT_TOKEN_LIMIT = 512  # Hard cap for safety

# Deduplication
DUPLICATE_THRESHOLD = 0.82  # Lower threshold for larger model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight but effective

# System Behavior
MAX_RETRIES = 2  # For generation failures
RETRY_DELAY = 1.0  # Seconds between retries
INTERVAL_SECONDS = 4.0  # Increased for larger model

# Safety Nets
MEMORY_MONITOR_INTERVAL = 30  # Check VRAM every N thoughts
AUTO_REDUCE_ON_OOM = True  # Dynamically lower params
FALLBACK_MODEL_ENABLED = True

# Type Hints for Validation
ModelSize = Literal["2.7B", "1.3B", "125M"]
DevicePrecision = Literal["float32", "float16", "bfloat16"]