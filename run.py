#!/usr/bin/env python3
# run.py - Clean startup script
import gc
import torch
from orchestrator import main

if __name__ == "__main__":
    # Clean up before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run the main function
    main()
    
    # Clean up after finishing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()