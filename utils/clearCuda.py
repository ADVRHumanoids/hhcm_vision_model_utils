#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clear CUDA cache and display detailed GPU memory statistics.

Simple utility script to free unused GPU memory and print comprehensive memory
usage summary. Useful for debugging out-of-memory errors or monitoring GPU
memory usage during development.

Purpose:
    Release all unused cached GPU memory and display detailed memory statistics
    including allocated memory, reserved memory, and fragmentation info.

Usage:
    python3 clearCuda.py

Output:
    Prints complete CUDA memory summary showing:
    - Active memory allocations
    - Cached memory (reserved but not used)
    - Memory fragmentation
    - Per-tensor allocation details

Requirements:
    pip install torch

Notes:
    - Only clears UNUSED cached memory (active allocations are preserved)
    - Requires CUDA-capable GPU and PyTorch with CUDA support
    - If no GPU available, script will print empty summary

Author: tori
"""

import torch

# Release all unused cached memory
torch.cuda.empty_cache()

# Print detailed memory summary
print(torch.cuda.memory_summary(device=None, abbreviated=False))