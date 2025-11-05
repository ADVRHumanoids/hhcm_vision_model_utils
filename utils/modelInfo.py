#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load and display PyTorch model architecture and device information.

Simple utility script for inspecting saved PyTorch models. Loads a .pt model file,
automatically detects available device (CUDA/CPU), and displays the model architecture.
Useful for quick model inspection during development.

Arguments:
    --model: Path to .pt model file (required)

Requirements:
    pip install torch torchinfo

Author: tori
Modified by: Alessio Lovato, 04-11-2025
"""

import os
import torch
import argparse

parser = argparse.ArgumentParser(description='Display PyTorch Model Information')
parser.add_argument('--model', type=str, required=True, help='Path to the .pt model file')
args = parser.parse_args()

# Configure model to load (edit these variables)
model_path = os.path.join(args.model)
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load model with automatic device detection
if torch.cuda.is_available():
    device = torch.device('cuda')
    model = torch.load(model_path)
else:
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=torch.device('cpu'))

# Print model architecture
print(model.cuda())