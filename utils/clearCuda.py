#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:47:17 2022

@author: tori
"""

import torch
torch.cuda.empty_cache() # release all UNUSED cached memory

print(torch.cuda.memory_summary(device=None, abbreviated=False))