#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:37:21 2022

@author: tori
"""

import torch
from torchsummary import summary
import os

#model_name = "model1.pt"
model_name = "fasterrcnn_mobilenet_high_e10_b4_tvt702010.pt"
model_path = os.path.join("../../../learningStuff", model_name)

if torch.cuda.is_available():
    device = torch.device('cuda')
    model = torch.load(model_path)

else:
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
print(model.cuda())