#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:07:19 2023

@author: tori
"""
import os

import argparse

parser = argparse.ArgumentParser(description='')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('--train_file', type=str, doc="the yolov5 train.py file with location (e.g. <some_folder>/yolov5/train.py) taken from https://github.com/ultralytics/yolov5/")
required.add_argument('--data_file', type=str, doc="the yaml file according to the yoloV5 train data format (e.g. <some_folder>/data.yaml)")
optional = parser.add_argument_group('optional arguments')
optional.add_argument('--batch', type=str, default="1")
optional.add_argument('--epochs', type=str, default="3")
optional.add_argument('--weights', type=str, default="yolov5s6")
optional.add_argument('--img_size', type=str, default="1280", doc="if rectangular images, put the bigger dimension. --rect option is another thing, it does not make use of mosaic stuff who knows what it is")
optional.add_argument('--data_name', type=str, default="laserSpots", doc="used for naming the output model")

args, unknown = parser.parse_known_args()

yolo_train_file = args.train_file
img_size = args.img_size 
batch_size = args.batch
epochs = args.epochs
data_file = args.data_file
weights = args.weights + '.pt'

model_out_name = (args.weights + "_e" + epochs + "_b"+ batch_size + "_tvt" + 
               str(30) + 
               str(20) + 
               str(10) +
               "_" + args.data_name)

command = " ".join(['python', 
                    yolo_train_file,
                    '--img', img_size,
                    '--batch', batch_size,
                    '--epochs', epochs,
                    '--data', data_file,
                    '--weights', weights, 
                    '--name', model_out_name  
                    ])                             

os.system(command)
