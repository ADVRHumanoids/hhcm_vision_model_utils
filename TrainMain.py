#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:31:38 2022

@author: tori
"""
import os

from TrainModel  import run

if __name__ == "__main__":
    

    ######### You can train multiple model with different params
    ######### take care that if same params are used twice, old model is overwritten 

    ####### Non-YOLO models
    # Data format according to COCO: data_path should be a folder cointating images and annotation folders.    
    ## in images, all the images (not divided by train, val, test, this is done by the code)
    ## in annotations, an instances_default.json file 
    # run(data_path='../../data/laser_v3', batch_size=1, num_epochs=40, model_type = 'faster_rcnn_v1', val_percentage=0.20, test_percentage=0.1)
    # run(data_path='../../data/laser_v3', batch_size=1, num_epochs=30, model_type = 'faster_rcnn_v2', val_percentage=0.20, test_percentage=0.1)
    
    # run(data_path='../../data/laser_v3', batch_size=1, num_epochs=40, model_type = 'fasterrcnn_mobilenet_low', val_percentage=0.20, test_percentage=0.1)
    # run(data_path='../../data/laser_v3', batch_size=1, num_epochs=30, model_type = 'fasterrcnn_mobilenet_high', val_percentage=0.20, test_percentage=0.1)



    ######### YOLO models
    # os.system('python TrainYolo.py --train_file ../../yolov5/train.py --data_file ../../data/laser_v3/data.yaml --batch 8 --epochs 100 --weights yolov5s6')
	# os.system('python TrainYolo.py --train_file ../../yolov5/train.py --data_file ../../data/laser_v3/data.yaml --batch 8 --epochs 200 --weights yolov5s6')
	
	# os.system('python TrainYolo.py --train_file ../../yolov5/train.py --data_file ../../data/laser_v3/data.yaml --batch 8 --epochs 100 --weights yolov5m6')	
	# os.system('python TrainYolo.py --train_file ../../yolov5/train.py --data_file ../../data/laser_v3/data.yaml --batch 8 --epochs 200 --weights yolov5m6')
	
	# os.system('python TrainYolo.py --train_file ../../yolov5/train.py --data_file ../../data/laser_v3/data.yaml --batch 8 --epochs 100 --weights yolov5l6')	
	# os.system('python TrainYolo.py --train_file ../../yolov5/train.py --data_file ../../data/laser_v3/data.yaml --batch 8 --epochs 200 --weights yolov5l6')

    print("Done!")