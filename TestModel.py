#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:00:25 2022

@author: tori

From the testing part of https://www.kaggle.com/code/havinath/object-detection-using-pytorch-training
"""

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2

#from ensemble_boxes import weighted_boxes_fusion

from torchmetrics.detection.mean_ap import MeanAveragePrecision


detection_threshold = 0.5 #not used for test_simple
detection_threshold = 0.4
test_images_to_print = []
outputs_to_print = []

# def make_ensemble_predictions(images, models):
#     result = []
#     for net in models:
#         net.eval()
#         outputs = net(images)
#         result.append(outputs)
#     return result

# def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.55, skip_box_thr=0.4, weights=None):
#     boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
#     scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
#     labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
#     boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#     boxes = boxes*(image_size-1)
#     return boxes, scores, labels

# def test(test_data_loader, models, device):

#     with torch.no_grad():

#         for images, image_ids in test_data_loader:
#             images = list(image.to(device) for image in images)    
#             predictions = make_ensemble_predictions(images, models)
        
#             for i, image in enumerate(images):
#                 test_images.append(image) #Saving image values
#                 boxes, scores, labels = run_wbf(predictions, image_index=i)
        
#                 boxes = boxes.astype(np.int32).clip(min=0, max=1023)
                    
#                 preds = boxes
#                 preds_sorted_idx = np.argsort(scores)[::-1]
#                 preds_sorted = preds[preds_sorted_idx]
#                 boxes = preds
                
#                 output = {
#                     'boxes': boxes,
#                     'scores': scores
#                 }

#                 outputs.append(output) #Saving outputs and scores
#                 image_id = image_ids[i]
        
        
#         ########### show images results
#         fig, axs = plt.subplots(2, 2, figsize=(32, 16))
#         axs = axs.ravel()
#         for i in range(4):
#             #sample = test_images[i].permute(1,2,0).cpu().numpy()
#             sample = torchvision.transforms.functional.convert_image_dtype(
#                 test_images[i].cpu(), torch.uint8).numpy().transpose([1,2,0])
#             sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
#             boxes = outputs[i]['boxes']
#             scores = outputs[i]['scores']
#             boxes = boxes[scores >= detection_threshold].astype(np.int32)
        
#             for box in boxes:
#                 cv2.rectangle(sample,
#                               (box[0], box[1]),
#                               (box[2], box[3]),
#                               (220, 0, 0), 2)
        
#             axs[i].set_axis_off()
#             axs[i].imshow(sample)
            
def test_simple(data_loader_test, model, device, show_images=False):

    with torch.no_grad():
        
        model.to(device)
        model.eval()    
        
        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        
        for images, targets in data_loader_test:
            
            images = list(image.to(device) for image in images)    
            predictions = model(images)   
            
            for i, pred in enumerate(predictions):
                   
                if pred['scores'].size(0) > 0:
                    best_index = torch.argmax(pred['scores'])
                    
                    #after 2 hours lost, this is the only method I found to create
                    # the tensor as size (1, 4) for boxes and size (1,1) for lab and scores
                    pred['boxes'] = torch.Tensor([np.array(pred['boxes'][best_index].cpu().numpy())])
                    pred['labels'] = torch.IntTensor([pred['labels'][best_index]])
                    pred['scores'] = torch.Tensor([pred['scores'][best_index]])
                
                else: 
                    for key in pred:
                        pred['boxes'] = torch.Tensor([[0,0,0,0]])
                        pred['labels'] = torch.IntTensor([10])
                        pred['scores'] = torch.Tensor([0])
                        
                # to print images
                test_images_to_print.append(images[i]) #Saving image values        
                
                if (pred['boxes'].size(0) > 0):
                    output = {
                        'boxes': pred['boxes'][0].cpu().numpy().astype(np.int32),
                        'scores': pred['scores']
                    }
                else :
                    output = {
                        'boxes': [],
                        'scores': []
                    }
    
                outputs_to_print.append(output) #Saving outputs and scores
                
            #all to cpu
            for pred in predictions:
                pred['boxes'].cpu()
                pred['labels'].cpu()
                pred['scores'].cpu()

            for target in targets:
                target['boxes'].cpu()
                target['labels'].cpu()
                
            metric.update(preds=predictions, target=targets)
            metric_result = metric.compute()
        
        if show_images:
            ########### show images results
            fig, axs = plt.subplots(2, 2, figsize=(32, 16))
            axs = axs.ravel()
            nImgToSee = min(4, len(test_images_to_print))
            for i in range(nImgToSee):
                #sample = test_images[i].permute(1,2,0).cpu().numpy()
                sample = torchvision.transforms.functional.convert_image_dtype(
                    test_images_to_print[i].cpu(), torch.uint8).numpy().transpose([1,2,0])
                sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                box = outputs_to_print[i]['boxes']
                score = outputs_to_print[i]['scores']
                
                if len(box) > 0:
                    cv2.rectangle(sample,
                                  (box[0], box[1]),
                                  (box[2], box[3]),
                                  (220, 0, 0), 2)
                    
                    cv2.putText(sample, str(score[0].detach().numpy()), (round(box[0].item()), round(box[3].item()+20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
                axs[i].set_axis_off()
                axs[i].imshow(sample)
            
        return metric_result