#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:46:19 2022

@author: tori

example taken from https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
"""
import torch
import torch.utils.data
import torchvision
import sys
import math

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

sys.path.insert(1, 'detection')
from engine import train_one_epoch, evaluate
import utils

from CustomCocoDataset import CustomCocoDataset

import numpy as np
import matplotlib.pyplot as plt

from TestModel import test_simple


"""
example taken from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
def get_model_mobilenet(num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model

"""
example taken from 
https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
https://pytorch.org/vision/stable/models.html#object-detection
"""
def get_model_fasterrcnn(num_classes, version=1):
    
    # load an instance segmentation model pre-trained pre-trained on COCO
    # Default is COCO, and it seems no other weights are available now
    
    if version == 1:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    elif version == 2:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weigths=weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

"""
this is for mobile uses?
"""
def get_model_fasterrcnn_mobilenet(num_classes, version='high'):
    
    # load an instance segmentation model pre-trained pre-trained on COCO
    # Default is COCO, and it seems no other weights are available now
    
    if version == 'high':
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weigths=weights)
        
    elif version == 'low':
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weigths=weights)
        
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_model_fcos(num_classes):

    weights = torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fcos_resnet50_fpn(weigths=weights)
    
    # Fcos has not roi_heads... so it is ok like this or we need other stuff?

    return model 

def get_model_retinanet(num_classes, version=1):

    if version == 1:
        weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.retinanet_resnet50_fpn(weigths=weights)        
    elif version == 2:
        weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weigths=weights)        
    
    # retinanet has not ros_heads, so to replace the last layer(s) I found:
    # https://datascience.stackexchange.com/questions/92724/fine-tune-the-retinanet-model-in-pytorch    
    # but it uses attributes that are not part of the object. Who knows

    return model  

def get_model_ssd(num_classes):

    weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    model = torchvision.models.detection.ssd300_vgg16(weigths=weights)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model 

def get_model_ssdlite(num_classes):

    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weigths=weights)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model 


"""
from https://pytorch.org/vision/0.13/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
TODO
"""


# def get_transform(train=False):
#     custom_transforms = []
#     custom_transforms.append(torchvision.transforms.ToTensor())
#     if train:
#         x=3
#         #custom_transforms.append(torchvision.transforms.RandomHorizontalFlip())
#         #custom_transforms.append(torchvision.transforms.RandomVerticalFlip())
#         #custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(1.3))
#         #custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(0.7))
#         #custom_transforms.append(torchvision.transforms.RandomAutocontrast())
#     return torchvision.transforms.Compose(custom_transforms)

def get_transform(train=False):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    if train:
        custom_transforms.append(torchvision.transforms.RandomHorizontalFlip())
        custom_transforms.append(torchvision.transforms.RandomVerticalFlip())
        custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(1.3))
        custom_transforms.append(torchvision.transforms.RandomAdjustSharpness(0.7))
        custom_transforms.append(torchvision.transforms.RandomAutocontrast())
    return torchvision.transforms.Compose(custom_transforms)

def run(data_path, data_name="laserSpots", batch_size=1, num_epochs=1, model_type = 'faster_rcnn_v1', val_percentage=0.20, test_percentage=0.10) :
#if __name__ == "__main__":
#    batch_size=1
#    num_epochs=10
#    model_type = 'fasterrcnn_mobilenet_low'
#    val_percentage=0.20
#    test_percentage=0.10
    

    torch.manual_seed(14)
    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 2 classes; Only target class or background
    num_classes = 2
        
    data_dir_images = data_path + '/images'
    data_dir_annotations = data_path + '/annotations/instances_default.json'

    # create own Dataset
    dataset = CustomCocoDataset(root=data_dir_images,
                              annotation=data_dir_annotations,
                              transforms=get_transform(True)
                              )
    dataset_val = CustomCocoDataset(root=data_dir_images,
                              annotation=data_dir_annotations,
                              transforms=get_transform(False)
                              )
    dataset_test = CustomCocoDataset(root=data_dir_images,
                          annotation=data_dir_annotations,
                          transforms=get_transform(False)
                          )
    
    # split the dataset in train val and test set
    
    train_percentage = 1.0 - val_percentage - test_percentage
    
    indices = torch.randperm(len(dataset)).tolist()
    
    index_train = math.ceil(len(indices) * train_percentage)
    index_val = math.ceil(len(indices) * val_percentage)
    
    dataset = torch.utils.data.Subset(dataset, indices[0:index_train])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[index_train:index_train+index_val])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[index_train+index_val:])
    
    # own DataLoader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=utils.collate_fn)
    
    data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                   batch_size=batch_size, 
                                                   shuffle=False, 
                                                   num_workers=4,
                                                   collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                   batch_size=batch_size, 
                                                   shuffle=False, 
                                                   num_workers=4,
                                                   drop_last=False,
                                                   collate_fn=utils.collate_fn)
    

    # DataLoader is iterable over Dataset
    #for imgs, annotations in data_loader:
    #    imgs = list(img.to(device) for img in imgs)
    #    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #    print(annotations)
       
   # model_types = ['faster_rcnn_v1', 'faster_rcnn_v2', 'fasterrcnn_mobilenet_high', 'fasterrcnn_mobilenet_low',
    #               'fcos', 'retinanet_v1', 'retinanet_v2', 'ssd', 'ssd_lite']
   # model_type = model_types[0]
    
    model = None
    
    if model_type == 'faster_rcnn_v1':
        model = get_model_fasterrcnn(num_classes, version=1)
        
    elif model_type == 'faster_rcnn_v2':
        model = get_model_fasterrcnn(num_classes, version=2)

    elif model_type == 'fasterrcnn_mobilenet_high':
        model = get_model_fasterrcnn_mobilenet(num_classes, version='high')

    elif model_type == 'fasterrcnn_mobilenet_low':
        model = get_model_fasterrcnn_mobilenet(num_classes, version='low')

    #elif model_type == 'fcos':
        #model = get_model_fcos(num_classes)
        
    #elif model_type == 'retinanet_v1':
       # model = get_model_retinanet(num_classes, version=1)
        
    #elif model_type == 'retinanet_v2':
       # model = get_model_retinanet(num_classes, version=2)
        
    #elif model_type == 'ssd':
    #    model = get_model_ssd(num_classes)
        
    #elif model_type == 'ssd_lite':
    #    model = get_model_ssdlite(num_classes)
    
    else:
        print(f'model {model_type} not recognized')
        return
    
    # move model to the right device
    model.to(device)
    
    # construct optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
        
    ############# PLOT UTILS ###############
    loss_vals = []
    loss_vals_eval = []
    ##################################
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        if len(data_loader_val)>0:
            # evaluate on the validation dataset
            coco_evaluator, metric_logger_eval = evaluate(model, data_loader_val, device=device)
        
            ### Plotting?
            loss_vals.append(metric_logger.loss.value)
            loss_vals_eval.append(metric_logger_eval.loss.value)

    torch.save(model, model_type+ "_e"+str(num_epochs) + "_b"+str(batch_size) + "_tvt" + 
               str(math.floor(train_percentage*100)) + 
               str(math.floor(val_percentage*100)) + 
               str(math.floor(test_percentage*100)) +
               "_" + data_name +".pt")
    

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.linspace(1, num_epochs, num_epochs), loss_vals, label='train')
    ax1.plot(np.linspace(1, num_epochs, num_epochs), loss_vals_eval, label='validation')
    ax1.legend()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_xticks(np.linspace(1, num_epochs, num_epochs))
        
    #Perform test on test set
    if len(data_loader_test) > 0 :
        test_result = test_simple(data_loader_test, model, device)
        
        test_labels = []
        test_data = []
        for key,value in test_result.items():
            if value >= 0 and key.startswith('map'):
                test_labels.append(key)
                test_data.append(value.item())
    
        #test_labels = ['map', 'map_50', 'map_75', 'map_s', 'map_m', 'map_l']
        #test_data = [test_result['map'], test_result['map_50'], test_result['map_75'],
        #             test_result['map_small'], test_result['map_medium'], test_result['map_large']]
        ax2.bar(test_labels, test_data)
        ax2.set_xticks(range(len(test_labels)), test_labels, rotation='vertical')
     
    fig.savefig(model_type+ "_e"+str(num_epochs) + "_b"+str(batch_size) + "_tvt" +
                str(math.floor(train_percentage*100)) + 
                str(math.floor(val_percentage*100)) +
                str(math.floor(test_percentage*100)) + 
                "_" + data_name + ".png", bbox_inches='tight')