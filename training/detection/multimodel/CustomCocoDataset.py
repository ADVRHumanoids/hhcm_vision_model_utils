#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom PyTorch Dataset class for loading COCO format object detection datasets.

Provides a PyTorch-compatible dataset interface for COCO annotations, handling
image loading, bounding box format conversion (COCO to PyTorch), and optional
data transformations. Suitable for training object detection models with COCO datasets.

@author: tori, 16-08-2022
Modified by: Alessio Lovato, 31-10-2025
"""
import os
import torch
from pycocotools.coco import COCO
from PIL import Image

class CustomCocoDataset(torch.utils.data.Dataset):
    """
    Custom COCO dataset for object detection tasks.

    Loads images and annotations from COCO format files and provides them
    in PyTorch-compatible format for training detection models.
    """

    def __init__(self, root, annotation, transforms=None):
        """
        Initialize the COCO dataset.

        Args:
            root (str): Root directory containing images
            annotation (str): Path to COCO format annotation JSON file
            transforms (callable, optional): Optional transforms to apply to images. Defaults to None.
        """
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Get a single image and its annotations by index.

        Converts COCO bounding box format [xmin, ymin, width, height] to
        PyTorch format [xmin, ymin, xmax, ymax].

        Args:
            index (int): Index of the image to retrieve

        Returns:
            tuple: (image, annotation_dict) where:
                - image: PIL Image or transformed tensor
                - annotation_dict: Dictionary containing:
                    - boxes (Tensor): Bounding boxes in [xmin, ymin, xmax, ymax] format
                    - labels (Tensor): Class labels (all 1 for single-class detection)
                    - image_id (Tensor): Image ID
                    - area (Tensor): Bounding box areas
                    - iscrowd (Tensor): Crowd flags (all 0)
        """
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        
        # if no label, we must anyway fill the object
        if num_objs == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
        
        else:
            for i in range(num_objs):
                xmin = coco_annotation[i]['bbox'][0]
                ymin = coco_annotation[i]['bbox'][1]
                xmax = xmin + coco_annotation[i]['bbox'][2]
                ymax = ymin + coco_annotation[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        
        # Size of bbox (Rectangular)
        areas = []
        if num_objs == 0:
            areas = torch.zeros((0,1), dtype=torch.float32)
            
        else:
            for i in range(num_objs):
                areas.append(coco_annotation[i]['area'])
            areas = torch.as_tensor(areas, dtype=torch.float32)
            
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset
        """
        return len(self.ids)
    
