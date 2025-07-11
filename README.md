# hhcm_yolo_training

Scripts to train YOLO and not only models with datasets. The repo also include other scripts to manipulate COCO and YOLO dataset.

## Training Non-YOLO models

Taken from https://github.com/pytorch/vision/tree/main

- Download files from https://github.com/pytorch/vision/tree/main/references/detection and put them in this folder, inside a "detection" folder. 
  They will be used by `TrainModel.py`

- Run `TrainModel.py`. Examples about how to run are given in TrainMain.py

## Training YOLOv5 models

Taken from https://github.com/ultralytics/yolov5/

You can either follow the https://github.com/ultralytics/yolov5/ tutorial (Training part)

OR, clone that repo somewhere and use the `TrainYolo.py`. Example about how to use the MainYolo are in `TrainMain.py`

## Training YOLOv11
*TrainYolov11.py*, check arguments in the file.

## COCO-YOLO utils
### COCO-to-YOLO
```extractcocoexportyolo.py``` is a python script to easily download cocosubsets. At the beginning of the file you can find its arguments to set. Pretty rough at this stage, poorly written, and arguments are hardcoded, but it worked for me so far

#### Why
Ever wanted to download only a subset of coco to be used for YOLO? With only certain classes? And with a balanced (more or less) number of sample for each category? This is the script for you. I made this because 
[fiftyone](https://docs.voxel51.com/) was limited in this.  
In brief, it downloads coco images and labes, convert them to yolo format and store it in your pc, ready to be used.

#### How
- `cocoTrain` and `cocoVal` paths to the instances_trainXXX.json and instances_val2017.json, to be downloaded in advance from YOLO website
- The script downloads images into `img_folder`. 
- Inside `output_name`, dataset is created, containing images and label. Images are first downloaded into `img_folder` and then copied into `output_name`. In such a way, keeping the `img_folder`, multiple subsets can be creted running multiple time the script, avoiding to download a particular image multiple time, but only once when necessary storing it in the general `img_folder`, saving time.
- `model_type` det or seg, the format of the labels. While det info should be always present in the COCO dataset, seg may be not. If seg is requested, det bbox info will be used as a segmentation label.
  - det: label, center-x, center-y, width, height
  - seg: label, x1, y1, ..., xn, yn
- `N_sample_train` and `N_sample_valid`. YOLO dataset has a big number of "person" instances with respect to other categories. So it may end up training your network unbalancing the categories. This args tell how many images do you want *for each category*. The script also assures that training and validation images are always different.
  - Note 1: Take care that for certain categories, images may be not enough, hence a smaller number will be available. At least, the script will try to take from the YOLO trainining images some to reach the requested `N_sample_valid`, if not enough.
  - Note 2: for each category, it is assured that AT LEAST this number of occurencies exist (because on same image multiple categories can be present, and also multiple occurencies of same category).
- `my_categories` categories you want, with id that will be used as reference for each of them. Ids must one after the other (no holes), so original YOLO IDs will be modified if you "skip" some category.  

### Rename ID of labels (yolo format)
```renameIdClassYoloLabel.py``` simple raw script to rename the id of all the labels in a given dataset.

### Visualize mask-bboxes
```visualize_masks.py```. Just set the image and label path. Its is fine with both YOLO det and YOLO seg formats

### Convert YOLO segmentation format to YOLO detection format
```convert_yolo_seg_to_det.py``` one-shot chatgpt script to convert the segmentation mask into a detection bounding box, for all label files in a given folder. (remember to do it for both the train and valid (and test eventually) folders).
