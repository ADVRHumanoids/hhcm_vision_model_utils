from ultralytics import YOLO
import os, time

import argparse

parser = argparse.ArgumentParser(description='')
parser._action_groups.pop()
optional = parser.add_argument_group('optional arguments')
optional.add_argument('--weights', type=str, default="yolo11m-seg")
optional.add_argument('--data_path', type=str, default="/home/tori/YOLO/data/")
optional.add_argument('--data_file', type=str, default="coco_lasers_nicla_combo_640")
optional.add_argument('--batch', type=int, default="16")
optional.add_argument('--epochs', type=int, default="500")
optional.add_argument('--img_size', type=int, default="640")

args, unknown = parser.parse_known_args()

weights = args.weights
data_path = args.data_path
data_file = args.data_file
img_size = args.img_size 
batch_size = args.batch
epochs = args.epochs

model_out_name = (args.weights + "_" + data_file + "_e" + str(epochs) + "_b" + str(batch_size) + "_" + str(int(time.time())))
# Load a model
model = YOLO(weights+".pt")

# Train the model
train_results = model.train(
    #data="/home/tori/ws/src/nn_laser_spot_tracking/training/coco_subset_1733499037/data.yaml",  # path to dataset YAML
    data=data_path+data_file+".yaml",
    epochs=epochs,  # number of training epochs
    imgsz=img_size,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch = batch_size, #between 0-1 for % GPU utilization
    name = model_out_name
)

# Evaluate model performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model