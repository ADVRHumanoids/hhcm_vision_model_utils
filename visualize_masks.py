import cv2
import numpy as np
from PIL import Image, ImageDraw,ImageFont
import random
from matplotlib import pyplot as plt

image_path = "/home/tori/YOLO/data/det_laser_nicla_320x320/train/images/lab_images_frame0297_jpg.rf.092e74106af6ddc1bc104e14a21f5ce0.jpg"  # path of the image, change it
annotation_path = "/home/tori/YOLO/data/det_laser_nicla_320x320/train/labels/lab_images_frame0297_jpg.rf.092e74106af6ddc1bc104e14a21f5ce0.txt"  # path of the annotation text file, change it

# The Helper functions below assume that the image size is (640,640).Hence resizing the image.
#Open the image
img = Image.open(image_path)
#Resize the image to 640 by 640
img = img.resize((640, 640))
#if you want then you can save the resized image by img.save('resized_image.jpg')

# <--------- Helper functions starts here ----------------------------------->
def maskVisualize(image,mask):
  fontsize = 18
  f, ax = plt.subplots(2, 1, figsize=(8, 8))
  ax[0].imshow(image)
  ax[1].imshow(mask)

#Define the boundary coordinates as a list of (x, y) tuples
'''
def draw_points_on_image(image, points):
  #resize image to 640*640
  resimg = cv2.resize(image, (640,640))
  #iterate for each mask
  for mask in points:
    #Draw each point on the image
    for point in mask:
        cv2.circle(resimg, tuple(point), 1, (0,0,255), -1,)
  #Display the image
  cv2.imshow("resimg", resimg)
'''
#convert the mask from the txt file(annotation_path is path of txt file) to array of points making that mask.
def generate_points(annotation_path=''):
  labels=[] # this will store labels
  #we are assuming that the image is of dimension (640,640). then you have annotated it.
  with open(annotation_path, "r") as file:
    points=[]
    for line in file:
      label,bbox,lis=line.split()[0],line.split()[1:5], line.split()[5:]
      labels.append(label)
      lis=list(map(float,lis))
      for i in range(len(lis)):
        lis[i]=int(lis[i]*640)
      newlis=[]
      i=0
      while(i<len(lis)):
        #appendint the coordinates as a tuple (x,y)
        newlis.append((lis[i],lis[i+1]))
        i+=2
      points.append(newlis)
    return labels,points


#the below function convert the boundary coordinates to mask array (it shows mask if you pass 1 at show)
#the mask array is required when we want to augument the mask also using albumentation
def convert_boundary_to_mask_array(labels,points, show=0):
  #Create a new image with the same size as the desired mask
  mask = Image.new("L", (640, 640), 0)
  draw = ImageDraw.Draw(mask)
  for i,boundary_coords in enumerate(points):
    #boundary_coords represent boundary of one polygon
    #Draw the boundary on the mask image
    if len(boundary_coords) == 0:
      continue
    draw.polygon(boundary_coords,fill=1)
    #Also put the label as text
    #Compute the centroid of the polygon
    centroid_x = sum(x for x, _ in boundary_coords) / len(boundary_coords)
    centroid_y = sum(y for _, y in boundary_coords) / len(boundary_coords)
    centroid = (int(centroid_x), int(centroid_y))
    #Write the name at the centroid
    text = str(labels[i])
    #Write the label at the centroid
    font = ImageFont.load_default()
    font.size = 30
    #text_w, text_h = draw.textsize(text, font=font)
    text_w = draw.textlength(text, font=font)
    text_h = font.size * 1
    text_pos = (centroid[0] - text_w/2, centroid[1] - text_h/2)
    draw.text(text_pos, text, font=font, fill='black')
  #Convert the mask image to a numpy array
  mask_array = np.array(mask)*255
  #Show the mask image
  if(show==1):
    #Image.fromarray(mask_array).show()
    cv2.imshow("mask_array", mask_array)
  return mask_array

#function that takes mask path (yolov8 seg txt file) and return mask of an image (shape of mask == shape of image)
def generate_mask(annotation_path='',show=0):
  #pass show=1 for showing the generated mask
  #firstly we generate the points (coordinates) from the annotations
  labels,points=generate_points(annotation_path)
  #once we get the points we will now generate the mask image from these points (binary mask image (black/white))
  #mask is represented by white and ground is represented as black
  mask_array=convert_boundary_to_mask_array(labels,points,show)
  return mask_array
# <---------- Helper Functions Ends here ------------------------------------------------------------->

mask_array=generate_mask(annotation_path=annotation_path,show=0)
maskVisualize(np.array(img),mask_array)

##Show bboxes
img = cv2.imread(image_path)
dh, dw, _ = img.shape

with open(annotation_path, "r") as file:
  points=[]
  for dt in file:
    label,bbox=dt.split()[0],dt.split()[1:5]

    # Split string to float
    x = float(bbox[0])
    y = float(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)
    cv2.putText(img, label, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)


plt.imshow(img)
plt.show()
