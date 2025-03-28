import kagglehub
from ultralytics import YOLO
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import torch
import matplotlib.patches as mpatches
from transformers import pipeline
from PIL import Image
import jellyfish
from ultralytics.utils.metrics import bbox_iou

# download dataset from kaggle
#path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
#print("Path to dataset files:", path)

# load the annotation data
annos = sio.loadmat(r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_annos.mat')
type(annos)
annos.keys()
annotation = annos["annotations"]
class_def = annos["class_names"]
type(annotation)

annotation[0][0]

# transfer annotation data from dictionary data type to list data type
annotation_list = []

for i in annotation[0]:
    annotation_sub_list = []
    for j in i:
        annotation_sub_list.append(j.flatten()[0])
    annotation_list.append(annotation_sub_list)
    

# create dataframe for annotation data
data = pd.DataFrame(annotation_list, columns=annotation.dtype.names)
data.head(5)

# function for displaying image together with the class name and their bounding boxes

def load_image(name, path):
    img_path = path + '\\' + name 
    img = cv2.imread(img_path)
    return img

def plot_image(img, car_names):
    plt.imshow(img)
    plt.title(car_names)

def plot_grid(img_names, car_names, img_root, rows=5, cols=5):
    fig = plt.figure(figsize=(25,25))
    
    i = 1
    for iname, cname in zip(img_names, car_names):
        fig.add_subplot(rows,cols,i)
        img = load_image(iname, img_root)
        plot_image(img, cname)
        i += 1
        
    plt.show()

IMG_ROOT = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_train\cars_train'

# create label dictionary for decoding label data in dataframe
label_dict = {i+1: name[0] for i, name in enumerate(class_def[0])}
data['class_name'] = data['class'].map(label_dict)

# plot data
try:
    plot_grid(data['fname'][:25], data['class_name'][:25], IMG_ROOT)
except:
    pass

# problem with the annotation file, car name not accurate finding the source of truth of data.
# obtained new annotation files in https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input

# the code below similar as before 
label = sio.loadmat(r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_meta.mat')
label.keys()
class_def = []

for i in label['class_names'][0]:
    class_def.append(i[0])

def annotation_pipeline(path_annotation):
    annos = sio.loadmat(path_annotation)
    annos.keys()
    annotation = annos["annotations"]
    annotation_list = []

    for i in annotation[0]:
        annotation_sub_list = []
        for j in i:
            annotation_sub_list.append(j.flatten()[0])
        annotation_list.append(annotation_sub_list)
        
    data = pd.DataFrame(annotation_list, columns=annotation.dtype.names)

    label_dict = {i+1: name for i, name in enumerate(class_def)}
    label_dict.keys()
    data['class_name'] = data['class'].map(label_dict)

    return data

def load_image(name, path):
    img_path = path + '\\' + name 
    img = cv2.imread(img_path)
    return img

def plot_image(img, car_names, bbox):
    plt.imshow(img)
    plt.title(car_names)
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    rect = mpatches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

def plot_grid(data_plot, num_disp, img_root, rows=5, cols=5):
    fig = plt.figure(figsize=(25,25))
    data = data_plot[:num_disp]
    img_names = data['fname']
    car_names = data['class_name']
    x1 = data['bbox_x1']
    y1 = data['bbox_y1']
    x2 = data['bbox_x2']
    y2 = data['bbox_y2']
    
    i = 1
    for iname, cname, x1, y1, x2, y2 in zip(img_names, car_names, x1, y1, x2, y2):
        fig.add_subplot(rows,cols,i)
        img = load_image(iname, img_root)
        bbox = (x1, y1, x2, y2)
        plot_image(img, cname, bbox)
        i += 1
        
    plt.show()

# define the path of image and label for training and testing dataset
img_train_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_train\cars_train'
label_train_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_train\label_train'

img_test_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_test\cars_test'
label_test_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_test\label_test'

# create dataframe of training and testing data as well as plot the car image together with bounding boxes and their label
data_train = annotation_pipeline(r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_train_annos.mat')
plot_grid(data_train, 25, img_train_path)

data_test = annotation_pipeline(r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_test_annos_withlabels.mat')
plot_grid(data_test, 25, img_test_path)

# YOLO from ultralytics require annotation in the txt file
# function below create text file based on the annotation dataset
def create_txt_files(dataframe, image_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename, group in dataframe.groupby('fname'):
        image_path = os.path.join(image_dir, filename)

        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size

        except FileNotFoundError:
            print(f"Warning: Image not found: {image_path}")
            continue 

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)

        with open(txt_filepath, 'w') as f:
            for _, row in group.iterrows():
                class_id = 0 
                x_min = row['bbox_x1']
                y_min = row['bbox_y1']
                x_max = row['bbox_x2']
                y_max = row['bbox_y2']
                
                # normalization to 0 to 1
                x_center = ((x_min + x_max) / 2) / width
                y_center = ((y_min + y_max) / 2) / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height

                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

create_txt_files(data_train, img_train_path, label_train_path)
create_txt_files(data_test, img_test_path, label_test_path)

# move the folder of files to appropriate directory
def move_folder(source_fold, dest_fold):

    os.makedirs(dest_fold, exist_ok=True) 
    dest_path = os.path.join(dest_fold, os.path.basename(source_fold))

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path) 

    shutil.move(source_fold, dest_path)

# move folder to car_data
move_folder(img_train_path, r'C:\Users\user\car_data\image')
move_folder(label_train_path, r'C:\Users\user\car_data\label')
move_folder(img_test_path, r'C:\Users\user\car_data\image')
move_folder(label_test_path, r'C:\Users\user\car_data\label')

# function below is to move txt annotation data to where image is being stored.
def move_txt_to_image_dir(label_dir, image_dir):

    for filename in os.listdir(label_dir):
        if filename.lower().endswith(".txt"):
            source_path = os.path.join(label_dir, filename)
            destination_path = os.path.join(image_dir, filename)
            shutil.move(source_path, destination_path)

move_txt_to_image_dir(r'C:\Users\user\car_data\label\label_train', r'C:\Users\user\car_data\image\cars_train')
move_txt_to_image_dir(r'C:\Users\user\car_data\label\label_test', r'C:\Users\user\car_data\image\cars_test')

# create yaml file, required by ultralytics YOLO
yaml_content = f"""
    train: car_data/image/cars_train
    val: car_data/image/cars_test

    nc: 1

    names: ['car']
    """

with open('cars_data.yaml', "w") as f:
    f.write(yaml_content)

# model initilization, pretrained model YOLO11
model = YOLO("yolo11n.pt")  

# train the model, transfer learning for 3 hour, my laptop can't do more. 
# resizing image will be handled automatically by ultralytics
results = model.train(data="cars_data.yaml", epochs=100, imgsz=640, time = 3, save = True)

# called model that have been trained
model = YOLO(r"runs\detect\train5\weights\best.pt")

#evaluate the model
metrics = model.val(data="cars_data.yaml")

# predict the testing data
results = model(r'C:\Users\user\car_data\image\cars_test')

# evaluate the model using interception of union
ious = []
for i in range(len(data_test)):
    ground_truth_bbox = torch.tensor([data_test[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].iloc[i].to_list()], dtype=torch.float32)

    if results[i] is None or len(results[i].boxes.xyxy) == 0: 
        ious.append(0.0)
    else:
        predicted_bboxes = results[i].boxes.xyxy
        iou_values = bbox_iou(ground_truth_bbox, predicted_bboxes)

        if len(iou_values) > 0:
            max_iou = torch.max(iou_values).item() 
            ious.append(max_iou)
        else:
            ious.append(0.0)

iou_mean = np.mean(ious)

print(metrics.results_dict)

dict_metrics = metrics.results_dict
dict_metrics['IOU'] = iou_mean

# model initilization pretrained car classifier
# no need to train this model, this is optional task
car_classifier = pipeline("image-classification", model="SriramSridhar78/sriram-car-classifier")

# evaluate the classifier
img_path = r'C:\Users\user\car_data\image\cars_test'

predicted_class_name = []
for i in data_test['fname'].to_list():
    image_path = img_path + '\\' + i
    print(image_path)
    image = Image.open(image_path)
    result = car_classifier(image)[0]['label'].replace('_', ' ')
    predicted_class_name.append(result)


class_name = list(data_test['class_name'].copy())
dict = {'class_name': class_name, 'predicted_class_name': predicted_class_name} 
class_data = pd.DataFrame(dict)

# use similarity score, we have ground truth label and predicted label, while accuracy is not suitable in this task due to its data type which is string
# similarity score is the best metrics to compared between two string
similarity = class_data.apply(lambda row: jellyfish.jaro_winkler_similarity(str(row['class_name']), str(row['predicted_class_name'])), axis=1)
similarity = np.mean(similarity)

dict_metrics['Similarity Score'] = similarity

# evaluation metrics plotting
labels = list(dict_metrics.keys())
values = list(dict_metrics.values())
plt.figure(figsize=(10, 6))

colors = ['blue'] * len(labels)
similarity_index = labels.index('Similarity Score')
colors[similarity_index] = 'red'
bars = plt.bar(labels, values, color=colors) 

plt.ylabel('Value')
plt.title('Cars Detector & Classifier Performance Metrics')
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom')

plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='blue'), plt.Rectangle((0, 0), 1, 1, color='red')],
           labels=['Detector Metrics', 'Classifier Metrics'])

plt.show()