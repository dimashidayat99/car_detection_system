import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import os
import kagglehub
from ultralytics import YOLO
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import matplotlib.patches as mpatches
import torchvision.transforms as T

#path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
#print("Path to dataset files:", path)

annos = sio.loadmat(r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_annos.mat')
type(annos)
annos.keys()
annotation = annos["annotations"]
class_def = annos["class_names"]
type(annotation)

annotation[0][0]

annotation_list = []

for i in annotation[0]:
    annotation_sub_list = []
    for j in i:
        annotation_sub_list.append(j.flatten()[0])
    annotation_list.append(annotation_sub_list)
    
data = pd.DataFrame(annotation_list, columns=annotation.dtype.names)
data.head(5)

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

label_dict = {i+1: name[0] for i, name in enumerate(class_def[0])}
data['class_name'] = data['class'].map(label_dict)
try:
    plot_grid(data['fname'][:25], data['class_name'][:25], IMG_ROOT)
except:
    pass

# problem with the annotation file, car name not accurate finding the source of truth of data.
# obtained new annotation files in https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input

#implementation code start from here

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

img_train_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_train\cars_train'
label_train_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_train\label_train'

img_test_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_test\cars_test'
label_test_path = r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_test\label_test'

data_train = annotation_pipeline(r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_train_annos.mat')
plot_grid(data_train, 25, img_train_path)

data_test = annotation_pipeline(r'C:\Users\user\.cache\kagglehub\datasets\jessicali9530\stanford-cars-dataset\versions\2\cars_test_annos_withlabels.mat')
plot_grid(data_test, 25, img_test_path)

# initiate the dataset preparetion
class CarDataset(Dataset):
    def __init__(self, dataframe, image_folder, transforms=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transforms = transforms
        self.image_ids = self.dataframe['fname'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_folder, image_id)
        image = Image.open(image_path).convert("RGB")

        records = self.dataframe[self.dataframe['fname'] == image_id]
        boxes = records[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((records.shape[0],), dtype=torch.int64) # All cars, so label is 1.

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((records.shape[0],), dtype=torch.int64)
        }

        if self.transforms is not None:
            image = self.transforms(image)
            target['boxes'] = target['boxes']
            target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])

        return T.ToTensor()(image), target


# import pretrain faster RCNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# set number of class & change the last layer
num_classes = 2 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# initialize the model parameters and optimizaer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

dataset = CarDataset(data_train, img_train_path)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

num_epochs = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# train the model
# however, this require high computation power where my laptops can't handle so i proceed with YOLO model only
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}')

# Save the trained model
torch.save(model.state_dict(), 'fasterrcnn_cars.pth')

