import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
from transformers import pipeline
import regex
import re
import cv2

# import the car detection and car classifier model
model_path = "runs/detect/train5/weights/best.pt"
model = YOLO(model_path)

car_classifier = pipeline("image-classification", model="SriramSridhar78/sriram-car-classifier")

# set title
st.title("Car Detection App")

# set file uploader that accept multiple image
uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # set container for each image
        with st.container(): 
            # open and detect car in the image
            image = Image.open(uploaded_file)
            results_detection = model(image)
            annotated_image = results_detection[0].plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # crop the image of detected car, sometimes, multiple cars can be detected in one image
            cropped_images = []
            for box in results_detection[0].boxes.xyxy.numpy().astype(int):
                x1, y1, x2, y2 = box
                cropped_car = image.crop((x1, y1, x2, y2))
                cropped_images.append(cropped_car)

            # display the cropped image
            if cropped_images:
                st.subheader(f"Identified Cars in file: {uploaded_file.name}:")
                st.image(annotated_image, caption=f'Cars Detected with Bounding Boxes in file: {uploaded_file.name}')

                # classifier all the car within the image
                for i, cropped_car in enumerate(cropped_images):
                    car_classifier_results = car_classifier(cropped_car)
                    manufacturer = re.search(r'[^_]*', car_classifier_results[0]['label']).group(0)
                    model_name = re.search(r'[^_]+_(.*)', car_classifier_results[0]['label']).group(1).replace('_', ' ')
                    st.image(cropped_car, caption=f"Manufacturer: {manufacturer}, Model: {model_name}", use_column_width=True)

            else:
                st.write(f"No Cars Detected in file: {uploaded_file.name}.")
