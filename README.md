# Car Detection and Classification App

This Streamlit application detects cars in uploaded images and classifies them using YOLO11 for object detection and a Hugging Face Transformers model for car classification.

## Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/dimashidayat99/car_detection_system.git
    cd <your_repository_directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    -   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Required Packages:**
    ```bash
    pip install streamlit pillow ultralytics transformers opencv-python regex
    ```

4.  **Download the YOLOv8 Model:**
    -   Ensure the YOLOv8 model file (`best.pt`) is located in the `runs/detect/train5/weights/` directory. If you have a different path, make sure to change the path inside the streamlit app code.
    -   If you need to train your own model, follow the ultralytics documentation.

## Deployment on Streamlit Cloud

1.  **Prepare your Repository:**
    -   Ensure your repository contains:
        -   `app.py` (your Streamlit application script).
        -   `requirements.txt` (a list of your Python dependencies).
        -   The trained YOLOv8 model file (`runs/detect/train5/weights/best.pt` or your custom path).
        -   Any other necessary files.

2.  **Create `requirements.txt`:**
    -   Create a file named `requirements.txt` in the root of your repository.
    -   Add the following dependencies to the `requirements.txt` file:
        ```
        streamlit
        pillow
        ultralytics
        transformers
        opencv-python
        regex
        ```
    -   If you have specific versions, add them like this: `streamlit==1.20.0`

3.  **Streamlit Cloud Setup:**
    -   Go to [Streamlit Cloud](https://streamlit.io/cloud).
    -   Sign up or log in with your GitHub account.
    -   Click "New app".
    -   Connect your GitHub repository to Streamlit Cloud.
    -   Specify the following:
        -   **Repository:** Your GitHub repository.
        -   **Branch:** The branch containing your app (usually `main` or `master`).
        -   **Main file path:** `app.py`.
        -   Click "Deploy!".

4.  **Handling Model Files:**
    -   Streamlit Cloud has limitations on file sizes. If your YOLOv8 model (`best.pt`) is large, consider these options:
        -   **Git LFS (Large File Storage):** If your model exceeds 100MB, use Git LFS to manage large files in your repository.
            ```bash
            git lfs install
            git lfs track "runs/detect/train5/weights/best.pt"
            git add .gitattributes
            git add runs/detect/train5/weights/best.pt
            git commit -m "Add model using Git LFS"
            git push
            ```
        -   **External Hosting (Not Recommended but Possible):** You can host the model file on an external service (e.g., Google Drive, AWS S3) and download it in your Streamlit app using a URL. However, this is not recommended for streamlit cloud as it will download the model every time the app starts, causing long load times.
        -   **Reduce Model Size (Recommended):** If possible, try to reduce the model size by quantization or pruning. Or retrain the model with smaller input image sizes.
    -   Make sure that the path to the model in the app.py code is correct.

5.  **Troubleshooting:**
    -   Check the Streamlit Cloud logs for any errors.
    -   Ensure all dependencies are correctly listed in `requirements.txt`.
    -   Verify that the model file is accessible by your application.
    -   Make sure the file paths are correct.

## Usage

1.  Open the deployed Streamlit app in your browser.
2.  Click "Browse files" to upload one or more image files.
3.  The app will display the images with bounding boxes around detected cars and classify each car.
4.  The manufacturer and model of each classified car will be shown below the cropped car image.

https://cardetectionsystem-assessment.streamlit.app/
