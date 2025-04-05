import os
import cv2
import torch
import numpy as np
import torch

from flask import Flask, request, render_template, url_for
import torchvision.transforms as transforms
import k_fold_cnn as k_fold
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from removal import process_and_save
from preprocess_ak1n import process_image
import allmodels as modelss
app = Flask(__name__)

# Set up folders for uploads and results
UPLOAD_FOLDER = 'WebApp/static/uploads'
RESULT_FOLDER = 'WebApp/static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Configure static folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the burn classification model (assumes the model file is in the same directory)
MODEL_PATH = r'C:\Users\atess\OneDrive\Masaüstü\best_modelRes1ilkepoch iyiverdi.pth'
model = modelss.BurnResNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cuda')))
model.eval()

# Define a mapping for burn degree predictions (adjust as needed)
class_names = {0: "First Degree Burn", 1: "Second Degree Burn", 2: "Third Degree Burn"}

# Define the image transform: convert numpy array to tensor, resize, and normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Change to the expected input size of your CNN
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file part", 400
        file = request.files['image']
        if file.filename == '':
            return "No file selected", 400

        # Save the uploaded file
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_path)

        # ----- Step 1: Background Removal -----
        # Remove the background using removal.py.
        # The process_and_save function takes the input file path and writes the result.
        removed_path = os.path.join(RESULT_FOLDER, 'removed_' + file.filename)
        process_and_save(upload_path, removed_path)

        # ----- Step 2: Preprocess with preprocess_ak1n.py -----
        # This function processes the image and returns a processed image.
        preprocessed_image = process_image(removed_path)
        if preprocessed_image is None:
            return "Error during preprocessing", 500

        # The output from preprocess_ak1n.py (using cv2.imread) is in BGR order.
        # Convert to RGB for the model if needed.
        if preprocessed_image.shape[2] == 3:
            preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
        elif preprocessed_image.shape[2] == 4:
            preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGRA2RGB)



        # ----- Step 3: Burn Classification -----
        # Transform the image and add a batch dimension
        input_tensor = transform(preprocessed_image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            burn_degree = class_names.get(predicted.item(), "Unknown")

        # Optionally, save the preprocessed image for display
        result_filename = 'final_' + file.filename
        result_image_path = os.path.join(RESULT_FOLDER, result_filename)
        # Convert back to BGR for saving with OpenCV
        cv2.imwrite(result_image_path, cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))

        # Generate a URL that Flask can serve
        result_image_url = url_for('static', filename=f'results/{result_filename}')

        return render_template('result.html', burn_degree=burn_degree, result_image=result_image_url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)