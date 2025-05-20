import os
import cv2
import timm
import torch

from flask import Flask, request, render_template, url_for

import torchvision.transforms as transforms
from removal import process_and_save
from preprocess_ak1n import process_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

MODEL_PATH = "regnet94valacc.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('regnety_080', pretrained=True)
model.reset_classifier(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
class_names = {0: "First Degree Burn", 1: "Second Degree Burn", 2: "Third Degree Burn"}

# Burn info dictionary
burn_info = {
    "First Degree Burn": {
        "title": "First-Degree Burn",
        "causes": "Mild sunburn, brief contact with something hot.",
        "symptoms": "Red, dry skin. Mild swelling. Pain or tenderness.",
        "treatment": (
            "Cool the burn under cool (not cold) running water for 10–20 minutes. "
            "Take off jewelry or tight items before swelling starts. "
            "Cover with a clean, non-stick bandage or cloth. "
            "Do not use ice, butter, or ointments. Do not pop blisters. "
            "Usually heals in about a week. "
            "See a doctor if you are unsure or if the burn is large."
        ),
        "when_to_seek_help": (
            "See a doctor if the burn is bigger than 3 inches (8 cm), on the face, hands, feet, groin, or joints, "
            "or if there are signs of infection (more pain, redness, swelling, or oozing)."
        )
    },
    "Second Degree Burn": {
        "title": "Second-Degree Burn",
        "causes": "Scalds from hot liquids, severe sunburn, contact with flames, or chemicals.",
        "symptoms": "Blisters. Deep redness. Swelling. Wet, shiny skin. Severe pain.",
        "treatment": (
            "Cool the burn under cool running water for 10–20 minutes. "
            "Cover with a clean, non-stick bandage. "
            "Do not break blisters. "
            "Get medical help if the burn is large or on sensitive areas."
        ),
        "when_to_seek_help": (
            "Get medical help if the burn is bigger than 3 inches (8 cm), on the face, hands, feet, groin, or joints, "
            "or if there are signs of infection."
        )
    },
    "Third Degree Burn": {
        "title": "Third-Degree Burn",
        "causes": "Prolonged exposure to flames, electricity, strong chemicals.",
        "symptoms": "White, charred, or leathery skin. May feel numb. Swelling.",
        "treatment": (
            "Call emergency services right away. "
            "Do not self-treat. "
            "Cover with a clean cloth and keep the person warm."
        ),
        "when_to_seek_help": (
            "Always call emergency services for third-degree burns."
        )
    }
}
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file part", 400

        files = request.files.getlist('image')
        if len(files) == 0:
            return "No files selected", 400

        burn_results = []

        for file in files:
            if file.filename == '':
                continue

            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            removed_path = os.path.join(app.config['RESULT_FOLDER'], 'removed_' + filename)
            process_and_save(upload_path, removed_path)

            preprocessed_image = process_image(removed_path)
            if preprocessed_image is None:
                continue

            if preprocessed_image.shape[2] == 3:
                preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
            elif preprocessed_image.shape[2] == 4:
                preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGRA2RGB)

            input_tensor = transform(preprocessed_image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                burn_degree = class_names.get(predicted.item(), "Unknown")

            result = {
                "filename": filename,
                "burn_degree": burn_degree,
                "causes": burn_info[burn_degree]["causes"],
                "symptoms": burn_info[burn_degree]["symptoms"],
                "treatment": burn_info[burn_degree]["treatment"],
                "when_to_seek_help": burn_info[burn_degree]["when_to_seek_help"],
                "image_url": url_for('static', filename='uploads/' + filename)
            }

            burn_results.append(result)

        return render_template('result.html', burn_results=burn_results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)