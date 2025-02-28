from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/": {"origins": "*"}})  # Allow all origins

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_pytorch_model():
    # Define the ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Defective & Good

    # Move model to device
    model = model.to(device)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'tire_defect_model.pth')

    print(f"Looking for model at: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} was not found")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

# Load the model once at startup
model = load_pytorch_model()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing image to match ResNet18 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for ResNet18
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Update this route to handle both GET and POST
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Handle the GET request (for displaying the form or info)
        return render_template('predict_form.html')

    if request.method == 'POST':
        file = request.files.get('file')  # Use get() to avoid KeyError
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        print(f"Received file: {file.filename}")  # Log file name for debugging

        # Read the image as a numpy array
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert NumPy array (OpenCV image) to PIL Image
        pil_img = Image.fromarray(img)

        # Apply the transformation
        img = transform(pil_img)
        img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Make the prediction without tracking gradients
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)  # Get the predicted class

            # Map the prediction to class labels (assuming 0 is 'Good' and 1 is 'Defective')
            result = "Defective" if predicted.item() == 1 else "Good"

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
