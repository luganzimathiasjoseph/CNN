from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/": {"origins": "*"}})  # Allow all origins

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🛠 CNN Model Definition
class TyreClassifier(nn.Module):
    def __init__(self):
        super(TyreClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected layers
        self.fc1 = nn.Linear(256 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)  # Output: 2 classes (Good & Defective)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No softmax (CrossEntropyLoss applies it)

        return x

# Initialize the model
model = TyreClassifier().to(device)

# Get script directory & load model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'defective_tire_detection_cnn.pth')

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-like input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        print(f"📸 Received file: {file.filename}")  # Debugging log

        # Convert file to numpy array & OpenCV format
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Convert BGR (OpenCV format) to RGB (PIL format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Apply image transformations
        img = transform(pil_img)
        img = img.unsqueeze(0).to(device)  # Add batch dimension & move to device

        # Run prediction
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)  # Get predicted class

            # Map prediction to class labels
            result = "Defective" if predicted.item() == 1 else "Good"

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
