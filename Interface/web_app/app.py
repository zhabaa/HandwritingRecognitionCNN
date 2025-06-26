from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from torchvision import transforms
import base64
import io
from PIL import Image

import torch
# Замените относительный импорт на абсолютный
from modelNN.model import DigitRecognizer

app = Flask(__name__)

# # Загружаем модель
model = DigitRecognizer()
# model.load_state_dict(torch.load('../model/best_model_weights.pth', map_location='cpu'))

import os
from pathlib import Path

# Получаем абсолютный путь к файлу с весами
MODEL_PATH = Path(__file__).parent.parent / "modelNN" / "best_model_weights.pth"
MODEL_PATH = "modelNN/weights/best_model_weights.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

model.eval()

# Преобразование изображения
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'No image data provided'}), 400

        # Получаем изображение
        data = request.json['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(data))).convert('L')
        
        # Подготовка изображения
        image = image.resize((28, 28))
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Трансформация и предсказание
        with torch.no_grad():
            tensor_image = transform(image_array).unsqueeze(0).float() # type: ignore
            output = model(tensor_image)
            probs = torch.softmax(output, dim=1).numpy()[0]
        
        return jsonify({
            'predicted': int(np.argmax(probs)),
            'probabilities': [float(p) for p in probs]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)