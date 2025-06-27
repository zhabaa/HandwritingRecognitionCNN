import io
import base64

import numpy as np
from PIL import Image
from flask import Flask, render_template, jsonify, request

from modelNN.model import DigitPredictor
from config import WEIGHT_PATH

app = Flask(__name__)

predictor = DigitPredictor(WEIGHT_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.json or "image" not in request.json:
            return jsonify(
                {"error": "No image data provided"}), 400

        data = request.json["image"].split(",")[1]

        image = Image.open(io.BytesIO(base64.b64decode(data))).convert("L")
        image = image.resize((28, 28))

        image_array = np.array(image, dtype=np.float32) / 255.0

        predictor = DigitPredictor()
        probs, predicted_idx = predictor.predict(image_array)

        return jsonify(
            {
                "predicted": int(np.argmax(probs)),
                "probabilities": [float(p) for p in probs],
            }
        )

    except Exception as e:
        return jsonify(
            {"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
