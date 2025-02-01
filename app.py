import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from classifier import classify

app = Flask(__name__)

STATIC_FOLDER = 'static'
UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Завантаження моделі
cnn_model = tf.keras.models.load_model(os.path.join(STATIC_FOLDER, "models", "model.h5"))

@app.route('/')
def home():
    return render_template('index.html')

@app.post("/classify")
@app.route('/classify', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Файл не знайдено"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "Файл не обрано"}), 400

        # Збереження файлу
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        os.makedirs(os.path.dirname(upload_image_path), exist_ok=True)
        file.save(upload_image_path)

        # Виклик функції класифікації
        label, score = classify(cnn_model, upload_image_path)

        return jsonify({
            "label": label,
            "score": float(round(score * 100, 2))  # Серіалізація у float
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
