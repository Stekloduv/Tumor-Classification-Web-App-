import tensorflow as tf
import numpy as np


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["1", "2", "3"]  # Відповідно до папок, у яких зберігалися дані під час тренування


def load_and_preprocess(path: str, target_size=(224, 224)):  # змінено розмір на 224x224
    """Завантажує та нормалізує зображення під потрібний розмір."""
    image = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Додаємо batch dimension
    img_array /= 255.0  # Масштабуємо значення пікселів до [0,1]
    return img_array


def classify(model, image_path: str):
    """Класифікує зображення, підлаштовуючи вхідні дані під архітектуру моделі."""
    preprocessed_image = load_and_preprocess(image_path)

    # Друкуємо архітектуру моделі, якщо потрібно
    model.summary()

    # Отримуємо очікуваний розмір вхідних даних
    input_shape = model.input_shape
    print(f"Model expects input shape: {input_shape}")

    # Перевіряємо, чи потрібно "сплющити" вхідні дані
    if len(input_shape) == 2:  # Очікується вектор (1, X)
        preprocessed_image = preprocessed_image.reshape(1, -1)
        print(f"Reshaped image to: {preprocessed_image.shape}")

    predictions = model.predict(preprocessed_image)
    score = np.max(predictions)  # Найбільша ймовірність серед класів
    label = CLASS_NAMES[np.argmax(predictions)]

    return label, score
