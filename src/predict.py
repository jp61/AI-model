from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 150
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# načítanie modelu
model = keras.models.load_model(os.path.join(SCRIPT_DIR, "cats_dogs_model.keras"))

# zisti všetky obrázky v priečinku
img_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in img_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = load_and_preprocess(img_path)
    pred = model.predict(img)
    label = "Dog" if pred[0][0] > 0.5 else "Cat"
    print(f"{img_name}: {label}")

    # --- Zobrazenie obrázka s predikciou ---
    img_display = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    plt.imshow(img_display)
    plt.title(f"{img_name}: {label}")
    plt.axis('off')
    plt.show()