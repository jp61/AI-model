"""Command-line cat-vs-dog prediction.

Usage:
    python src/predict.py <image-or-directory>

If no argument is given, runs over src/images/.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cats_dogs_model.keras")
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
IMG_SIZE = 150
EXTS = (".jpg", ".jpeg", ".png")


def load_and_preprocess(img_path):
    # Match training preprocessing exactly: decode full-res, then bilinear resize, /255.
    # Using load_img(target_size=...) would apply PIL's default nearest-neighbor, which
    # diverges from training (tf.image.resize -> bilinear) and the web demo (resizeBilinear).
    img = image.load_img(img_path)
    arr = image.img_to_array(img)
    arr = tf.image.resize(arr, [IMG_SIZE, IMG_SIZE], method='bilinear').numpy()
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0), arr


def collect_paths(arg):
    if os.path.isdir(arg):
        return sorted(
            os.path.join(arg, f)
            for f in os.listdir(arg)
            if f.lower().endswith(EXTS)
        )
    if os.path.isfile(arg):
        return [arg]
    raise SystemExit(f"Not a file or directory: {arg}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGES_DIR
    paths = collect_paths(target)
    if not paths:
        raise SystemExit(f"No images found in {target}")

    model = keras.models.load_model(MODEL_PATH)

    for p in paths:
        batch, tensor = load_and_preprocess(p)
        raw = float(model.predict(batch, verbose=0)[0][0])
        label = "Dog" if raw > 0.5 else "Cat"
        conf = raw if raw > 0.5 else 1 - raw
        print(
            f"{os.path.basename(p):30s}  {label:4s}  raw={raw:.4f}  conf={conf*100:.1f}%  "
            f"tensor[min={tensor.min():.3f} max={tensor.max():.3f} mean={tensor.mean():.3f}]"
        )


if __name__ == "__main__":
    main()
