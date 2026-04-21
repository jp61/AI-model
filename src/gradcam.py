"""Grad-CAM diagnostic.

Visualizes where the last Conv2D layer looks when predicting cat vs dog.
Selvaraju et al. 2017, "Grad-CAM: Visual Explanations from Deep Networks."

Run:
    python src/gradcam.py                  # uses src/images/
    python src/gradcam.py path/to/img.jpg  # single image
"""

import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(REPO_ROOT, "model", "cats_dogs_model.keras")
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "gradcam")
IMG_SIZE = 150
EXTS = (".jpg", ".jpeg", ".png")


def last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
    raise RuntimeError("No Conv2D layer found")


def preprocess(path):
    img = keras.preprocessing.image.load_img(path)
    arr = keras.preprocessing.image.img_to_array(img)
    arr = tf.image.resize(arr, [IMG_SIZE, IMG_SIZE], method="bilinear").numpy()
    return arr / 255.0


def gradcam(model, img, conv_layer):
    conv_index = model.layers.index(conv_layer)
    pre = model.layers[: conv_index + 1]
    post = model.layers[conv_index + 1 :]

    x = tf.convert_to_tensor(np.expand_dims(img, 0).astype(np.float32))
    with tf.GradientTape() as tape:
        h = x
        for l in pre:
            h = l(h, training=False)
        conv_out = h
        tape.watch(conv_out)
        h = conv_out
        for l in post:
            h = l(h, training=False)
        preds = h
        score = preds[:, 0] if preds[0, 0] > 0.5 else 1.0 - preds[:, 0]
    grads = tape.gradient(score, conv_out)[0]            # (H, W, C)
    conv = conv_out[0]                                   # (H, W, C)
    weights = tf.reduce_mean(grads, axis=(0, 1))         # (C,)
    cam = tf.reduce_sum(conv * weights, axis=-1)         # (H, W)
    cam = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
    ) / 255.0
    return cam, float(preds[0, 0])


def colorize(cam):
    # Simple jet-like colormap without matplotlib: r = cam, g = fade, b = 1-cam.
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(img, cam, alpha=0.45):
    heat = colorize(cam)
    return np.clip(img * (1 - alpha) + heat * alpha, 0, 1)


def save_side_by_side(orig, overlayed, label, raw, conf, out_path):
    combined = np.concatenate([orig, overlayed], axis=1)
    combined = (combined * 255).astype(np.uint8)
    im = Image.fromarray(combined)
    im.save(out_path)
    print(f"  saved {out_path}  label={label}  raw={raw:.4f}  conf={conf*100:.1f}%")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGES_DIR
    if os.path.isdir(target):
        paths = sorted(
            os.path.join(target, f) for f in os.listdir(target)
            if f.lower().endswith(EXTS)
        )
    else:
        paths = [target]

    os.makedirs(OUT_DIR, exist_ok=True)
    model = keras.models.load_model(MODEL_PATH)
    conv_layer = last_conv_layer(model)
    print(f"Using last conv layer: {conv_layer.name}  output={conv_layer.output.shape}")

    for p in paths:
        img = preprocess(p)
        cam, raw = gradcam(model, img, conv_layer)
        label = "Dog" if raw > 0.5 else "Cat"
        conf = raw if raw > 0.5 else 1 - raw
        out_name = os.path.splitext(os.path.basename(p))[0] + "_gradcam.png"
        save_side_by_side(img, overlay(img, cam), label, raw, conf,
                          os.path.join(OUT_DIR, out_name))


if __name__ == "__main__":
    main()
