"""Command-line cat-vs-dog prediction.

Usage:
    python src/predict.py <image-or-directory> [--tta]

If no argument is given, runs over src/images/.
--tta enables 5-view test-time augmentation (original + hflip + 4 corner crops),
averaging sigmoids. Reduces variance on borderline cases (Krizhevsky et al. 2012).
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cats_dogs_model.keras")
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
IMG_SIZE = 150
CROP_SIZE = int(IMG_SIZE * 0.875)  # standard 87.5% center-crop ratio
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


def tta_views(tensor):
    """Return 5 IMG_SIZE×IMG_SIZE views of a single (H, W, 3) tensor.

    Views: original + horizontal flip + 4 corner crops (upscaled to IMG_SIZE).
    """
    H = tensor.shape[0]
    assert H == IMG_SIZE, "expected tensor at IMG_SIZE for crop arithmetic"
    d = H - CROP_SIZE  # pixels outside each corner crop

    views = [tensor, np.ascontiguousarray(tensor[:, ::-1, :])]
    for (y, x) in [(0, 0), (0, d), (d, 0), (d, d)]:
        crop = tensor[y:y + CROP_SIZE, x:x + CROP_SIZE]
        views.append(
            tf.image.resize(crop, [IMG_SIZE, IMG_SIZE], method='bilinear').numpy()
        )
    return np.stack(views, axis=0).astype(np.float32)


def predict_one(model, tensor, use_tta):
    if not use_tta:
        return float(model.predict(np.expand_dims(tensor, 0), verbose=0)[0][0])
    batch = tta_views(tensor)
    raws = model.predict(batch, verbose=0).reshape(-1)
    return float(raws.mean())


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", nargs="?", default=DEFAULT_IMAGES_DIR)
    p.add_argument("--tta", action="store_true", help="Enable test-time augmentation.")
    args = p.parse_args()

    paths = collect_paths(args.target)
    if not paths:
        raise SystemExit(f"No images found in {args.target}")

    model = keras.models.load_model(MODEL_PATH)

    for pth in paths:
        _, tensor = load_and_preprocess(pth)
        raw = predict_one(model, tensor, args.tta)
        label = "Dog" if raw > 0.5 else "Cat"
        conf = raw if raw > 0.5 else 1 - raw
        tag = " [TTA]" if args.tta else ""
        print(
            f"{os.path.basename(pth):30s}  {label:4s}  raw={raw:.4f}  conf={conf*100:.1f}%{tag}  "
            f"tensor[min={tensor.min():.3f} max={tensor.max():.3f} mean={tensor.mean():.3f}]"
        )


if __name__ == "__main__":
    main()
