"""Convert trained Keras model to TensorFlow.js Layers format.

Produces model.json + weight shard .bin files that tf.loadLayersModel() can load.
Does NOT depend on the tensorflowjs package (which has broken transitive deps).

Also writes calibration.json with deterministic reference predictions so the web
demo can verify weights loaded correctly.
"""

import json
import os
import sys

import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(REPO_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cats_dogs_model.keras")
# TF.js export lives inside src/web_demo/ so the web page can load it with a
# short relative path and the server can be started from that directory.
OUT_DIR = os.path.join(SCRIPT_DIR, "web_demo", "model")
SHARD_SIZE = 4 * 1024 * 1024  # 4 MB per shard
IMG_SIZE = 150

# Layers that carry trainable weights with canonical TF.js variable names.
# Extend this when you add layer types that have weights.
EXPECTED_WEIGHTS = {
    "Conv2D": ["kernel", "bias"],
    "Dense": ["kernel", "bias"],
    "BatchNormalization": ["gamma", "beta", "moving_mean", "moving_variance"],
}


def canonical_var_name(weight):
    """Strip ':0' suffix and any layer-scope prefix from a weight's name."""
    return weight.name.split(":")[0].split("/")[-1]


def build_tfjs_topology(model):
    layers_json = []
    for layer in model.layers:
        cfg = layer.get_config()
        ltype = type(layer).__name__

        if isinstance(cfg.get("dtype"), dict):
            cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")

        for key in list(cfg.keys()):
            val = cfg[key]
            if isinstance(val, dict) and "module" in val:
                cfg[key] = {
                    "class_name": val["class_name"],
                    "config": val.get("config", {}),
                }

        cfg.pop("quantization_config", None)

        if ltype == "InputLayer":
            continue

        if len(layers_json) == 0:
            cfg["batch_input_shape"] = list(model.input_shape)

        layers_json.append({"class_name": ltype, "config": cfg})

    return {
        "class_name": "Sequential",
        "config": {"name": model.name, "layers": layers_json},
        "keras_version": "2.15.0",
        "backend": "tensorflow",
    }


def serialize_weights(model, out_dir, shard_size):
    entries = []
    raw_bytes = bytearray()

    for layer in model.layers:
        for w in layer.weights:
            arr = w.numpy().astype(np.float32)
            name = f"{layer.name}/{canonical_var_name(w)}"
            entries.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})
            raw_bytes.extend(arr.tobytes())

    total = len(raw_bytes)
    num_shards = max(1, (total + shard_size - 1) // shard_size)
    paths = []

    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, total)
        fname = f"group1-shard{i + 1}of{num_shards}.bin"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(raw_bytes[start:end])
        paths.append(fname)

    return [{"paths": paths, "weights": entries}]


def validate_manifest(model, manifest):
    """Verify every expected weight name is present and shapes are non-trivial.

    TF.js silently falls back to a random initializer when a layer can't find
    its weight by name, which produces a model that loads cleanly but predicts
    garbage. Catching naming mismatches here prevents that whole class of bug.
    """
    written = {e["name"]: tuple(e["shape"]) for e in manifest[0]["weights"]}
    missing = []
    empty = []

    for layer in model.layers:
        ltype = type(layer).__name__
        expected = EXPECTED_WEIGHTS.get(ltype)
        if expected is None:
            continue
        if not layer.weights:
            continue
        for var_name in expected:
            # A Conv2D/Dense with use_bias=False legitimately has no bias weight.
            if var_name == "bias" and not getattr(layer, "use_bias", True):
                continue
            full = f"{layer.name}/{var_name}"
            if full not in written:
                missing.append(full)
            elif not written[full] or any(d == 0 for d in written[full]):
                empty.append((full, written[full]))

    if missing or empty:
        msg = ["Weight manifest validation failed:"]
        if missing:
            msg.append(f"  Missing: {missing}")
            msg.append(f"  Written: {sorted(written.keys())}")
        if empty:
            msg.append(f"  Empty shapes: {empty}")
        raise SystemExit("\n".join(msg))


def write_calibration(model, out_dir):
    """Predict on deterministic inputs so the browser can verify weight load.

    If TF.js loaded the same weights correctly, running the same fixed tensors
    through the model must produce the same raw sigmoid (within float tolerance).
    """
    zeros = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    ones = np.ones((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    half = np.full((1, IMG_SIZE, IMG_SIZE, 3), 0.5, dtype=np.float32)

    cases = []
    for name, tensor in [("zeros", zeros), ("ones", ones), ("half", half)]:
        raw = float(model.predict(tensor, verbose=0)[0][0])
        cases.append({"name": name, "fill": float(tensor.flat[0]), "expected": raw})

    raw_values = [c["expected"] for c in cases]
    if max(raw_values) - min(raw_values) < 1e-6:
        print("  WARNING: calibration values are identical across inputs — "
              "model may be degenerate.", file=sys.stderr)

    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({"img_size": IMG_SIZE, "tolerance": 1e-3, "cases": cases}, f, indent=2)

    return cases


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun 'python src/train.py' first.")

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)

    topology = build_tfjs_topology(model)
    manifest = serialize_weights(model, OUT_DIR, SHARD_SIZE)
    validate_manifest(model, manifest)
    calibration = write_calibration(model, OUT_DIR)

    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.keras.__version__}",
        "convertedBy": "convert_to_tfjs.py",
        "modelTopology": topology,
        "weightsManifest": manifest,
    }

    with open(os.path.join(OUT_DIR, "model.json"), "w") as f:
        json.dump(model_json, f)

    total_bytes = sum(os.path.getsize(os.path.join(OUT_DIR, p)) for p in manifest[0]["paths"])
    print(f"\nConversion complete:")
    print(f"  Output:      {OUT_DIR}/")
    print(f"  Shards:      {len(manifest[0]['paths'])}")
    print(f"  Weights:     {len(manifest[0]['weights'])} tensors, {total_bytes / 1e6:.1f} MB")
    print(f"  Layers:      {len(topology['config']['layers'])}")
    cal_summary = ", ".join(f"{c['name']}={c['expected']:.4f}" for c in calibration)
    print(f"  Calibration: {cal_summary}")


if __name__ == "__main__":
    main()
