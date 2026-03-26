"""Convert trained Keras model to TensorFlow.js Layers format.

Produces model.json + weight shard .bin files that tf.loadLayersModel() can load.
Does NOT depend on the tensorflowjs package (which has broken transitive deps).
Instead, it reads the trained model's weights and builds the TF.js-compatible
model.json in Keras 2 format directly.
"""

import json
import os
import struct

import numpy as np
import tensorflow as tf

MODEL_PATH = "cats_dogs_model.keras"
OUT_DIR = "web_demo/tfjs_model"
SHARD_SIZE = 4 * 1024 * 1024  # 4 MB per shard


def build_tfjs_topology(model):
    """Build a TF.js-compatible model topology dict in Keras 2 format."""

    def initializer_config(name):
        return {"class_name": name, "config": {}}

    layers_json = []
    for i, layer in enumerate(model.layers):
        cfg = layer.get_config()
        ltype = type(layer).__name__

        # Flatten dtype to string if it's a dict (Keras 3 DTypePolicy)
        if isinstance(cfg.get("dtype"), dict):
            cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")

        # Flatten initializer objects: strip module/registered_name (Keras 3 keys)
        for key in list(cfg.keys()):
            val = cfg[key]
            if isinstance(val, dict) and "module" in val:
                cfg[key] = {
                    "class_name": val["class_name"],
                    "config": val.get("config", {}),
                }

        # Remove Keras 3 only keys
        cfg.pop("quantization_config", None)

        # Skip InputLayer (Keras 3 adds explicit InputLayer to Sequential)
        if ltype == "InputLayer":
            continue

        # Add batch_input_shape to the first real layer
        if len(layers_json) == 0:
            input_shape = model.input_shape  # (None, 150, 150, 3)
            cfg["batch_input_shape"] = list(input_shape)

        layers_json.append({"class_name": ltype, "config": cfg})

    return {
        "class_name": "Sequential",
        "config": {
            "name": model.name,
            "layers": layers_json,
        },
        "keras_version": "2.15.0",
        "backend": "tensorflow",
    }


def serialize_weights(model, out_dir, shard_size):
    """Extract model weights and write as binary shards."""

    weights_entries = []
    raw_bytes = bytearray()

    for layer in model.layers:
        for w in layer.weights:
            arr = w.numpy()
            # TF.js expects: layer_name/variable_name (e.g. conv2d/kernel)
            # Keras 3 w.name is just "kernel"/"bias", so prefix with layer name
            var_name = w.name.split(":")[ 0]  # strip ":0" if present
            name = f"{layer.name}/{var_name}"

            weights_entries.append({
                "name": name,
                "shape": list(arr.shape),
                "dtype": "float32",
            })
            raw_bytes.extend(arr.astype(np.float32).tobytes())

    # Split into shards
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

    manifest = [{"paths": paths, "weights": weights_entries}]
    return manifest


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Build topology
    topology = build_tfjs_topology(model)

    # Serialize weights
    manifest = serialize_weights(model, OUT_DIR, SHARD_SIZE)

    # Write model.json
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.keras.__version__}",
        "convertedBy": "convert_to_tfjs.py",
        "modelTopology": topology,
        "weightsManifest": manifest,
    }

    model_json_path = os.path.join(OUT_DIR, "model.json")
    with open(model_json_path, "w") as f:
        json.dump(model_json, f)

    # Report
    total_bytes = sum(
        os.path.getsize(os.path.join(OUT_DIR, p))
        for p in manifest[0]["paths"]
    )
    print(f"\nConversion complete:")
    print(f"  Output:  {OUT_DIR}/")
    print(f"  Shards:  {len(manifest[0]['paths'])}")
    print(f"  Weights: {len(manifest[0]['weights'])} tensors, {total_bytes / 1e6:.1f} MB")
    print(f"  Layers:  {len(topology['config']['layers'])}")


if __name__ == "__main__":
    main()
