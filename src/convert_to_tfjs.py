"""
Convert trained Keras model (.keras) to TensorFlow.js Layers format.

Produces:
1. model.json -> The structural "blueprint" of the model.
2. .bin files -> Binary shards containing the actual trained weights (knowledge).
These can be loaded in JS using tf.loadLayersModel().
"""

import json
import os
import struct
import numpy as np
import tensorflow as tf

# Define paths and shard size
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(SCRIPT_DIR, "cats_dogs_model.keras")
MODEL_PATH = os.path.join(SCRIPT_DIR, "cats_dogs_model_fine_tuned.keras")
OUT_DIR = os.path.join(SCRIPT_DIR, "web_demo", "tfjs_model")
SHARD_SIZE = 4 * 1024 * 1024  # 4 MB per shard (standard for web delivery)


def build_tfjs_topology(model):
    """Build a TF.js-compatible model topology dict in Keras 2 format."""

    def initializer_config(name):
        return {"class_name": name, "config": {}}

    layers_json = []
    # Iterate through all layers (Conv2D, MaxPooling, Dense, etc.)
    for i, layer in enumerate(model.layers):
        cfg = layer.get_config()
        ltype = type(layer).__name__

        # Flatten dtype to string (Keras 3 compatibility fix)
        if isinstance(cfg.get("dtype"), dict):
            cfg["dtype"] = cfg["dtype"].get("config", {}).get("name", "float32")

        # Clean up initializer objects for the browser
        for key in list(cfg.keys()):
            val = cfg[key]
            if isinstance(val, dict) and "module" in val:
                cfg[key] = {
                    "class_name": val["class_name"],
                    "config": val.get("config", {}),
                }

        # Remove Keras 3 exclusive keys that TF.js doesn't recognize
        cfg.pop("quantization_config", None)

        # Skip InputLayer (TF.js defines inputs within the first real layer)
        if ltype == "InputLayer":
            continue

        # Add batch_input_shape to the first layer so the browser knows the input size
        if len(layers_json) == 0:
            input_shape = model.input_shape  # e.g., (None, 150, 150, 3)
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
    """Extract model weights and write them as binary shard files."""

    weights_entries = [] # List of weight metadata (names, shapes, types)
    raw_bytes = bytearray() # Buffer for the actual binary data

    for layer in model.layers:
        for w in layer.weights:
            arr = w.numpy()
            # Format the name for TF.js (e.g., conv2d/kernel)
            var_name = w.name.split(":")[0]  
            name = f"{layer.name}/{var_name}"

            weights_entries.append({
                "name": name,
                "shape": list(arr.shape),
                "dtype": "float32",
            })
            # Convert numbers to raw binary bytes
            raw_bytes.extend(arr.astype(np.float32).tobytes())

    # Split the large binary chunk into smaller 4MB files (shards)
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

    # The manifest links the binary files back to the JSON structure
    manifest = [{"paths": paths, "weights": weights_entries}]
    return manifest


def main():
    print(f"Loading model from {MODEL_PATH}...")
    # Load your trained Python model
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Build the architectural blueprint (topology)
    topology = build_tfjs_topology(model)

    # Step 2: Save the "knowledge" (weights) as binary shards
    manifest = serialize_weights(model, OUT_DIR, SHARD_SIZE)

    # Step 3: Combine everything into the final model.json
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

    # Final report on export size
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