# AI-model

Train a CNN to classify cats vs dogs, then run inference in the browser using TensorFlow.js.

## Prerequisites

- **Python 3.10–3.12** (TensorFlow does not support 3.13+)
- [pyenv](https://github.com/pyenv/pyenv) (recommended for managing Python versions)

## Installation

```bash
# Install Python 3.12 via pyenv (if you don't have a compatible version)
pyenv install 3.12
pyenv local 3.12

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install .
```

## Usage

### 1. Train the model

```bash
cd src
python train.py
```

This downloads the `cats_vs_dogs` dataset, trains a CNN for 10 epochs, and saves the model as `cats_dogs_model.h5` and `cats_dogs_model.keras`.

### 2. Run Python predictions (optional)

Place test images in `src/images/` and run:

```bash
python predict.py
```

### 3. Convert to TensorFlow.js

```bash
python convert_to_tfjs.py
```

This converts the trained Keras model into TensorFlow.js Layers format under `web_demo/tfjs_model/` (a `model.json` file and binary weight shards).

### 4. Launch the web demo

```bash
cd web_demo
python3 -m http.server 8000
```

Open http://localhost:8000 in your browser. Drag and drop an image (or click to browse) to classify it as a cat or dog. All inference runs client-side in the browser.

## Project structure

```
src/
  train.py              # Train the CNN
  predict.py            # Python inference on local images
  convert_to_tfjs.py    # Convert Keras model to TF.js format
  images/               # Sample test images
  web_demo/
    index.html          # Web UI
    style.css           # Styling
    app.js              # Client-side inference logic
    tfjs_model/         # Generated TF.js model (after conversion)
```
