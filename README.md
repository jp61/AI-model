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

All commands are run from the repository root. Model files and intermediate artifacts are written to `model/`.

### 1. Train the model

```bash
python src/train.py
```

Downloads the `cats_vs_dogs` dataset, trains a CNN with augmentation + dropout + L2 + EarlyStopping, and saves `model/cats_dogs_model.h5` and `model/cats_dogs_model.keras`.

Regularization is controlled by flags (`--augment/--no-augment`, `--dropout`, `--l2`, `--patience`, `--epochs`, `--batch-size`). Run `python src/train.py --help` for the full list, or see [`docs/regularization.md`](docs/regularization.md) for what each flag does, what to expect, and how to troubleshoot.

### 2. Run Python predictions (optional)

```bash
python src/predict.py src/images/cat2.jpg      # single image
python src/predict.py src/images/               # whole directory
python src/predict.py                           # defaults to src/images/
```

Prints the predicted label, raw sigmoid output, and confidence for each image.

### 3. Convert to TensorFlow.js

```bash
python src/convert_to_tfjs.py
```

Writes `src/web_demo/model/model.json` plus binary weight shards and `calibration.json`. Everything the browser needs sits next to `app.js`.

### 4. Launch the web demo

Serve from `src/web_demo/`:

```bash
cd src/web_demo && python3 -m http.server 8000
```

Then open http://localhost:8000/ in your browser. Drag and drop an image (or click to browse) to classify it. All inference runs client-side.

## Project structure

```
model/                    # Trained Keras model (gitignored)
  cats_dogs_model.h5
  cats_dogs_model.keras
src/
  train.py                # Train the CNN
  predict.py              # Python CLI inference
  convert_to_tfjs.py      # Keras → TF.js converter (outputs to web_demo/model/)
  images/                 # Sample test images
  web_demo/
    index.html
    style.css
    app.js                # Client-side inference
    model/                # TF.js export (gitignored)
      model.json
      calibration.json
      group1-shard*.bin
```
