# Convert model to TensorFlow.js and run demo

This project contains a Keras model (`cats_dogs_model.h5`) and a tiny web demo (`web_demo/`) that expects TFJS artifacts at `web_demo/tfjs_model/` (i.e. `web_demo/tfjs_model/model.json`).

If `tensorflowjs` installation fails in your current Python (e.g. Python 3.13 + NumPy 2.x incompatibilities), use one of the options below.

Option A — Use Python 3.11 (recommended):

1. Create a new venv with Python 3.11:

```bash
python3.11 -m venv venv_tfjs_311
source venv_tfjs_311/bin/activate
pip install --upgrade pip
pip install tensorflowjs
```

2. Convert the Keras HDF5 file to TFJS format:

```bash
tensorflowjs_converter --input_format=keras cats_dogs_model.h5 web_demo/tfjs_model
```

After this completes you'll have `web_demo/tfjs_model/model.json` and weight shard files.

Option B — Docker (isolated conversion):

```bash
docker run --rm -v "$PWD":/work -w /work python:3.11-bullseye /bin/bash -lc "pip install --no-cache-dir tensorflowjs && tensorflowjs_converter --input_format=keras cats_dogs_model.h5 web_demo/tfjs_model"
```

Option C — If you must use the included `venv_tfjs` with Python 3.13, the `tensorflowjs` package may require older NumPy wheels; creating a fresh Python 3.11 venv is easier.

Run the demo locally:

1. Serve the repo root (so `web_demo/tfjs_model` is reachable). From the project root run:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/web_demo/index.html
```

Notes:
- The demo expects the TFJS model to output a single probability (shape [1,1]) like the original Keras model.
- If your Keras model uses a different output shape or preprocessing, update `web_demo/main.js` accordingly.
