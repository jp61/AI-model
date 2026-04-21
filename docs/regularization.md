# Regularization: the cat2.jpg incident and what we did about it

## The problem

The first trained model classified `src/images/cat2.jpg` as **"Dog, 100% confidence"** in the web demo. A friend reviewing the project expected the model to get this right — it's an obvious cat to a human.

## Investigation (2026-04-20)

We spent significant effort ruling out pipeline bugs before concluding this was a model-quality issue, not an infrastructure issue:

| Suspected cause | How we checked | Result |
| --- | --- | --- |
| Keras → TF.js converter corrupts weights | Built a Node TF.js harness (`tfjs_sim/verify.js`) running a calibration suite on synthetic zero/one/half tensors, comparing against Python predictions. | Worst diff **2.98e-8** — bit-faithful. Not the bug. |
| Browser color transforms (EXIF, ICC profile) | Inspected `cat2.jpg` metadata. | No EXIF orientation, no ICC profile. Not the bug. |
| Preprocessing mismatch (PIL nearest vs bilinear) | Rewrote `predict.py` to use `tf.image.resize(bilinear)` matching the training pipeline. Then ran CLI on cat2.jpg. | CLI now also said **Dog, 75.2%**. Pipeline and CLI agree. |
| Weight loading failure in the browser (silent) | Added calibration check in `app.js` that throws if the loaded model's output on synthetic tensors diverges by more than 1e-3. | Passes. Weights load correctly. |
| Model generalization | Compared train/val accuracy. | **Train accuracy hit 99%+** within a few epochs; the model had memorized the training set. |

### Verdict

The "100% Dog" was not a bug in any pipeline stage. The model was legitimately outputting a sigmoid around **0.75** for cat2.jpg — the UI was just rounding/displaying a high-but-legitimate wrong answer. The web UI said "100%" because the original code did not include precision in the display; the raw sigmoid was, again, ~0.75.

Root cause: the model overfit. With ~16M parameters (most in `Dense(512)`) and ~20k training images, no augmentation, no dropout, no weight decay, 10 epochs — the model memorized training data and failed on cat2.jpg, a held-out example that sits near its decision boundary.

## Industry-standard mitigations

These are the standard tools for image classifiers trained from scratch on modest datasets:

1. **Data augmentation** — flip/rotate/zoom/crop the training images so each epoch sees a slightly different version of each sample. Encodes invariances ("a mirrored cat is still a cat") and effectively multiplies dataset size.
2. **Dropout** — during training, randomly zero a fraction of activations. Prevents neurons from co-adapting; acts like training an ensemble of thinned networks and averaging them at inference.
3. **L2 weight decay** — add `λ · Σ‖w‖²` to the loss. Pulls weights toward zero, capping the expressive capacity of the model. The classical "ridge regression" regularizer, applied to neural-net kernels.
4. **EarlyStopping** — stop training when validation loss stops improving; restore the weights from the best epoch. Regularization via optimization rather than model structure.
5. **Transfer learning** (higher ceiling, more complex) — start from a pretrained feature extractor (MobileNetV2, EfficientNetB0) and fine-tune the classifier head. Dodges the whole "too few images for a from-scratch CNN" problem. Not implemented here; see "Beyond the current flags" below.

These compose well because they constrain the model along different axes: augmentation constrains the *training distribution*, dropout/L2 constrain the *parameters*, EarlyStopping constrains the *optimization trajectory*.

## What's implemented

All four are wired into `src/train.py` and controlled by flags:

| Flag | Default | What it does | Disable with |
| --- | --- | --- | --- |
| `--augment` / `--no-augment` | `--augment` | Applies `RandomFlip("horizontal")`, `RandomRotation(0.1)`, `RandomZoom(0.1)` in the `tf.data` pipeline (not inside the model — see note below). | `--no-augment` |
| `--dropout FLOAT` | `0.5` | Adds `Dropout(rate)` before `Dense(512)`. | `--dropout 0` |
| `--l2 FLOAT` | `1e-4` | Adds `kernel_regularizer=l2(λ)` to every Conv2D and Dense. | `--l2 0` |
| `--patience INT` | `3` | `EarlyStopping(val_loss, patience, restore_best_weights=True)`. | `--patience 0` |
| `--epochs INT` | `30` | Upper bound on epochs. | — |
| `--batch-size INT` | `32` | Training batch size. | — |

### Important: augmentation lives in the data pipeline, not in the model

`RandomFlip`, `RandomRotation`, and `RandomZoom` are applied via `train_ds.map(augmenter)` **before** the data reaches the model. They are **not** layers inside the saved `Sequential`.

**Why this matters:** `@tensorflow/tfjs` (the browser runtime) does not register these preprocessing layer classes. If they are baked into the model, `tf.loadLayersModel()` fails with `Unknown layer: RandomFlip`, the browser gets stuck showing "Loading model... 100%", and the only fix is to strip the layers and reconvert. The pipeline-side approach sidesteps this entirely: the exported model is pure Conv/Dense/MaxPool/Dropout/Flatten, which TF.js handles natively. Dropout stays inside the model because it's a no-op at inference and TF.js supports it.

## Usage

**Default (all regularization on, recommended):**
```bash
python src/train.py
```

**Baseline (no regularization, reproduces the original overfit model):**
```bash
python src/train.py --no-augment --dropout 0 --l2 0 --patience 0 --epochs 10
```

**Stronger regularization (if val_loss still gap above train_loss):**
```bash
python src/train.py --dropout 0.6 --l2 5e-4
```

**Weaker regularization (if the model is underfitting):**
```bash
python src/train.py --dropout 0.3 --l2 1e-5
```

**Quick smoke test:**
```bash
python src/train.py --epochs 3 --no-augment --dropout 0 --l2 0 --patience 0
```

After training, always reconvert and re-verify:
```bash
python src/convert_to_tfjs.py
python src/predict.py src/images/cat2.jpg
# then reload the web demo
```

## What to expect

### Baseline (everything off)
- Train accuracy climbs to 99%+ in 5-7 epochs.
- Validation accuracy plateaus around 80-85%.
- Large train/val gap — classic overfitting.
- cat2.jpg likely misclassified.

### Default (all on)
- Train accuracy climbs more slowly, often tops out in the low 90s.
- Validation accuracy matches or exceeds train accuracy (dropout makes training "harder" than inference).
- Gap is small — sign that the model is generalizing.
- EarlyStopping typically kicks in between epoch 10 and 20.
- cat2.jpg should classify as Cat, though confidence may be modest (60-80%).

### With L2 on top of dropout
- Loss starts higher (penalty term adds to the reported loss — don't panic).
- Weights stay smaller in magnitude. Makes the model more robust to input perturbations.
- Accuracy change is usually modest on top of dropout+augmentation, but it can stabilize training.

## Troubleshooting

| Symptom | Likely cause | What to try |
| --- | --- | --- |
| Train acc ≈ val acc, both low (<85%) | **Underfitting** — too much regularization, or model too small. | `--dropout 0.3 --l2 1e-5`, or drop `--l2 0` entirely. Remove augmentation as a test. |
| Train acc much higher than val acc | **Overfitting** — regularization too weak. | Increase `--dropout` to 0.6, `--l2` to 5e-4. Confirm `--augment` is on. |
| Loss is `nan` after a few batches | Usually LR × L2 too aggressive on first step. | Lower `--l2` by 10×, or lower the Adam learning rate in `train.py`. |
| Val loss oscillates wildly | Batch size too small or LR too high. | `--batch-size 64`. |
| EarlyStopping never triggers | `--patience` too high, or val_loss genuinely still improving. | Harmless if epochs are bounded. Lower patience if you want a quicker exit. |
| Val loss improves then plateaus immediately | Normal. EarlyStopping will restore the plateau point as best. | No action. |
| cat2.jpg still wrong after retraining | Model capacity may not be enough for this dataset from scratch. | Consider transfer learning (see below). Also try more epochs + more aggressive augmentation. |
| Converter (`convert_to_tfjs.py`) fails on the new model | Unlikely — augmentation is outside the model, dropout has no weights. | Open an issue with the converter's error message. |
| Web demo calibration fails after retraining | The model file was updated but the TF.js export wasn't regenerated, or the TF.js export is stale. | Re-run `python src/convert_to_tfjs.py` and hard-refresh the browser. |
| Web demo hangs at "Loading model... 100%" with no error banner | Either the http server is running from the wrong directory (must be `src/web_demo/`), the model artifacts haven't been generated yet (`src/web_demo/model/` missing), or the model contains unsupported layer types (e.g. someone reintroduced `RandomFlip` inside the model). | Kill the server, run `cd src/web_demo && python3 -m http.server 8000`, and hard-refresh at `http://localhost:8000/`. Re-run `python src/convert_to_tfjs.py` if `src/web_demo/model/model.json` is missing. If still stuck, check the browser console for `Unknown layer: ...` and confirm augmentation stays in the data pipeline. |
| Browser console says `Unknown layer: RandomFlip` (or RandomRotation/RandomZoom) | Augmentation layers were accidentally placed inside the model. | Make sure `build_model()` in `src/train.py` does not include any `Random*` layers, then retrain and reconvert. |

## Beyond the current flags

If augmentation + dropout + L2 + EarlyStopping don't close the gap on cat2.jpg, the next move is **transfer learning**. Replace the hand-rolled CNN with a pretrained backbone:

```python
base = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3), include_top=False, weights='imagenet'
)
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])
```

This typically gets 95%+ validation accuracy on cats-vs-dogs in a few epochs because the backbone already knows what edges, textures, and object parts look like from ImageNet. It's not wired into flags here because it's a structural change (different input shape, different preprocessing) rather than a knob.

## Conceptual reference

**How are these different from L2 in linear regression (`C = error + λΣw²`)?**

- **L2 / weight decay**: identical concept — add a penalty on the sum of squared weights to the loss. Caps parameter magnitude.
- **Dropout**: acts on parameters too, but stochastically. Randomly zeros activations during training; roughly equivalent to averaging an ensemble of sub-networks.
- **Augmentation**: doesn't touch parameters or the loss. Modifies the *training distribution* by applying label-preserving transformations (a flipped cat is still a cat). Encodes domain knowledge as invariances.

For linear models with Gaussian-noise augmentation there's an exact equivalence between augmentation and L2. For nonlinear nets with structured augmentations like rotation the equivalence breaks down and augmentation is strictly more expressive — it encodes symmetries L2 cannot.
