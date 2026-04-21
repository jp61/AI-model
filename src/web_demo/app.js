const IMG_SIZE = 150;
const MODEL_PATH = './model/model.json';
const CALIBRATION_PATH = './model/calibration.json';
const EXPECTED_INPUT_SHAPE = [null, IMG_SIZE, IMG_SIZE, 3];
const EXPECTED_OUTPUT_SHAPE = [null, 1];

// DOM refs
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPrompt = document.getElementById('upload-prompt');
const previewImg = document.getElementById('preview-img');
const statusEl = document.getElementById('status');
const modelSpinner = document.getElementById('model-spinner');
const statusText = document.getElementById('status-text');
const inferStatus = document.getElementById('infer-status');
const inferSpinner = document.getElementById('infer-spinner');
const inferText = document.getElementById('infer-text');
const resultEl = document.getElementById('result');
const resultLabel = document.getElementById('result-label');
const confidenceText = document.getElementById('confidence-text');
const confidenceBar = document.getElementById('confidence-bar');
const tryAnotherBtn = document.getElementById('try-another');
const errorBanner = document.getElementById('error-banner');
const errorTitle = document.getElementById('error-title');
const errorDetails = document.getElementById('error-details');

let modelPromise = null;
let modelBroken = false;

function showError(title, details) {
  console.error('[classifier]', title, details);
  errorTitle.textContent = title;
  errorDetails.textContent = details || '';
  errorBanner.classList.remove('hidden');
  statusEl.classList.add('hidden');
  inferStatus.classList.add('hidden');
}

function clearError() {
  errorBanner.classList.add('hidden');
}

function shapeMatches(actual, expected) {
  if (!Array.isArray(actual) || actual.length !== expected.length) return false;
  return actual.every((d, i) => expected[i] === null ? true : d === expected[i]);
}

async function fetchJSON(url, what) {
  let resp;
  try {
    resp = await fetch(url);
  } catch (e) {
    throw new Error(`Could not reach ${what} at ${url} (${e.message}). Are you serving from the repo root?`);
  }
  if (!resp.ok) {
    throw new Error(`Could not fetch ${what}: HTTP ${resp.status} at ${url}`);
  }
  try {
    return await resp.json();
  } catch (e) {
    throw new Error(`${what} is not valid JSON (${e.message})`);
  }
}

async function runCalibration(model) {
  const cal = await fetchJSON(CALIBRATION_PATH, 'calibration.json');
  const tol = cal.tolerance ?? 1e-3;

  for (const c of cal.cases) {
    const input = tf.fill([1, cal.img_size, cal.img_size, 3], c.fill);
    let pred, actual;
    try {
      pred = model.predict(input);
      actual = (await pred.data())[0];
    } finally {
      input.dispose();
      if (pred) pred.dispose();
    }

    if (!Number.isFinite(actual)) {
      throw new Error(`Calibration "${c.name}" produced ${actual} — weights likely failed to load.`);
    }
    const diff = Math.abs(actual - c.expected);
    if (diff > tol) {
      throw new Error(
        `Calibration "${c.name}" mismatch: expected ${c.expected.toFixed(6)}, got ${actual.toFixed(6)} ` +
        `(diff ${diff.toExponential(2)} > tol ${tol}). The converted model does not match the trained model — ` +
        `re-run 'python src/convert_to_tfjs.py'.`
      );
    }
  }
}

async function loadModel() {
  statusText.textContent = 'Loading model...';
  modelSpinner.classList.remove('hidden');
  statusEl.classList.remove('hidden');

  let model;
  try {
    model = await tf.loadLayersModel(MODEL_PATH, {
      onProgress: (fraction) => {
        statusText.textContent = `Loading model... ${Math.round(fraction * 100)}%`;
      }
    });
  } catch (e) {
    throw new Error(`tf.loadLayersModel failed: ${e.message}`);
  }

  if (!shapeMatches(model.inputs[0].shape, EXPECTED_INPUT_SHAPE)) {
    throw new Error(
      `Unexpected input shape ${JSON.stringify(model.inputs[0].shape)}, ` +
      `expected ${JSON.stringify(EXPECTED_INPUT_SHAPE)}.`
    );
  }
  if (!shapeMatches(model.outputs[0].shape, EXPECTED_OUTPUT_SHAPE)) {
    throw new Error(
      `Unexpected output shape ${JSON.stringify(model.outputs[0].shape)}, ` +
      `expected ${JSON.stringify(EXPECTED_OUTPUT_SHAPE)}.`
    );
  }

  statusText.textContent = 'Verifying model...';
  await runCalibration(model);

  statusText.textContent = 'Model ready';
  modelSpinner.classList.add('hidden');
  setTimeout(() => statusEl.classList.add('hidden'), 1500);
  return model;
}

function ensureModel() {
  if (!modelPromise) {
    modelPromise = loadModel().catch((e) => {
      modelBroken = true;
      showError('Model failed to load', e.message);
      throw e;
    });
  }
  return modelPromise;
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(new Error('Could not read file'));
    r.readAsDataURL(file);
  });
}

function showPreview(dataURL) {
  return new Promise((resolve, reject) => {
    previewImg.onload = () => resolve();
    previewImg.onerror = () => reject(new Error('Image could not be decoded (unsupported or corrupt)'));
    previewImg.src = dataURL;
    previewImg.hidden = false;
    uploadPrompt.hidden = true;
    dropZone.classList.add('has-image');
  });
}

function preprocess(imgElement) {
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(imgElement).toFloat();
    t = tf.image.resizeBilinear(t, [IMG_SIZE, IMG_SIZE]);
    t = t.div(255.0);
    return t.expandDims(0);
  });
}

async function tensorStats(t) {
  const [minT, maxT, meanT] = [t.min(), t.max(), t.mean()];
  const [min, max, mean] = await Promise.all([minT.data(), maxT.data(), meanT.data()]);
  minT.dispose(); maxT.dispose(); meanT.dispose();
  return { min: min[0], max: max[0], mean: mean[0] };
}

async function classify(imgElement) {
  if (modelBroken) {
    throw new Error('Model is not loaded — see the error above.');
  }

  inferStatus.classList.remove('hidden');
  inferText.textContent = 'Classifying...';
  inferSpinner.classList.remove('hidden');

  const model = await ensureModel();

  const input = preprocess(imgElement);
  let pred, val, stats;
  try {
    stats = await tensorStats(input);
    pred = model.predict(input);
    val = (await pred.data())[0];
  } finally {
    input.dispose();
    if (pred) pred.dispose();
  }

  inferStatus.classList.add('hidden');

  console.log('[classifier] input tensor stats:', stats);
  console.log('[classifier] raw sigmoid:', val);

  if (!Number.isFinite(val)) {
    throw new Error(`Inference produced ${val}. The model is not usable.`);
  }
  if (val < 0 || val > 1) {
    throw new Error(`Inference produced out-of-range sigmoid ${val}. The model output layer is wrong.`);
  }

  const isDog = val > 0.5;
  const label = isDog ? 'Dog' : 'Cat';
  const confidence = isDog ? val : 1 - val;
  showResult(label, confidence, val, stats);
}

function showResult(label, confidence, raw, stats) {
  const cls = label.toLowerCase();
  resultLabel.textContent = label;
  resultLabel.className = 'result-label ' + cls;

  const pct = Math.round(confidence * 100);
  const statStr = stats
    ? ` · tensor[${stats.min.toFixed(2)}, ${stats.max.toFixed(2)}] μ=${stats.mean.toFixed(3)}`
    : '';
  confidenceText.textContent = `${pct}% confidence (raw ${raw.toFixed(4)})${statStr}`;

  confidenceBar.className = 'confidence-bar-fill ' + cls;
  confidenceBar.style.width = '0%';

  resultEl.classList.add('visible');

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceBar.style.width = pct + '%';
    });
  });
}

function reset() {
  resultEl.classList.remove('visible');
  confidenceBar.style.width = '0%';
  previewImg.hidden = true;
  previewImg.src = '';
  uploadPrompt.hidden = false;
  dropZone.classList.remove('has-image');
  inferStatus.classList.add('hidden');
  fileInput.value = '';
  if (!modelBroken) clearError();
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showError('Not an image', `File type "${file.type || 'unknown'}" is not supported. Drop a JPEG or PNG.`);
    return;
  }

  resultEl.classList.remove('visible');
  confidenceBar.style.width = '0%';
  if (!modelBroken) clearError();

  try {
    const dataURL = await readFileAsDataURL(file);
    await showPreview(dataURL);
    await classify(previewImg);
  } catch (e) {
    showError('Classification failed', e.message);
  }
}

let dragCounter = 0;

dropZone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

dropZone.addEventListener('dragleave', () => {
  dragCounter--;
  if (dragCounter === 0) dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropZone.classList.remove('dragover');
  handleFile(e.dataTransfer.files[0]);
});

dropZone.addEventListener('click', () => {
  if (!dropZone.classList.contains('has-image')) fileInput.click();
});

dropZone.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && !dropZone.classList.contains('has-image')) {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
tryAnotherBtn.addEventListener('click', reset);

window.addEventListener('error', (e) => showError('Uncaught error', e.message));
window.addEventListener('unhandledrejection', (e) => {
  const msg = e.reason?.message || String(e.reason);
  showError('Unhandled promise rejection', msg);
});

ensureModel().catch(() => {});
