const IMG_SIZE = 150;
const MODEL_PATH = 'tfjs_model/model.json';

// DOM references
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

// Model loading
let modelPromise = null;

function ensureModel() {
  if (!modelPromise) modelPromise = loadModel();
  return modelPromise;
}

async function loadModel() {
  statusText.textContent = 'Loading model...';
  modelSpinner.classList.remove('hidden');
  statusEl.classList.remove('hidden');

  const model = await tf.loadLayersModel(MODEL_PATH, {
    onProgress: (fraction) => {
      const pct = Math.round(fraction * 100);
      statusText.textContent = `Loading model... ${pct}%`;
    }
  });

  statusText.textContent = 'Model ready';
  modelSpinner.classList.add('hidden');
  setTimeout(() => statusEl.classList.add('hidden'), 2000);

  return model;
}

// File reading
function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

// Show preview, returns promise that resolves when image loads
function showPreview(dataURL) {
  return new Promise((resolve) => {
    previewImg.onload = () => resolve();
    previewImg.src = dataURL;
    previewImg.hidden = false;
    uploadPrompt.hidden = true;
    dropZone.classList.add('has-image');
  });
}

// Preprocessing (identical to original main.js)
function preprocess(imgElement) {
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(imgElement).toFloat();
    t = tf.image.resizeBilinear(t, [IMG_SIZE, IMG_SIZE]);
    t = t.div(255.0);
    t = t.expandDims(0);
    return t;
  });
}

// Inference
async function classify(imgElement) {
  // Show classifying indicator
  inferStatus.classList.remove('hidden');
  inferText.textContent = 'Classifying...';
  inferSpinner.classList.remove('hidden');

  const model = await ensureModel();

  const input = preprocess(imgElement);
  const pred = model.predict(input);
  const val = (await pred.data())[0];
  input.dispose();
  pred.dispose();

  const isDog = val > 0.5;
  const label = isDog ? 'Dog' : 'Cat';
  const confidence = isDog ? val : 1 - val;

  // Hide inference spinner
  inferStatus.classList.add('hidden');

  showResult(label, confidence);
}

// Display result
function showResult(label, confidence) {
  const cls = label.toLowerCase();
  resultLabel.textContent = label;
  resultLabel.className = 'result-label ' + cls;

  const pct = Math.round(confidence * 100);
  confidenceText.textContent = `${pct}% confidence`;

  confidenceBar.className = 'confidence-bar-fill ' + cls;
  confidenceBar.style.width = '0%';

  resultEl.classList.add('visible');

  // Animate bar after a brief delay so the transition fires
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceBar.style.width = pct + '%';
    });
  });
}

// Reset
function reset() {
  resultEl.classList.remove('visible');
  confidenceBar.style.width = '0%';
  previewImg.hidden = true;
  previewImg.src = '';
  uploadPrompt.hidden = false;
  dropZone.classList.remove('has-image');
  inferStatus.classList.add('hidden');
  fileInput.value = '';
}

// Handle file
async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    statusText.textContent = 'Please upload an image file';
    statusEl.classList.remove('hidden');
    modelSpinner.classList.add('hidden');
    setTimeout(() => statusEl.classList.add('hidden'), 2500);
    return;
  }

  // Reset any previous result
  resultEl.classList.remove('visible');
  confidenceBar.style.width = '0%';

  const dataURL = await readFileAsDataURL(file);
  await showPreview(dataURL);
  await classify(previewImg);
}

// Drag and drop
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
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

// Click to browse
dropZone.addEventListener('click', () => {
  if (!dropZone.classList.contains('has-image')) {
    fileInput.click();
  }
});

// Keyboard accessibility
dropZone.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && !dropZone.classList.contains('has-image')) {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  handleFile(file);
});

// Try another
tryAnotherBtn.addEventListener('click', reset);

// Start loading model immediately
ensureModel();
