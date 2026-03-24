const IMG_SIZE = 150;
let model;

async function loadModel() {
  model = await tf.loadLayersModel('tfjs_model/model.json');
  console.log('Model loaded');
}

function readFileAsDataURL(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

function preprocessImage(imgElement) {
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(imgElement).toFloat();
    t = tf.image.resizeBilinear(t, [IMG_SIZE, IMG_SIZE]);
    t = t.div(255.0);
    t = t.expandDims(0);
    return t;
  });
}


document.getElementById('file').addEventListener('change', async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const url = await readFileAsDataURL(f);
  const img = new Image();
  img.src = url;
  img.onload = async () => {
    document.getElementById('preview').innerHTML = '';
    document.getElementById('preview').appendChild(img);
    if (!model) await loadModel();
    const input = preprocessImage(img);
    const pred = model.predict(input);
    const val = (await pred.data())[0];
    input.dispose();
    pred.dispose();
    const label = val > 0.5 ? 'Dog' : 'Cat';
    document.getElementById('result').textContent = `Prediction: ${label} (score=${val.toFixed(3)})`;
  };
});
