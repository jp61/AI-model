// Simulate the web preprocessing path on cat2.jpg: decode JPEG, resize bilinear, /255, predict.
const tf = require('@tensorflow/tfjs');
const jpeg = require('jpeg-js');
const fs = require('fs');

const MODEL_URL = 'http://localhost:8000/model/model.json';
const IMG_PATH = '/home/yin/Projects/Others/AI-model/src/images/cat2.jpg';
const IMG_SIZE = 150;

async function main() {
  const model = await tf.loadLayersModel(MODEL_URL);

  const raw = jpeg.decode(fs.readFileSync(IMG_PATH), { useTArray: true });
  console.log(`Decoded JPEG: ${raw.width}x${raw.height} (RGBA bytes = ${raw.data.length})`);

  // Drop alpha, build tensor like tf.browser.fromPixels would
  const pixels = new Uint8Array((raw.data.length / 4) * 3);
  for (let i = 0, j = 0; i < raw.data.length; i += 4, j += 3) {
    pixels[j]     = raw.data[i];
    pixels[j + 1] = raw.data[i + 1];
    pixels[j + 2] = raw.data[i + 2];
  }

  const result = tf.tidy(() => {
    let t = tf.tensor3d(pixels, [raw.height, raw.width, 3], 'int32').toFloat();
    t = tf.image.resizeBilinear(t, [IMG_SIZE, IMG_SIZE]);
    t = t.div(255.0);
    const batch = t.expandDims(0);
    const min = t.min().dataSync()[0];
    const max = t.max().dataSync()[0];
    const mean = t.mean().dataSync()[0];
    const pred = model.predict(batch);
    const val = pred.dataSync()[0];
    return { min, max, mean, val };
  });

  console.log(`Tensor: min=${result.min.toFixed(3)} max=${result.max.toFixed(3)} mean=${result.mean.toFixed(3)}`);
  console.log(`Raw sigmoid: ${result.val.toFixed(4)}  ->  ${result.val > 0.5 ? 'Dog' : 'Cat'}`);
}
main().catch(e => { console.error(e); process.exit(1); });
