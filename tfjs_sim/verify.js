const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const MODEL_URL = 'http://localhost:8000/model/model.json';
const CAL_PATH = '/home/yin/Projects/Others/AI-model/src/web_demo/model/calibration.json';

async function main() {
  console.log('Loading model from', MODEL_URL);
  let model;
  try {
    model = await tf.loadLayersModel(MODEL_URL);
  } catch (e) {
    console.error('LOAD FAILED:', e.message);
    process.exit(2);
  }
  console.log('Input shape:', JSON.stringify(model.inputs[0].shape));
  console.log('Output shape:', JSON.stringify(model.outputs[0].shape));

  const cal = JSON.parse(fs.readFileSync(CAL_PATH));
  console.log('\nCalibration:');
  let worstDiff = 0;
  for (const c of cal.cases) {
    const input = tf.fill([1, cal.img_size, cal.img_size, 3], c.fill);
    const pred = model.predict(input);
    const val = (await pred.data())[0];
    const diff = Math.abs(val - c.expected);
    worstDiff = Math.max(worstDiff, diff);
    const status = diff < cal.tolerance ? 'OK' : 'MISMATCH';
    console.log(`  ${c.name.padEnd(8)} expected=${c.expected.toFixed(6)} got=${val.toFixed(6)} diff=${diff.toExponential(2)}  ${status}`);
    input.dispose();
    pred.dispose();
  }
  console.log(`\nWorst diff: ${worstDiff.toExponential(2)}  tolerance: ${cal.tolerance}`);
  process.exit(worstDiff < cal.tolerance ? 0 : 3);
}
main();
