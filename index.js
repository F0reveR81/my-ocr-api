
import express from 'express';
import ort from 'onnxruntime-node';
import Jimp from 'jimp';

const app = express();
app.use(express.json({ limit: '2mb' }));

const charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const modelPath = './model/common.onnx';

const getImageTensorFromBase64 = async (base64) => {
  const buffer = Buffer.from(base64, 'base64');
  const image = await Jimp.read(buffer);
  image.resize(120, 32).grayscale();

  const pixels = [];
  image.scan(0, 0, image.bitmap.width, image.bitmap.height, function(x, y, idx) {
    const value = this.bitmap.data[idx];
    pixels.push(value / 255.0);
  });

  const tensor = new ort.Tensor('float32', new Float32Array(pixels), [1, 1, 32, 120]);
  return tensor;
};

const decodeOutput = (output) => {
  const indices = output.data;
  const chars = [];
  let last = -1;
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    if (idx !== last && idx < charset.length) {
      chars.push(charset[idx]);
      last = idx;
    }
  }
  return chars.join('');
};

app.post('/', async (req, res) => {
  try {
    const { image } = req.body;
    const tensor = await getImageTensorFromBase64(image);
    const session = await ort.InferenceSession.create(modelPath);
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    const output = results.output;
    const code = decodeOutput(output);
    res.json({ result: code });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'OCR failed', detail: err.message });
  }
});

app.listen(8080, () => {
  console.log('OCR API running on http://localhost:8080');
});
