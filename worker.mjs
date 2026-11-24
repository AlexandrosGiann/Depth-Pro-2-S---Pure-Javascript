import { DepthPro2S } from "./depthpro2.mjs";

async function downloadWeights(response) {
  const size = Number(response.headers.get("Content-Length"));
  const encoding = response.headers.get("Content-Encoding");
  const total = !encoding && Number.isSafeInteger(size) && size > 0 ? size : 0;
  const buffer = total ? new Uint8Array(total) : null;
  const chunks = [];
  let received = 0;
  let lastUpdate = -Infinity;
  for await (const chunk of response.body) {
    if (buffer) buffer.set(chunk, received);
    else chunks.push(chunk);
    received += chunk.byteLength;
    const now = performance.now();
    if (now - lastUpdate >= 100 || received === total) {
      self.postMessage({ type: "download-progress", received, total });
      lastUpdate = now;
    }
  }
  if (total && received !== total) throw new Error("Incomplete model download");
  self.postMessage({ type: "download-progress", received, total: received, done: true });
  return buffer ?? new Uint8Array(await new Blob(chunks).arrayBuffer());
}

async function loadModel() {
  try {
    const response = await fetch("./depthpro2-s.safetensors");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const model = new DepthPro2S(await downloadWeights(response));
    self.postMessage({ type: "ready" });
    return model;
  } catch (error) {
    self.postMessage({ type: "load-error", message: error.message });
    return null;
  }
}

const loading = loadModel();
self.onmessage = async ({ data: { rgb, width, height } }) => {
  const model = await loading;
  if (!model) return;
  try {
    const { inverseDepth } = model.predict(rgb, width, height, undefined, progress => {
      self.postMessage({ type: "depth-progress", ...progress });
    });
    self.postMessage({ type: "map", inverseDepth, width, height }, [inverseDepth.buffer]);
  } catch (error) {
    self.postMessage({ type: "predict-error", message: error.message });
  }
};
