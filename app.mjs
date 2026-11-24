const worker = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });
const fileInput = document.querySelector("#photo-file");
const run = document.querySelector("#predict");
const modelStatus = document.querySelector("#model-status");
const status = document.querySelector("#status");
const modelProgress = document.querySelector("#model-progress");
const depthProgress = document.querySelector("#depth-progress");
const modelPercent = document.querySelector("#model-percent");
const depthPercent = document.querySelector("#depth-percent");
const source = document.querySelector("#source");
const result = document.querySelector("#result");
const MAX_SIDE = 320;
let ready = false;
let busy = false;
let imageReady = false;
let selection = 0;

function setProgress(bar, label, completed, total) {
  if (!total) {
    bar.removeAttribute("value");
    label.textContent = `${(completed / 1_000_000).toFixed(1)} MB`;
    return;
  }
  const percent = Math.min(100, Math.floor(100 * completed / total));
  bar.value = percent;
  label.textContent = `${percent}%`;
}

function updateControls() {
  run.disabled = !ready || !imageReady || busy;
  fileInput.disabled = busy;
}

fileInput.addEventListener("change", async () => {
  const id = ++selection;
  const file = fileInput.files[0];
  imageReady = false;
  source.hidden = true;
  result.hidden = true;
  setProgress(depthProgress, depthPercent, 0, 100);
  status.textContent = "";
  updateControls();
  if (!file) return;
  try {
    const bitmap = await createImageBitmap(file);
    try {
      if (id !== selection) return;
      const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
      source.width = Math.max(1, Math.round(bitmap.width * scale));
      source.height = Math.max(1, Math.round(bitmap.height * scale));
      source.getContext("2d", { willReadFrequently: true }).drawImage(bitmap, 0, 0, source.width, source.height);
      imageReady = true;
      source.hidden = false;
      status.textContent = `${source.width} × ${source.height} px`;
    } finally {
      bitmap.close();
    }
  } catch {
    if (id === selection) status.textContent = "Η εικόνα δεν άνοιξε. Δοκίμασε JPEG, PNG ή WebP.";
  } finally {
    updateControls();
  }
});

run.addEventListener("click", () => {
  const { width, height } = source;
  const rgba = source.getContext("2d").getImageData(0, 0, width, height).data;
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    rgb.set(rgba.subarray(i * 4, i * 4 + 3), i * 3);
  }
  busy = true;
  result.hidden = true;
  setProgress(depthProgress, depthPercent, 0, 100);
  status.textContent = "Υπολογισμός βάθους…";
  updateControls();
  worker.postMessage({ rgb, width, height }, [rgb.buffer]);
});

function drawMap(values, width, height) {
  const sorted = values.slice().sort();
  const low = sorted[Math.floor(sorted.length * 0.02)];
  const high = sorted[Math.floor(sorted.length * 0.98)];
  const span = high - low;
  result.width = width;
  result.height = height;
  const context = result.getContext("2d");
  const image = context.createImageData(width, height);
  for (let i = 0; i < values.length; i++) {
    const gray = span > 0 ? 255 * (values[i] - low) / span : values[i] > 0 ? 128 : 0;
    image.data.set([gray, gray, gray, 255], i * 4);
  }
  context.putImageData(image, 0, 0);
  result.hidden = false;
}

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case "download-progress":
      setProgress(modelProgress, modelPercent, data.received, data.total);
      if (data.done) modelStatus.textContent = "Προετοιμασία μοντέλου…";
      break;
    case "ready":
      ready = true;
      setProgress(modelProgress, modelPercent, 100, 100);
      modelStatus.textContent = "Μοντέλο έτοιμο";
      break;
    case "load-error":
      modelStatus.textContent = `Η φόρτωση απέτυχε (${data.message}). Ανανεώστε τη σελίδα για νέα προσπάθεια.`;
      break;
    case "depth-progress":
      setProgress(depthProgress, depthPercent, data.completed, data.total);
      break;
    case "map":
      drawMap(data.inverseDepth, data.width, data.height);
      setProgress(depthProgress, depthPercent, 100, 100);
      busy = false;
      status.textContent = "Ο χάρτης είναι έτοιμος. Οι φωτεινές περιοχές αντιστοιχούν σε κοντινότερα σημεία.";
      break;
    case "predict-error":
      busy = false;
      status.textContent = `Ο υπολογισμός απέτυχε (${data.message}).`;
      break;
  }
  updateControls();
};
worker.onerror = () => {
  ready = false;
  busy = false;
  modelStatus.textContent = "Ο browser δεν μπόρεσε να εκτελέσει το μοντέλο.";
  updateControls();
};
