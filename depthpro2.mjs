const DIM = 384;
const HEAD = 64;
const PREFIX = 5;

function readWeights(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 8) throw new Error("Incomplete SafeTensors file");
  const headerSize = Number(view.getBigUint64(0, true));
  const start = 8 + headerSize;
  if (start > bytes.length || headerSize > 100_000_000) throw new Error("Invalid SafeTensors header");
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, start)));
  const weights = new Map();
  for (const [name, entry] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets: [lo, hi] } = entry;
    const length = shape.reduce((a, b) => a * b, 1);
    if (dtype !== "F32" || !Number.isSafeInteger(lo) || lo < 0 ||
        hi - lo !== length * 4 || hi > bytes.length - start) {
      throw new Error(`Invalid F32 tensor: ${name}`);
    }
    const values = new Float32Array(length);
    for (let i = 0; i < length; i++) values[i] = view.getFloat32(start + lo + i * 4, true);
    weights.set(name, { shape, values });
  }
  return weights;
}

function dot(a, ai, b, bi, n) {
  let sum = 0;
  for (let i = 0; i < n; i++) sum += a[ai + i] * b[bi + i];
  return sum;
}

function norm(x, scale, bias) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += DIM) {
    let mean = 0;
    for (let c = 0; c < DIM; c++) mean += x[i + c];
    mean /= DIM;
    let variance = 0;
    for (let c = 0; c < DIM; c++) variance += (x[i + c] - mean) ** 2;
    const inv = 1 / Math.sqrt(variance / DIM + 1e-5);
    for (let c = 0; c < DIM; c++) out[i + c] = (x[i + c] - mean) * inv * scale[c] + bias[c];
  }
  return out;
}

// Approximate erf for the exact-GELU form, with absolute erf error below 1.5e-7.
function gelu(x) {
  const z = Math.abs(x) * Math.SQRT1_2;
  const t = 1 / (1 + 0.3275911 * z);
  const tail = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
    0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * x * (x < 0 ? tail : 2 - tail);
}

function rotary(height, width) {
  const table = new Float32Array(height * width * HEAD);
  const extent = Math.max(height, width);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let i = 0; i < HEAD / 2; i++) {
        const coordinate = i < HEAD / 4 ? y : x;
        const angle = 2 * Math.PI * (2 * (coordinate + 0.5) / extent - 1) /
          100 ** ((i % (HEAD / 4)) / (HEAD / 4));
        const offset = (y * width + x) * HEAD + i;
        table[offset] = Math.sin(angle);
        table[offset + HEAD / 2] = Math.cos(angle);
      }
    }
  }
  return table;
}

function attention(qkv, rope) {
  const rows = qkv.length / (3 * DIM);
  const out = new Float32Array(rows * DIM);
  const scores = new Float64Array(rows);
  for (let row = PREFIX; row < rows; row++) {
    const position = ((row - PREFIX) % (rope.length / HEAD)) * HEAD;
    for (let head = 0; head < 2 * DIM; head += HEAD) {
      const base = row * 3 * DIM + head;
      for (let i = 0; i < HEAD / 2; i++) {
        const a = qkv[base + i];
        const b = qkv[base + i + HEAD / 2];
        const sin = rope[position + i];
        const cos = rope[position + i + HEAD / 2];
        qkv[base + i] = a * cos - b * sin;
        qkv[base + i + HEAD / 2] = b * cos + a * sin;
      }
    }
  }
  for (let row = 0; row < rows; row++) {
    for (let head = 0; head < DIM; head += HEAD) {
      let max = -Infinity;
      for (let k = 0; k < rows; k++) {
        scores[k] = dot(qkv, row * 3 * DIM + head, qkv, k * 3 * DIM + DIM + head, HEAD) / 8;
        max = Math.max(max, scores[k]);
      }
      let total = 0;
      for (let k = 0; k < rows; k++) {
        scores[k] = Math.exp(scores[k] - max);
        total += scores[k];
      }
      for (let c = 0; c < HEAD; c++) {
        let sum = 0;
        for (let k = 0; k < rows; k++) sum += scores[k] * qkv[k * 3 * DIM + 2 * DIM + head + c];
        out[row * DIM + head + c] = sum / total;
      }
    }
  }
  return out;
}

function relu(values) {
  return values.map(x => Math.max(0, x));
}

// Feature maps use contiguous channels at each pixel.
export class DepthPro2S {
  #weights;

  constructor(bytes) {
    this.#weights = readWeights(bytes);
    this.#tensor("encoder.cls_token", [1, 1, DIM]);
    this.#tensor("encoder.reg_token", [1, PREFIX - 1, DIM]);
  }

  #tensor(name, shape) {
    const tensor = this.#weights.get(name);
    if (!tensor || tensor.shape.length !== shape.length || shape.some((d, i) => d !== tensor.shape[i])) {
      throw new Error(`Missing or incorrectly shaped tensor: ${name} (${shape.join("×")})`);
    }
    return tensor.values;
  }

  #linear(x, name, input, output, hasBias = true) {
    const w = this.#tensor(name + ".weight", [output, input]);
    const b = hasBias ? this.#tensor(name + ".bias", [output]) : null;
    const rows = x.length / input;
    const result = new Float32Array(rows * output);
    for (let r = 0; r < rows; r++) {
      for (let o = 0; o < output; o++) {
        result[r * output + o] = dot(x, r * input, w, o * input, input) + (b ? b[o] : 0);
      }
    }
    return result;
  }

  #conv(image, name, channels, kernel, stride = 1, pad = 0, transpose = false, hasBias = true) {
    const { data, h, w, c } = image;
    const weight = this.#tensor(name + ".weight",
      transpose ? [c, channels, kernel, kernel] : [channels, c, kernel, kernel]);
    const bias = hasBias ? this.#tensor(name + ".bias", [channels]) : new Float32Array(channels);
    const height = transpose ? h * stride : Math.floor((h + 2 * pad - kernel) / stride) + 1;
    const width = transpose ? w * stride : Math.floor((w + 2 * pad - kernel) / stride) + 1;
    const output = new Float32Array(height * width * channels);
    const patch = new Float32Array(c * kernel * kernel);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const offset = (y * width + x) * channels;
        if (transpose) {
          // All transposed convolutions in S have non-overlapping kernels (kernel == stride).
          const source = (Math.floor(y / stride) * w + Math.floor(x / stride)) * c;
          const tap = (y % stride) * kernel + x % stride;
          for (let o = 0; o < channels; o++) {
            let sum = bias[o];
            for (let i = 0; i < c; i++) sum += data[source + i] * weight[(i * channels + o) * kernel ** 2 + tap];
            output[offset + o] = sum;
          }
        } else {
          for (let ky = 0; ky < kernel; ky++) {
            for (let kx = 0; kx < kernel; kx++) {
              const yy = y * stride + ky - pad;
              const xx = x * stride + kx - pad;
              const inBounds = yy >= 0 && yy < h && xx >= 0 && xx < w;
              for (let i = 0; i < c; i++) {
                patch[(i * kernel + ky) * kernel + kx] = inBounds ? data[(yy * w + xx) * c + i] : 0;
              }
            }
          }
          for (let o = 0; o < channels; o++) {
            output[offset + o] = dot(patch, 0, weight, o * patch.length, patch.length) + bias[o];
          }
        }
      }
    }
    return { data: output, h: height, w: width, c: channels };
  }

  #block(x, index, rope) {
    const name = `encoder.blocks.${index}`;
    const normalize = (x, part) => norm(x, this.#tensor(name + part + ".weight", [DIM]),
      this.#tensor(name + part + ".bias", [DIM]));
    const qkv = this.#linear(normalize(x, ".norm1"), name + ".attn.qkv", DIM, 3 * DIM, false);
    const attended = this.#linear(attention(qkv, rope), name + ".attn.proj", DIM, DIM);
    const scale1 = this.#tensor(name + ".gamma_1", [DIM]);
    const middle = x.map((v, i) => v + attended[i] * scale1[i % DIM]);
    const hidden = this.#linear(normalize(middle, ".norm2"), name + ".mlp.fc1", DIM, 4 * DIM).map(gelu);
    const projected = this.#linear(hidden, name + ".mlp.fc2", 4 * DIM, DIM);
    const scale2 = this.#tensor(name + ".gamma_2", [DIM]);
    return middle.map((v, i) => v + projected[i] * scale2[i % DIM]);
  }

  #encode(images, advance) {
    const patches = images.map(image => this.#conv(image, "encoder.patch_embed.proj", DIM, 16, 16));
    advance("encoder");
    const { h, w } = patches[0];
    const prefixSize = PREFIX * DIM;
    const patchSize = h * w * DIM;
    const rope = rotary(h, w);
    let tokens = patches.map(patch => {
      const x = new Float32Array(prefixSize + patchSize);
      x.set(this.#tensor("encoder.cls_token", [1, 1, DIM]));
      x.set(this.#tensor("encoder.reg_token", [1, PREFIX - 1, DIM]), DIM);
      x.set(patch.data, prefixSize);
      return x;
    });
    const features = [];
    for (let index = 0; index < 12; index++) {
      if ([5, 7, 9, 11].includes(index)) {
        const joint = new Float32Array(prefixSize + patchSize * tokens.length);
        for (let i = 0; i < prefixSize; i++) {
          joint[i] = tokens.reduce((sum, token) => sum + token[i], 0) / tokens.length;
        }
        tokens.forEach((token, i) => joint.set(token.subarray(prefixSize), prefixSize + i * patchSize));
        const next = this.#block(joint, index, rope);
        tokens.forEach((token, i) => token.set(
          next.subarray(prefixSize + i * patchSize, prefixSize + (i + 1) * patchSize), prefixSize));
      } else {
        tokens = tokens.map(token => this.#block(token, index, rope));
      }
      if ([2, 5, 8, 11].includes(index)) {
        features.push(tokens.map(token => ({ h, w, c: DIM, data: token.slice(prefixSize) })));
      }
      advance("encoder");
    }
    return features;
  }

  #residual(image, name) {
    let out = this.#conv({ ...image, data: relu(image.data) }, name + ".residual.1", 64, 3, 1, 1);
    out = this.#conv({ ...out, data: relu(out.data) }, name + ".residual.3", 64, 3, 1, 1);
    out.data = out.data.map((v, i) => v + image.data[i]);
    return out;
  }

  #decode(features, advance) {
    const channels = [48, 96, 192, 384];
    let result;
    for (let i = 3; i >= 0; i--) {
      const name = `encoder.reassemble.${i}`;
      let level = this.#conv(features[i], name + ".0", channels[i], 1);
      if (i < 2) {
        const scale = 4 >> i;
        level = this.#conv(level, name + ".1", channels[i], scale, scale, 0, true);
      }
      if (i === 3) level = this.#conv(level, name + ".1", channels[i], 3, 2, 1);
      level = this.#conv(level, `decoder.convs.${i}`, 64, 3, 1, 1, false, false);
      const fusion = `decoder.fusions.${i}`;
      if (result) {
        const skip = this.#residual(level, fusion + ".resnet1");
        result.data = result.data.map((v, j) => v + skip.data[j]);
      } else {
        result = level;
      }
      result = this.#residual(result, fusion + ".resnet2");
      result = this.#conv(result, fusion + ".deconv", 64, 2, 2, 0, true, false);
      result = this.#conv(result, fusion + ".out_conv", 64, 1);
      advance("decoder");
    }
    result = this.#conv(result, "head.0", 32, 3, 1, 1);
    advance("head");
    result = this.#conv(result, "head.1", 32, 2, 2, 0, true);
    advance("head");
    result = this.#conv(result, "head.2", 32, 3, 1, 1);
    advance("head");
    result.data = relu(result.data);
    result = this.#conv(result, "head.4", 1, 1);
    result.data = relu(result.data);
    return result;
  }

  predict(rgb, width, height, focalPx, onProgress) {
    return this.predictClip([rgb], width, height, focalPx, onProgress)[0];
  }

  predictClip(frames, width, height, focalPx, onProgress) {
    if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1 ||
        frames.length === 0 || frames.some(rgb => !(rgb instanceof Uint8Array) || rgb.length !== width * height * 3)) {
      throw new Error("Expected one or more packed RGB8 frames of matching positive dimensions");
    }
    if (focalPx !== undefined && !(Number.isFinite(focalPx) && focalPx > 0)) {
      throw new Error("Focal length in pixels must be positive and finite");
    }
    // Preprocessing, patch embedding, 12 transformer blocks, then 8 steps per frame.
    const total = 14 + 8 * frames.length;
    let completed = 0;
    const advance = stage => onProgress?.({ completed: ++completed, total, stage });
    const w = Math.ceil(width / 32) * 32;
    const h = Math.ceil(height / 32) * 32;
    const mean = [0.485, 0.456, 0.406].map(Math.fround);
    const std = [0.229, 0.224, 0.225].map(Math.fround);
    const images = frames.map(rgb => {
      const data = new Float32Array(w * h * 3);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const base = (Math.min(y, height - 1) * width + Math.min(x, width - 1)) * 3;
          for (let c = 0; c < 3; c++) {
            data[(y * w + x) * 3 + c] = Math.fround(Math.fround(rgb[base + c] / 255) - mean[c]) / std[c];
          }
        }
      }
      return { data, w, h, c: 3 };
    });
    advance("preprocess");
    const features = this.#encode(images, advance);
    return frames.map((_, i) => {
      const map = this.#decode(features.map(level => level[i]), advance);
      const inverseDepth = new Float32Array(width * height);
      for (let y = 0; y < height; y++) inverseDepth.set(map.data.subarray(y * w, y * w + width), y * width);
      const depthMeters = focalPx === undefined ? undefined : inverseDepth.map(x => focalPx / (width * x));
      advance("output");
      return { width, height, inverseDepth, depthMeters };
    });
  }
}
