import { readFileSync, writeFileSync } from "node:fs";
import { DepthPro2S } from "./depthpro2.mjs";

try {
  const [weights, file, widthText, heightText, output, focalText, ...extra] = process.argv.slice(2);
  if (!output || extra.length) {
    throw new Error("Usage: node predict.mjs WEIGHTS INPUT.rgb WIDTH HEIGHT OUTPUT.f32 [FOCAL_PX]");
  }
  const [width, height] = [Number(widthText), Number(heightText)];
  const frameBytes = width * height * 3;
  const input = readFileSync(file);
  if (!Number.isSafeInteger(frameBytes) || width <= 0 || height <= 0 || !input.length || input.length % frameBytes) {
    throw new Error("INPUT.rgb must contain whole packed RGB8 frames");
  }
  const frames = [];
  for (let offset = 0; offset < input.length; offset += frameBytes) {
    frames.push(input.subarray(offset, offset + frameBytes));
  }
  const focal = focalText === undefined ? undefined : Number(focalText);
  const model = new DepthPro2S(readFileSync(weights));
  const maps = model.predictClip(frames, width, height, focal);
  const bytes = Buffer.alloc(width * height * 4 * maps.length);
  let offset = 0;
  for (const map of maps) {
    for (const value of map.depthMeters ?? map.inverseDepth) {
      bytes.writeFloatLE(value, offset);
      offset += 4;
    }
  }
  writeFileSync(output, bytes);
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
