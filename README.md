# Depth Pro 2 S in JavaScript

An unofficial JavaScript implementation of Apple's Depth Pro 2 S. It runs on the CPU in a browser or in Node.js without npm dependencies.

## Web app

You need Node.js and the **97.3 MB** [official F32 SafeTensors weights](https://huggingface.co/apple/DepthPro2-S/resolve/main/model.safetensors). Save the weights as `depthpro2-s.safetensors` next to `server.mjs`, then run this command from the project directory.

```sh
node server.mjs
```

Open [http://127.0.0.1:8080/](http://127.0.0.1:8080/). To make the app available on your local network, replace the example IP below with the host computer's Wi-Fi or Ethernet IPv4 address.

```sh
node server.mjs 192.168.1.20 8080
```

From a device on the same network, open `http://192.168.1.20:8080/` using the same IP you passed to the server. You can pass a different path to the weights as the third argument.

The model starts downloading as soon as the page opens. Select a photo and click «Υπολόγισε βάθος» (Calculate depth) once the model is ready. One progress bar tracks bytes downloaded and the other tracks completed inference stages. The browser processes the photo locally and scales it down when its longest side exceeds 320 pixels. Brighter areas in the inverse depth map indicate closer points.

## Command line

Use FFmpeg to convert a photo to packed RGB8. Replace `640 480` with the actual image dimensions.

```sh
ffmpeg -i photo.jpg -frames:v 1 -pix_fmt rgb24 -f rawvideo photo.rgb
node predict.mjs depthpro2-s.safetensors photo.rgb 640 480 depth.f32
```

The `.rgb` file can contain one frame or a sequence of frames with the same dimensions. The `.f32` output contains little-endian float32 values in row-major order, with frames in input order. Pass the focal length in pixels as the final argument to get depth in meters along the optical axis. Omit it to get normalized inverse depth.

To use the model in your own code, import `DepthPro2S` from `depthpro2.mjs`. Pass the weights as a `Uint8Array` and call `predict` for one frame or `predictClip` for a clip.

## Credits and model license

This port implements the encoder, decoder, and depth head in JavaScript and uses Apple's original checkpoint. Apple's separate focal-length estimator and CLI streaming modes are outside the scope of this port. [Apple source code](https://github.com/apple-aiml-research/ml-depth-pro-2) · [DINOv3 in timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva.py)

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement. The weights are available for non-commercial scientific research and academic development. [Apple model license](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL)
