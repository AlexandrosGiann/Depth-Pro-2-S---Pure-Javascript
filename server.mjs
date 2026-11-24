import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";

const [host = "127.0.0.1", portText = "8080", checkpoint] = process.argv.slice(2);
const port = Number(portText);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Invalid port");

const modelFile = checkpoint ?? new URL("./depthpro2-s.safetensors", import.meta.url);
await stat(modelFile).catch(() => {
  throw new Error("Save depthpro2-s.safetensors next to server.mjs or pass the checkpoint path as the third argument");
});

const html = [new URL("./index.html", import.meta.url), "text/html; charset=utf-8"];
const routes = new Map([
  ["/", html],
  ["/index.html", html],
  ["/app.mjs", [new URL("./app.mjs", import.meta.url), "text/javascript; charset=utf-8"]],
  ["/worker.mjs", [new URL("./worker.mjs", import.meta.url), "text/javascript; charset=utf-8"]],
  ["/depthpro2.mjs", [new URL("./depthpro2.mjs", import.meta.url), "text/javascript; charset=utf-8"]],
  ["/depthpro2-s.safetensors", [modelFile, "application/octet-stream"]],
]);

createServer(async (req, res) => {
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405, { Allow: "GET, HEAD" }).end();
    return;
  }
  const path = req.url.split("?")[0];
  const entry = routes.get(path);
  if (!entry) {
    res.writeHead(404).end();
    return;
  }
  const [file, type] = entry;
  try {
    const info = await stat(file);
    res.writeHead(200, {
      "Content-Type": type,
      "Content-Length": info.size,
      "Last-Modified": info.mtime.toUTCString(),
      "Cache-Control": path.endsWith(".safetensors") ? "public, max-age=3600" : "no-cache",
      "X-Content-Type-Options": "nosniff",
    });
    if (req.method === "HEAD") res.end();
    else {
      const stream = createReadStream(file);
      stream.on("error", () => res.destroy());
      res.on("close", () => stream.destroy());
      stream.pipe(res);
    }
  } catch {
    res.writeHead(404).end();
  }
}).listen(port, host, () => console.log(`Depth Pro 2 S: http://${host}:${port}`));
