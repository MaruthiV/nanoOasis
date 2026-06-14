// nanoOasis server-fallback engine (Phase 7, W6). Same interface as inference.js (init/reseed/step), but
// the model + referee run on the server (server/ws.py); we just send actions and draw the streamed frames.
// main.js loads this instead of inference.js when the browser has no WebGPU.

// set after `modal deploy server/ws.py` (Modal prints the URL; the WS endpoint is wss://<that-host>/ws)
const SERVER_URL = "wss://REPLACE-WITH-MODAL-URL/ws";

let sock, pending, W, H, lastRgb;

export async function init(base = "assets") {
  const M = await (await fetch(`${base}/manifest.json`)).json();
  W = M.game.img_w; H = M.game.img_h;
  const dev = location.hostname === "localhost" || location.hostname === "127.0.0.1";
  sock = new WebSocket(dev ? `ws://${location.hostname}:8000/ws` : SERVER_URL);
  await new Promise((res, rej) => { sock.onopen = res; sock.onerror = () => rej(new Error("WS connect failed")); });
  sock.onmessage = (e) => { const p = pending; pending = null; if (p) p(JSON.parse(e.data)); };
  return { webgpu: false, mode: "server" };
}

function rpc(msg) {
  return new Promise((res) => { pending = res; sock.send(JSON.stringify(msg)); });
}

// base64 PNG -> RGB Uint8ClampedArray (H*W*3), matching the local engine's frame format so main.draw is shared
async function pngToRgb(b64) {
  const img = new Image();
  img.src = "data:image/png;base64," + b64;
  await img.decode();
  const cx = new OffscreenCanvas(W, H).getContext("2d");
  cx.drawImage(img, 0, 0);
  const d = cx.getImageData(0, 0, W, H).data;
  const rgb = new Uint8ClampedArray(W * H * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2]; }
  return rgb;
}

export async function reseed() {
  const r = await rpc({ type: "retry" });
  lastRgb = await pngToRgb(r.frame);
  return { rgb: lastRgb, action: r.heading ?? 0 };
}

export async function step(a) {
  const t0 = performance.now();
  const r = await rpc({ type: "action", a });
  if (r.dead) return { rgb: lastRgb, dead: true, reason: r.reason, genMs: performance.now() - t0 };
  lastRgb = await pngToRgb(r.frame);
  return { rgb: lastRgb, dead: false, apples: r.apples, length: r.length, genMs: performance.now() - t0 };
}
