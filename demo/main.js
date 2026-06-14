// nanoOasis demo loop (Phase 7, W5): input + canvas + the ~4/s tick + the game-over/RETRY flow.
// Engine is chosen at boot: WebGPU -> local in-browser inference; otherwise -> the WebSocket server (W6).
let engine;

async function pickEngine() {
  const forceServer = new URLSearchParams(location.search).has("server");   // ?server -> test the fallback
  if (!forceServer && navigator.gpu) {
    try { if (await navigator.gpu.requestAdapter()) return await import("./inference.js"); } catch {}
  }
  return await import("./remote.js");                 // server fallback (no WebGPU) -- doesn't load ORT
}

const SCALE = 3, TICK_MS = 250;                       // ~4 cells/s (the play-test cadence, SNAKE_DESIGN)
const KEY = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3,
              w: 0, s: 1, a: 2, d: 3 };

const view = document.getElementById("view");
const v2d = view.getContext("2d");
const overlay = document.getElementById("overlay");
const hud = { prov: document.getElementById("prov"), gen: document.getElementById("gen"),
              fps: document.getElementById("fps"), score: document.getElementById("score"),
              best: document.getElementById("best") };

let off, o2d, imgData, W, H;
let running = false, lastAction = 0, pending = null;
let score = 0, best = 0, genAvg = 0, lastTick = 0;

function draw(rgb) {
  const px = imgData.data;
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    px[j] = rgb[i]; px[j + 1] = rgb[i + 1]; px[j + 2] = rgb[i + 2]; px[j + 3] = 255;
  }
  o2d.putImageData(imgData, 0, 0);
  v2d.imageSmoothingEnabled = false;                  // crisp blocks, no interpolation
  v2d.drawImage(off, 0, 0, W, H, 0, 0, view.width, view.height);
}

async function tick() {
  if (!running) return;
  const a = pending != null ? pending : lastAction;
  pending = null; lastAction = a;

  const r = await engine.step(a);
  genAvg = genAvg ? genAvg * 0.8 + r.genMs * 0.2 : r.genMs;
  if (r.dead) { running = false; gameOver(r.reason); return; }

  draw(r.rgb);
  score = r.apples;
  best = Math.max(best, score);
  const now = performance.now();
  hud.gen.textContent = genAvg.toFixed(0);
  hud.fps.textContent = lastTick ? (1000 / (now - lastTick)).toFixed(1) : "—";
  hud.score.textContent = score; hud.best.textContent = best;
  lastTick = now;

  setTimeout(tick, Math.max(0, TICK_MS - r.genMs));   // don't overlap async steps
}

function gameOver(reason) {
  document.getElementById("reason").textContent = reason ? `ended: ${reason}` : "";
  console.log("[nanoOasis] game over —", reason);
  overlay.classList.add("show");
}

async function newGame() {
  overlay.classList.remove("show");
  const { rgb, action } = await engine.reseed();
  draw(rgb);
  lastAction = action; pending = null; score = 0; lastTick = 0;
  running = true;
  setTimeout(tick, TICK_MS);
}

addEventListener("keydown", (e) => {
  if (e.key in KEY) { pending = KEY[e.key]; e.preventDefault(); }
  else if ((e.key === " " || e.key === "Enter") && !running) { e.preventDefault(); newGame(); }
});
overlay.addEventListener("click", () => { if (!running) newGame(); });

(async function boot() {
  try {
    engine = await pickEngine();
    const cfg = await engine.init("assets");
    const m = await (await fetch("assets/manifest.json")).json();
    W = m.game.img_w; H = m.game.img_h;
    view.width = W * SCALE; view.height = H * SCALE;
    off = document.createElement("canvas"); off.width = W; off.height = H;
    o2d = off.getContext("2d"); imgData = o2d.createImageData(W, H);
    hud.prov.textContent = cfg.mode === "server" ? "server (WS)" : (cfg.webgpu ? "WebGPU" : "WASM");
    document.getElementById("loading").classList.remove("show");
    newGame();
  } catch (e) {
    console.error(e);
    document.getElementById("loading").textContent = "couldn't start — see console";
  }
})();
