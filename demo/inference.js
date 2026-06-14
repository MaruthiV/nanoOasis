// nanoOasis browser engine (Phase 7, W4 + referee). The model GENERATES every pixel you see (DiT denoiser
// euler-4 + VAE decoder run the AR loop here). A thin deterministic "referee" tracks head/heading/body and
// adjudicates collisions by the real Snake rule -- because a diffusion model can't reliably render discrete
// termination (a head off-grid has no pixels), so reading death from its output is unreliable, especially
// walls. The referee uses the rule (reliable); the model stays the renderer (D030, option 1).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.bundle.min.mjs";

ort.env.wasm.numThreads = 1;     // single-threaded -> runs under a plain static server (no COOP/COEP)

let M, dit, dec, seeds;          // manifest, sessions, seed latents (flat Float32Array)
let ctx, acts, lastRgb;          // model context (T latents) + action window + last good frame
let head, heading, body, apple, eaten, badHead;   // referee state (head/body cells are [col,row])

const frameLen = () => M.latent.C * M.latent.Hp * M.latent.Wp;
const BAD_HEAD_LIMIT = 5;        // model lost the head this many frames in a row = breakdown (soft safety)

export async function init(base = "assets") {
  M = await (await fetch(`${base}/manifest.json`)).json();
  const opt = { executionProviders: ["webgpu", "wasm"] };
  dit = await ort.InferenceSession.create(`${base}/dit.onnx`, opt);
  dec = await ort.InferenceSession.create(`${base}/vae_dec.onnx`, opt);
  seeds = new Float32Array(await (await fetch(`${base}/${M.seeds.file}`)).arrayBuffer());
  return { webgpu: !!navigator.gpu };
}

function randn(n) {                                   // Box-Muller; inference noise needs no determinism
  const a = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.random() || 1e-9, v = Math.random();
    const r = Math.sqrt(-2 * Math.log(u));
    a[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) a[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return a;
}

// re-seed a clean context + the exact ground-truth referee state (the retry / new-game path).
export async function reseed() {
  const T = M.context_frames, fr = frameLen();
  const k = Math.floor(Math.random() * M.n_seeds);
  ctx = [];
  for (let t = 0; t < T; t++) {
    const off = (k * T + t) * fr;
    ctx.push(seeds.slice(off, off + fr));
  }
  acts = M.seeds.actions[k].slice();
  const st = M.seeds.states[k];
  head = st.head.slice(); heading = st.heading; apple = st.apple.slice();
  body = st.body.map((b) => b.slice());
  eaten = 0; badHead = 0;
  lastRgb = await decode(ctx[T - 1]);
  return { rgb: lastRgb, action: acts[T - 1] };
}

// one tick: the referee decides death/eat by rule FIRST; only if alive do we ask the model to render.
export async function step(a) {
  const t0 = performance.now();
  const GC = M.game.grid_cols, GR = M.game.grid_rows, DIRS = M.game.dirs, REV = M.game.reverse;

  if (a !== REV[heading]) heading = a;                // direct reversal is ignored (classic Snake)
  const d = DIRS[heading];
  const nh = [head[0] + d[0], head[1] + d[1]];

  // wall death -- by rule (the model can't render this, so we must compute it)
  if (nh[0] < 0 || nh[0] >= GC || nh[1] < 0 || nh[1] >= GR)
    return { rgb: lastRgb, dead: true, reason: "wall", genMs: performance.now() - t0 };

  const willEat = apple && nh[0] === apple[0] && nh[1] === apple[1];
  // self death -- by rule. the tail vacates this tick (unless eating), so it isn't an obstacle.
  const trunk = willEat ? body : body.slice(0, body.length - 1);
  if (trunk.some((b) => b[0] === nh[0] && b[1] === nh[1]))
    return { rgb: lastRgb, dead: true, reason: "self", genMs: performance.now() - t0 };

  // alive -> the model renders the frame
  const x = await sample(a);
  const rgb = await decode(x);
  const grid = readGrid(rgb);
  const heads = cellsOf(grid, 2);

  // HEAD-RESYNC: snap the referee head to the model's ACTUALLY-rendered head each tick, so referee/model
  // drift can't accumulate into a false wall/self death -- the head the next tick checks against is the one
  // you SEE. Fall back to the rule position only when the head isn't cleanly readable (D030 reserve lever).
  const newHead = heads.length === 1 ? heads[0] : nh;
  body.unshift(newHead); if (!willEat) body.pop();
  head = newHead; if (willEat) eaten++;
  ctx.shift(); ctx.push(x);
  acts.shift(); acts.push(a);
  lastRgb = rgb;

  // apple tracked from the model so the next eat targets the VISIBLE apple (hold last on a noisy read)
  const ap = cellsOf(grid, 3);
  if (ap.length === 1) apple = ap[0];
  // soft safety: head unreadable for a sustained run = the model broke down -> end
  badHead = heads.length === 1 ? 0 : badHead + 1;
  if (badHead >= BAD_HEAD_LIMIT)
    return { rgb: lastRgb, dead: true, reason: "breakdown", genMs: performance.now() - t0 };

  return { rgb, dead: false, apples: eaten, length: body.length, genMs: performance.now() - t0 };
}

// euler-4 sampler over the model context (infer.sample_next_frame, validated by the ONNX rollout).
async function sample(a) {
  const T = M.context_frames, fr = frameLen(), ctxN = T - 1;
  const { C, Hp, Wp } = M.latent, sched = M.sampler.sigma_schedule, ss = M.sampler.sigma_stab;
  const noisy = new Float32Array(ctxN * fr);
  for (let j = 0; j < ctxN; j++) {
    const src = ctx[j + 1], nz = randn(fr);
    for (let i = 0; i < fr; i++) noisy[j * fr + i] = src[i] + ss * nz[i];
  }
  const actT = new ort.Tensor("int32", Int32Array.from(acts.slice(1).concat([a])), [1, T]);
  let x = randn(fr);
  for (let i = 0; i < fr; i++) x[i] *= sched[0];
  for (let s = 0; s < sched.length - 1; s++) {
    const sc = sched[s], sn = sched[s + 1];
    const xin = new Float32Array(T * fr);
    xin.set(noisy, 0); xin.set(x, ctxN * fr);
    const sig = new Float32Array(T).fill(ss); sig[T - 1] = sc;
    const out = await dit.run({
      x_in: new ort.Tensor("float32", xin, [1, T, C, Hp, Wp]),
      sigma: new ort.Tensor("float32", sig, [1, T]),
      action: actT,
    });
    const x0 = out.x0.data;
    for (let i = 0; i < fr; i++) x[i] += (sn - sc) * ((x[i] - x0[i]) / sc);
  }
  return x;
}

// latent (C,Hp,Wp C-major) -> RGB Uint8ClampedArray (H*W*3). token order row-major Hp-then-Wp (decode_latent)
async function decode(latent) {
  const { C, Hp, Wp, N } = M.latent;
  const z = new Float32Array(N * C);
  for (let hp = 0; hp < Hp; hp++)
    for (let wp = 0; wp < Wp; wp++) {
      const n = hp * Wp + wp;
      for (let c = 0; c < C; c++) z[n * C + c] = latent[c * Hp * Wp + hp * Wp + wp];
    }
  const out = await dec.run({ z: new ort.Tensor("float32", z, [1, N, C]) });
  const img = out.img.data;                            // (1, H, W, 3) in [-1, 1]
  const rgb = new Uint8ClampedArray(img.length);
  for (let i = 0; i < img.length; i++) rgb[i] = (img[i] + 1) * 127.5;
  return rgb;
}

// nearest-color cell readout (infer.read_grid): 0 empty, 1 body, 2 head, 3 apple. used for apple + safety.
function readGrid(rgb) {
  const g = M.game, W = g.img_w, CELL = g.cell, INS = g.sample_inset, SZ = g.sample_size;
  const pal = [g.colors.empty, g.colors.body, g.colors.head, g.colors.apple];
  const grid = new Int8Array(g.grid_rows * g.grid_cols);
  for (let r = 0; r < g.grid_rows; r++)
    for (let c = 0; c < g.grid_cols; c++) {
      let mr = 0, mg = 0, mb = 0, cnt = 0;
      for (let y = r * CELL + INS; y < r * CELL + INS + SZ; y++)
        for (let x = c * CELL + INS; x < c * CELL + INS + SZ; x++) {
          const o = (y * W + x) * 3;
          mr += rgb[o]; mg += rgb[o + 1]; mb += rgb[o + 2]; cnt++;
        }
      mr /= cnt; mg /= cnt; mb /= cnt;
      let best = 0, bd = 1e9;
      for (let k = 0; k < 4; k++) {
        const dd = Math.abs(pal[k][0] - mr) + Math.abs(pal[k][1] - mg) + Math.abs(pal[k][2] - mb);
        if (dd < bd) { bd = dd; best = k; }
      }
      grid[r * g.grid_cols + c] = best;
    }
  return grid;
}

function cellsOf(grid, state) {
  const GC = M.game.grid_cols, GR = M.game.grid_rows, out = [];
  for (let r = 0; r < GR; r++)
    for (let c = 0; c < GC; c++)
      if (grid[r * GC + c] === state) out.push([c, r]);
  return out;
}
