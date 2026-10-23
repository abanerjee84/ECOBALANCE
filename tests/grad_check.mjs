// Finite-difference check of every analytic gradient in ppo.js updateChunk().
//
// The scalar objective being checked (mirrors the intended PPO objective, averaged
// over the whole batch, exactly as updateChunk scales by 1/B):
//
//   L = mean_s [ -min( A·ratio , A·clip(ratio) ) ]        policy (PPO-clip)
//     + vfCoef · mean_s [ 0.5·(V - R)^2 ]                  value
//     - entCoef · mean_s [ H_categorical + logσ0 + logσ1 ] entropy bonus
//
// We run one updateChunk() with the Adam steps stubbed out, read the accumulated
// analytic gradients, then central-difference each parameter. Any sign error in
// the policy/entropy/logStd gradients shows up immediately.
//
// Run: node tests/grad_check.mjs
import { World, OBS_DIM, criticDimFor } from '../src/sim/world.js';
import { PPO } from '../src/rl/ppo.js';
import { RolloutBuffer } from '../src/rl/buffer.js';
import { RNG } from '../src/util/rng.js';

const N = 4, T = 4;                 // 16 samples in the batch
const EPS = 1e-4;                   // central-difference step
const world = new World({ seed: 5, population: N });
const ppo = new PPO({
  obsDim: OBS_DIM, criticDim: world.criticDim, hidden: 8, seed: 3,
  minibatch: 16, epochs: 1, entCoef: 0.03, vfCoef: 0.5, clip: 0.2,
  lrActor: 3e-4, lrCritic: 1e-3,
});
const c = ppo.cfg;
const buf = new RolloutBuffer(T, N, OBS_DIM, world.criticDim);
const rng = new RNG(11);

// ---- fill the buffer from a real short rollout (realistic obs/actions/rewards) ----
const holders = Array.from({ length: N }, () => ({ mvx: 0, mvz: 0, act: 0, logp: 0, probs: new Float32Array(5) }));
const obsSnap = new Float32Array(N * OBS_DIM);
const cinSnap = new Float32Array(N * world.criticDim);
const mvx = new Float32Array(N), mvz = new Float32Array(N);
const acts = new Int32Array(N), logp = new Float32Array(N);
const vals = new Float32Array(N), boot = new Float32Array(N);
let t = 0;
while (!ppo.updating) {
  obsSnap.set(world.obs);
  cinSnap.set(world.critic);
  for (let i = 0; i < N; i++) {
    const h = ppo.act(obsSnap.subarray(i * OBS_DIM, (i + 1) * OBS_DIM), holders[i]);
    mvx[i] = h.mvx; mvz[i] = h.mvz; acts[i] = h.act; logp[i] = h.logp;
    vals[i] = ppo.value(cinSnap.subarray(i * world.criticDim, (i + 1) * world.criticDim));
  }
  world.step(mvx, mvz, acts);
  buf.write(t, obsSnap, cinSnap, mvx, mvz, acts, logp, vals, world.rewards, world.dones);
  t++;
  if (t >= T) {
    for (let i = 0; i < N; i++) boot[i] = ppo.value(world.critic.subarray(i * world.criticDim, (i + 1) * world.criticDim));
    ppo.startUpdate(buf, boot);
  }
}
const B = buf.size;

// ---- run the update with the optimisers stubbed out so gradients accumulate ----
ppo.actor.step = () => 0;
ppo.critic.step = () => 0;
ppo.logStd.step = () => 0;
const r = ppo.updateChunk();
if (!r.done) throw new Error('expected the single minibatch to cover the whole batch');

// ---- collect analytic gradients into flat arrays ----
function flattenParams(net) {
  const parts = [];
  for (const p of net.params) { parts.push(p.W); parts.push(p.b); }
  return parts;
}
function flattenGrads(net) {
  const parts = [];
  for (const g of net.grads) { parts.push(g.W); parts.push(g.b); }
  return parts;
}
const groups = [
  { name: 'actor', p: flattenParams(ppo.actor), g: flattenGrads(ppo.actor) },
  { name: 'critic', p: flattenParams(ppo.critic), g: flattenGrads(ppo.critic) },
  { name: 'logStd', p: [ppo.logStd.p], g: [ppo.logStd.g] },
];
const analytic = [];
for (const gr of groups) for (const arr of gr.g) analytic.push(...arr);
const totalParams = analytic.length;

// ---- scalar loss, computed independently from the same buffers ----
function scalarLoss() {
  let L = 0;
  const eps = c.clip;
  for (let s = 0; s < buf.size; s++) {
    const i = buf.order[s];
    const obs = buf.obs.subarray(i * c.obsDim, i * c.obsDim + c.obsDim);
    const cin = buf.critic.subarray(i * c.criticDim, i * c.criticDim + c.criticDim);

    const o = ppo.actor.forward(obs);
    const mu0 = c.moveMax * Math.tanh(o[0]);
    const mu1 = c.moveMax * Math.tanh(o[1]);
    const s0 = Math.exp(ppo.logStd.p[0]);
    const s1 = Math.exp(ppo.logStd.p[1]);
    const a0 = buf.mvx[i], a1 = buf.mvz[i], act = buf.act[i];
    const z0 = (a0 - mu0) / s0, z1 = (a1 - mu1) / s1;

    let mx = -Infinity, sum = 0;
    const pr = [];
    for (let k = 0; k < c.nActions; k++) { pr[k] = o[2 + k]; if (pr[k] > mx) mx = pr[k]; }
    for (let k = 0; k < c.nActions; k++) { pr[k] = Math.exp(pr[k] - mx); sum += pr[k]; }
    const logSumExp = mx + Math.log(sum);
    for (let k = 0; k < c.nActions; k++) pr[k] /= sum;

    const logpNew = -0.5 * (z0 * z0 + z1 * z1) - Math.log(s0) - Math.log(s1) - 1.8378770664093453
      + o[2 + act] - logSumExp;
    const ratio = Math.exp(logpNew - buf.logp[i]);
    const A = buf.adv[i];
    const clipped = Math.max(1 - eps, Math.min(1 + eps, ratio));
    L += -Math.min(ratio * A, clipped * A);

    const v = ppo.critic.forward(cin)[0];
    L += c.vfCoef * 0.5 * (v - buf.ret[i]) * (v - buf.ret[i]);

    let H = 0;
    for (let k = 0; k < c.nActions; k++) if (pr[k] > 1e-12) H -= pr[k] * Math.log(pr[k]);
    L -= c.entCoef * H;
    L -= c.entCoef * (ppo.logStd.p[0] + ppo.logStd.p[1]);
  }
  return L / buf.size;
}

// ---- central differences ----
const numeric = new Float64Array(totalParams);
let idx = 0;
const problems = [];
for (const gr of groups) {
  for (let a = 0; a < gr.p.length; a++) {
    const arr = gr.p[a];
    for (let j = 0; j < arr.length; j++) {
      const orig = arr[j];
      const e = Math.max(EPS, Math.abs(orig) * 1e-3);
      arr[j] = orig + e; const lp = scalarLoss();
      arr[j] = orig - e; const lm = scalarLoss();
      arr[j] = orig;
      numeric[idx] = (lp - lm) / (2 * e);
      idx++;
    }
  }
}

// ---- compare ----
let worst = 0, worstIdx = -1, nBad = 0;
for (let j = 0; j < totalParams; j++) {
  const a = analytic[j], n = numeric[j];
  const diff = Math.abs(a - n);
  const tol = 2e-3 * (1 + Math.abs(n) + Math.abs(a));
  if (diff > tol) nBad++;
  if (diff / (1 + Math.abs(n)) > worst) { worst = diff / (1 + Math.abs(n)); worstIdx = j; }
}
const rel = (j) => (Math.abs(analytic[j] - numeric[j]) / (1 + Math.abs(numeric[j]))).toFixed(5);
console.log(`params checked: ${totalParams} (batch ${B} samples)`);
console.log(`max |analytic - numeric| / (1+|numeric|) = ${worst.toFixed(6)} at index ${worstIdx}`);
console.log(`mismatches beyond tolerance: ${nBad}`);
if (worstIdx >= 0) console.log(`  analytic=${analytic[worstIdx]} numeric=${numeric[worstIdx]} rel=${rel(worstIdx)}`);

// Show a per-slice breakdown so a sign error is attributable to a specific head.
{
  let i = 0;
  for (const gr of groups) {
    let sliceBad = 0, sliceMax = 0;
    for (const arr of gr.p) {
      for (let j = 0; j < arr.length; j++) {
        const d = Math.abs(analytic[i] - numeric[i]) / (1 + Math.abs(numeric[i]));
        if (d > 2e-3) sliceBad++;
        if (d > sliceMax) sliceMax = d;
        i++;
      }
    }
    console.log(`  ${gr.name.padEnd(7)} max-rel ${sliceMax.toFixed(6)}  bad ${sliceBad}`);
  }
}

// direction check: are the signs, on average, flipped?
let dot = 0, na = 0, nn = 0;
for (let j = 0; j < totalParams; j++) { dot += analytic[j] * numeric[j]; na += analytic[j] ** 2; nn += numeric[j] ** 2; }
const cos = dot / (Math.sqrt(na * nn) + 1e-12);
console.log(`cosine(analytic, numeric) = ${cos.toFixed(4)}  (must be ~+1, a value near -1 means a sign error)`);

if (nBad > 0 || cos < 0.99) {
  console.error(`\nFAIL — ${nBad} gradient(s) disagree with finite differences (cos=${cos.toFixed(4)})`);
  process.exit(1);
}
console.log('\nPASS — analytic gradients match central finite differences');
