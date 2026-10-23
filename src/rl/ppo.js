// Parameter-sharing PPO with a centralized critic (MAPPO-style).
//
//  * ONE actor network is shared by every agent (parameter sharing).
//    It outputs: movement intent (2-D Gaussian head) + discrete mode head
//    (gather / trade / share / store / migrate).
//  * The critic is centralized: V(s, o_i) = f([o_i ; o_1..o_N ; global features])
//    so each agent's advantage is computed against the joint economic state.
//  * PPO-clip objective with GAE, per-minibatch Adam steps, global grad clipping.
//
// The update is chunked: call updateChunk() once per animation frame to keep the
// renderer responsive while a large batch is optimised over several epochs.

import { MLP, LogStd, LN_2PI } from './net.js';
import { RNG } from '../util/rng.js';

export const N_ACTIONS = 5;
export const ACT_NAMES = ['gather', 'trade', 'share', 'store', 'migrate'];
export const ACT = { GATHER: 0, TRADE: 1, SHARE: 2, STORE: 3, MIGRATE: 4 };

export class PPO {
  constructor(opts) {
    this.cfg = Object.assign({
      obsDim: 28, criticDim: 284, nActions: N_ACTIONS, moveMax: 1.0,
      hidden: 128, lrActor: 3e-4, lrCritic: 1e-3, clip: 0.2,
      gamma: 0.995, lambda: 0.96, entCoef: 0.012, vfCoef: 0.5,
      epochs: 3, minibatch: 128, maxGradNorm: 0.5, seed: 7,
    }, opts);
    const c = this.cfg;
    this.actor = new MLP([c.obsDim, c.hidden, c.hidden, 2 + c.nActions], { seed: c.seed });
    this.critic = new MLP([c.criticDim, c.hidden, c.hidden, 1], { seed: c.seed + 977 });
    this.logStd = new LogStd(2, -0.7);
    this.rng = new RNG(c.seed + 13);

    this.actorOut = new Float32Array(2 + c.nActions);
    this.probs = new Float32Array(c.nActions);
    this.gradA = new Float32Array(2 + c.nActions);
    this.gradC = new Float32Array(1);

    this.buffer = null;
    this.updating = false;
    this.cursor = 0;
    this.stats = {
      updates: 0, samples: 0, policyLoss: 0, valueLoss: 0, entropy: 0,
      approxKl: 0, clipFrac: 0, explainedVar: 0, gradNorm: 0, lastBatchMs: 0,
      progress: 0, nan: false,
    };
  }

  get paramCount() { return this.actor.numParams + this.critic.numParams + 2; }

  resetRng(seed) { this.rng.reseed(seed + 13); }

  /* ---------------- acting ---------------- */

  /**
   * Sample an action from the shared policy.
   * @param {Float32Array} obs  agent observation (obsDim)
   * @param {object} out  reused per-agent holder {mvx,mvz,act,logp,probs}
   */
  act(obs, out) {
    const c = this.cfg;
    const o = this.actor.forward(obs);
    const mvMax = c.moveMax;
    const mu0 = mvMax * Math.tanh(o[0]);
    const mu1 = mvMax * Math.tanh(o[1]);
    const s0 = Math.exp(this.logStd.p[0]);
    const s1 = Math.exp(this.logStd.p[1]);
    const a0 = mu0 + s0 * this.rng.gaussian();
    const a1 = mu1 + s1 * this.rng.gaussian();

    // softmax over the discrete head
    const probs = this.probs;
    let mx = -Infinity;
    for (let k = 0; k < c.nActions; k++) if (o[2 + k] > mx) mx = o[2 + k];
    let sum = 0;
    for (let k = 0; k < c.nActions; k++) { const e = Math.exp(o[2 + k] - mx); probs[k] = e; sum += e; }
    let logSumExp = mx + Math.log(sum);
    for (let k = 0; k < c.nActions; k++) probs[k] /= sum;

    // Gumbel-max sampling
    let best = 0, bestS = -Infinity;
    for (let k = 0; k < c.nActions; k++) {
      const u = 1 - this.rng.next();
      const g = -Math.log(-Math.log(u));
      const s = o[2 + k] + g;
      if (s > bestS) { bestS = s; best = k; }
    }

    const z0 = (a0 - mu0) / s0, z1 = (a1 - mu1) / s1;
    const logp = -0.5 * (z0 * z0 + z1 * z1) - Math.log(s0) - Math.log(s1) - LN_2PI
      + o[2 + best] - logSumExp;

    out.mvx = a0; out.mvz = a1; out.act = best; out.logp = logp;
    if (out.probs) out.probs.set(probs);
    return out;
  }

  /** Centralized value estimate V(s, o_i). */
  value(criticIn) { return this.critic.forward(criticIn)[0]; }

  /* ---------------- PPO update (chunked) ---------------- */

  startUpdate(buffer, bootstrap) {
    const c = this.cfg;
    buffer.finalize(bootstrap, c.gamma, c.lambda);
    buffer.shuffle(() => this.rng.next());
    this.buffer = buffer;
    this.cursor = 0;
    this.updating = true;

    // explained variance of the value function on this rollout
    const n = buffer.size;
    let sr = 0, sr2 = 0, se = 0, se2 = 0;
    for (let i = 0; i < n; i++) {
      const r = buffer.ret[i], e = buffer.ret[i] - buffer.value[i];
      sr += r; sr2 += r * r; se += e; se2 += e * e;
    }
    const vr = sr2 / n - (sr / n) ** 2;
    const ve = se2 / n - (se / n) ** 2;
    this.stats.explainedVar = vr > 1e-9 ? 1 - ve / vr : 0;
    this._agg = { policyLoss: 0, valueLoss: 0, entropy: 0, approxKl: 0, clipFrac: 0, gradNorm: 0, batches: 0, ms: 0 };
    this._epoch = 0;
    return buffer.size;
  }

  /**
   * Process a single minibatch. Call repeatedly until done === true.
   * @returns {{done:boolean, progress:number, metrics:object}}
   */
  updateChunk() {
    const c = this.cfg, buf = this.buffer;
    if (!buf || !this.updating) return { done: true, progress: 1, metrics: this.stats };

    const t0 = performance.now();
    if (this.cursor >= buf.size) { this.updating = false; return { done: true, progress: 1, metrics: this.stats }; }

    const B = Math.min(c.minibatch, buf.size - this.cursor);
    const scale = 1 / B;
    this.actor.zeroGrad();
    this.critic.zeroGrad();
    this.logStd.zeroGrad();

    const a = this.actorOut, gA = this.gradA, gC = this.gradC, probs = this.probs;
    const nA = c.nActions, mvMax = c.moveMax, eps = c.clip;
    let policyLoss = 0, valueLoss = 0, ent = 0, kl = 0, clipFrac = 0;
    let nan = false;

    for (let s = this.cursor; s < this.cursor + B; s++) {
      const i = buf.order[s];
      const obs = buf.obs.subarray(i * c.obsDim, i * c.obsDim + c.obsDim);
      const cin = buf.critic.subarray(i * c.criticDim, i * c.criticDim + c.criticDim);

      // ---- actor forward ----
      const o = this.actor.forward(obs);
      const mu0 = mvMax * Math.tanh(o[0]);
      const mu1 = mvMax * Math.tanh(o[1]);
      const s0 = Math.exp(this.logStd.p[0]);
      const s1 = Math.exp(this.logStd.p[1]);
      const a0 = buf.mvx[i], a1 = buf.mvz[i], act = buf.act[i];
      const z0 = (a0 - mu0) / s0, z1 = (a1 - mu1) / s1;

      let mx = -Infinity;
      for (let k = 0; k < nA; k++) if (o[2 + k] > mx) mx = o[2 + k];
      let sum = 0;
      for (let k = 0; k < nA; k++) { const e = Math.exp(o[2 + k] - mx); probs[k] = e; sum += e; }
      const logSumExp = mx + Math.log(sum);
      for (let k = 0; k < nA; k++) probs[k] /= sum;

      const logpNew = -0.5 * (z0 * z0 + z1 * z1) - Math.log(s0) - Math.log(s1) - LN_2PI
        + o[2 + act] - logSumExp;
      const logr = logpNew - buf.logp[i];
      const ratio = Math.exp(logr);
      const A = buf.adv[i];

      // ---- PPO-clip surrogate coefficient ----
      let coef = A * ratio;
      if ((ratio > 1 + eps && A > 0) || (ratio < 1 - eps && A < 0)) coef = 0;

      const minSur = Math.min(ratio * A, Math.max(1 - eps, Math.min(1 + eps, ratio)) * A);
      policyLoss += -minSur;
      const r1 = ratio - 1;
      kl += r1 - logr;
      if (Math.abs(ratio - 1) > eps) clipFrac++;

      // ---- entropy (discrete) ----
      let H = 0;
      for (let k = 0; k < nA; k++) { const p = probs[k]; H += p > 1e-12 ? -p * Math.log(p) : 0; }
      ent += H;

      // ---- gradients: mean head (through the tanh squash) ----
      // ---- gradients of the MINIMISED loss L = -surrogate - entCoef·H ----
      // (Adam performs descent, so every gradient here must be dL/dθ.)
      gA.fill(0);
      const d0 = muMaxDeriv(mu0, mvMax), d1 = muMaxDeriv(mu1, mvMax);
      gA[0] = -coef * (z0 / s0) * d0;
      gA[1] = -coef * (z1 / s1) * d1;
      for (let k = 0; k < nA; k++) {
        const p = probs[k];
        const lp = Math.log(p > 1e-12 ? p : 1e-12);
        // d/dlogit of -A·ratio is -coef·(δ-p);  dH/dlogit = -p·(log p + H)
        gA[2 + k] = -coef * ((k === act ? 1 : 0) - p) + c.entCoef * p * (lp + H);
      }
      this.actor.backward(gA, scale, null);

      // ---- log-std:  dlogπ/dlogσ = z² − 1 ;  dH/dlogσ = 1 ----
      this.logStd.g[0] += scale * (-coef * (z0 * z0 - 1) - c.entCoef);
      this.logStd.g[1] += scale * (-coef * (z1 * z1 - 1) - c.entCoef);

      // ---- critic ----
      const v = this.critic.forward(cin)[0];
      const R = buf.ret[i];
      valueLoss += 0.5 * (v - R) * (v - R);
      gC[0] = c.vfCoef * (v - R);
      this.critic.backward(gC, scale, null);

      if (!isFinite(logpNew) || !isFinite(v)) nan = true;
    }

    const actorNorm = this.actor.step(c.lrActor, c.maxGradNorm);
    const criticNorm = this.critic.step(c.lrCritic, c.maxGradNorm);
    this.logStd.step(c.lrActor, c.maxGradNorm);
    const gNorm = Math.max(actorNorm, criticNorm);

    this.cursor += B;
    const agg = this._agg;
    agg.policyLoss += policyLoss / B;
    agg.valueLoss += valueLoss / B;
    agg.entropy += ent / B;
    agg.approxKl += kl / B;
    agg.clipFrac += clipFrac / B;
    agg.gradNorm += gNorm;
    agg.batches++;
    agg.ms += performance.now() - t0;

    const st = this.stats;
    st.policyLoss = agg.policyLoss / agg.batches;
    st.valueLoss = agg.valueLoss / agg.batches;
    st.entropy = agg.entropy / agg.batches;
    st.approxKl = agg.approxKl / agg.batches;
    st.clipFrac = agg.clipFrac / agg.batches;
    st.gradNorm = agg.gradNorm / agg.batches;
    st.lastBatchMs = agg.ms / agg.batches;
    st.progress = this.cursor / buf.size;
    if (nan || !isFinite(st.policyLoss) || !isFinite(st.valueLoss)) { st.nan = true; }

    const done = this.cursor >= buf.size;
    if (done) {
      this.updating = false;
      st.updates++;
      st.samples += buf.size;
      st.progress = 1;
    }
    return { done, progress: st.progress, metrics: st };
  }
}

/** d(mu)/d(raw) for the squashed Gaussian mean: mu = mvMax * tanh(raw). */
function muMaxDeriv(mu, mvMax) {
  const t = mu / mvMax;
  return mvMax * (1 - t * t);
}
