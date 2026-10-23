// Rollout storage + Generalized Advantage Estimation (per agent stream).

export class RolloutBuffer {
  /** @param {number} T rollout length in environment steps @param {number} N population */
  constructor(T, N, obsDim, criticDim) {
    this.T = T; this.N = N; this.obsDim = obsDim; this.criticDim = criticDim;
    const TS = T * N;
    this.obs = new Float32Array(TS * obsDim);
    this.critic = new Float32Array(TS * criticDim);
    this.mvx = new Float32Array(TS);
    this.mvz = new Float32Array(TS);
    this.act = new Int32Array(TS);
    this.logp = new Float32Array(TS);
    this.value = new Float32Array(TS);
    this.reward = new Float32Array(TS);
    this.done = new Float32Array(TS);
    this.adv = new Float32Array(TS);
    this.ret = new Float32Array(TS);
    this.order = new Int32Array(TS);
    for (let i = 0; i < TS; i++) this.order[i] = i;
    this.size = TS;
    this.filled = 0;
    this.advMean = 0;
    this.advStd = 1;
  }

  /** Store one environment step (all N agents at once). */
  write(t, obs, critic, mvx, mvz, act, logp, value, reward, done) {
    const N = this.N, b = t * N;
    this.obs.set(obs, b * this.obsDim);
    this.critic.set(critic, b * this.criticDim);
    this.mvx.set(mvx, b);
    this.mvz.set(mvz, b);
    this.act.set(act, b);
    this.logp.set(logp, b);
    this.value.set(value, b);
    this.reward.set(reward, b);
    this.done.set(done, b);
    this.filled = Math.max(this.filled, (t + 1) * N);
  }

  /**
   * GAE over every agent stream, then advantage normalization.
   * @param {Float32Array} bootstrap V(s_T, o_i) for each agent (bootstrapping value)
   */
  finalize(bootstrap, gamma, lambda) {
    const T = this.T, N = this.N;
    for (let i = 0; i < N; i++) {
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const idx = t * N + i;
        const nonterminal = 1 - this.done[idx];
        const nextV = t === T - 1 ? bootstrap[i] : this.value[idx + N];
        const delta = this.reward[idx] + gamma * nextV * nonterminal - this.value[idx];
        gae = delta + gamma * lambda * nonterminal * gae;
        this.adv[idx] = gae;
        this.ret[idx] = gae + this.value[idx];
      }
    }
    const n = T * N;
    let m = 0;
    for (let i = 0; i < n; i++) m += this.adv[i];
    m /= n;
    let s = 0;
    for (let i = 0; i < n; i++) { const d = this.adv[i] - m; s += d * d; }
    const std = Math.sqrt(s / n);
    this.advMean = m;
    this.advStd = std;
    if (std > 1e-6) {
      for (let i = 0; i < n; i++) this.adv[i] = (this.adv[i] - m) / std;
    } else {
      this.adv.fill(0);
    }
    return { advMean: m, advStd: std };
  }

  /** Shuffle sample order (Fisher–Yates, seeded by caller). */
  shuffle(rand) {
    const a = this.order, n = this.size;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
  }
}
