// Minimal, dependency-free MLP with hand-written backprop + Adam.
// Used for both the parameter-shared actor and the centralized critic.
//
// Gradients are accumulated per sample and scaled by 1/batchSize at
// backward() time, so step() can be called once per minibatch.

import { mulberry32 } from '../util/rng.js';

export const LN_2PI = 1.8378770664093453; // ln(2*pi)

function makeCache(sizes) {
  const a = [], z = [];
  for (let l = 0; l < sizes.length; l++) a.push(new Float32Array(sizes[l]));
  for (let l = 0; l < sizes.length - 1; l++) z.push(new Float32Array(sizes[l + 1]));
  return { a, z };
}

export class MLP {
  /**
   * @param {number[]} sizes  e.g. [in, h1, h2, out]
   * @param {{activation?:'tanh'|'relu', seed?:number}} opts
   */
  constructor(sizes, opts = {}) {
    this.sizes = sizes.slice();
    this.actName = opts.activation || 'tanh';
    this.params = [];
    this.grads = [];
    const rand = mulberry32((opts.seed ?? 1) >>> 0);

    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const scale = Math.sqrt(2.0 / (nin + nout)); // Glorot/Xavier-ish
      const W = new Float32Array(nout * nin);
      for (let i = 0; i < W.length; i++) W[i] = (rand() * 2 - 1) * scale;
      this.params.push({ W, b: new Float32Array(nout) });
      this.grads.push({ W: new Float32Array(nout * nin), b: new Float32Array(nout) });
    }

    // Adam moments
    this.m = this.params.map(p => ({ W: new Float32Array(p.W.length), b: new Float32Array(p.b.length) }));
    this.v = this.params.map(p => ({ W: new Float32Array(p.W.length), b: new Float32Array(p.b.length) }));
    this.t = 0;

    const maxN = Math.max(...sizes);
    this._dz = new Float32Array(maxN);   // dLoss/dz for the current layer
    this._g0 = new Float32Array(maxN);   // ping-pong buffers for dLoss/da
    this._g1 = new Float32Array(maxN);

    this.cache = makeCache(sizes);
    this.numParams = this.params.reduce((s, p) => s + p.W.length + p.b.length, 0);
  }

  _activate(name, z, out, n) {
    if (name === 'relu') {
      for (let i = 0; i < n; i++) out[i] = z[i] > 0 ? z[i] : 0;
    } else { // tanh
      for (let i = 0; i < n; i++) out[i] = Math.tanh(z[i]);
    }
  }

  /** Forward pass; result is written into cache.a[last] and returned (a view). */
  forward(x) {
    const c = this.cache;
    c.a[0].set(x);
    const L = this.params.length;
    for (let l = 0; l < L; l++) {
      const { W, b } = this.params[l];
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const prev = c.a[l], z = c.z[l];
      for (let o = 0; o < nout; o++) {
        const row = o * nin;
        let s = b[o];
        for (let i = 0; i < nin; i++) s += W[row + i] * prev[i];
        z[o] = s;
      }
      if (l === L - 1) c.a[l + 1].set(z); // linear output layer
      else this._activate(this.actName, z, c.a[l + 1], nout);
    }
    return c.a[L];
  }

  /**
   * Backward pass: consume grad w.r.t. the network output, accumulate parameter
   * gradients (scaled by `scale`, typically 1/batchSize) and write grad w.r.t. input.
   * NOTE: `scale` is applied exactly once — at the output layer. Every deeper layer
   * receives the already-scaled gradient through `cur`, otherwise it would be
   * multiplied by scale once per layer.
   */
  backward(gradOut, scale, gradIn) {
    const c = this.cache;
    const L = this.params.length;
    const dz = this._dz;
    let cur = gradOut;              // dLoss/da_{l+1} (scaled at the output layer)
    let buf = this._g0;             // scratch for dLoss/da_l
    for (let l = L - 1; l >= 0; l--) {
      const { W, b } = this.params[l];
      const gW = this.grads[l].W, gb = this.grads[l].b;
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const prev = c.a[l];

      if (l === L - 1) {
        for (let o = 0; o < nout; o++) dz[o] = cur[o] * scale;
      } else if (this.actName === 'relu') {
        const z = c.z[l];
        for (let o = 0; o < nout; o++) dz[o] = z[o] > 0 ? cur[o] : 0;
      } else { // tanh' = 1 - tanh(z)^2  (cur already carries the batch scale)
        const a = c.a[l + 1];
        for (let o = 0; o < nout; o++) dz[o] = cur[o] * (1 - a[o] * a[o]);
      }

      for (let o = 0; o < nout; o++) {
        const row = o * nin, d = dz[o];
        gb[o] += d;
        for (let i = 0; i < nin; i++) gW[row + i] += d * prev[i];
      }

      const dst = l === 0 ? gradIn : buf;
      if (dst) {
        for (let i = 0; i < nin; i++) {
          let s = 0;
          for (let o = 0; o < nout; o++) s += W[o * nin + i] * dz[o];
          dst[i] = s;
        }
      }
      if (l > 0) {
        cur = dst;
        buf = buf === this._g0 ? this._g1 : this._g0;
      }
    }
    return gradIn;
  }

  zeroGrad() {
    for (const g of this.grads) { g.W.fill(0); g.b.fill(0); }
  }

  gradNorm() {
    let s = 0;
    for (const g of this.grads) {
      for (let i = 0; i < g.W.length; i++) s += g.W[i] * g.W[i];
      for (let i = 0; i < g.b.length; i++) s += g.b[i] * g.b[i];
    }
    return Math.sqrt(s);
  }

  /** Adam step with global-norm gradient clipping. */
  step(lr, clipNorm = 0.5, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    const norm = this.gradNorm();
    const scale = clipNorm > 0 && norm > clipNorm ? clipNorm / (norm + 1e-8) : 1;
    this.t++;
    const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
    for (let l = 0; l < this.params.length; l++) {
      const p = this.params[l], g = this.grads[l], m = this.m[l], v = this.v[l];
      this._adamInto(p.W, g.W, m.W, v.W, lr, scale, b1, b2, eps, bc1, bc2);
      this._adamInto(p.b, g.b, m.b, v.b, lr, scale, b1, b2, eps, bc1, bc2);
    }
    return norm;
  }

  _adamInto(p, g, m, v, lr, scale, b1, b2, eps, bc1, bc2) {
    for (let i = 0; i < p.length; i++) {
      const gi = g[i] * scale;
      if (gi === 0 && m[i] === 0 && v[i] === 0) continue;
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      const mh = m[i] / bc1, vh = v[i] / bc2;
      p[i] -= lr * mh / (Math.sqrt(vh) + eps);
    }
  }

  /** Forgetting of gradient state (not parameters). */
  resetGrad() { this.zeroGrad(); }
}

/**
 * Small helper for the state-independent Gaussian policy std.
 * logStd is optimised with the same Adam moments pattern.
 */
export class LogStd {
  constructor(dim, init = -0.4) {
    this.p = new Float32Array(dim).fill(init);
    this.g = new Float32Array(dim);
    this.m = new Float32Array(dim);
    this.v = new Float32Array(dim);
    this.t = 0;
    this.dim = dim;
  }
  step(lr, clipNorm = 0.5, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    let n = 0;
    for (let i = 0; i < this.dim; i++) n += this.g[i] * this.g[i];
    n = Math.sqrt(n);
    const scale = clipNorm > 0 && n > clipNorm ? clipNorm / (n + 1e-8) : 1;
    this.t++;
    const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
    for (let i = 0; i < this.dim; i++) {
      const gi = this.g[i] * scale;
      this.m[i] = b1 * this.m[i] + (1 - b1) * gi;
      this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi;
      this.p[i] -= lr * (this.m[i] / bc1) / (Math.sqrt(this.v[i] / bc2) + eps);
      // keep std in a sane range: sigma in [0.03, 3]
      if (this.p[i] > 1.1) this.p[i] = 1.1;
      if (this.p[i] < -3.5) this.p[i] = -3.5;
    }
  }
  zeroGrad() { this.g.fill(0); }
}
