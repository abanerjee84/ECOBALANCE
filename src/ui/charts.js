// Small canvas chart helpers (no dependencies, sized for the HUD panels).

function fit(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || canvas.width;
  const h = canvas.clientHeight || canvas.height;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/**
 * Multi-series line chart with automatic per-series scaling.
 * series: [{label, color, values:number[], kind?:'line'|'bars'}]
 */
export class Chart {
  constructor(canvas, series) {
    this.canvas = canvas;
    this.series = series;
    this.maxPoints = 240;
  }

  legend() {
    return this.series.map(s => {
      const v = s.values && s.values.length ? s.values[s.values.length - 1] : NaN;
      const txt = Number.isFinite(v) ? fmt(v, s.dp ?? 3) : '–';
      return `<span style="color:${s.color}">■</span> ${s.label} <b style="color:#d7e3f2">${txt}</b>`;
    }).join(' &nbsp; ');
  }

  draw() {
    const { ctx, w, h } = fit(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const padL = 4, padR = 34, padT = 5, padB = 4;
    const iw = w - padL - padR, ih = h - padT - padB;
    if (iw <= 4 || ih <= 4) return;

    // collect bounds (per-series normalization)
    let n = 0;
    for (const s of this.series) n = Math.max(n, s.values ? s.values.length : 0);
    if (n < 2) {
      ctx.fillStyle = '#5c6f86';
      ctx.font = '10px Segoe UI';
      ctx.fillText('collecting…', padL + 2, h / 2);
      return;
    }
    const take = Math.min(n, this.maxPoints);
    const bounds = this.series.map(s => {
      const arr = s.values.slice(n - take);
      let mn = Infinity, mx = -Infinity;
      for (const v of arr) { if (!Number.isFinite(v)) continue; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (!Number.isFinite(mn)) { mn = 0; mx = 1; }
      if (mn === mx) { mn -= 0.5; mx += 0.5; }
      if (s.zeroBase !== false && mn > 0) mn = 0;
      const pad = (mx - mn) * 0.08;
      return { mn: mn - (mn > 0 ? pad : 0), mx: mx + pad };
    });

    // grid
    ctx.strokeStyle = 'rgba(120,170,220,0.12)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = padT + (ih * i) / 3;
      ctx.beginPath(); ctx.moveTo(padL, y + 0.5); ctx.lineTo(padL + iw, y + 0.5); ctx.stroke();
    }

    this.series.forEach((s, si) => {
      const arr = s.values.slice(n - take);
      const { mn, mx } = bounds[si];
      const yOf = (v) => padT + ih - ((v - mn) / (mx - mn)) * ih;
      const xOf = (i) => padL + (i / Math.max(1, arr.length - 1)) * iw;

      if (s.kind === 'bars') {
        ctx.fillStyle = s.color;
        const bw = Math.max(1, iw / arr.length - 1);
        for (let i = 0; i < arr.length; i++) {
          const v = Number.isFinite(arr[i]) ? arr[i] : 0;
          const y = yOf(Math.max(v, mn));
          const y0 = yOf(Math.max(mn, 0));
          ctx.globalAlpha = 0.55;
          ctx.fillRect(xOf(i) - bw / 2, Math.min(y, y0), bw, Math.max(1, Math.abs(y0 - y)));
        }
        ctx.globalAlpha = 1;
      } else {
        // area fill under the line (series can opt out with fill:false)
        if (s.fill !== false) {
          ctx.beginPath();
          let started = false;
          for (let i = 0; i < arr.length; i++) {
            const v = Number.isFinite(arr[i]) ? arr[i] : mn;
            const x = xOf(i), y = yOf(v);
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
          }
          ctx.lineTo(xOf(arr.length - 1), padT + ih);
          ctx.lineTo(xOf(0), padT + ih);
          ctx.closePath();
          const grad = ctx.createLinearGradient(0, padT, 0, padT + ih);
          grad.addColorStop(0, hexA(s.color, 0.26));
          grad.addColorStop(1, hexA(s.color, 0));
          ctx.fillStyle = grad;
          ctx.fill();
        }
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.4;
        ctx.shadowColor = hexA(s.color, 0.55);
        ctx.shadowBlur = 4;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < arr.length; i++) {
          const v = Number.isFinite(arr[i]) ? arr[i] : mn;
          const x = xOf(i), y = yOf(v);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // right-hand range labels
      ctx.fillStyle = s.color;
      ctx.font = '9px Consolas, monospace';
      ctx.fillText(fmt(mx, s.dp ?? 2), padL + iw + 3, padT + 7);
      ctx.fillStyle = 'rgba(150,170,195,0.75)';
      ctx.fillText(fmt(mn, s.dp ?? 2), padL + iw + 3, padT + ih);
    });
  }
}

/** Horizontal stacked reward-contribution bar with a zero baseline. */
export function drawRewardBar(canvas, values, colors) {
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  let pos = 0, neg = 0;
  for (const v of values) { if (v >= 0) pos += v; else neg -= v; }
  const span = Math.max(pos, neg, 1e-6);
  const cx = w * (neg / (pos + neg || 1));
  ctx.fillStyle = 'rgba(120,170,220,0.35)';
  ctx.fillRect(cx - 0.5, 0, 1, h);
  let x = cx;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const wpx = (Math.abs(v) / span) * (v >= 0 ? w - cx : cx);
    if (wpx <= 0) continue;
    ctx.fillStyle = colors[i];
    ctx.fillRect(v >= 0 ? x : x - wpx, 4, Math.max(wpx, 0.5), h - 8);
    if (v >= 0) x += wpx; else x -= wpx;
  }
}

/** Per-step action-history strip for the selected agent. */
export function drawActionHistory(canvas, actions, colors) {
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  const n = actions.length;
  if (!n) return;
  const bw = w / n;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = colors[actions[i]] || '#555';
    ctx.fillRect(i * bw + 0.5, 2, Math.max(bw - 1, 1), h - 4);
  }
}

/** '#rrggbb' + alpha → 'rgba(r,g,b,a)' (for canvas gradients/shadows). */
function hexA(hex, a) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function fmt(v, dp = 3) {
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(dp);
  return v.toFixed(dp + 1);
}
