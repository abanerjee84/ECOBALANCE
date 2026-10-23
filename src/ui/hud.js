// HUD controller: metric rows, reward mapping, emergence metrics, time-series
// charts, live configuration panel and the selected-agent inspector.

import { Chart, drawRewardBar, drawActionHistory } from './charts.js';
import { ACTION_NAMES } from '../sim/world.js';

const el = (id) => document.getElementById(id);

const REWARD_COMPONENTS = [
  { key: 'survival', label: 'survival  α·ΔE/Emax', color: '#59e391' },
  { key: 'inventory', label: 'inventory  β·Δ(food+ore·p)', color: '#5ad1ff' },
  { key: 'prosperity', label: 'prosperity  γ·Δstore', color: '#ffcc5c' },
  { key: 'alive', label: 'alive  δ', color: '#9fb3c8' },
  { key: 'death', label: 'death  −ε', color: '#ff6b6b' },
];
const ACTION_COLORS = [0x59e391, 0x5ad1ff, 0xa97bff, 0xffcc5c, 0xff945c]
  .map(c => '#' + c.toString(16).padStart(6, '0'));

const CONFIG_SPEC = [
  { g: 'World', scope: 'world', key: 'population', label: 'population (reset)', min: 20, max: 100, step: 1, reset: true, fmt: v => v.toFixed(0) },
  { g: 'World', scope: 'world', key: 'floraR', label: 'flora regrowth r', min: 0.002, max: 0.08, step: 0.001, fmt: v => v.toFixed(3) },
  { g: 'World', scope: 'world', key: 'metab', label: 'metabolism /step', min: 0.4, max: 3, step: 0.1, fmt: v => v.toFixed(1) },
  { g: 'World', scope: 'world', key: 'foodDecay', label: 'food decay /step', min: 0, max: 0.05, step: 0.002, fmt: v => v.toFixed(3) },
  { g: 'World', scope: 'world', key: 'tradeRange', label: 'trade range', min: 2, max: 16, step: 0.5, fmt: v => v.toFixed(1) },
  { g: 'World', scope: 'world', key: 'priceEta', label: 'price adaptation η', min: 0.01, max: 0.3, step: 0.01, fmt: v => v.toFixed(2) },
  { g: 'Reward', scope: 'world', key: 'wSurvival', label: 'α survival', min: 0, max: 3, step: 0.05, fmt: v => v.toFixed(2) },
  { g: 'Reward', scope: 'world', key: 'wInventory', label: 'β inventory', min: 0, max: 2, step: 0.05, fmt: v => v.toFixed(2) },
  { g: 'Reward', scope: 'world', key: 'wProsperity', label: 'γ prosperity', min: 0, max: 2, step: 0.05, fmt: v => v.toFixed(2) },
  { g: 'Reward', scope: 'world', key: 'wAlive', label: 'δ alive bonus', min: 0, max: 0.1, step: 0.002, fmt: v => v.toFixed(3) },
  { g: 'Reward', scope: 'world', key: 'wDeath', label: 'ε death penalty', min: 0, max: 3, step: 0.05, fmt: v => v.toFixed(2) },
  { g: 'PPO', scope: 'ppo', key: 'lrActor', label: 'actor lr (×1e-4)', min: 0.1, max: 30, step: 0.1, scale: 1e-4, fmt: v => v.toExponential(1) },
  { g: 'PPO', scope: 'ppo', key: 'lrCritic', label: 'critic lr (×1e-4)', min: 0.1, max: 30, step: 0.1, scale: 1e-4, fmt: v => v.toExponential(1) },
  { g: 'PPO', scope: 'ppo', key: 'clip', label: 'PPO clip ε', min: 0.05, max: 0.5, step: 0.01, fmt: v => v.toFixed(2) },
  { g: 'PPO', scope: 'ppo', key: 'gamma', label: 'discount γ', min: 0.9, max: 0.999, step: 0.001, fmt: v => v.toFixed(3) },
  { g: 'PPO', scope: 'ppo', key: 'lambda', label: 'GAE λ', min: 0.8, max: 0.99, step: 0.01, fmt: v => v.toFixed(2) },
  { g: 'PPO', scope: 'ppo', key: 'entCoef', label: 'entropy coef', min: 0, max: 0.05, step: 0.002, fmt: v => v.toFixed(3) },
  { g: 'PPO', scope: 'ppo', key: 'epochs', label: 'epochs / rollout', min: 1, max: 6, step: 1, fmt: v => v.toFixed(0) },
  { g: 'PPO', scope: 'ppo', key: 'minibatch', label: 'minibatch', min: 32, max: 512, step: 32, fmt: v => v.toFixed(0) },
  { g: 'PPO', scope: 'buf', key: 'T', label: 'rollout length T', min: 16, max: 128, step: 8, reset: true, fmt: v => v.toFixed(0) },
];

function buildRows(container, specs) {
  const map = {};
  container.innerHTML = '';
  for (const s of specs) {
    const div = document.createElement('div');
    div.className = 'row';
    const k = document.createElement('span');
    k.className = 'k'; k.textContent = s.k;
    const v = document.createElement('span');
    v.className = 'v'; v.textContent = '—';
    div.appendChild(k); div.appendChild(v);
    container.appendChild(div);
    map[s.id] = v;
  }
  return map;
}

export class HUD {
  constructor() {
    this.cache = {};
    this.status = el('status-line');
    this.rmEq = el('rm-eq');

    this.sim = buildRows(el('sim-rows'), [
      { id: 'step', k: 'Sim step' },
      { id: 'pop', k: 'Population' },
      { id: 'deaths', k: 'Deaths (starve/age)' },
      { id: 'flora', k: 'Flora biomass' },
      { id: 'ore', k: 'Ore remaining' },
      { id: 'energy', k: 'Mean energy' },
      { id: 'food', k: 'Food / store / wealth' },
      { id: 'price', k: 'Ore price (food)' },
      { id: 'ims', k: 'Intents buy / sell' },
      { id: 'fps', k: 'FPS · speed · steps/s' },
    ]);

    this.ppo = buildRows(el('ppo-rows'), [
      { id: 'updates', k: 'PPO updates · samples' },
      { id: 'cadence', k: 'Learn every · batch' },
      { id: 'params', k: 'Network parameters' },
      { id: 'pol', k: 'Policy loss' },
      { id: 'val', k: 'Value loss' },
      { id: 'ent', k: 'Policy entropy' },
      { id: 'kl', k: 'Approx KL · clip frac' },
      { id: 'ev', k: 'Explained variance' },
      { id: 'gn', k: 'Grad norm · batch ms' },
      { id: 'adv', k: 'Advantage mean/std' },
    ]);

    this.rm = buildRows(el('rm-rows'),
      REWARD_COMPONENTS.map((c, i) => ({ id: 'rc' + i, k: c.label })).concat([{ id: 'rct', k: 'total (EMA)' }]));

    this.em = buildRows(el('em-rows'), [
      { id: 'gini', k: 'Wealth Gini' },
      { id: 'top10', k: 'Top-10% wealth share' },
      { id: 'oretop', k: 'Top-10% ore share' },
      { id: 'vol', k: 'Trade volume /step' },
      { id: 'traders', k: 'Traders · active dyads' },
      { id: 'shares', k: 'Share events · reciprocity' },
      { id: 'over', k: 'Overharvest (harvest/regrow)' },
      { id: 'lives', k: 'Mean lifetime (steps)' },
      { id: 'mrew', k: 'Mean reward /step' },
    ]);

    this.aRows = buildRows(el('a-rows'), [
      { id: 'aen', k: 'energy' },
      { id: 'afo', k: 'food / store' },
      { id: 'aor', k: 'ore (worth)' },
      { id: 'age', k: 'age · wealth' },
      { id: 'atr', k: 'trades · shares g/r' },
      { id: 'are', k: 'last reward' },
    ]);
    this.aRew = buildRows(el('a-rew'),
      REWARD_COMPONENTS.map((c, i) => ({ id: 'arc' + i, k: c.label.split('  ')[0] })));

    // action-probability bars
    this.probWrap = el('a-probs');
    this.probWrap.innerHTML = '';
    this.probEls = ACTION_NAMES.map((n, i) => {
      const d = document.createElement('div');
      d.className = 'prob';
      d.innerHTML = `<span class="n"><i class="chip" style="background:${ACTION_COLORS[i]}"></i>${n}</span>
        <span class="b"><i style="background:${ACTION_COLORS[i]}"></i></span>
        <span class="p">0%</span>`;
      this.probWrap.appendChild(d);
      return { fill: d.querySelector('.b > i'), pct: d.querySelector('.p') };
    });

    // charts
    this.charts = {
      mort: new Chart(el('c-mort'), [
        { label: 'starve', color: '#ff6b6b', values: [], kind: 'bars', dp: 2 },
        { label: 'reward', color: '#59e391', values: [], dp: 3 },
      ]),
      res: new Chart(el('c-res'), [
        { label: 'flora', color: '#59e391', values: [], dp: 2 },
        { label: 'ore', color: '#c39a6b', values: [], dp: 2 },
      ]),
      mkt: new Chart(el('c-mkt'), [
        { label: 'price', color: '#5ad1ff', values: [], dp: 2 },
        { label: 'vol', color: '#4a7fb5', values: [], kind: 'bars', dp: 2 },
      ]),
      ineq: new Chart(el('c-ineq'), [
        { label: 'gini', color: '#ffcc5c', values: [], dp: 2 },
        { label: 'top10', color: '#ff6b6b', values: [], dp: 2 },
      ]),
      learn: new Chart(el('c-learn'), [
        { label: 'entropy', color: '#a97bff', values: [], dp: 2 },
        { label: 'vloss', color: '#59e391', values: [], dp: 2 },
      ]),
    };
    this.rewardBar = el('c-reward');
    this.ahist = el('c-ahist');
    this.ppoMode = el('ppo-mode');
    this.rolloutBar = el('rollout-bar');
    this.cfgRows = el('cfg-rows');
    this.agentPanel = el('p-agent');
    this.agentTitle = el('a-title');
    this.selected = -1;

    // reward equation text (weights refreshed live)
    this.refreshEq = (w) => {
      this.rmEq.innerHTML =
        `r<sub>i</sub> = <b style="color:#59e391">${w.wSurvival.toFixed(2)}</b>·ΔE/100` +
        ` + <b style="color:#5ad1ff">${w.wInventory.toFixed(2)}</b>·Δ(food+ore·price)/20` +
        ` + <b style="color:#ffcc5c">${w.wProsperity.toFixed(2)}</b>·Δstore/20` +
        ` + <b style="color:#9fb3c8">${w.wAlive.toFixed(2)}</b>·alive` +
        ` − <b style="color:#ff6b6b">${w.wDeath.toFixed(2)}</b>·death`;
    };
    this.refreshEq({ wSurvival: 1, wInventory: 0.5, wProsperity: 0.5, wAlive: 0.02, wDeath: 1 });
  }

  /* ---------------- configuration panel ---------------- */

  buildConfig(hooks) {
    const wrap = this.cfgRows;
    wrap.innerHTML = '';
    let group = null;
    this.cfgInputs = [];
    for (const spec of CONFIG_SPEC) {
      if (spec.g !== group) {
        group = spec.g;
        const h = document.createElement('div');
        h.style.cssText = 'font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#5ad1ff;margin:7px 0 3px;border-bottom:1px solid rgba(120,170,220,.16);padding-bottom:3px';
        h.textContent = group;
        wrap.appendChild(h);
      }
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:7px;padding:2px 0';
      const lab = document.createElement('label');
      lab.textContent = spec.label;
      lab.style.cssText = 'flex:1;color:#8095ad;font-size:11px';
      const inp = document.createElement('input');
      inp.type = 'range';
      inp.min = spec.min; inp.max = spec.max; inp.step = spec.step;
      const current = hooks.get(spec.scope, spec.key);
      const sliderVal = spec.scale ? (current / spec.scale) : current;
      inp.value = sliderVal;
      const val = document.createElement('span');
      val.style.cssText = 'width:56px;text-align:right;font-variant-numeric:tabular-nums;font-size:11px';
      const show = (v) => { val.textContent = spec.fmt(spec.scale ? v * spec.scale : v); };
      show(sliderVal);
      inp.addEventListener('input', () => {
        const raw = parseFloat(inp.value);
        show(raw);
        hooks.set(spec.scope, spec.key, spec.scale ? raw * spec.scale : raw, spec);
        if (spec.reset) this.markResetNeeded();
      });
      row.appendChild(lab); row.appendChild(inp); row.appendChild(val);
      wrap.appendChild(row);
      this.cfgInputs.push({ spec, inp, show, hooks });
    }
  }

  markResetNeeded() {
    const btn = el('btn-reset');
    btn.classList.add('on');
    btn.textContent = 'Apply + reset world';
    setTimeout(() => { }, 0);
  }

  clearResetNeeded() {
    const btn = el('btn-reset');
    btn.classList.remove('on');
    btn.textContent = 'New world + reset nets';
  }

  /* ---------------- main update ---------------- */

  update(d, now) {
    const s = d.stats, p = d.ppoStats;
    this.set(this.sim, 'step', s.step.toLocaleString());
    this.set(this.sim, 'pop', `${s.population} alive`);
    this.set(this.sim, 'deaths', `${s.deathsStarve} / ${s.deathsAge}`,
      s.deathsStarve > 0 ? 'bad' : 'good');
    this.set(this.sim, 'flora', `${(s.floraFraction * 100).toFixed(1)}%`,
      s.floraFraction > 0.5 ? 'good' : s.floraFraction > 0.25 ? 'warn' : 'bad');
    this.set(this.sim, 'ore', `${(s.oreFraction * 100).toFixed(1)}%`,
      s.oreFraction > 0.5 ? 'good' : s.oreFraction > 0.2 ? 'warn' : 'bad');
    this.set(this.sim, 'energy', s.meanEnergy.toFixed(1),
      s.meanEnergy > 55 ? 'good' : s.meanEnergy > 30 ? 'warn' : 'bad');
    this.set(this.sim, 'food', `${s.meanFood.toFixed(1)} / ${s.meanStore.toFixed(1)} / ${s.meanWealth.toFixed(1)}`);
    this.set(this.sim, 'price', `${s.price.toFixed(3)} food/ore`);
    this.set(this.sim, 'ims', `${s.demand} buy · ${s.supply} sell`);
    this.set(this.sim, 'fps', `${d.fps} · ${d.speed}× · ${d.stepsPerSec}`);

    this.set(this.ppo, 'updates', `${p.updates} · ${p.samples.toLocaleString()}`);
    // buffer cadence: PPO learns once per full rollout, not every step
    this.set(this.ppo, 'cadence', `every ${d.rolloutTotal} steps · ${(d.rolloutTotal * s.population).toLocaleString()} agent-steps`,
      'good');
    this.set(this.ppo, 'params', `${d.paramCount.toLocaleString()} (${d.obsDim}→${d.criticDim} critic)`);
    this.set(this.ppo, 'pol', fmtS(p.policyLoss), p.nan ? 'bad' : '');
    this.set(this.ppo, 'val', fmtS(p.valueLoss), p.nan ? 'bad' : '');
    this.set(this.ppo, 'ent', p.entropy.toFixed(3));
    this.set(this.ppo, 'kl', `${p.approxKl.toFixed(4)} · ${(p.clipFrac * 100).toFixed(1)}%`);
    this.set(this.ppo, 'ev', p.explainedVar.toFixed(3),
      p.explainedVar > 0.3 ? 'good' : p.explainedVar > 0 ? 'warn' : '');
    this.set(this.ppo, 'gn', `${p.gradNorm.toFixed(2)} · ${p.lastBatchMs.toFixed(1)}ms`);
    this.set(this.ppo, 'adv', `${d.advMean.toFixed(3)} / ${d.advStd.toFixed(3)}`);

    // reward mapping
    this.refreshEq(s.params);
    const parts = s.rewardParts;
    for (let i = 0; i < 5; i++) {
      this.set(this.rm, 'rc' + i, (parts[i] >= 0 ? '+' : '') + parts[i].toFixed(4));
    }
    const total = parts.reduce((a, b) => a + b, 0);
    this.set(this.rm, 'rct', (total >= 0 ? '+' : '') + total.toFixed(4), total >= 0 ? 'good' : 'bad');
    drawRewardBar(this.rewardBar, parts, REWARD_COMPONENTS.map(c => c.color));

    // emergence metrics
    this.set(this.em, 'gini', s.gini.toFixed(3), s.gini > 0.6 ? 'bad' : s.gini > 0.4 ? 'warn' : 'good');
    this.set(this.em, 'top10', `${(s.top10Share * 100).toFixed(1)}%`, s.top10Share > 0.7 ? 'bad' : '');
    this.set(this.em, 'oretop', `${(s.oreTop10 * 100).toFixed(1)}%`);
    this.set(this.em, 'vol', `${s.volumeEma.toFixed(2)} (Σ ${s.tradesTotal})`);
    this.set(this.em, 'traders', `${s.uniqueTraders}/${s.population} · ${s.activeTradeEdges}`);
    this.set(this.em, 'shares', `${s.sharesTotal} · ${s.reciprocity.toFixed(2)}×`);
    this.set(this.em, 'over', s.overharvest.toFixed(3),
      s.overharvest > 1.15 ? 'bad' : s.overharvest > 0.9 ? 'warn' : 'good');
    this.set(this.em, 'lives', Number.isFinite(s.meanLifetime) ? s.meanLifetime.toFixed(0) : '—');
    this.set(this.em, 'mrew', (s.meanReward >= 0 ? '+' : '') + s.meanReward.toFixed(4),
      s.meanReward >= 0 ? 'good' : 'bad');

    // status + progress
    this.status.textContent = d.mode === 'update'
      ? `PPO updating ${(d.rollout * 100).toFixed(0)}% · ${p.updates} updates`
      : (d.running ? `learning · rollout ${d.rolloutCount}/${d.rolloutTotal} · ${d.speed}×` : 'paused');
    this.ppoMode.textContent = d.mode === 'update' ? 'optimising' : (d.running ? 'collecting' : 'paused');
    this.ppoMode.classList.toggle('live', d.mode === 'update');
    this.rolloutBar.style.width = `${(d.rollout * 100).toFixed(1)}%`;

    // charts (throttled by caller)
    if (now - (this.lastChart || 0) > 700) {
      this.lastChart = now;
      const h = d.history;
      push(this.charts.mort.series[0].values, h.starve);
      push(this.charts.mort.series[1].values, h.reward);
      push(this.charts.res.series[0].values, h.flora);
      push(this.charts.res.series[1].values, h.ore);
      push(this.charts.mkt.series[0].values, h.price);
      push(this.charts.mkt.series[1].values, h.vol);
      push(this.charts.ineq.series[0].values, h.gini);
      push(this.charts.ineq.series[1].values, h.top10);
      push(this.charts.learn.series[0].values, d.entropyHistory);
      push(this.charts.learn.series[1].values, d.valueHistory);
      for (const c of Object.values(this.charts)) c.draw();
      el('lg-mort').innerHTML = this.charts.mort.legend();
      el('lg-res').innerHTML = this.charts.res.legend();
      el('lg-mkt').innerHTML = this.charts.mkt.legend();
      el('lg-ineq').innerHTML = this.charts.ineq.legend();
      el('lg-learn').innerHTML = this.charts.learn.legend();
    }

    // plain-English narrative (throttled — string work, not needed every tick)
    if (now - (this.lastStory || 0) > 1200) {
      this.lastStory = now;
      this.updateStory(d, s);
    }

    if (this.selected >= 0 && d.agent) this.updateAgent(d.agent, s);
  }

  /* ---------------- plain-English "what's happening" ---------------- */

  updateStory(d, s) {
    const H = d.history, p = d.ppoStats;
    const L = [];
    const n = H.starve.length;

    // 1) learning / mortality — compare early vs recent starvation
    if (s.step < 400) {
      L.push(['', `<b>Early days</b> — the brain knows nothing yet; expect mistakes and deaths (${s.deathsStarve} so far).`]);
    } else if (n >= 20) {
      const win = Math.min(30, Math.floor(n / 2));
      const mean = (arr, from, to) => {
        let t = 0, k = 0;
        for (let i = from; i < to; i++) { t += arr[i]; k++; }
        return k ? t / k : 0;
      };
      const early = mean(H.starve, 0, win) * 10;   // deaths per 100 steps
      const recent = mean(H.starve, n - win, n) * 10;
      if (recent < Math.max(0.5, early * 0.3)) {
        L.push(['good', `<b>Learning is working</b> — starvation fell from ${early.toFixed(1)} to ${recent.toFixed(1)} deaths / 100 steps over the run.`]);
      } else if (recent > 5) {
        L.push(['bad', `<b>Agents are starving</b> — ${recent.toFixed(1)} deaths / 100 steps. The policy hasn't cracked survival yet (${p.updates} updates in).`]);
      } else {
        L.push(['warn', `<b>Survival is mixed</b> — ${recent.toFixed(1)} deaths / 100 steps recently (was ${early.toFixed(1)} early on).`]);
      }
    }

    // 2) the commons
    if (s.overharvest > 1.1) {
      L.push(['bad', `<b>Overharvesting</b> — plants are cut ${s.overharvest.toFixed(2)}× faster than they regrow; the commons has fallen to ${(s.floraFraction * 100).toFixed(0)}%.`]);
    } else if (s.floraFraction > 0.45) {
      L.push(['good', `<b>The commons is healthy</b> — plant cover ${(s.floraFraction * 100).toFixed(0)}%, harvest roughly matches regrowth.`]);
    } else {
      L.push(['warn', `<b>The commons is thinned</b> — plant cover ${(s.floraFraction * 100).toFixed(0)}%, harvest ${s.overharvest.toFixed(2)}× regrowth; watch whether it recovers.`]);
    }

    // 3) the market
    if (s.volumeEma >= 0.05) {
      L.push(['good', `<b>Markets are active</b> — ~${s.volumeEma.toFixed(2)} deals/step at ${s.price.toFixed(2)} food per ore.`]);
    } else if (s.demand + s.supply > 0) {
      L.push(['', `<b>Market is quiet</b> — ${s.demand} buy / ${s.supply} sell intents this step, but few matches; price ${s.price.toFixed(2)}. Trade is a scarcity phenomenon: it picks up when food gets tight.`]);
    } else {
      L.push(['', `<b>Market is idle</b> — no buy/sell intents this step; with food plentiful there is little to gain from trading.`]);
    }

    // 4) cooperation & inequality
    const bits = [];
    if (s.sharesTotal > 0) bits.push(`${s.sharesTotal} food gifts, reciprocity ${s.reciprocity.toFixed(2)}×`);
    bits.push(s.gini < 0.35 ? `wealth spread fairly evenly (Gini ${s.gini.toFixed(2)})`
      : s.gini < 0.55 ? `inequality building (Gini ${s.gini.toFixed(2)})`
      : `high inequality (Gini ${s.gini.toFixed(2)}, top 10% own ${(s.top10Share * 100).toFixed(0)}%)`);
    L.push([s.gini > 0.55 ? 'warn' : '', `<b>Between agents</b> — ${bits.join('; ')}.`]);

    const html = L.map(([c, t]) => `<div class="sline ${c}">${t}</div>`).join('');
    if (this.storyHTML !== html) { this.storyHTML = html; el('story-lines').innerHTML = html; }
  }

  /* ---------------- selected-agent inspector ---------------- */

  setSelected(i) {
    this.selected = i;
    this.agentPanel.style.display = i >= 0 ? 'block' : 'none';
  }

  updateAgent(a, s) {
    this.agentTitle.textContent = `Agent #${a.index} — action distribution`;
    const r = this.aRows;
    this.set(r, 'aen', `${a.energy.toFixed(1)} / ${s && a.Emax ? a.Emax : 100}`);
    this.set(r, 'afo', `${a.food.toFixed(1)} / ${a.store.toFixed(1)}`);
    this.set(r, 'aor', `${a.ore.toFixed(1)} (≈${(a.ore * a.price).toFixed(1)} food)`);
    this.set(r, 'age', `${a.age.toFixed(0)} · ${a.wealth.toFixed(1)}`);
    this.set(r, 'atr', `${a.trades} · ${a.given.toFixed(1)}/${a.received.toFixed(1)}`);
    this.set(r, 'are', (a.lastReward >= 0 ? '+' : '') + a.lastReward.toFixed(4),
      a.lastReward >= 0 ? 'good' : 'bad');

    for (let i = 0; i < 5; i++) {
      this.set(this.aRew, 'arc' + i, (a.parts[i] >= 0 ? '+' : '') + a.parts[i].toFixed(4));
      const pct = Math.max(0, Math.min(1, a.probs[i])) * 100;
      this.probEls[i].fill.style.width = pct.toFixed(1) + '%';
      this.probEls[i].pct.textContent = pct.toFixed(0) + '%';
    }
    drawActionHistory(this.ahist, a.history, ACTION_COLORS);
  }

  /* ---------------- helpers ---------------- */

  set(map, id, text, cls) {
    const e = map[id];
    if (!e) return;
    if (e.__t !== text) { e.__t = text; e.textContent = text; }
    const c = cls || '';
    if (e.__c !== c) { e.__c = c; e.className = 'v' + (c ? ' ' + c : ''); }
  }
}

function push(arr, values) {
  // charts receive the full history array each refresh; replace contents in place
  arr.length = 0;
  for (const v of values) arr.push(v);
  if (arr.length > 900) arr.splice(0, arr.length - 900);
}

function fmtS(v) {
  if (!Number.isFinite(v)) return 'NaN';
  return (v >= 0 ? '+' : '') + v.toFixed(4);
}
