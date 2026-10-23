// --------------------------------------------------------------------------------
// Emergent Economies — bootstrap & orchestration.
// Wires  World (economy)  →  PPO (parameter-shared actor + centralized critic)
//        →  Renderer3D (Three.js)  →  HUD (metrics, reward mapping, charts)
// --------------------------------------------------------------------------------

import { World, OBS_DIM, criticDimFor } from './sim/world.js';
import { PPO } from './rl/ppo.js';
import { RolloutBuffer } from './rl/buffer.js';
import { Renderer3D } from './render/scene.js';
import { HUD } from './ui/hud.js';

const el = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const num = (k, d) => (params.has(k) ? Number(params.get(k)) : d);

const CFG = {
  population: Math.max(20, Math.min(100, num('pop', 60))),
  seed: num('seed', 42) | 0,
  T: Math.max(16, num('T', 32) | 0),
  speed: [1, 2, 4, 8].includes(num('speed', 4)) ? num('speed', 4) : 4,
  hidden: num('hidden', 128) | 0,
  floraR: Math.max(0.002, Math.min(0.08, num('floraR', 0.05))),
  fx: ['low', 'high', 'auto'].includes(params.get('fx')) ? params.get('fx') : 'auto',
  render: params.get('render') !== '0',
  autostart: params.get('start') !== '0',
};

// ------------------------------ state ------------------------------
let world, ppo, buffer, hud, r3d;
let running = CFG.autostart;
let speed = CFG.speed;
let t = 0;                       // rollout cursor (environment steps)
let frames = 0, stepsThisSec = 0, fps = 60, stepsPerSec = 0;
let secMark = performance.now();
let lastFrame = performance.now();
let hudMark = 0;
let needsRebuffer = false;
let selectedAgent = -1;
let entropyHist = [], valueHist = [];
let obsSnap, criticSnap, mvx, mvz, acts, logp, vals, boot, holders;

function fatal(err) {
  console.error(err);
  el('loading').classList.add('hidden');
  const f = el('fatal');
  f.style.display = 'flex';
  f.textContent = 'Runtime error:\n' + (err && err.stack ? err.stack : String(err));
}
window.addEventListener('error', (e) => fatal(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fatal(e.reason));

// ------------------------------ construction ------------------------------

function buildPPO() {
  ppo = new PPO({
    obsDim: OBS_DIM,
    criticDim: criticDimFor(world.population),
    hidden: CFG.hidden,
    seed: (CFG.seed + 101) | 0,
  });
}

function buildBuffers() {
  buffer = new RolloutBuffer(CFG.T, world.population, OBS_DIM, world.criticDim);
  const N = world.population;
  obsSnap = new Float32Array(N * OBS_DIM);
  criticSnap = new Float32Array(N * world.criticDim);
  mvx = new Float32Array(N); mvz = new Float32Array(N);
  acts = new Int32Array(N); logp = new Float32Array(N);
  vals = new Float32Array(N); boot = new Float32Array(N);
  holders = Array.from({ length: N }, () => ({ mvx: 0, mvz: 0, act: 0, logp: 0, probs: new Float32Array(5) }));
  t = 0;
}

function resetWorld({ seed = CFG.seed, newNets = false } = {}) {
  const prevN = world.population;
  world.reset(seed);                       // honours world.cfg.population (config slider)
  const needNets = newNets || !ppo || prevN !== world.population;
  if (needNets) buildPPO();
  buildBuffers();
  if (r3d) { r3d.setWorld(world); r3d.setSelected(-1); }
  hud.setSelected(-1);
  selectedAgent = -1;
  hud.clearResetNeeded();
}

function init() {
  world = new World({ population: CFG.population, seed: CFG.seed, floraR: CFG.floraR });
  buildPPO();
  buildBuffers();

  if (CFG.render) {
    r3d = new Renderer3D(el('app'), world, { fx: CFG.fx });
    r3d.onSelect = (idx) => {
      selectedAgent = idx;
      hud.setSelected(idx);
    };
  }

  hud = new HUD();
  hud.buildConfig({
    get: (scope, key) => (scope === 'world' ? world.cfg[key] : scope === 'ppo' ? ppo.cfg[key] : CFG[key]),
    set: (scope, key, v, spec) => {
      if (scope === 'world') world.cfg[key] = v;
      else if (scope === 'ppo') ppo.cfg[key] = v;
      else if (scope === 'buf') { CFG[key] = v; needsRebuffer = true; }
    },
  });

  wireButtons();
  wireKeys();

  el('loading').classList.add('hidden');

  window.__EE = {
    get world() { return world; },
    get ppo() { return ppo; },
    get buffer() { return buffer; },
    get config() { return CFG; },
    get running() { return running; },
    set running(v) { setRunning(v); },
    get speed() { return speed; },
    set speed(v) { setSpeed(v); },
    stats: () => snapshot(),
    /** Run `n` PPO updates as fast as possible (used by the automated tests). */
    fastForward: (nUpdates, maxSteps = 400000) => {
      const start = ppo.stats.updates;
      let steps = 0;
      while (ppo.stats.updates < start + nUpdates && steps < maxSteps) {
        if (ppo.updating) ppo.updateChunk();
        else { collectStep(); steps++; }
      }
      return { updates: ppo.stats.updates - start, steps };
    },
    reset: (opts = {}) => resetWorld(opts),
    setPopulation: (n) => { world.cfg.population = Math.max(20, Math.min(100, n | 0)); resetWorld({}); },
    /** Viewport-pixel position of agent i — used by the automated tests to click precisely. */
    screenPosOfAgent: (i) => (r3d ? r3d.screenPosOfAgent(i) : null),
  };

  requestAnimationFrame(frame);
}

function snapshot() {
  const s = world.stats();
  return {
    step: s.step,
    population: s.population,
    updates: ppo.stats.updates,
    samples: ppo.stats.samples,
    policyLoss: ppo.stats.policyLoss,
    valueLoss: ppo.stats.valueLoss,
    entropy: ppo.stats.entropy,
    approxKl: ppo.stats.approxKl,
    clipFrac: ppo.stats.clipFrac,
    explainedVar: ppo.stats.explainedVar,
    nan: ppo.stats.nan,
    meanReward: s.meanReward,
    meanEnergy: s.meanEnergy,
    deathsStarve: s.deathsStarve,
    deathsAge: s.deathsAge,
    gini: s.gini,
    top10Share: s.top10Share,
    price: s.price,
    tradesTotal: s.tradesTotal,
    sharesTotal: s.sharesTotal,
    volumeEma: s.volumeEma,
    uniqueTraders: s.uniqueTraders,
    activeTradeEdges: s.activeTradeEdges,
    floraFraction: s.floraFraction,
    oreFraction: s.oreFraction,
    overharvest: s.overharvest,
    rewardParts: s.rewardParts,
    running, speed, T: CFG.T,
  };
}

// ------------------------------ one environment step ------------------------------

function collectStep() {
  obsSnap.set(world.obs);
  criticSnap.set(world.critic);
  const N = world.population;

  for (let i = 0; i < N; i++) {
    const o = obsSnap.subarray(i * OBS_DIM, (i + 1) * OBS_DIM);
    const c = criticSnap.subarray(i * world.criticDim, (i + 1) * world.criticDim);
    const h = ppo.act(o, holders[i]);
    mvx[i] = h.mvx; mvz[i] = h.mvz; acts[i] = h.act; logp[i] = h.logp;
    vals[i] = ppo.value(c);
  }

  world.step(mvx, mvz, acts);
  buffer.write(t, obsSnap, criticSnap, mvx, mvz, acts, logp, vals, world.rewards, world.dones);
  if (r3d) r3d.pushEvents(world.events);
  stepsThisSec++;
  t++;

  if (t >= CFG.T) {
    for (let i = 0; i < N; i++) {
      boot[i] = ppo.value(world.critic.subarray(i * world.criticDim, (i + 1) * world.criticDim));
    }
    ppo.startUpdate(buffer, boot);   // finalizes GAE + shuffles internally
    t = 0;
  }
}

// ------------------------------ main loop ------------------------------

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;
  frames++;

  if (needsRebuffer) { needsRebuffer = false; buildBuffers(); }

  try {
    if (running) {
      if (ppo.updating) {
        const chunks = speed >= 8 ? 3 : speed >= 4 ? 2 : 1;
        for (let c = 0; c < chunks && ppo.updating; c++) {
          ppo.updateChunk();
          if (!ppo.updating) {
            entropyHist.push(ppo.stats.entropy);
            valueHist.push(ppo.stats.valueLoss);
            if (entropyHist.length > 900) entropyHist.shift();
            if (valueHist.length > 900) valueHist.shift();
          }
        }
      } else {
        for (let k = 0; k < speed; k++) collectStep();
      }
    }

    if (r3d) {
      r3d.update(dt, now / 1000);
      r3d.render();
    }
  } catch (err) {
    running = false;
    fatal(err);
  }

  if (now - secMark >= 1000) {
    fps = Math.round((frames * 1000) / (now - secMark));
    stepsPerSec = Math.round((stepsThisSec * 1000) / (now - secMark));
    frames = 0; stepsThisSec = 0; secMark = now;
  }

  if (now - hudMark > 120) {
    hudMark = now;
    try { hudUpdate(now); } catch (err) { running = false; fatal(err); }
  }
}

function hudUpdate(now) {
  const s = world.stats();
  const agent = selectedAgent >= 0 ? buildAgentInfo(selectedAgent) : null;
  hud.update({
    stats: s,
    ppoStats: ppo.stats,
    fps, speed, stepsPerSec,
    running,
    mode: ppo.updating ? 'update' : 'collect',
    rollout: ppo.updating ? ppo.stats.progress : t / CFG.T,
    rolloutCount: t,
    rolloutTotal: CFG.T,
    advMean: buffer.advMean || 0,
    advStd: buffer.advStd || 1,
    paramCount: ppo.paramCount,
    obsDim: OBS_DIM,
    criticDim: world.criticDim,
    history: world.history,
    entropyHistory: entropyHist,
    valueHistory: valueHist,
    agent,
  }, now);
}

function buildAgentInfo(i) {
  const w = world;
  const probs = holders[i] ? holders[i].probs : null;
  const hist = [];
  for (let k = 0; k < w.histLen; k++) {
    const idx = (w.histCursor + k) % w.histLen;
    hist.push(w.actionHist[i * w.histLen + idx]);
  }
  const o = i * 5;
  return {
    index: i,
    energy: w.energy[i], food: w.food[i], ore: w.ore[i], store: w.store[i],
    age: w.age[i], wealth: w.wealth[i], price: w.price, Emax: w.cfg.EMAX,
    lastReward: w.lastReward[i],
    trades: w.tradeCount[i], given: w.given[i], received: w.received[i],
    parts: [w.parts[o], w.parts[o + 1], w.parts[o + 2], w.parts[o + 3], w.parts[o + 4]],
    probs: probs ? Array.from(probs) : [0, 0, 0, 0, 0],
    history: hist,
  };
}

// ------------------------------ controls ------------------------------

function setRunning(v) {
  running = v;
  const b = el('btn-pause');
  b.textContent = v ? 'Pause' : 'Resume';
  b.classList.toggle('on', !v);
}

function setSpeed(v) {
  speed = v;
  document.querySelectorAll('[data-sp]').forEach(b => b.classList.toggle('on', Number(b.dataset.sp) === v));
}

function wireButtons() {
  el('btn-pause').addEventListener('click', () => setRunning(!running));
  document.querySelectorAll('[data-sp]').forEach(b => {
    b.addEventListener('click', () => setSpeed(Number(b.dataset.sp)));
  });
  el('btn-cfg').addEventListener('click', () => {
    const p = el('p-config');
    p.style.display = p.style.display === 'block' ? 'none' : 'block';
    el('btn-cfg').classList.toggle('on', p.style.display === 'block');
  });
  el('btn-cam').addEventListener('click', () => {
    if (!r3d) return;
    const on = !r3d.follow;
    r3d.setFollow(on && selectedAgent >= 0);
    el('btn-cam').classList.toggle('on', r3d.follow);
  });
  el('btn-guide').addEventListener('click', toggleGuide);
  el('btn-reset').addEventListener('click', () => resetWorld({ seed: CFG.seed, newNets: true }));
  el('btn-rand').addEventListener('click', () => {
    CFG.seed = (Math.random() * 1e9) | 0;
    resetWorld({ seed: CFG.seed, newNets: false });
  });
}

function toggleGuide() {
  const g = el('p-guide');
  const on = g.style.display !== 'block';
  g.style.display = on ? 'block' : 'none';
  el('btn-guide').classList.toggle('on', on);
}

function wireKeys() {
  window.addEventListener('keydown', (e) => {
    if (e.target && /input|textarea/i.test(e.target.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); setRunning(!running); }
    else if (e.key === '1') setSpeed(1);
    else if (e.key === '2') setSpeed(2);
    else if (e.key === '3') setSpeed(4);
    else if (e.key === '4') setSpeed(8);
    else if (e.key === '?' || e.key === '/' || e.key === 'h' || e.key === 'H') toggleGuide();
    else if (e.key === 'Escape') {
      // close the guide first, then the agent inspector
      if (el('p-guide').style.display === 'block') { toggleGuide(); return; }
      selectedAgent = -1;
      if (r3d) { r3d.setSelected(-1); r3d.setFollow(false); }
      hud.setSelected(-1);
      el('btn-cam').classList.remove('on');
    }
  });
}

init();
