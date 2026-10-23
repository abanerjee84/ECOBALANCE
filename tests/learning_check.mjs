// Headless learning check: run PPO for N updates and report the learning curve,
// benchmarked against a uniform-random policy run for the SAME number of steps
// in an identically-seeded world.
//
// Pass criteria (evidence of learning):
//   * no NaN / non-finite statistics anywhere
//   * final-window mortality < 60% of the random baseline, OR
//     final-window mean reward > random baseline + 5e-4
//
// Usage: node tests/learning_check.mjs [updates] [seed]
// Env:   N (population, default 60), T (rollout length, default 32)
import { World, OBS_DIM, criticDimFor } from '../src/sim/world.js';
import { PPO } from '../src/rl/ppo.js';
import { RolloutBuffer } from '../src/rl/buffer.js';

const TOTAL = Number(process.argv[2] || 80);
const SEED = Number(process.argv[3] || 7);
const N = Number(process.env.N || 60);
const T = Number(process.env.T || 32);
const STEPS = TOTAL * T;                    // env steps the trained policy visits
const WIN0 = Math.floor(STEPS * 0.6);       // window over which we score both runs

/* ---------------- random baseline: same seed, same step budget ---------------- */
function runRandomBaseline(steps, seed, win0) {
  const w = new World({ seed, population: N });
  const mvx = new Float32Array(N), mvz = new Float32Array(N), acts = new Int32Array(N);
  let deaths = 0, rewSum = 0, n = 0;
  for (let t = 0; t < steps; t++) {
    for (let i = 0; i < N; i++) {
      mvx[i] = w.rng.range(-1, 1); mvz[i] = w.rng.range(-1, 1); acts[i] = w.rng.int(5);
    }
    const before = w.deathsStarve;
    w.step(mvx, mvz, acts);
    if (t >= win0) { deaths += w.deathsStarve - before; rewSum += w.meanRewardOverStep; n++; }
  }
  const s = w.stats();
  return { mortality: deaths / Math.max(1, n), reward: rewSum / Math.max(1, n),
           finalFlora: s.floraFraction, finalEnergy: s.meanEnergy };
}

console.log(`random baseline: ${STEPS} steps, window [${WIN0}, ${STEPS}) ...`);
const base = runRandomBaseline(STEPS, SEED, WIN0);
console.log(`  mortality ${base.mortality.toFixed(4)}/step   reward ${base.reward.toFixed(4)}   flora ${(base.finalFlora * 100).toFixed(1)}%   E ${base.finalEnergy.toFixed(1)}`);

/* ------------------------------ trained run ------------------------------ */
const world = new World({ seed: SEED, population: N });
const ppo = new PPO({ obsDim: OBS_DIM, criticDim: world.criticDim, seed: SEED + 101, hidden: 128 });
const buf = new RolloutBuffer(T, N, OBS_DIM, world.criticDim);

const holders = Array.from({ length: N }, () => ({ mvx: 0, mvz: 0, act: 0, logp: 0, probs: new Float32Array(5) }));
const obsSnap = new Float32Array(N * OBS_DIM);
const cinSnap = new Float32Array(N * world.criticDim);
const mvx = new Float32Array(N), mvz = new Float32Array(N);
const acts = new Int32Array(N), logp = new Float32Array(N);
const vals = new Float32Array(N), boot = new Float32Array(N);

let winDeaths = 0, winRewSum = 0, winN = 0;

function collect() {
  obsSnap.set(world.obs);
  cinSnap.set(world.critic);
  for (let i = 0; i < N; i++) {
    const h = ppo.act(obsSnap.subarray(i * OBS_DIM, (i + 1) * OBS_DIM), holders[i]);
    mvx[i] = h.mvx; mvz[i] = h.mvz; acts[i] = h.act; logp[i] = h.logp;
    vals[i] = ppo.value(cinSnap.subarray(i * world.criticDim, (i + 1) * world.criticDim));
  }
  const before = world.deathsStarve;
  world.step(mvx, mvz, acts);
  if (world.step_ > WIN0) { winDeaths += world.deathsStarve - before; winRewSum += world.meanRewardOverStep; winN++; }
  buf.write(t, obsSnap, cinSnap, mvx, mvz, acts, logp, vals, world.rewards, world.dones);
  t++;
  if (t >= T) {
    for (let i = 0; i < N; i++) boot[i] = ppo.value(world.critic.subarray(i * world.criticDim, (i + 1) * world.criticDim));
    ppo.startUpdate(buf, boot);
    t = 0;
  }
}

const finishUpdate = () => { while (ppo.updating) ppo.updateChunk(); };

let t = 0;
const t0 = Date.now();
const REPORT = Math.max(10, Math.round(TOTAL / 4));

for (let u = 0; u < TOTAL; u++) {
  while (!ppo.updating) collect();
  finishUpdate();
  if ((u + 1) % REPORT === 0 || u + 1 === TOTAL) {
    const s = world.stats();
    console.log(JSON.stringify({
      updates: ppo.stats.updates,
      steps: s.step,
      reward: +s.meanReward.toFixed(4),
      energy: +s.meanEnergy.toFixed(1),
      food: +s.meanFood.toFixed(2),
      store: +s.meanStore.toFixed(2),
      starve: s.deathsStarve,
      gini: +s.gini.toFixed(2),
      price: +s.price.toFixed(2),
      trades: s.tradesTotal,
      shares: s.sharesTotal,
      flora: +(s.floraFraction * 100).toFixed(1),
      ore: +(s.oreFraction * 100).toFixed(1),
      ent: +ppo.stats.entropy.toFixed(3),
      vloss: +ppo.stats.valueLoss.toFixed(4),
      pol: +ppo.stats.policyLoss.toFixed(4),
      ev: +ppo.stats.explainedVar.toFixed(2),
      nan: ppo.stats.nan,
    }));
  }
}

const final = world.stats();
const elapsed = (Date.now() - t0) / 1000;
const mortTrained = winDeaths / Math.max(1, winN);
const rewTrained = winRewSum / Math.max(1, winN);

console.log(`\n${TOTAL} updates (${world.step_} env steps) in ${elapsed.toFixed(1)}s  ->  ${(TOTAL / elapsed).toFixed(2)} updates/s`);
console.log(`window [${WIN0}, ${world.step_}]:`);
console.log(`  random    mortality ${base.mortality.toFixed(4)}/step  reward ${base.reward.toFixed(4)}`);
console.log(`  trained   mortality ${mortTrained.toFixed(4)}/step  reward ${rewTrained.toFixed(4)}`);
console.log(`  final: reward ${final.meanReward.toFixed(4)}  E ${final.meanEnergy.toFixed(1)}  food ${final.meanFood.toFixed(2)}  store ${final.meanStore.toFixed(2)}  price ${final.price.toFixed(2)}  trades ${final.tradesTotal}  shares ${final.sharesTotal}  gini ${final.gini.toFixed(2)}  flora ${(final.floraFraction * 100).toFixed(1)}%  ore ${(final.oreFraction * 100).toFixed(1)}%  ent ${ppo.stats.entropy.toFixed(3)}  ev ${ppo.stats.explainedVar.toFixed(2)}`);

const problems = [];
if (ppo.stats.nan) problems.push('PPO reported NaN metrics');
for (const [k, v] of Object.entries(final)) {
  if (typeof v === 'number' && !Number.isFinite(v)) problems.push(`stats.${k} = ${v}`);
}

const mortRatio = mortTrained / Math.max(1e-6, base.mortality);
const rewDelta = rewTrained - base.reward;
console.log(`\nmortality ratio (trained/random): ${mortRatio.toFixed(3)}   reward delta: ${rewDelta.toFixed(5)}`);

const mortImproved = mortTrained < base.mortality * 0.6;
const rewardImproved = rewDelta > 5e-4;
if (!mortImproved && !rewardImproved) {
  problems.push(`no evidence of learning vs random baseline (mortality ${base.mortality.toFixed(4)}->${mortTrained.toFixed(4)}, reward Δ ${rewDelta.toFixed(5)})`);
}
if (!mortImproved && !rewardImproved) {
  console.log('  hint: the trained policy must beat a random walk on survival or reward');
}

if (problems.length) {
  console.error('\nFAIL:'); for (const p of problems) console.error(' - ' + p);
  process.exit(1);
}
console.log('\nPASS — learning check (trained policy beats the random baseline)');
