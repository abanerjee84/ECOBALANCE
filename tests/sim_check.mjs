// Headless sanity harness: steps the world with random actions and runs one PPO
// rollout + update cycle, checking every numeric array for NaN/Infinity.
// Run: node tests/sim_check.mjs
import { World, OBS_DIM, criticDimFor, ACTIONS, ACTION_NAMES } from '../src/sim/world.js';
import { PPO, ACT_NAMES } from '../src/rl/ppo.js';
import { RolloutBuffer } from '../src/rl/buffer.js';
import { RNG } from '../src/util/rng.js';

const problems = [];
const bad = (name, arr) => {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) { problems.push(`${name}[${i}] = ${arr[i]}`); return; }
  }
};

const N = 16, T = 12;
const world = new World({ seed: 7, population: N });
const ppo = new PPO({ obsDim: OBS_DIM, criticDim: world.criticDim, seed: 3, hidden: 64, minibatch: 32, epochs: 2 });
const buf = new RolloutBuffer(T, N, OBS_DIM, world.criticDim);
const rng = new RNG(11);

console.log(`world: N=${N} obsDim=${OBS_DIM} criticDim=${world.criticDim} actorParams=${ppo.actor.numParams} criticParams=${ppo.critic.numParams}`);

// --- random rollout ---
const holders = Array.from({ length: N }, () => ({ mvx: 0, mvz: 0, act: 0, logp: 0, probs: new Float32Array(5) }));
let rewardSum = 0;
for (let t = 0; t < T; t++) {
  const obs = new Float32Array(N * OBS_DIM);
  const cin = new Float32Array(N * world.criticDim);
  obs.set(world.obs); cin.set(world.critic);
  const mvx = new Float32Array(N), mvz = new Float32Array(N), acts = new Int32Array(N);
  const logp = new Float32Array(N), vals = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const a = ppo.act(obs.subarray(i * OBS_DIM, i * OBS_DIM + OBS_DIM), holders[i]);
    mvx[i] = a.mvx; mvz[i] = a.mvz; acts[i] = a.act; logp[i] = a.logp;
    vals[i] = ppo.value(cin.subarray(i * world.criticDim, (i + 1) * world.criticDim));
  }
  const rewards = world.step(mvx, mvz, acts);
  const dones = world.dones;
  buf.write(t, obs, cin, mvx, mvz, acts, logp, vals, rewards, dones);
  for (let i = 0; i < N; i++) rewardSum += rewards[i];
}
bad('obs', world.obs);
bad('critic', world.critic);
bad('rewards', world.rewards);
bad('parts', world.parts);
bad('buffer.adv (pre-finalize)', buf.adv);

// --- GAE + PPO update ---
const boot = new Float32Array(N);
for (let i = 0; i < N; i++) boot[i] = ppo.value(world.critic.subarray(i * world.criticDim, (i + 1) * world.criticDim));
ppo.startUpdate(buf, boot);
bad('buffer.adv', buf.adv);
bad('buffer.ret', buf.ret);
let chunks = 0;
for (;;) {
  const r = ppo.updateChunk();
  chunks++;
  if (r.done) break;
  if (chunks > 500) { problems.push('updateChunk never finished'); break; }
}
console.log(`chunks=${chunks} updates=${ppo.stats.updates} policyLoss=${ppo.stats.policyLoss.toFixed(4)} valueLoss=${ppo.stats.valueLoss.toFixed(4)} entropy=${ppo.stats.entropy.toFixed(3)} kl=${ppo.stats.approxKl.toFixed(4)} clip=${ppo.stats.clipFrac.toFixed(3)} gradNorm=${ppo.stats.gradNorm.toFixed(3)} nan=${ppo.stats.nan}`);
if (ppo.stats.nan) problems.push('PPO reported NaN');

const s = world.stats();
console.log('stats:', JSON.stringify({
  step: s.step, flora: s.floraFraction.toFixed(3), ore: s.oreFraction.toFixed(3),
  price: s.price.toFixed(3), gini: s.gini.toFixed(3), meanReward: s.meanReward.toFixed(4),
  rewardParts: s.rewardParts.map(v => +v.toFixed(4)), deaths: s.deathsStarve + s.deathsAge,
  trades: s.tradesTotal, shares: s.sharesTotal, edges: s.activeTradeEdges,
}));
console.log('action names:', ACT_NAMES.join(','));
if (!Number.isFinite(s.gini) || !Number.isFinite(s.price)) problems.push('non-finite stats');
if (s.trades === 0) console.log('note: no trades occurred in a 12-step random rollout (not necessarily a bug)');

// --- longer random run to exercise deaths/respawns/market ---
for (let t = 0; t < 600; t++) {
  const mvx = new Float32Array(N), mvz = new Float32Array(N), acts = new Int32Array(N);
  for (let i = 0; i < N; i++) { mvx[i] = rng.gaussian(0, 0.6); mvz[i] = rng.gaussian(0, 0.6); acts[i] = rng.int(5); }
  world.step(mvx, mvz, acts);
}
const s2 = world.stats();
console.log('after 600 random steps:', JSON.stringify({
  step: s2.step, deathsStarve: s2.deathsStarve, deathsAge: s2.deathsAge,
  price: s2.price.toFixed(3), gini: s2.gini.toFixed(3), ore: s2.oreFraction.toFixed(3),
  flora: s2.floraFraction.toFixed(3), trades: s2.tradesTotal, shares: s2.sharesTotal,
  meanEnergy: s2.meanEnergy.toFixed(1), meanReward: s2.meanReward.toFixed(4),
}));
bad('obs@600', world.obs);
bad('critic@600', world.critic);
bad('rewards@600', world.rewards);

if (problems.length) {
  console.error(`\nFAIL (${problems.length}):`);
  for (const p of problems.slice(0, 20)) console.error(' - ' + p);
  process.exit(1);
}
console.log('\nPASS — sim + PPO cycle numerically clean');
