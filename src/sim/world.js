// --------------------------------------------------------------------------------
// World: a continuous 2-D economy on a coarse resource grid.
//
// Resources
//   * Flora  — renewable, logistically regrowing biomass on a patchy grid
//              (harvest > regrowth  =>  common-pool over-exploitation can emerge)
//   * Ore    — non-renewable nodes with finite stock (classic common-pool asset)
//
// Agents (population N, fixed slots; deaths are immediately repopulated so the
// centralized critic keeps a constant input width)
//   energy   metabolism + action costs; food is auto-consumed for energy
//   food     perishable (decays), the numeraire of the economy
//   ore      durable asset, valued at the current market price
//   store    personal stockpile ("bank"): food deposited, no perish decay
//
// Actions (discrete mode, combined with continuous movement intent)
//   GATHER / TRADE / SHARE / STORE / MIGRATE
//
// Market: local bilateral matching (spatial trading networks) with an adaptive
// reference price driven by the buy/sell imbalance of trade intents.
// --------------------------------------------------------------------------------

import { RNG, clamp } from '../util/rng.js';

export const ACTIONS = { GATHER: 0, TRADE: 1, SHARE: 2, STORE: 3, MIGRATE: 4 };
export const ACTION_NAMES = ['gather', 'trade', 'share', 'store', 'migrate'];
export const OBS_DIM = 28;
export const COMPACT_DIM = 4;   // per-agent slice inside the centralized critic input
export const GLOBAL_DIM = 16;

export const criticDimFor = (N) => OBS_DIM + N * COMPACT_DIM + GLOBAL_DIM;

const DEFAULTS = {
  seed: 42,
  population: 60,
  worldSize: 96,
  grid: 48,
  floraSeedFraction: 0.49,
  floraK: 15,
  // ---- carrying-capacity budget (why these numbers hang together) ----
  //   seeded cells      C = 48·48·0.49 ≈ 1128  (seeding honours this exactly)
  //   max regrowth/cell at b = K/2    r·K/4 = 3.75·r
  //   => world's peak food production = 1128 · 3.75 · r = 4230·r  per tick
  //   population metabolic need = N · metab/eatEff = 60·1/4 = 15    per tick
  //   a competent learned policy settles near ~100 harvest/tick
  //   full greed (every agent gathering) = 60 · 1.6 = 96            per tick
  // With r = 0.05 the peak supply is ~212/tick. Verified by tests/carrying_probe:
  // uniform harvest H settles at the logistic equilibrium
  // x* = (1 + √(1 − 4H/(C·r·K)))/2 while H < peak and collapses above it.
  // Real agents harvest *spatially heterogeneously* (they concentrate on rich
  // patches, drain them to local extinction and move on), so the effective
  // equilibrium is much lower than the uniform x* — the learned policy sits at
  // ~21-27% fill, stable but well below the uniform prediction. The tragedy of
  // the commons is reachable by lowering r (at r=0.02 peak supply drops to
  // ~85/tick, below the learned harvest rate) — both claims are asserted in
  // tests/learning_check.mjs and tests/carrying_probe.mjs. The reward is purely
  // individual, so sustainability emerges from learning, not from design.
  floraR: 0.05,
  floraImmigration: 0.003,    // recolonisation: local extinction recoverable in ~150 ticks
  oreNodes: 26,
  oreStock: 420,
  // --- agent physiology / economy ---
  EMAX: 100,
  metab: 1.0,
  eatEff: 4,                  // 1 food = 4 energy => steady-state need 0.25 food/agent/tick
  eatRate: 1.0,
  foodDecay: 0.025,           // perishable inventory: banking (STORE) now has real utility
  storeDecay: 0.002,
  speed: 1.25,
  gatherFoodCost: 1.4,
  gatherFoodYield: 1.6,
  gatherOreCost: 2.6,
  gatherOreYield: 0.5,
  reach: 1.9,
  shareRange: 2.5,
  shareAmount: 1.0,
  shareCost: 0.1,
  storeCost: 0.15,
  tradeCost: 0.35,
  tradeRange: 8.0,
  migrateCost: 2.6,
  migrateDist: 6.5,
  migrateSense: 3.0,
  maxAge: 6000,
  // --- market ---
  priceInit: 1.0,
  priceEta: 0.09,
  priceMin: 0.25,
  priceMax: 12,
  sellFoodTrigger: 6.0,
  buyFoodMargin: 1.0,
  oreCap: 6.0,
  // --- reward weights (see HUD "Reward mapping") ---
  wSurvival: 1.0,
  wInventory: 0.5,
  wProsperity: 0.5,
  wAlive: 0.02,
  wDeath: 1.0,
  refInventory: 20,
};

export class World {
  constructor(cfg = {}) {
    this.cfg = Object.assign({}, DEFAULTS, cfg);
    this.reset(this.cfg.seed);
  }

  reset(seed) {
    const c = this.cfg;
    c.seed = seed >>> 0;
    this.rng = new RNG(c.seed);
    const N = this.population = c.population;
    const G = c.grid;
    const cell = this.cell = c.worldSize / G;
    this.half = c.worldSize / 2;
    this.G = G;

    // ---- flora grid ----
    this.flora = new Float32Array(G * G);
    this.floraSeeded = new Uint8Array(G * G);
    this.totalSeededK = 0;
    this._seedFlora();

    // ---- ore nodes ----
    this.nodes = [];
    const rng = this.rng;
    let guard = 0;
    while (this.nodes.length < c.oreNodes && guard++ < 5000) {
      const x = rng.range(-this.half + 5, this.half - 5);
      const z = rng.range(-this.half + 5, this.half - 5);
      let ok = true;
      for (const n of this.nodes) {
        const dx = n.x - x, dz = n.z - z;
        if (dx * dx + dz * dz < 36) { ok = false; break; }
      }
      if (ok) this.nodes.push({ x, z, stock: c.oreStock, max: c.oreStock });
    }
    this.oreInitial = this.nodes.reduce((s, n) => s + n.max, 0);

    // ---- agent state ----
    const A = (n) => new Float32Array(n);
    this.energy = A(N); this.food = A(N); this.ore = A(N); this.store = A(N);
    this.age = A(N); this.px = A(N); this.pz = A(N);
    this.prevE = A(N); this.prevF = A(N); this.prevO = A(N); this.prevS = A(N);
    this.dE = A(N); this.lastReward = A(N); this.wealth = A(N);
    this.deathPenalty = A(N);
    this.given = A(N); this.received = A(N); this.tradeCount = A(N);
    this.returnAcc = A(N);
    this.lastAction = new Int32Array(N).fill(0);
    this.intent = new Int32Array(N);          // -1 sell ore, +1 buy ore, 0 none
    this.matched = new Int32Array(N);
    this.dones = new Float32Array(N);
    this.rewards = A(N);
    this.parts = A(N * 5);                    // reward decomposition per agent
    this.partsEma = new Float32Array(5);
    this.histLen = 48;
    this.actionHist = new Int32Array(N * this.histLen);
    this.histCursor = 0;

    // ---- buffers ----
    const D = OBS_DIM, C = criticDimFor(N);
    this.criticDim = C;
    this.obs = A(N * D);
    this.critic = A(N * C);
    this.mvx = A(N); this.mvz = A(N); this.acts = new Int32Array(N);
    this.holder = { mvx: 0, mvz: 0, act: 0, logp: 0, probs: new Float32Array(5) };
    this.bootstrap = A(N);

    // ---- market ----
    this.price = c.priceInit;
    this.prevPrice = c.priceInit;
    this.volumeEma = 0;
    this.tradesTotal = 0;
    this.sharesTotal = 0;
    this.deathsStarve = 0;
    this.deathsAge = 0;
    this.starveWin = 0;                       // starvation deaths since last history record
    this.step_ = 0;
    this.meanRewardOverStep = 0;
    this.meanRewardEma = 0;
    this.meanReward = 0;

    // ---- event queues (consumed by the renderer each step) ----
    this.events = { trades: [], shares: [], deaths: [], harvests: [] };

    // ---- rolling statistics ----
    this.harvestFloraEma = 0;
    this.harvestOreEma = 0;
    this.regrowthEma = 0;
    this.deathsEma = 0;
    this.sharesEma = 0;
    this.meanLifetimeEma = 0;
    this.gini = 0;
    this.top10Share = 0;
    this.meanWealth = 0;
    this.meanEnergy = 0;
    this.meanFood = 0;
    this.meanStore = 0;
    this.meanOre = 0;
    this.uniqueTraders = 0;
    this.activeTradeEdges = 0;
    this.tradeEdges = new Map();              // "min:max" -> count (rolling)
    this.reciprocity = 0;
    this.history = {
      t: [], flora: [], ore: [], price: [], vol: [], gini: [], top10: [],
      reward: [], death: [], shares: [], starve: [],
    };
    this.histCounter = 0;
    this.histInterval = 10;
    this.historyMax = 900;

    // initial values
    for (let i = 0; i < N; i++) this._spawn(i, true);
    this._capturePrev();
    this._globals();
    this.computeObs();
  }

  /* ------------------------------ seeding ------------------------------ */

  _seedFlora() {
    const c = this.cfg, G = this.G, rng = this.rng;
    const targetCells = Math.floor(G * G * c.floraSeedFraction);
    const centers = [];
    const nClusters = Math.floor((G * G * c.floraSeedFraction) / 9);
    for (let k = 0; k < nClusters; k++) centers.push({ x: rng.int(G), z: rng.int(G), r: rng.range(2.0, 4.5) });
    for (const cl of centers) {
      const r = cl.r;
      for (let dz = -Math.ceil(r); dz <= Math.ceil(r); dz++) {
        for (let dx = -Math.ceil(r); dx <= Math.ceil(r); dx++) {
          const gx = cl.x + dx, gz = cl.z + dz;
          if (gx < 0 || gz < 0 || gx >= G || gz >= G) continue;
          const d = Math.hypot(dx, dz);
          if (d > r) continue;
          const p = 1 - d / r;
          if (rng.next() < p * 1.4) {
            const idx = gz * G + gx;
            this.floraSeeded[idx] = 1;
            this.flora[idx] = Math.max(this.flora[idx], c.floraK * (0.4 + 0.6 * rng.next()) * p);
          }
        }
      }
    }
    // clusters can overshoot the target area; trim the excess so that
    // floraSeedFraction is EXACTLY honoured (the carrying-capacity budget in
    // DEFAULTS depends on the seeded cell count being what it claims)
    let seeded = 0;
    for (let i = 0; i < G * G; i++) if (this.floraSeeded[i]) seeded++;
    while (seeded > targetCells) {
      const idx = rng.int(G * G);
      if (this.floraSeeded[idx]) { this.floraSeeded[idx] = 0; this.flora[idx] = 0; seeded--; }
    }
    // top up to the target area if the clusters fell short
    let guard = 0;
    while (seeded < targetCells && guard++ < 100000) {
      const idx = rng.int(G * G);
      if (!this.floraSeeded[idx]) {
        this.floraSeeded[idx] = 1;
        this.flora[idx] = c.floraK * rng.range(0.3, 1.0);
        seeded++;
      }
    }
    for (let i = 0; i < G * G; i++) {
      if (this.floraSeeded[i]) this.totalSeededK += c.floraK;
    }
  }

  _spawn(i, initial = false) {
    const c = this.cfg;
    // bias spawn towards productive ground
    let x = 0, z = 0, ok = false;
    for (let k = 0; k < 12 && !ok; k++) {
      x = this.rng.range(-this.half + 3, this.half - 3);
      z = this.rng.range(-this.half + 3, this.half - 3);
      ok = initial || this._cellBiomassAt(x, z) > 0.4;
    }
    this.px[i] = x; this.pz[i] = z;
    this.energy[i] = c.EMAX * (initial ? this.rng.range(0.5, 0.9) : 0.8);
    this.food[i] = initial ? this.rng.range(3, 9) : 4;
    this.ore[i] = 0;
    this.store[i] = 0;
    this.age[i] = 0;
    this.dE[i] = 0;
    this.lastReward[i] = 0;
    this.returnAcc[i] = 0;
    this.lastAction[i] = ACTIONS.GATHER;
  }

  _capturePrev() {
    const N = this.population;
    for (let i = 0; i < N; i++) {
      this.prevE[i] = this.energy[i]; this.prevF[i] = this.food[i];
      this.prevO[i] = this.ore[i]; this.prevS[i] = this.store[i];
    }
    this.prevPrice = this.price;
  }

  /* ------------------------------ helpers ------------------------------ */

  _cellIndexAt(x, z) {
    const G = this.G;
    let gx = Math.floor((x + this.half) / this.cell);
    let gz = Math.floor((z + this.half) / this.cell);
    gx = clamp(gx, 0, G - 1); gz = clamp(gz, 0, G - 1);
    return gz * G + gx;
  }
  _cellBiomassAt(x, z) { return this.flora[this._cellIndexAt(x, z)]; }

  _nearestAgent(i, maxDist) {
    const N = this.population, dx = this.px, dz = this.pz;
    let best = -1, bd = maxDist * maxDist;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const ex = dx[j] - this.px[i], ez = dz[j] - this.pz[i];
      const d2 = ex * ex + ez * ez;
      if (d2 < bd) { bd = d2; best = j; }
    }
    return { j: best, dist: Math.sqrt(bd) };
  }

  /* ------------------------------ actions ------------------------------ */

  _doGather(i) {
    const c = this.cfg, R2 = c.reach * c.reach;
    let node = -1, nd = R2;
    for (let k = 0; k < this.nodes.length; k++) {
      const n = this.nodes[k];
      if (n.stock <= 0) continue;
      const dx = n.x - this.px[i], dz = n.z - this.pz[i];
      const d2 = dx * dx + dz * dz;
      if (d2 < nd) { nd = d2; node = k; }
    }
    // best flora cell within reach (3x3 cells around the agent)
    let cellIdx = -1, bestB = 0.05, cd = R2;
    const G = this.G;
    const gx = Math.floor((this.px[i] + this.half) / this.cell);
    const gz = Math.floor((this.pz[i] + this.half) / this.cell);
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = gx + dx, cz = gz + dz;
        if (cx < 0 || cz < 0 || cx >= G || cz >= G) continue;
        const idx = cz * G + cx;
        const b = this.flora[idx];
        if (b <= 0.05) continue;
        const mx = (cx + 0.5) * this.cell - this.half;
        const mz = (cz + 0.5) * this.cell - this.half;
        const ex = mx - this.px[i], ez = mz - this.pz[i];
        const d2 = ex * ex + ez * ez;
        if (d2 <= R2 && d2 < cd && b > bestB) { cd = d2; bestB = b; cellIdx = idx; }
      }
    }

    const dNode = node >= 0 ? Math.sqrt(nd) : Infinity;
    const dCell = cellIdx >= 0 ? Math.sqrt(cd) : Infinity;

    if (dNode <= dCell && node >= 0) {
      const n = this.nodes[node];
      const take = Math.min(c.gatherOreYield, n.stock);
      n.stock -= take;
      this.ore[i] += take;
      this.energy[i] -= c.gatherOreCost;
      this.harvestOrePending += take;
      this.events.harvests.push({ i, kind: 'ore', x: n.x, z: n.z, amt: take });
    } else if (cellIdx >= 0) {
      const take = Math.min(c.gatherFoodYield, this.flora[cellIdx]);
      this.flora[cellIdx] -= take;
      this.food[i] += take * 0.95;
      this.energy[i] -= c.gatherFoodCost;
      this.harvestFloraPending += take;
      this.events.harvests.push({ i, kind: 'flora', x: this.px[i], z: this.pz[i], amt: take });
    } else {
      this.energy[i] -= c.gatherFoodCost * 0.35; // wasted effort: no resource in reach
    }
  }

  _tradeSide(i) {
    const c = this.cfg;
    if (this.ore[i] >= 1 && this.food[i] < c.sellFoodTrigger) return -1;   // need food -> sell ore
    if (this.food[i] >= this.price + c.buyFoodMargin && this.ore[i] < c.oreCap) return 1; // surplus -> buy ore
    return 0;
  }

  _doTrade(i) {
    const c = this.cfg;
    const side = this.intent[i];
    if (side === 0) { this.energy[i] -= c.tradeCost * 0.4; return; }
    const { j } = this._nearestAgent(i, c.tradeRange);
    if (j < 0 || this.intent[j] === 0 || this.intent[j] === side || this.matched[j]) {
      this.energy[i] -= c.tradeCost * 0.5; // no counterparty: opportunity cost
      return;
    }
    // execute 1 ore at the reference price (seller -> buyer)
    const seller = side === -1 ? i : j;
    const buyer = side === 1 ? i : j;
    if (this.ore[seller] < 1 || this.food[buyer] < this.price + 0.5) {
      this.energy[i] -= c.tradeCost * 0.5;
      return;
    }
    this.ore[seller] -= 1;
    this.food[buyer] += 1;
    this.food[seller] += this.price;
    this.food[buyer] -= this.price;
    this.matched[i] = 1; this.matched[j] = 1;
    this.energy[i] -= c.tradeCost;
    this.tradeCount[i]++; this.tradeCount[j]++;
    this.tradesTotal++;
    this.volumePending++;
    const a = Math.min(i, j), b = Math.max(i, j);
    const key = a + ':' + b;
    this.tradeEdges.set(key, (this.tradeEdges.get(key) || 0) + 1);
    this.events.trades.push({ i, j, price: this.price, x: this.px[i], z: this.pz[i], x2: this.px[j], z2: this.pz[j] });
  }

  _doShare(i) {
    const c = this.cfg;
    const { j } = this._nearestAgent(i, c.shareRange);
    if (j >= 0 && this.food[i] >= c.shareAmount + 0.5) {
      const worse = (this.food[j] + this.energy[j] / c.EMAX * 4) <
                    (this.food[i] + this.energy[i] / c.EMAX * 4);
      if (worse) {
        const amt = Math.min(c.shareAmount, this.food[i] - 0.5);
        if (amt > 0.05) {
          this.food[i] -= amt; this.food[j] += amt;
          this.given[i] += amt; this.received[j] += amt;
          this.sharesPending++;
          this.sharesTotal++;
          this.events.shares.push({ i, j, amt, x: this.px[i], z: this.pz[i], x2: this.px[j], z2: this.pz[j] });
        }
      }
    }
    this.energy[i] -= c.shareCost;
  }

  _doStore(i) {
    const c = this.cfg;
    if (this.food[i] >= 1) { this.food[i] -= 1; this.store[i] += 1; }
    else if (this.store[i] >= 1) { this.store[i] -= 1; this.food[i] += 1; }
    this.energy[i] -= c.storeCost;
  }

  _doMigrate(i) {
    const c = this.cfg;
    const sense = c.reach * c.migrateSense * 4;
    const G = this.G, R = Math.ceil(sense / this.cell);
    const gx = Math.floor((this.px[i] + this.half) / this.cell);
    const gz = Math.floor((this.pz[i] + this.half) / this.cell);
    let bestScore = 0, bx = 0, bz = 0, found = false;
    for (let dz = -R; dz <= R; dz++) {
      for (let dx = -R; dx <= R; dx++) {
        const cx = gx + dx, cz = gz + dz;
        if (cx < 0 || cz < 0 || cx >= G || cz >= G) continue;
        const idx = cz * G + cx;
        const b = this.flora[idx];
        if (b > bestScore) {
          bestScore = b; found = true;
          bx = (cx + 0.5) * this.cell - this.half;
          bz = (cz + 0.5) * this.cell - this.half;
        }
      }
    }
    for (const n of this.nodes) {
      if (n.stock <= 0) continue;
      const dxn = n.x - this.px[i], dzn = n.z - this.pz[i];
      if (dxn * dxn + dzn * dzn > sense * sense) continue;
      const score = n.stock * this.price;
      if (score > bestScore) { bestScore = score; found = true; bx = n.x; bz = n.z; }
    }
    const cur = this._cellBiomassAt(this.px[i], this.pz[i]);
    if (found && bestScore > cur + 0.5) {
      const dx = bx - this.px[i], dz = bz - this.pz[i];
      const d = Math.hypot(dx, dz) || 1;
      const step = Math.min(c.migrateDist, d);
      this.px[i] += (dx / d) * step;
      this.pz[i] += (dz / d) * step;
      this.energy[i] -= c.migrateCost;
    } else {
      this.energy[i] -= c.migrateCost * 0.5; // no better ground: wasted trip
    }
  }

  /* ------------------------------ main step ------------------------------ */

  step(mvx, mvz, acts) {
    const c = this.cfg, N = this.population;
    this.mvx.set(mvx); this.mvz.set(mvz); this.acts.set(acts);
    this._capturePrev();
    this.harvestFloraPending = 0;
    this.harvestOrePending = 0;
    this.regrowthPending = 0;
    this.volumePending = 0;
    this.sharesPending = 0;
    this.intent.fill(0);
    this.matched.fill(0);
    this.dones.fill(0);
    this.events.trades.length = 0;
    this.events.shares.length = 0;
    this.events.deaths.length = 0;
    this.events.harvests.length = 0;

    // ---- 1. movement + discrete modes ----
    for (let i = 0; i < N; i++) {
      const a = acts[i];
      let dx = mvx[i] * c.speed, dz = mvz[i] * c.speed;
      if (a === ACTIONS.GATHER) {
        this._doGather(i);
      } else if (a === ACTIONS.TRADE) {
        this.intent[i] = this._tradeSide(i);
      } else if (a === ACTIONS.SHARE) {
        this._doShare(i);
      } else if (a === ACTIONS.STORE) {
        this._doStore(i);
      } else if (a === ACTIONS.MIGRATE) {
        this._doMigrate(i);
      }
      let nx = this.px[i] + dx, nz = this.pz[i] + dz;
      const lim = this.half - 1.5;
      this.px[i] = clamp(nx, -lim, lim);
      this.pz[i] = clamp(nz, -lim, lim);
    }

    // second pass: all trade intents are known, now match bilateral partners
    for (let i = 0; i < N; i++) {
      if (this.acts[i] === ACTIONS.TRADE && !this.matched[i]) this._doTrade(i);
    }

    // ---- 2. adaptive reference price (tâtonnement on trade intents) ----
    let demand = 0, supply = 0;
    for (let i = 0; i < N; i++) {
      if (this.acts[i] !== ACTIONS.TRADE) continue;
      if (this.intent[i] > 0) demand++;
      else if (this.intent[i] < 0) supply++;
    }
    this.demand = demand; this.supply = supply;
    if (demand + supply > 0) {
      this.price *= Math.exp(c.priceEta * (demand - supply) / (demand + supply));
      this.price = clamp(this.price, c.priceMin, c.priceMax);
    }

    // ---- 3. flora regrowth (logistic + immigration for seeded cells) ----
    const G = this.G, K = c.floraK, r = c.floraR;
    const flora = this.flora, seeded = this.floraSeeded;
    let regrow = 0;
    for (let idx = 0; idx < flora.length; idx++) {
      const b = flora[idx];
      if (b <= 0) {
        if (seeded[idx]) { flora[idx] = c.floraImmigration; regrow += c.floraImmigration; }
        continue;
      }
      const inc = r * b * (1 - b / K) + (seeded[idx] ? c.floraImmigration * 0.25 : 0);
      if (b + inc <= K) { flora[idx] = b + inc; regrow += inc; }
      else { flora[idx] = K; regrow += K - b; }
    }
    this.regrowthPending = regrow;

    // ---- 4. metabolism, eating, decay, ageing, death ----
    for (let i = 0; i < N; i++) {
      this.energy[i] -= c.metab;
      if (this.energy[i] < c.EMAX && this.food[i] > 0) {
        const need = (c.EMAX - this.energy[i]) / c.eatEff;
        const eat = Math.min(this.food[i], need, c.eatRate);
        this.food[i] -= eat;
        this.energy[i] += eat * c.eatEff;
      }
      this.food[i] *= (1 - c.foodDecay);
      this.store[i] *= (1 - c.storeDecay);
      this.age[i] += 1;
      this.dE[i] = this.energy[i] - this.prevE[i];

      let dead = 0, aged = 0;
      if (this.energy[i] <= 0) { dead = 1; this.deathsStarve++; this.starveWin++; }
      else if (this.age[i] >= c.maxAge) { aged = 1; this.deathsAge++; }
      if (dead || aged) {
        this.dones[i] = 1;
        this.meanLifetimeEma = this.meanLifetimeEma === 0
          ? this.age[i] : this.meanLifetimeEma * 0.995 + this.age[i] * 0.005;
        this.events.deaths.push({ i, x: this.px[i], z: this.pz[i], cause: dead ? 'starve' : 'age' });
        this._spawn(i);
      }
    }

    // ---- 5. reward decomposition ----
    const wS = c.wSurvival, wI = c.wInventory, wP = c.wProsperity, wA = c.wAlive, wD = c.wDeath;
    const RI = c.refInventory, EMAX = c.EMAX;
    let sumReward = 0;
    for (let i = 0; i < N; i++) {
      const dE = this.energy[i] - this.prevE[i];
      const dInv = (this.food[i] + this.ore[i] * this.price) -
                   (this.prevF[i] + this.prevO[i] * this.prevPrice);
      const dStore = this.store[i] - this.prevS[i];
      const dead = this.dones[i] > 0;
      const p0 = wS * (dE / EMAX);
      const p1 = wI * (dInv / RI);
      const p2 = wP * (dStore / RI);
      const p3 = dead ? 0 : wA;
      const p4 = dead ? -wD : 0;
      const o = i * 5;
      this.parts[o] = p0; this.parts[o + 1] = p1; this.parts[o + 2] = p2;
      this.parts[o + 3] = p3; this.parts[o + 4] = p4;
      this.rewards[i] = p0 + p1 + p2 + p3 + p4;
      this.lastReward[i] = this.rewards[i];
      this.returnAcc[i] += this.rewards[i];
      this.lastAction[i] = acts[i];
      this.actionHist[i * this.histLen + this.histCursor] = acts[i];
      sumReward += this.rewards[i];
      for (let k = 0; k < 5; k++) {
        this.partsEma[k] = this.partsEma[k] === 0
          ? this.parts[o + k]
          : this.partsEma[k] * 0.999 + this.parts[o + k] * 0.001;
      }
    }
    this.meanRewardOverStep = sumReward / N;
    this.histCursor = (this.histCursor + 1) % this.histLen;

    // ---- 6. rolling emas + book-keeping ----
    const ema = (prev, v, a = 0.02) => prev * (1 - a) + v * a;
    this.harvestFloraEma = ema(this.harvestFloraEma, this.harvestFloraPending);
    this.harvestOreEma = ema(this.harvestOreEma, this.harvestOrePending);
    this.regrowthEma = ema(this.regrowthEma, this.regrowthPending);
    this.deathsEma = ema(this.deathsEma, this.dones.reduce((s, v) => s + v, 0) / N * 20);
    this.sharesEma = ema(this.sharesEma, this.sharesPending);
    this.volumeEma = ema(this.volumeEma, this.volumePending);

    // prune old trade edges (rolling window ~1500 events)
    if (this.tradeEdges.size > 1500) {
      let n = 0;
      for (const [k, v] of this.tradeEdges) {
        this.tradeEdges.set(k, v * 0.9);
        if (v * 0.9 < 0.05) this.tradeEdges.delete(k);
        if (++n > 400) break;
      }
    }

    this.step_++;
    this._globals();
    this.computeObs();
    this._maybeRecordHistory();
    return this.rewards;
  }

  /* ------------------------------ statistics ------------------------------ */

  _globals() {
    const c = this.cfg, N = this.population;
    let biomass = 0, ore = 0;
    for (let i = 0; i < this.flora.length; i++) biomass += this.flora[i];
    for (const n of this.nodes) ore += n.stock;
    this.floraBiomass = biomass;
    this.floraFraction = this.totalSeededK > 0 ? biomass / this.totalSeededK : 0;
    this.oreFraction = this.oreInitial > 0 ? ore / this.oreInitial : 0;

    let sumE = 0, sumF = 0, sumO = 0, sumSt = 0, sumW = 0;
    const w = this.wealth;
    for (let i = 0; i < N; i++) {
      w[i] = this.food[i] + this.store[i] + this.ore[i] * this.price;
      sumE += this.energy[i]; sumF += this.food[i]; sumO += this.ore[i];
      sumSt += this.store[i]; sumW += w[i];
    }
    this.meanEnergy = sumE / N; this.meanFood = sumF / N;
    this.meanOre = sumO / N; this.meanStore = sumSt / N;
    this.meanWealth = sumW / N;

    // Gini coefficient of wealth (rank-based, O(N log N))
    const sorted = Float32Array.from(w).sort();
    let g = 0;
    for (let i = 0; i < N; i++) g += (2 * (i + 1) - N - 1) * sorted[i];
    this.gini = sumW > 1e-9 ? g / (N * sumW) : 0;

    // top-10% wealth share (monopoly index)
    const k = Math.max(1, Math.round(N * 0.1));
    let top = 0;
    for (let i = N - k; i < N; i++) top += sorted[i];
    this.top10Share = sumW > 1e-9 ? top / sumW : 0;

    // ore concentration: share of all ore held by the top decile
    let oreSum = 0;
    for (let i = 0; i < N; i++) oreSum += this.ore[i];
    if (oreSum > 1e-6) {
      const so = Float32Array.from(this.ore).sort();
      let to = 0;
      for (let i = N - k; i < N; i++) to += so[i];
      this.oreTop10 = to / oreSum;
    } else this.oreTop10 = 0;

    // trade-participation stats
    let traders = 0;
    for (let i = 0; i < N; i++) if (this.tradeCount[i] > 0) traders++;
    this.uniqueTraders = traders;
    let edges = 0;
    for (const v of this.tradeEdges.values()) if (v >= 1) edges++;
    this.activeTradeEdges = edges;

    // reciprocity: how much of what was given came back (aggregate)
    let given = 0, received = 0;
    for (let i = 0; i < N; i++) { given += this.given[i]; received += this.received[i]; }
    this.reciprocity = given > 1e-6 ? received / given : 0;
    this.totalGiven = given; this.totalReceived = received;

    // exponential moving average of the per-step mean reward
    this.meanRewardEma = this.meanRewardEma === undefined ? this.meanRewardOverStep
      : this.meanRewardEma * 0.97 + this.meanRewardOverStep * 0.03;
    this.meanReward = this.meanRewardEma;
  }

  _maybeRecordHistory() {
    if (this.step_ % this.histInterval !== 0) return;
    const h = this.history;
    h.t.push(this.step_);
    h.flora.push(this.floraFraction);
    h.ore.push(this.oreFraction);
    h.price.push(this.price);
    h.vol.push(this.volumeEma);
    h.gini.push(this.gini);
    h.top10.push(this.top10Share);
    h.reward.push(this.meanRewardOverStep);
    h.death.push(this.deathsEma);
    h.shares.push(this.sharesEma);
    h.starve.push(this.starveWin);            // raw starvation deaths per histInterval steps
    this.starveWin = 0;
    for (const k of Object.keys(h)) if (h[k].length > this.historyMax) h[k].shift();
  }

  /* ------------------------------ observations ------------------------------ */

  computeObs() {
    const c = this.cfg, N = this.population, D = OBS_DIM, G = this.G;
    const sense = 22, sense2 = sense * sense;
    const obs = this.obs;
    let meanW = this.meanWealth || 1;

    for (let i = 0; i < N; i++) {
      const o = i * D;
      const x = this.px[i], z = this.pz[i];
      obs[o] = this.energy[i] / c.EMAX;
      obs[o + 1] = this.food[i] / 20;
      obs[o + 2] = this.ore[i] / 12;
      obs[o + 3] = this.store[i] / 40;
      obs[o + 4] = this.age[i] / c.maxAge;
      obs[o + 5] = clamp(this.dE[i] / 10, -1, 1);

      // nearest flora
      let fdx = 0, fdz = 0, fd = sense, found = false;
      const R = Math.ceil(sense / this.cell);
      const gx = Math.floor((x + this.half) / this.cell);
      const gz = Math.floor((z + this.half) / this.cell);
      let localB = 0;
      for (let dz = -R; dz <= R; dz++) {
        for (let dx = -R; dx <= R; dx++) {
          const cx = gx + dx, cz = gz + dz;
          if (cx < 0 || cz < 0 || cx >= G || cz >= G) continue;
          const idx = cz * G + cx;
          const b = this.flora[idx];
          if (Math.abs(dx) <= 1 && Math.abs(dz) <= 1) localB = Math.max(localB, b);
          if (b < 0.5) continue;
          const mx = (cx + 0.5) * this.cell - this.half;
          const mz = (cz + 0.5) * this.cell - this.half;
          const ex = mx - x, ez = mz - z;
          const d2 = ex * ex + ez * ez;
          if (d2 < fd * fd && d2 <= sense2) { fd = Math.sqrt(d2); fdx = ex; fdz = ez; found = true; }
        }
      }
      const fw = found ? (1 - fd / sense) / (fd || 1) : 0;
      obs[o + 6] = fdx * fw; obs[o + 7] = fdz * fw;
      obs[o + 8] = found ? fd / sense : 1;
      obs[o + 9] = localB / c.floraK;

      // nearest ore node
      let odx = 0, odz = 0, od = sense, onode = -1;
      for (let k = 0; k < this.nodes.length; k++) {
        const n = this.nodes[k];
        if (n.stock <= 0) continue;
        const ex = n.x - x, ez = n.z - z;
        const d2 = ex * ex + ez * ez;
        if (d2 < od * od && d2 <= sense2) { od = Math.sqrt(d2); odx = ex; odz = ez; onode = k; }
      }
      const ow = onode >= 0 ? (1 - od / sense) / (od || 1) : 0;
      obs[o + 10] = odx * ow; obs[o + 11] = odz * ow;
      obs[o + 12] = onode >= 0 ? od / sense : 1;
      obs[o + 13] = onode >= 0 ? this.nodes[onode].stock / c.oreStock : 0;

      // nearest agent
      let adx = 0, adz = 0, ad = 30, aj = -1;
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const ex = this.px[j] - x, ez = this.pz[j] - z;
        const d2 = ex * ex + ez * ez;
        if (d2 < ad * ad) { ad = Math.sqrt(d2); adx = ex; adz = ez; aj = j; }
      }
      const near30 = ad < 30 ? 1 / 30 : 0;
      obs[o + 14] = adx * near30; obs[o + 15] = adz * near30;
      obs[o + 16] = Math.min(ad / 30, 1);
      obs[o + 17] = aj >= 0 ? this.energy[aj] / c.EMAX : 0.5;
      obs[o + 18] = aj >= 0 ? Math.min(this.food[aj] / 10, 1.5) : 0;

      // local crowding
      let crowd = 0;
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const ex = this.px[j] - x, ez = this.pz[j] - z;
        if (ex * ex + ez * ez < 36) crowd++;
      }
      obs[o + 19] = Math.min(crowd / 8, 1);

      // economy context
      obs[o + 20] = Math.min(this.price / 3, 3);
      obs[o + 21] = this.floraFraction;
      obs[o + 22] = this.oreFraction;
      obs[o + 23] = this.lastAction[i] / 4;
      obs[o + 24] = clamp(this.lastReward[i], -1, 1);
      obs[o + 25] = clamp((this.wealth[i] - meanW) / (meanW + 5), -1, 1);
      obs[o + 26] = this._hasTradePartner(i) ? 1 : 0;
      obs[o + 27] = this._hasShareTarget(i) ? 1 : 0;
    }

    // ---- centralized critic input: [ own obs ; joint compact ; global ] ----
    const critic = this.critic, C = this.criticDim;
    for (let i = 0; i < N; i++) {
      let p = i * C;
      critic.set(obs.subarray(i * D, i * D + D), p);
      p += D;
      for (let j = 0; j < N; j++) {
        critic[p] = this.energy[j] / c.EMAX;
        critic[p + 1] = this.food[j] / 20;
        critic[p + 2] = this.ore[j] / 12;
        critic[p + 3] = this.store[j] / 40;
        p += COMPACT_DIM;
      }
      const g = this._globalFeatures();
      critic.set(g, p);
    }
  }

  _hasTradePartner(i) {
    const c = this.cfg;
    const side = this._tradeSide(i);
    if (side === 0) return false;
    const N = this.population, R2 = c.tradeRange * c.tradeRange;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const ex = this.px[j] - this.px[i], ez = this.pz[j] - this.pz[i];
      if (ex * ex + ez * ez > R2) continue;
      const sj = this._tradeSide(j);
      if (sj !== 0 && sj !== side) return true;
    }
    return false;
  }

  _hasShareTarget(i) {
    const c = this.cfg;
    if (this.food[i] < 1.5) return false;
    const N = this.population, R2 = c.shareRange * c.shareRange;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const ex = this.px[j] - this.px[i], ez = this.pz[j] - this.pz[i];
      if (ex * ex + ez * ez > R2) continue;
      if ((this.food[j] + this.energy[j] / c.EMAX * 4) <
          (this.food[i] + this.energy[i] / c.EMAX * 4)) return true;
    }
    return false;
  }

  _globalBuf = null;
  _globalFeatures() {
    if (!this._globalBuf) this._globalBuf = new Float32Array(GLOBAL_DIM);
    const g = this._globalBuf, c = this.cfg;
    g[0] = this.floraFraction;
    g[1] = this.oreFraction;
    g[2] = Math.min(this.price / 3, 3);
    g[3] = Math.min(this.volumeEma / 3, 2);
    g[4] = this.meanEnergy / c.EMAX;
    g[5] = this.meanFood / 20;
    g[6] = this.meanStore / 40;
    g[7] = this.gini;
    g[8] = this.meanOre / 12;
    g[9] = Math.min(this.harvestFloraEma / 8, 2);
    g[10] = Math.min(this.harvestOreEma / 3, 2);
    g[11] = Math.min(this.sharesEma / 2, 2);
    g[12] = Math.min(this.deathsEma, 2);
    g[13] = (this.step_ % 1000) / 1000;
    g[14] = clamp((this.price / (this.prevPrice || 1)) - 1, -1, 1) * 5;
    g[15] = this.top10Share;
    return g;
  }

  /* ------------------------------ API for the trainer ------------------------------ */

  stats() {
    const c = this.cfg, N = this.population;
    return {
      step: this.step_,
      population: N,
      deathsStarve: this.deathsStarve,
      deathsAge: this.deathsAge,
      meanLifetime: this.meanLifetimeEma,
      floraFraction: this.floraFraction,
      oreFraction: this.oreFraction,
      meanEnergy: this.meanEnergy,
      meanFood: this.meanFood,
      meanOre: this.meanOre,
      meanStore: this.meanStore,
      meanWealth: this.meanWealth,
      gini: this.gini,
      top10Share: this.top10Share,
      oreTop10: this.oreTop10,
      price: this.price,
      volumeEma: this.volumeEma,
      tradesTotal: this.tradesTotal,
      uniqueTraders: this.uniqueTraders,
      activeTradeEdges: this.activeTradeEdges,
      sharesTotal: this.sharesTotal,
      sharesEma: this.sharesEma,
      reciprocity: this.reciprocity,
      harvestFloraEma: this.harvestFloraEma,
      harvestOreEma: this.harvestOreEma,
      regrowthEma: this.regrowthEma,
      overharvest: this.regrowthEma > 1e-4 ? this.harvestFloraEma / this.regrowthEma : 0,
      deathsEma: this.deathsEma,
      meanReward: this.meanReward,
      rewardParts: Array.from(this.partsEma),
      demand: this.demand || 0,
      supply: this.supply || 0,
      params: { wSurvival: c.wSurvival, wInventory: c.wInventory, wProsperity: c.wProsperity, wAlive: c.wAlive, wDeath: c.wDeath },
    };
  }
}
