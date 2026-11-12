/**
 * pumpfun-top5-forecast-01 entry.
 *
 * Reads the house recorder's marks (/input/history.ndjson, /input/task.json) and
 * writes one probabilistic forecast (/output/forecast.json): softmax rank
 * probabilities for places 1..5 over a shortlist of coins, plus a log-normal
 * mixture over the 82 market-cap size bins, plus a deterministic top5.
 *
 * Stdlib only, no network, seeded-free (fully deterministic).
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

type Row = {
  rank: number;
  mint: string;
  market_cap_lamports: string;
  venue?: string;
  slot?: number;
  name?: string;
  symbol?: string;
  created_at?: string;
  complete?: boolean;
  mayhem?: boolean;
  quote_reserves_lamports?: string;
};

type Mark = { at: string; rows: Row[] };

type Task = {
  reads: string[];
  places: number;
  chart_size: number;
  size: { edges: number[] };
  history_until?: string;
};

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR || '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR || '/output';

const LAMPORTS = 1e9;
const MAX_ROWS_PER_PLACE = 100;
const NAMED_MASS = 0.62; // rest stays unnamed: newcomers + model error
// Tuning knobs (env overrides are for our backtest only; defaults are what the grader runs)
const NAMED_1 = Number(process.env.PF_NAMED1 || 0.58);
const NAMED_K = Number(process.env.PF_NAMEDK || 0.62);
const SIGMA_RANK = Number(process.env.PF_SIGRANK || 3.2);
const PRIOR_W = Number(process.env.PF_PRIORW || 1);
const PROJ_W = Number(process.env.PF_PROJW || 0.55);
const SIZE_S1 = Number(process.env.PF_S1 || 0.45);
const SIZE_S2 = Number(process.env.PF_S2 || 0.95);
const SIZE_W1 = Number(process.env.PF_W1 || 0.7);
const SHORTLIST = 26;

function readJson<T>(file: string): T {
  return JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as T;
}

function readHistory(): Mark[] {
  const raw = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8');
  const marks: Mark[] = [];
  for (const line of raw.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    const m = JSON.parse(t) as Mark & { kind?: string; ok?: boolean };
    if (m.kind && m.kind !== 'mark') continue;
    if (m.ok === false) continue;
    if (!Array.isArray(m.rows) || m.rows.length === 0) continue;
    marks.push(m);
  }
  marks.sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  return marks;
}

const ms = (iso: string): number => Date.parse(iso);
const hours = (a: number, b: number): number => (b - a) / 3_600_000;
const capSol = (r: Row): number => Number(r.market_cap_lamports) / LAMPORTS;
const log10 = (x: number): number => Math.log(x) / Math.LN10;

type Coin = {
  mint: string;
  cap: number; // SOL, latest seen
  rank: number; // last rank seen
  slope: number; // log10(SOL) per hour over the recent window
  ageH: number;
  pool: boolean;
  seen: number;
};

function buildCoins(marks: Mark[], tNow: number): Map<string, Coin> {
  const trail = new Map<string, { t: number; cap: number; rank: number }[]>();
  const window = marks.slice(Math.max(0, marks.length - 8));
  for (const m of window) {
    const t = ms(m.at);
    for (const r of m.rows) {
      const cap = capSol(r);
      if (!(cap > 0)) continue;
      const arr = trail.get(r.mint) || [];
      arr.push({ t, cap, rank: r.rank });
      trail.set(r.mint, arr);
    }
  }
  const latest = marks[marks.length - 1];
  const createdOf = new Map<string, number>();
  for (const m of marks) {
    for (const r of m.rows) {
      if (r.created_at && !createdOf.has(r.mint)) createdOf.set(r.mint, ms(r.created_at));
    }
  }
  const coins = new Map<string, Coin>();
  for (const r of latest.rows) {
    const series = (trail.get(r.mint) || []).slice().sort((a, b) => a.t - b.t);
    let slope = 0;
    if (series.length >= 2) {
      const first = series[Math.max(0, series.length - 4)];
      const last = series[series.length - 1];
      const dh = hours(first.t, last.t);
      if (dh > 0.01) slope = (log10(last.cap) - log10(first.cap)) / dh;
    }
    const created = createdOf.get(r.mint);
    coins.set(r.mint, {
      mint: r.mint,
      cap: capSol(r),
      rank: r.rank,
      slope: Math.max(-0.45, Math.min(0.45, slope)),
      ageH: created ? Math.max(0, hours(created, tNow)) : 12,
      pool: r.venue === 'pool' || r.complete === true,
      seen: series.length,
    });
  }
  return coins;
}

/** Predicted log10(SOL) of a coin dtHours after the last mark. */
function project(coin: Coin, dtHours: number): number {
  const base = log10(Math.max(coin.cap, 1));
  // Momentum decays: a coin that ran hard for 80 minutes rarely keeps it for 13 hours.
  const tau = coin.pool ? 2.2 : 3.4;
  const decayed = coin.slope * tau * (1 - Math.exp(-dtHours / tau));
  // Young coins on the curve keep some launch premium, old ones fade.
  const ageAdj = coin.ageH < 1 ? 0.15 : coin.ageH < 6 ? 0.05 : -0.02;
  return base + decayed + ageAdj;
}

type Prediction = { mint: string; rank: number; mu: number };

function shortlist(coins: Map<string, Coin>, task: Task, horizonH: number): Prediction[] {
  const preds: Prediction[] = [];
  for (const c of coins.values()) {
    preds.push({ mint: c.mint, rank: 0, mu: project(c, horizonH) });
  }
  preds.sort((a, b) => (b.mu - a.mu) || (a.mint < b.mint ? -1 : a.mint > b.mint ? 1 : 0));
  for (let i = 0; i < Math.min(preds.length, Math.max(SHORTLIST, task.places * 4)); i++) {
    preds[i].rank = i + 1;
  }
  return preds.slice(0, SHORTLIST);
}

function placeRows(preds: Prediction[], places: number, coins: Map<string, Coin>) {
  const rows: { mint: string; place: number; p: number }[] = [];
  const sigmaPlace = SIGMA_RANK; // how far a coin's predicted rank can drift over 7-13 h
  const byLastRank = new Map<number, string>();
  for (const c of coins.values()) byLastRank.set(c.rank, c.mint);
  for (let place = 1; place <= places; place++) {
    // Two views: what the last mark's order says (a strong prior for 7-13 h out)
    // and what the momentum projection says. Blend, then normalise.
    const prior = new Map<string, number>();
    for (const [r, mint] of byLastRank) {
      const d = Math.abs(r - place);
      if (d === 0) prior.set(mint, 0.3 * PRIOR_W);
      else if (d === 1) prior.set(mint, 0.1 * PRIOR_W);
    }
    const scored = preds.map((p) => {
      const d = p.rank - place;
      const proj = Math.exp(-(d * d) / (2 * sigmaPlace * sigmaPlace));
      return { mint: p.mint, w: PROJ_W * proj + (prior.get(p.mint) || 0) };
    });
    for (const [mint, w] of prior) {
      if (!scored.some((s) => s.mint === mint)) scored.push({ mint, w });
    }
    const total = scored.reduce((s, x) => s + x.w, 0) || 1;
    const named = place === 1 ? NAMED_1 : NAMED_K;
    const take = scored
      .slice()
      .sort((a, b) => b.w - a.w)
      .slice(0, MAX_ROWS_PER_PLACE);
    const scaled = take.map((x) => ({ mint: x.mint, place, p: Math.floor((x.w / total) * named * 1e6) / 1e6 }));
    for (const row of scaled) if (row.p > 0) rows.push(row);
    if (!scaled.some((r) => r.p > 0)) rows.push({ mint: preds[0].mint, place, p: Math.floor((named / 2) * 1e6) / 1e6 });
  }
  return rows;
}

function sizeRows(preds: Prediction[], task: Task) {
  const edges = task.size.edges; // 81 edges, bins 0..81
  const rows: { place: number; bin: number; p: number }[] = [];
  const at = (place: number) => preds[Math.min(preds.length - 1, Math.max(0, place - 1))].mu;
  for (let place = 1; place <= task.places; place++) {
    const mu = at(place);
    const parts = [
      { w: SIZE_W1, mu, sigma: SIZE_S1 },
      { w: 1 - SIZE_W1, mu: mu + 0.15, sigma: SIZE_S2 },
    ];
    const ps: number[] = [];
    for (let bin = 0; bin <= 81; bin++) {
      let p = 0;
      for (const comp of parts) {
        p += comp.w * binMass(bin, edges, comp.mu, comp.sigma);
      }
      ps.push(p);
    }
    const total = ps.reduce((s, x) => s + x, 0) || 1;
    let written = 0;
    for (let bin = 0; bin <= 81; bin++) {
      const p = Math.floor((ps[bin] / total) * 0.999 * 1e6) / 1e6;
      if (p <= 0) continue;
      rows.push({ place, bin, p });
      written += p;
      if (written > 0.985) break;
    }
  }
  return rows;
}

/** Probability mass a normal(mu, sigma) in log10(SOL) puts in one bin. */
function binMass(bin: number, edges: number[], mu: number, sigma: number): number {
  const lo = bin === 0 ? -Infinity : edges[bin - 1];
  const hi = bin === 81 ? Infinity : edges[bin];
  return cdf((hi - mu) / sigma) - cdf((lo - mu) / sigma);
}

function cdf(z: number): number {
  if (!Number.isFinite(z)) return z > 0 ? 1 : 0;
  // Abramowitz & Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp((-z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

function main(): void {
  const task = readJson<Task>('task.json');
  const marks = readHistory();
  const tLast = ms(marks[marks.length - 1].at);
  const horizon = task.reads && task.reads.length
    ? hours(tLast, ms(task.reads[Math.floor(task.reads.length / 2)]))
    : 8;

  const coins = buildCoins(marks, tLast);
  const preds = shortlist(coins, task, Math.max(0.5, horizon));

  const forecast = {
    probabilities: placeRows(preds, task.places || 5, coins),
    top5: preds.slice(0, 5).map((p) => p.mint),
    sizes: sizeRows(preds, task),
  };

  mkdirSync(OUTPUT_DIR, { recursive: true });
  writeFileSync(join(OUTPUT_DIR, 'forecast.json'), JSON.stringify(forecast) + '\n');
}

main();