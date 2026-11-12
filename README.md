# pumpfun-top5-forecast

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01)

A deterministic forecaster for make.fun's `pumpfun-top5-forecast-01`: given the
house recorder's marks, it writes a log-score-optimal forecast of the top 5
pump.fun coins by market cap at each of the seven answer reads, plus the size
distribution of each place's market cap.

## What it does

* reads `/input/history.ndjson` (66 marks, 10 min apart, 200 ranked coins each)
  and `/input/task.json` (the seven answer-read instants, the 82 size bins);
* rebuilds a per-coin trail over the last 8 marks (cap in SOL, rank, venue,
  launch time);
* projects `log10(cap)` forward to the middle answer read with momentum that
  decays with a per-venue time constant (a curve coin's pump fades faster than a
  pooled one), clipped to ±0.45 dex/hour;
* blends two views of each place: the last mark's order (a strong prior 7–13
  hours out) and the momentum ordering, then turns the blend into
  `P(coin at place k)` with a Gaussian rank kernel;
* leaves 38–42% of every place unnamed on purpose: coins launched after the
  history ends cannot be named, and the rules share each place's unnamed mass
  among them;
* builds each place's market cap in log10(SOL) as a two-component log-normal
  mixture (σ 0.45 around the projected place cap, plus a broad σ 0.95 tail),
  integrated over the task's 81 bin edges.

Deterministic: no randomness, no clock reads, no network. Two runs on the same
history produce byte-identical output.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Writes `out/forecast.json`:

```json
{"probabilities":[{"mint":"…","place":1,"p":0.137}],"top5":["…"],"sizes":[{"place":1,"bin":26,"p":0.31}]}
```

## Licence

MIT — see `LICENSE`.