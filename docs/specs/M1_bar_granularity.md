# M1 Research Bar-Granularity — Econometric Design Note

**Status:** DRAFT for sign-off (methodology). Resolves the Ops-M1 finding that the
unconstrained JB bar-type selector produced ~184k bars / 6 mo → ~68k CUSUM events /
config → SBRF `O(n²)` intractable **and** economically meaningless (over-trading).

The fix is **not** "pick a coarser bar." It is to **derive** the sampling
granularity from three mutually-consistent theoretical pillars, so the event
count lands — by construction — in the regime that is both *computationally
tractable* and *economically meaningful*.

---

## 0. The pathology, stated precisely

`select_bar_type` minimises the Jarque–Bera (JB) statistic over a candidate set
of bar parameters. Empirically it selected an information bar so fine that 6
months of EURUSD yielded **184,113 bars**, and the Symmetric-CUSUM filter then
fired **68,269 events** at threshold `k = 1.0σ` (still **19,864** at `k = 3.0σ`).

Two independent failures:
1. **Computational** — SBRF's sequential bootstrap is `≈O(n²)`; `n ≈ 68k` ⇒ ~4.6×10⁹ ops/fit ⇒ a single config did not finish in 34 min.
2. **Economic** — 68k events / 6 mo ≈ 545 trades/day is not a strategy; it is microstructure-noise harvesting that CPCV/DSR would reject anyway.

Both are symptoms of one cause: **over-sampling relative to the strategy's information horizon.**

---

## 1. Pillar I — Information-time sampling, and why JB-minimisation alone is degenerate

**Theory (Mandelbrot & Taylor 1967; Clark 1973; AFML Ch. 2.3).** Price is a
*subordinated* process: returns are near-Gaussian and near-IID when sampled on a
**business/information clock** (volume, dollar, tick) rather than the calendar
clock. This is *why* we use information bars at all — to recover the statistical
regularity the CLT promises.

**The trap.** The CLT also means that **finer bars are monotonically *more*
Gaussian**: aggregating more, smaller increments → lower JB, without bound, until
you hit pure microstructure noise (bid–ask bounce), whose *first* differences are
actually near-Gaussian by construction. So:

> **JB-minimisation is necessary but not sufficient.** Taken as an unconstrained
> objective it is *degenerate* — its infimum is at the finest possible bar. It
> must be **regularised by an economic scale**; JB then chooses the *most
> Gaussian bar among economically-admissible candidates*, not the globally finest.

This is the precise sense in which the current selector is mis-specified.

---

## 2. Pillar II — The CUSUM filter is a first-passage problem (the event-count law)

**Theory.** The Symmetric-CUSUM filter (López de Prado, AFML Snippet 2.4)
accumulates signed returns and fires when the running sum breaches a barrier
`h = kσ`, then resets. For a driftless process with per-bar variance `σ²`, the
number of bars between fires is the **first-passage time** of a random walk to a
symmetric barrier `±h`. The expected exit time of Brownian motion from `[−h, +h]`
started at 0 is the classical result

$$\mathbb{E}[\tau] = \frac{h^2}{\sigma^2} = \left(\frac{h}{\sigma}\right)^2 = k^2 .$$

Hence, over `B` bars, the **expected event count obeys**

$$\boxed{\; \mathbb{E}[\text{events}] \;\approx\; \frac{B}{k^2} \;}\qquad (k \gtrsim 1,\ \text{the selective regime}).$$

**Data validation (from the probe).** On 6 mo of EURUSD, `B = 184{,}113`:

| `k` | predicted `B/k²` | observed | ratio |
|---|---|---|---|
| 3.0 | 20,457 | 19,864 | **0.97** |
| 1.0 | 184,113 | 68,269 | 0.37 |

At the selective end (`k = 3`) the law is accurate to **3%**; at `k = 1` the
walk is in the discrete/saturating pre-asymptotic regime (and real returns have
volatility clustering), so events fall *below* the Gaussian bound — conservative
for us. **The math is empirically supported exactly where strategy events live
(selective `k`).**

**Consequence.** Event count is *jointly* set by `(B, k)`. Since the sweep fixes
the `k`-grid, **controlling `B` controls the event count deterministically.**

---

## 3. Pillar III — Holding-period coherence (market geometry)

A meta-labelled trade is held until a triple-barrier touch, capped at the
**vertical barrier** `V` bars. So the strategy's *maximum information horizon* is
`V` bars. Sampling at a granularity `Δ` such that `V·Δ ≪ H_min` (the shortest
economically-actionable holding horizon) means **each bar carries structure the
strategy structurally cannot act on** — it holds for `V` bars to reach horizon
`V·Δ`, so sub-`Δ` detail is, to this strategy, noise.

**Market-geometry anchor.** Fix a target holding horizon `H` (a *strategy-design*
choice — e.g. swing-trading these liquid instruments ⇒ `H ≈ 1 trading week`).
Coherence requires the bar duration

$$\boxed{\;\Delta^\* = \frac{H}{V}\;}$$

i.e. the vertical barrier (`V` bars) spans exactly the intended holding horizon.
Equivalently, the **target bar count** over a research window of trading-length `T` is

$$B^\* = \frac{T}{\Delta^\*} = \frac{T\,V}{H}.$$

---

## 4. The unified rule

1. **Anchor:** choose `H` (target holding horizon) and read `V` from the
   triple-barrier vertical → `Δ\* = H/V`, `B\* = T·V/H`.
2. **Sample:** build bars targeting `B\*` — **information bars** (tick/volume/dollar
   sized to `total_activity / B\*`) for the Pillar-I Gaussianity benefit, with time
   bars at `Δ\*` as a candidate. **JB chooses among these economically-admissible
   candidates** (Pillar I regularised).
3. **Bound events:** by Pillar II, across the `k`-grid `[k_min, k_max]`,

$$\text{events} \in \Big[\,\tfrac{B^\*}{k_{\max}^2},\ \sim B^\*\,\Big]
\;=\;\text{tractable } \wedge \text{ meaningful } \wedge \ge \text{ validity floor (500).}$$

So the event count is **engineered, not capped after the fact** — every config is
modelled on its *true, full* event set (no 1.5%-subset surrogate), so both the
surface and certification are full-fidelity. The earlier event-cap becomes a mere
safety net, never the mechanism.

---

## 5. Worked example (FX, 6-year window)

`H = 1 trading week ≈ 120 trading-hours`, `V = 20` ⇒ `Δ\* = 6 h`.
FX trading-time `T ≈ 312 wk × 120 h ≈ 37{,}440 h` ⇒ `B\* ≈ 6{,}240 bars`.

Event count across the CUSUM grid (`k ∈ [0.5, 3.0]`), via `B\*/k²` with
small-`k` saturation:

| `k` | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| events | ~4,400 | ~2,300 | ~1,250 | ~770 | ~560 | ~690→... |

All in **[~500, ~4,400]** — above the 500 validity floor, below the SBRF pain
threshold (≤ ~5k ⇒ `O(n²)` ≈ seconds–1 min/fit). The `V·Δ\* = 120 h ≈ 1-week`
max hold is an economically coherent swing horizon.

Per asset, `B\*` is realised through **information bars** (`expected_ticks =
total_ticks / B\*`), which auto-adapt to each instrument's activity (crypto's
24/7 vs FX's 24/5 vs index sessions) — so the *economic* horizon, not the
calendar, is held fixed.

---

## 6. Implementation delta (precompute)

`precompute_asset` stops calling the unconstrained `select_bar_type`. Instead:
1. Compute `B\* = T·V / H` from the configured `H` and `V`.
2. Construct candidate bar parameters **targeting `B\*`** (time interval ≈ `Δ\*`;
   `expected_ticks ≈ total_ticks / B\*`) and run the JB tournament **only over
   those** → JB picks the most-Gaussian *admissible* bar.
3. FFD `d\*` and everything downstream are unchanged.

`H` becomes a single, documented research parameter (default ≈ 1 week) — the lone
economic input; all granularity follows by the math above.

---

## 7. Falsifiable DoD checks
- **Bar count:** realised bars within, say, ±25% of `B\*`.
- **Event-count law:** on the pilot, fitted `events ≈ B/k²` holds (R² high) at selective `k` — the Pillar-II law is not just asserted but *measured*.
- **Range:** across the grid, events ∈ [500, safety-cap] for the pilot assets — no config is starved or intractable.
- **Wall-clock:** a representative trial completes in ≤ ~1 min (full-fidelity, no subsample).

---

## Open input for sign-off
`H` — the **target holding horizon** (the one strategy-design choice the
granularity hangs on). Proposed default: **~1 trading week** (swing horizon for
these liquid FX/index/metal/crypto instruments). A shorter `H` ⇒ finer bars ⇒
more events (heavier); longer `H` ⇒ coarser ⇒ fewer. Optionally `H` can later be
made *data-driven* (e.g. anchored to each asset's variance-ratio / mean-reversion
half-life), but a fixed swing `H` is the clean, defensible default.
