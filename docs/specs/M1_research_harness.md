# M1 Implementation Spec — Research Harness + Plateau Selector

**Status:** DRAFT (pre-spec; design before code). Elaborates `docs/OPERATIONS_ROADMAP.md` §M1 + §M1.1.
**Goal:** run real 2020–2025 Dukascopy data through Phases 1–6 across a hyperparameter sweep, log every trial to the Alpha Registry, select deployable configs via Neighborhood-Minimax, and certify survivors with CPCV/PBO/DSR.

---

## 1. Module layout — `src/afml/research/`

| Module | Responsibility |
|---|---|
| `grids.py` | Data-derived hyperparameter grid per alpha family. |
| `precompute.py` | Per-asset, config-independent stages (bars, FFD, rolling feature frame) — built once, cached. |
| `harness.py` | `run_trial(asset, family, config)` — one config → one `TrialResult`; the per-config Phase 2→5 flow. |
| `objective.py` | `oos_strategy_sharpe(...)` — the plateau objective `s(g)` from PWF-CV OOS folds. |
| `plateau.py` | `select_plateau(...)` — the Neighborhood-Minimax selector. |
| `sweep.py` | `run_sweep(asset, family)` — iterate grid → log trials → build surface → select → certify survivor. |
| `artifacts.py` | Persist per-stage artifacts + the model bundle stub (formalized in M2). |
| `report.py` | Emit `RESEARCH_RUN.md`. |
| (CLI) `afml.cli` | `afml research sweep / select / report`. |

New pytest marker `m1`; `make m1` mirrors `make m0`.

---

## 2. Per-asset precompute (config-independent, cached)

Built **once per asset** and reused across all configs (the big speed win):

1. **Bar type** — `select_bar_type(ticks)` (Jarque-Bera-min, Phase 1 DoD) picks time / tick-imbalance / tick-run; build that bar frame over 2020-01-01→2025-12-31. *(Bar type is a data property, not a strategy knob; bar frequency is a fixed default — a future sweep axis, noted §13.)*
2. **FFD** — `find_optimal_d(bar_close)` (ADF `p<0.05`, τ=1e-5) → `ffd_apply`; stationary price series cached.
3. **Rolling feature frame** — compute the **causal rolling features over bars once** (`.shift(1)` enforced); per config we only *sample* this frame at that config's event timestamps. (Requires a `compute_features` path that separates "roll over bars" from "sample at events"; if absent, add a thin `sample_features_at(events)` helper. Avoids recomputing 50+ rolling features per config.)

Cached to `artifacts/{asset}/_precompute/` keyed by a content hash of (asset, window, code version).

---

## 3. Per-config trial pipeline — `run_trial(asset, family, config)`

```
events   = alpha(config).detect(bars)                     # CUSUM/Bollinger/Donchian
labels   = apply_triple_barrier(bars, events,             # exit_timestamp = realized t1
              vol_span, profit_take_mult, stop_loss_mult, vertical_barrier_bars)
feats    = sample_features_at(rolling_frame, events)      # from §2 cache
X, y, w, t0, t1 = align_labels_to_features(labels, feats) # V2 burn-in; t1 = exit_timestamp
sel      = select_features(X, y, t0=t0, t1=t1,            # ONC + Clustered MDA
              registry=registry, experiment_metadata=meta)
if sel.halted_at_mda:                                     # zero survivors (V3 breaker)
    registry.record_failed_mda(...);  return TrialResult(status=FAILED_AT_MDA, ...)
brain2   = train_brain_two(X[sel.features], y, sample_weight=w, t0=t0, t1=t1, ...)
s_g      = oos_strategy_sharpe(brain2.folds, bars, labels)  # §5  (None if invalid)
registry.record_experiment(... brain_1_recall, brain_2_log_loss, orthogonality_score ...)
return TrialResult(config, s_g, recall, brier, n_events, status=COMPLETED)
```

- **Registry row per trial** — `hyperparameter_vector = {family, **config}` (stable dedup hash; order-independent). Dedup → `DuplicateHypothesisError` caught & skipped (resumability). Orthogonality vs. deployed via `is_orthogonal`/`max_correlation`.
- **Every grid point is a trial** (drives the DSR `K` count) even if it later loses plateau selection or fails MDA.

---

## 4. The objective `s(g)` — `objective.py`

`s(g)` = **median, over PurgedWalkForward OOS folds, of the meta-labeled strategy's annualized Sharpe.**

Per OOS fold: Brain-2 (trained on fold-train) emits calibrated `P(success)` on fold-test events → `calculate_bet_size(p)` → per-event strategy return `= bet_size · realized_return(event)`, where `realized_return = side · (price[exit_timestamp]/price[t0] − 1)` from bars. Fold Sharpe `= mean/std · √(events_per_year)`; **median across folds** = `s(g)`.

**Validity filter** (else `s(g) = −∞`, excluded from the surface): `n_events ≥ 500` (P2), `recall ≥ 0.70` (P2), `Brier < naive_baseline` (P5).

Annualization uses the config's realized average events/year (consistent factor reused in M3's OOS gate). *(Sub-decision §13.)*

---

## 5. Hyperparameter grids — `grids.py` (data-derived, ≥30/cohort)

Values are **relative / data-scaled** (anti-bias: no absolute price magic numbers); resolution chosen so each (asset, family) cohort has **≥ `DSR_MIN_TRIALS` (30)** trials.

| Family | Axes (ordinal lattice) | Example resolution |
|---|---|---|
| CUSUM | `vol_span` × `threshold_mult` (×σ_EWM) | 5 × 6 = 30 |
| Bollinger | `window` × `num_std` | 6 × 5 = 30 |
| Donchian | `window` × `vertical_bars` | 6 × 5 = 30 |
| (shared) triple-barrier | `pt_mult` × `sl_mult` (×vol) | folded into the above or a 3rd axis |

Thresholds expressed as multiples of the rolling EWM volatility / σ so they transfer across assets. Grid ranges proposed concretely in the M1 PR; the lattice dims `d` drive the plateau neighborhood.

---

## 6. Plateau selector — `plateau.py` (Neighborhood-Minimax)

```python
@dataclass(frozen=True)
class PlateauResult:
    selected: tuple[int, ...] | None     # ordinal coords; None ⇒ no stable config
    robustness: float                    # R(g*)
    plateau_size: int                    # connected good-cell count
    reason: str

def select_plateau(
    scores: dict[tuple[int, ...], float],   # ordinal coord -> s(g); invalid = -inf
    *, dims: int, s_floor: float = 0.0, delta: float = 0.1,
) -> PlateauResult: ...
```

Algorithm:
1. **Neighbors** `N(g)` = Chebyshev-distance-1 cells (the `3^d − 1` adjacent), valid only (`s > −∞`).
2. **Eligibility:** `g` eligible iff `|valid N(g)| ≥ ⌈(3^d − 1)/2⌉` (boundary guard — a corner can't fake a plateau).
3. **Robustness** `R(g) = min(s(g), min_{h∈N(g)} s(h))`.
4. **Select** `g* = argmax_{eligible g} R(g)`.
5. **Tie-break** (within `ε` of max `R`): largest connected component where `s ≥ s_max − delta` → parsimony (more conservative params) → lexicographic.
6. **Reject** → `selected=None` if `R(g*) < s_floor` **or** connected-plateau-size `< 2`.

**Unit-test DoD** (synthetic surfaces): spike(3.0)/plateau(2.0)→plateau; monotone ramp→interior; flat→centre; all-isolated→`None`; order-invariance.

---

## 7. Two-stage compute & certification — `sweep.py`

- **Pre-pass (cheap, whole grid):** `run_trial` with **reduced** PWF-CV folds + a capped forest → `s(g)` surface. Logs all trials (K count).
- **Select:** `select_plateau(surface)` → `g*` (or honest "no stable config" → log, skip asset/family).
- **Certify (expensive, `g*` + immediate neighbors):** full **CPCV → PBO → DSR** via `validate_strategy` + `build_cohort_performance_matrices` + `count_cohort_trials`; `record_validation(pbo, dsr)`. Confirms the plateau holds under the stronger validator.

---

## 8. CLI surface

```
afml research sweep   --asset EURUSD --family cusum [--all] [--start --end] [--out artifacts/]
afml research select  --asset EURUSD --family cusum     # re-run selection on a logged surface
afml research report                                    # emit RESEARCH_RUN.md from the registry
```
Pilot: `--asset EURUSD` and `--asset BTCUSD` first (FX + crypto), then `--all`. Local execution.

---

## 9. Determinism, resumability, artifacts
- Global seed; PWF-CV / SBRF seeded. Same seed ⇒ identical surface ⇒ identical `g*`.
- Resumable: a trial whose dedup hash is already in the registry is skipped; precompute + surface cached under `artifacts/`.
- Artifacts per survivor staged for M2's `ModelStore` (model bundle written in M2; M1 writes the metrics + surface + selection).

---

## 10. DoD (M1 gate)
- Sweep completes on the pilot pair (EURUSD + BTCUSD), then all 14; reproducible under seed.
- **Per-cohort** `count_cohort_trials(asset, family) ≥ 30` and `≥ MIN_COHORT_STRATEGIES`.
- CPCV/PBO/DSR computed + `record_validation` for every plateau survivor; `awaiting_signoff()` non-empty *or* an honest "no stable config" logged per cohort.
- Anti-leakage on a real slice: truncation-hash (P1/P3), index-intersection (P5), target-shuffling (P6).
- `plateau.py` unit tests (§6) green; `oos_strategy_sharpe` validated on a synthetic labelled set.
- `RESEARCH_RUN.md` evidence sheet per survivor (JB, events+recall, label balance, clusters, Brier vs baseline, PBO, DSR, **plateau coords + R(g*)**).

## 11. Compute & risk
- Per-config cost dominated by Clustered-MDA (Purged K-Fold) + SBRF training. Pre-pass uses reduced folds/forest; full settings only on `g*`. Precompute caching removes redundant bar/feature work.
- **Risk:** wall-clock on 14 assets. Mitigation: pilot 2 assets to measure, parallelize per (asset, family), checkpoint to registry.
- **Risk:** "no edge" is a valid outcome — the harness reports honest PBO/DSR / no-plateau, never manufactures a survivor.

## 12. Open sub-decisions for the M1 PR
1. Exact grid ranges per family (data-derived; proposed in PR).
2. PWF-CV fold count + embargo for the pre-pass vs. certification.
3. Sharpe annualization factor (events/year vs. calendar) — must match M3.
4. Reduced-forest size for the pre-pass (speed vs. surface fidelity).
5. Whether bar **frequency** becomes a sweep axis (default: fixed per asset).
