# Master Operational Roadmap — Autonomous Quant Lab (M0–M7)

**Status:** ✅ APPROVED (CEO sign-off 2026-05-20) — binding spec for Operational Milestones M0–M7
**Objective:** Operationalize the validated AFML build (Phases 0–9) into an autonomous, agentic ML trading firm.
**Relationship to the phases:** The AFML *build* Phases 0–9 (the math engine + control plane + crypto + Redis transport) are **shipped and validated**. These **Operational Milestones (M0–M7)** are the run/ops layer that turns that engine into a live, self-running lab. They are *not* the same numbering as the AFML phases.

**Execution rules (same discipline as the phases):**
- Strictly sequential. **One milestone per PR.** Do not advance until the milestone's DoD is green.
- Each milestone ends with a Lead-Quant audit gate before the next begins.
- `make lint type test` + frontend `typecheck/build/e2e` must be green; CLAUDE.md updated on every ship.
- Anti-bias / anti-leakage rules from the phases remain binding (see §Cross-cutting).

---

## Data windows (locked)

| Window | Dates | Use | Source |
|---|---|---|---|
| **Research / in-sample** | **2020-01-01 → 2025-12-31** | M1 sweep, training, CPCV/PBO/DSR certification | existing `*_2020_2025` (and `*_2016_2025`) parquets on disk |
| **Out-of-sample (OOS)** | **2026-01-01 → 2026-04-30** | M3 strict no-retrain walk-forward | `{DATA_ROOT}/2026_ytd/{ASSET}_2026-01-01_2026-04-30_DUKASCOPY.parquet` |

- Uniform **2020-01-01** research start across all 14 assets for cross-asset comparability. EURUSD/GBPUSD have 2016+ history available for an *optional* robustness check only.
- 2026 OOS is **never** touched during M1/M2 — it is the honest forward test.
- `config/assets.py` gains an `oos_data_path` (the `2026_ytd/` file) per `AssetSpec`; `afml doctor` (M0) and `OOSValidator` (M3) resolve OOS data through it.

---

## Governance contract (binding) — autonomous vs. Human-in-the-Loop

Two distinct flatten paths, never conflated:

| Action class | Examples | Authority |
|---|---|---|
| **Risk-*reducing*** | Circuit-breaker flatten on `MARKET_REGIME_BREAK` / `CONCEPT_DRIFT_ALERT` / drawdown breach; halt new orders | **Autonomous** — Agent 7 acts immediately, no signature, to protect capital |
| **Risk-*increasing*** | Promote a strategy Paper→Live; allocate capital | **HITL only** — CEO Ed25519 signature + TOTP, never autonomous |

- The CEO control-plane `/emergency/flatten` (signed) and Agent 7's **internal** auto-flatten are *separate code paths*. The internal path is triggered by a monitoring event off the Redis bus and carries no `signed_token`; it is logged + alerted but requires no human.
- Implementation note: introduce an internal `RISK_HALT` trigger consumed by Agent 7 distinct from the signed `EMERGENCY_FLATTEN` event. Auto-deploy is **never** permitted.

---

## M0 — Genesis & Operator Bootstrap

**Goal:** secure, launchable environment + CLI + infrastructural sanity checks.

**Deliverables**
- `afml` CLI (Typer; registered under `[project.scripts]`).
- `afml enroll-ceo` — generate Ed25519 keypair + TOTP seed → macOS Keychain; print the public-key hex (for `AFML_CP_CEO_PUBLIC_KEY_HEX`) and the `otpauth://` QR/URI **once**.
- `afml doctor` — verify: all required parquet files present (research **and** 2026 OOS); each file carries sufficient **warm-up history** to cold-start (≥ FFD `l*` + `max(feature_window) − 1` burn-in bars *before* the first usable event); Redis reachable; SQLite registry reachable; Keychain secrets present.
- **CI (G1):** GitHub Actions workflow running `make lint type test` + frontend `typecheck/build/e2e` on every push/PR.
- `.env.example` + a one-page local quickstart.

**DoD**
- `afml enroll-ceo` round-trips: sign a test message with the enrolled key → `verify_signature` true; the seed is loadable on a simulated reboot (persistence).
- The control plane boots with the enrolled public key; `POST /execution/approve` returns **403** (not 500) on a well-formed but unauthorized request. *(Full happy-path approval is exercised in M5, once strategies exist.)*
- `afml doctor` exits **0** on a healthy system and **1** with specific, actionable errors on missing data/services/warm-up.
- CI is green on the PR and **blocks merge on a red gate**.

**Depends on:** nothing. **Key files:** `src/afml/cli.py`, `pyproject.toml`, `.github/workflows/ci.yml`.

---

## M1 — Historical Ingestion Sweep (Research, 2020–2025)

**Goal:** run the math engine (Phases 1–6) across the research window to populate the Alpha Registry with **trials** (driving the DSR multiple-testing count) and to certify survivors via CPCV/PBO/DSR.

**Deliverables**
- `ResearchHarness` — per-asset driver: `load_ticks(asset, start=2020-01-01, end=2025-12-31)` → bars → {CUSUM, Bollinger, Donchian} events → triple-barrier labels → features → ONC+MDA selection → Brain-2 → validation. Deterministic (seeded), resumable, artifacts persisted per stage.
- Hyperparameter **sweep** per (asset, family); **every grid point logged to the registry as a trial** (`record_experiment` / `record_failed_mda`), with dedup, orthogonality, and `FAILED_AT_MDA` already handled.
- **Stable-Plateau selection** (§M1.1) chooses the deployable configuration per (asset, family) — *never the peak*.
- **Two-stage compute:** a cheap pre-pass scores the whole grid (PurgedWalkForwardCV OOS Sharpe) for plateau selection; the expensive full **CPCV/PBO/DSR** runs only on the selected plateau centre (+ its immediate neighbors, to confirm the plateau holds). Every grid point still counts as a trial for `K`.

**DoD**
- Sweep completes on all 14 assets for 2020–2025, **staged**: prove the `ResearchHarness` end-to-end on the pilot pair **EURUSD + BTCUSD** (FX + crypto — distinct microstructure) first, then fan out to the remaining 12. Runs **locally**.
- **Per-cohort trial floor (B2):** for every certified (asset, family) cohort, `count_cohort_trials(registry, asset, family) ≥ DSR_MIN_TRIALS (30)` and `≥ MIN_COHORT_STRATEGIES` for PBO — so DSR is **not** auto-quarantined and PBO is well-posed.
- CPCV, PBO, DSR computed and logged (`record_validation`) for every plateau survivor; `awaiting_signoff()` returns the survivors.
- Anti-leakage gates hold on **real** ticks in a `tests/integration` run on a slice: truncation-hash (P1/P3), index-intersection (P5), target-shuffling (P6).
- `RESEARCH_RUN.md` evidence sheet: per survivor — bar JB stat, event count + recall, label balance, surviving clusters, Brier vs. baseline, PBO, DSR, **and the selected plateau coordinates + its worst-neighbor score**.

**Depends on:** M0. **Key modules:** all phase libs; `AlphaRegistryRepository`; `afml.validation` (`build_cohort_performance_matrices`, `count_cohort_trials`, `validate_strategy`).

### M1.1 — Stable-Plateau Selection Algorithm (Neighborhood-Minimax)

*The anti-curve-fit rule made concrete. Chosen for our application because it has **no free tuning constant** (so the selector itself can't be meta-overfit), it is a direct statement of parameter robustness, and it is trivially unit-testable.*

**Inputs.** For a fixed (asset, family), a hyperparameter grid `G` (e.g. CUSUM `{vol_span × threshold_mult}`; triple-barrier `{pt_mult × sl_mult × vertical_bars}`). Each point `g` has a **robust objective** `s(g)` = the **median** over PurgedWalkForwardCV OOS folds of the meta-labeled strategy's annualized Sharpe. Median (not mean) → robust to one lucky fold.

**Validity filter.** `s(g) = −∞` (excluded) unless `g` is *valid*: events ≥ 500 (P2 DoD), Brain-1 recall ≥ 0.70 (P2 DoD), Brain-2 Brier < naive baseline (P5 DoD).

**Lattice.** Map each hyperparameter axis to **ordinal indices** (handles log/irregular spacing). Neighborhood `N(g)` = points within Chebyshev distance 1 (adjacent cells incl. diagonals).

**Robustness score (minimax).**
```
R(g) = min over {g} ∪ N(g) of s(·)
```
`R(g)` is high only if `g` *and all its neighbors* are strong → an isolated spike (≥1 weak neighbor) can never win. Equivalent to: *"the worst case under a ±1-step parameter perturbation is still profitable."*

**Selection.** `g* = argmax_g R(g)`.

**Boundary guard.** A point is plateau-eligible only if it has ≥ ⌈(3^d − 1)/2⌉ valid neighbors (`d` = #hyperparameters) — a corner with one neighbor cannot masquerade as a plateau.

**Tie-break** (within ε of max `R`): (1) largest **connected** plateau (size of the connected component where `s ≥ s_max − δ`); then (2) **parsimony** — more conservative params (larger thresholds / longer windows → fewer, higher-conviction events); then (3) deterministic lexicographic order.

**No-plateau rejection.** If `max_g R(g) < S_floor` (default `0.0` — a positive worst-neighbor Sharpe) **or** the selected plateau's connected size < 2, declare **no stable configuration** for (asset, family): log it, do **not** certify or deploy. *(A valid, expected outcome — not a failure.)*

**DoD (unit tests on synthetic surfaces):**
1. Tall narrow **spike** (3.0, neighbors 0.2) vs. broad **plateau** (2.0, neighbors 1.9) → selector returns the **plateau**, never the spike.
2. Monotone ramp → returns the high **interior** point with a full valid neighborhood, not the extreme corner.
3. All-flat surface → returns the **centre** (largest connected region).
4. All-isolated surface → returns the **no-stable-config** sentinel.
5. Selection is **invariant** to the grid evaluation order (determinism).

---

## M2 — Model Persistence & Artifact Lifecycle

**Goal:** serialize M1 survivors so live agents load them deterministically.

**Deliverables**
- `ModelStore` (local dir now, S3-ready): save/load **Brain-2 estimator + calibrator + selected ONC cluster spec + FFD `d*` + feature spec**, bundled under the `experiment_id` (matching the Alpha Registry), versioned with the package version + a content hash.
- **(B3) No GMM in the bundle** — the Phase-7 mixture is fit *per-batch at runtime* in `bet_sizes_for_batch` on the live z-distribution; it is ephemeral, not an artifact.
- `afml doctor` extension: every deployed registry row resolves to an intact, loadable bundle.

**DoD**
- Train → save → flush → reload reproduces identical Brier score + class probabilities on a fixed test array (≤ 1e-9, or bit-identical for the SBRF).
- Load **rejects** a feature-set / schema mismatch with a clear error (no silent wrong-model inference).
- Orphaned / missing bundles are flagged by `afml doctor`.

**Depends on:** M1. **Key files:** `src/afml/modeling/store.py`.

---

## M3 — 2026 Out-of-Sample Walk-Forward (offline)

**Goal:** prove M1 survivors hold up on **truly unseen 2026 Q1 data** before any live agent trusts them.

**Deliverables**
- `OOSValidator` — load approved bundles from M2; feed **2026-01-01 → 2026-04-30** ticks (from `{DATA_ROOT}/2026_ytd/`) through the *same* pipeline (ticks → info bars → features → Brain-2 probability → **offline** bet-sizing math) with **no retraining**.
- **(C2) Offline only:** uses the bet-sizing functions + mock broker; **no live dispatch**, no dependency on M4/M6.
- Log realized 2026 OOS performance vs. the M1 CPCV distribution.

**DoD**
- Pipeline runs 2026 data end-to-end with frozen models (no fit calls — asserted).
- **(C3) The OOS gate is breakdown detection, not performance confirmation.** Four months is too short to *confirm* a Sharpe (the estimate's standard error is large); it is only powerful enough to *detect a breakdown*. A strategy is auto-quarantined **only** if it shows **(a) significant degradation** — realized 2026 Sharpe **below the 5th percentile of its own M1 CPCV Sharpe path distribution** (a collapse beneath its own historical worst case — the overfit signature) — **or (b) decayed skill** — Brain-2 on 2026 fails `Brier < naive baseline`, or the SHAP importance rank-correlation vs. training falls below the drift threshold (0.5). A strategy with a **mildly negative but in-band** Sharpe and intact skill **survives** (flagged for a CEO note, not quarantined) — a bad-luck period is not an overfit. Real performance confirmation comes only from the M7 live soak + ongoing monitoring.
- A `OOS_2026.md` sheet: per strategy — projected (CPCV) vs. realized (2026) Sharpe, decision (survive/quarantine).

**Depends on:** M2. **Key modules:** M1 harness (reused), `afml.execution.bet_sizing`, `AlphaRegistryRepository`.

---

## M4 — Autonomous Agent Runtime

**Goal:** turn the static libraries into 8 always-on async workers on the existing Redis bus.

**Deliverables**
- `core/orchestrator.py` — async Agent base: `subscribe → handle → publish`, lifecycle, **heartbeats** (`AgentHeartbeat`), `structlog` structured logging, per-message error isolation, graceful shutdown, backpressure.
- `agents/agent1…8` daemons wiring each phase to its events (`BAR_GENERATED → EVENT_TRIGGERED → LABEL_COMPUTED → MODEL_TRAINED → STRATEGY_VALIDATED → BET_SIZED → ORDER_DISPATCHED`).
- `afml run-agent <n>` CLI.
- **(G3) Retrain policy (confirmed):** Agent 5 (Brain-2) retrains on a **drift trigger** (`CONCEPT_DRIFT_ALERT`) with a **scheduled weekly fallback**. A retrain is itself a new registry trial and re-enters M3-style OOS gating before redeploy.

**Audit injections**
- Agent 8 binds **strictly** to `BAR_GENERATED` (information bars), **never** raw ticks, and runs GSADF via `StructuralBreakMonitor.check_regime_async` (`run_in_executor`) — already built (polishing-audit P2).
- Agent 7 serializes fetch→size→dispatch via `ExecutionEngine.execute_batch_async` (`asyncio.Lock`) — already built (integration-audit V4).

**DoD**
- Synthetic ticks published to Redis flow through all 8 agents to a `STRATEGY_VALIDATED` / registry write — asserted in a `tests/integration` run on **real Redis** (`make redis-up`).
- 10k msg/s throughput retained, zero drops; a crashing handler is isolated (agent survives, error logged, heartbeat continues); all 8 heartbeats observable; CPU does not bottleneck during Agent 8 GSADF.

**Depends on:** M3. **Key modules:** `core/broker.py` (exists), `core/events.py`, `agents/`.

---

## M5 — Control-Plane Awakening

**Goal:** wire the React UI + FastAPI control plane to the live Redis bus to observe and govern the M4 agents.

**Deliverables**
- Route Alpha Registry data to the dashboard (surviving 2020–2025 models + their 2026 OOS performance + PBO/DSR/CPCV charts).
- **Redis-backed `EventPublisher`** (swap the in-memory default): `/approve` emits `CEO_APPROVAL`, `/emergency/flatten` emits `EMERGENCY_FLATTEN` to Agent 7.
- **(G2) Live telemetry feed:** SSE/poll endpoint(s) streaming `MARKET_REGIME_BREAK` / `CONCEPT_DRIFT_ALERT` / `AgentHeartbeat`; frontend GSADF + drift + heartbeat panels consume it.
- Persisted approval **audit log**.

**DoD**
- The UI renders CPCV/PBO/DSR charts and live regime/drift/heartbeat state (within bounded latency).
- Approving a strategy requires CEO **signature + TOTP**, updates the registry (`is_deployed`), **and** Agent 7 receives `CEO_APPROVAL` over the live bus — verified end-to-end (mock broker).
- `/emergency/flatten` over Redis closes all mock positions.
- **Playwright E2E against a *live* backend** with a real signature round-trip (closes the box left open in PR #18).

**Depends on:** M4. **Key files:** `control_plane/deps.py`, new `control_plane/feeds.py`, `apps/web/`.

---

## M6 — Broker Bridge (Paper Trading)

**Goal:** connect Agent 7 to MetaTrader 5 paper trading.

**Deliverables**
- Complete `MT5Adapter` (the 4 `NotImplementedError` methods + `connect`/`equity`) via the socket bridge to the **dedicated Windows VM** running the MT5 terminal (confirmed host).
- Route **live MT5 ticks** into Agent 1's ingestion queue — the *same* socket feeds features **and** execution (no feed/fill mismatch).
- Agent 7 dispatches dynamic lot sizes from the Phase-7 continuous bet-sizing math under ESMA/FTMO caps.

**Audit injection (State Rehydration)**
- On startup, Agent 7 queries MT5 for open positions and seeds its committed-margin / `c_95` via `ExecutionEngine.rehydrate_state` (already built) **before** sizing new bets.

**DoD**
- Against an MT5 **demo** account: live ticks generate information bars; Agent 7 dispatches a dynamically sized paper trade; positions reconcile after a simulated restart (rehydration scales the next bet correctly).
- A 50-signal max-confidence burst stays within the FTMO drawdown buffer (live re-verification of the Phase-7 guarantee).

**Depends on:** M2, M5. **Key files:** `execution/brokers/mt5.py`, `execution/feed.py`.

---

## M7 — Agentic Soak & Disaster Drill

**Goal:** final multi-week live paper soak + autonomous safeguard verification.

**Deliverables**
- Dockerize the full stack: `infra/` Dockerfiles (api, web, agents) + `docker-compose.yml` (redis + registry volume + api + web + agent workers); container-ready for AWS.
- Run the system live on paper money: **$100,000 paper account, 30-day continuous run**; KPI tracking (PnL, hit-rate, modeled vs. realized drawdown, slippage); daily reconciliation; operator **runbook**; disaster-recovery drill.
- **The Disaster Drill:** inject a synthetic "flash crash" tick array into the live feed.

**Soak pass thresholds — safety + correctness + stability, *not* profit** (30 days can't confirm edge — same logic as the M3 OOS gate; PnL is reported, never gated):
- **No FTMO breach (hard fail):** max daily loss < **5%** ($5,000); max overall drawdown < **10%** ($10,000).
- **Realized ≤ modeled risk:** realized max drawdown ≤ **1.25×** modeled; realized daily vol ≤ **1.5×** modeled.
- **Execution integrity:** **100%** of halt/flatten triggers executed correctly; daily broker-vs-internal-book reconciliation **exact**; median slippage within tolerance.
- **Ops stability:** **zero** unhandled agent crashes; heartbeat uptime ≥ **99%**.

**DoD (the ultimate test)**
1. The synthetic flash crash triggers Agent 8's GSADF break detection.
2. Agent 8 emits `MARKET_REGIME_BREAK`.
3. Agent 7 **autonomously** intercepts it (per the §Governance contract — risk-reducing, no CEO signature), halts new orders, and flattens all MT5 positions to protect capital.
4. The whole stack runs under `docker compose up`; killing any agent fires an alert + visible heartbeat gap.
5. All soak pass thresholds met; all halt paths (regime break, drift, FTMO/ESMA caps, CEO flatten) exercised; runbook + DR drill passed → **final CEO cryptographic sign-off → live capital.**

**Depends on:** M6.

---

## Cross-cutting (carried from the phases — still binding)
- **Anti-bias / anti-leakage:** stable-plateau selection (M1.1, never the peak); realized `exit_timestamp` purging; truncation-hash; target-shuffling; **zero hard-coded numerics** (all thresholds data-derived; domain constants in `src/afml/config/`).
- **PR-per-milestone**, `Co-Authored-By` sign-off, CLAUDE.md updated each ship, Lead-Quant audit gate between milestones.
- **Config/secrets** centralized in `src/afml/config/` + Keychain; never inline.
- **CI** (from M0) gates every PR.

## Critical path
**M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7.** M1 (with M1.1) is the highest-leverage and the real test of edge; M3's 2026 OOS is the certification gate *before* any live wiring. M4/M5 may partially parallelize once M2/M3 land.

---

## Decisions log (resolved)

| # | Decision | Resolution |
|---|---|---|
| 1 | M1 compute staging | Pilot **EURUSD + BTCUSD**, then fan to all 14; run **locally**. |
| 2 | MT5 hosting (M6) | Socket bridge to a **dedicated Windows VM** running the MT5 terminal. |
| 3 | Retrain cadence (M4) | **Drift-triggered** + **weekly** scheduled fallback; every retrain re-enters OOS gating. |
| 4 | OOS acceptance (M3) | **Breakdown detection, not performance confirmation.** Quarantine only on (a) 2026 Sharpe < 5th-pct of own CPCV path distribution, **or** (b) decayed skill (Brier ≥ baseline / SHAP rank-corr < 0.5). A bad-but-in-band period **survives**. |
| 5 | Plateau objective + grid (M1.1) | Objective = **median PurgedWalkForward OOS Sharpe**; per-family grid ranges (data-derived) proposed in the M1 PR. |
| 6 | 2026 OOS data placement | `{DATA_ROOT}/2026_ytd/{ASSET}_2026-01-01_2026-04-30_DUKASCOPY.parquet`; `config/assets.py` gains `oos_data_path`. |
| 7 | M7 soak | **$100,000** paper account, **30-day** run; pass on **safety/correctness/stability** (no FTMO breach, realized ≤ 1.25× modeled DD, 100% halts fire, exact reconciliation, ≥99% heartbeat uptime, drill passes) — **PnL reported, not gated**. |

All open decisions are resolved. On CEO sign-off this document becomes the binding spec for M0–M7 and is referenced from `CLAUDE.md`.
