# CLAUDE.md — AFML Quant Lab

This file is automatically loaded by every Claude Code session in this repo. **Keep it current**: amend the phase status table, conventions, and decision log as the build progresses. Every phase PR must update this file when it ships.

## Mission

Build an institutional-grade autonomous quantitative trading system implementing Marcos López de Prado's *Advances in Financial Machine Learning* "Meta-Labeling" framework end-to-end across 14 liquid FX/Indices/Metals/Crypto instruments, under cryptographic Human-in-the-Loop (HITL) CEO governance. Targets ICMarkets EU ($100K, ESMA-regulated) and FTMO (5 % daily / 10 % max drawdown) prop accounts. Strategies must pass `PBO < 5 %` and `DSR > 1.0` before any capital deployment.

## Binding specifications

Two documents in `docs/` define the work and supersede any other source of truth:

- [AI-First Quant Lab — Architecture & PRD](<docs/AI-First Quant Lab_ AFML Architecture & PRD.md>) — strategic / KPIs / governance.
- [Master Engineering Blueprint — Implementation Spec](<docs/Master Engineering Blueprint_ AFML Implementation Spec.md>) — exact algorithms, schemas, and Definition-of-Done unit tests for each phase.

When the PRD and Blueprint disagree, the Blueprint wins (it's the engineering contract).

Once the AFML build phases (0–9) shipped, the **operational** build (turning the validated engine into an autonomous live lab) is governed by a third binding doc:

- [Master Operational Roadmap (M0–M7)](docs/OPERATIONS_ROADMAP.md) — ✅ CEO-approved; deliverables + DoD per operational milestone, the locked data windows (research 2020–2025 / OOS 2026 YTD), the autonomous-vs-HITL governance contract, and the resolved decisions log.

## Phase status

| # | Phase | Agent | Status | Blueprint § | PR |
|---|---|---|---|---|---|
| 0 | Orchestration & Alpha Registry | — | ✅ shipped | §2 | [#1](https://github.com/denissalamanca/QLab/pull/1) |
| 1 | Structural Data Engineering | Agent 1 | ✅ shipped | §3 | [#3](https://github.com/denissalamanca/QLab/pull/3) |
| 2 | Primary Signals / Brain 1 | Agent 2 | ✅ shipped | §4 | [#4](https://github.com/denissalamanca/QLab/pull/4) |
| 3 | Microstructure Features | Agent 3 | ✅ shipped | §5 | [#5](https://github.com/denissalamanca/QLab/pull/5) |
| 4 | Feature Selection & ONC | Agent 4 | ✅ shipped | §6 | [#8](https://github.com/denissalamanca/QLab/pull/8) |
| 5 | Meta-Labeling / Brain 2 | Agent 5 | ✅ shipped | §7 | [#9](https://github.com/denissalamanca/QLab/pull/9) |
| 6 | Validation / CPCV / DSR | Agent 6 | ✅ shipped | §8 | [#12](https://github.com/denissalamanca/QLab/pull/12) |
| 7 | Bet Sizing & Execution | Agent 7 | ✅ shipped | §9 | [#15](https://github.com/denissalamanca/QLab/pull/15) |
| 8 | MLOps / Structural Breaks | Agent 8 | ✅ shipped | §10 | [#16](https://github.com/denissalamanca/QLab/pull/16) |
| 9 | Control Plane (React/FastAPI) | — | ✅ shipped | §11 | [#18](https://github.com/denissalamanca/QLab/pull/18) |

**Strict phase-by-phase build.** `make phase{N-1}` must be green before any code is written for phase N. No vertical-slice shortcuts. No relaxing of unit-test assertions to make them pass — fix the underlying code.

## Operational milestones (post-Phase-9)

The run/ops layer per [`docs/OPERATIONS_ROADMAP.md`](docs/OPERATIONS_ROADMAP.md). Same discipline: one milestone per PR, gated on its DoD, Lead-Quant audit between milestones.

| # | Milestone | Status | PR |
|---|---|---|---|
| M0 | Genesis & Operator Bootstrap (`afml` CLI, enroll-ceo, doctor, CI) | ✅ shipped | [#21](https://github.com/denissalamanca/QLab/pull/21) |
| M1 | Research Harness & Sweep (plateau selection, two-stage certification, registry population) | ✅ shipped | — |
| M2 | Model Persistence & Artifact Lifecycle | — | — |
| M3 | 2026 Out-of-Sample Walk-Forward | — | — |
| M4 | Autonomous Agent Runtime (orchestrator + 8 agents) | — | — |
| M5 | Control-Plane Awakening (Redis wiring + live feeds) | — | — |
| M6 | Broker Bridge (MT5 paper trading) | — | — |
| M7 | Agentic Soak & Disaster Drill → live | — | — |

### M1 research-harness contracts (`src/afml/research/`)

- **Bar granularity is regime-derived, not curve-fit** (`docs/specs/M1_bar_granularity.md`). A `HoldingRegime` fixes the economic timescale; the first-passage law gives `Δ = mean_hold / pt²` and `V = round(max_hold / Δ)`. The JB bar-type tournament is **constrained to that granularity** (pure JB minimisation is degenerate — CLT drives it to the finest bar). Default `day` (mean-hold 6h); the system sweeps `scalp/day/swing/position` — never hard-coded to day-trading.
- **Plateau selection (anti-curve-fit).** `select_plateau` — Neighborhood-Minimax `R(g)=min(s(g), neighbors)`; full-interior eligibility (a boundary point can't claim robustness); **no free tuning constant**; reject (`selected=None`) when `R(g*) < s_floor` — "no stable configuration" is a valid outcome, never a manufactured strategy.
- **Surface objective** `s(g)` = median PurgedWalkForward OOS **strategy** Sharpe (`oos_strategy_sharpe`: bet-size each OOS proba via `calculate_bet_size`, apply side, annualise). A config is valid for the surface only if `events ≥ 500` **and** calibrated Brier beats naive. Recall is intentionally absent (undefined OOS).
- **Estimator split (compute).** Surface = uniqueness-weighted **RandomForest** (`estimator="rf"`, `n_jobs=-1`, no bootstrap, O(n log n)). Certification cohort = a RF **complexity ladder** (+ XGBoost) — the right cohort for PBO (model complexity is the overfitting axis). The sequential-bootstrap **SBRF is O(n²)** → intractable inside CPCV/target-shuffling; the deployable weighted SBRF is fit in M2 on the certified config.
- **Two-stage certification.** `sweep_and_certify`: surface sweep → plateau centre → `certify` runs `validate_strategy` (CPCV → PBO, DSR, target-shuffling) and writes `(pbo, dsr)` back to the winner's registry row via `record_validation`. `n_trials` = per-`(asset, family)` cohort count (≥ 30 after one full grid → clears the DSR cold-start quarantine). Typed early-exits (`insufficient_events`, `halted_at_mda`, `data_leakage`, `degenerate_cpcv`) never crash a batch.
- **Every config is a registry trial** (`completed` / `FAILED_AT_MDA`) — drives the DSR `K`; the sweep is **resumable** through registry dedup.
- **CLI + artifacts.** `afml research sweep|select|report`. One JSON run artifact per `(asset, family)` (resumable, auditable) + the `RESEARCH_RUN.md` evidence sheet (bar JB, events, plateau coord + worst-neighbor `R`, Brier vs. naive, surviving clusters, PBO, DSR). Markers: `make m1`.

## Build commands

```bash
make install        # uv sync --all-groups
make sync           # uv sync (default groups only)
make phase0         # ruff + ruff format check + mypy --strict + pytest -m phase0
make phaseN         # same for phase N
make integration    # ruff + mypy + pytest -m integration (cross-phase end-to-end)
make lint           # ruff check + ruff format --check
make type           # mypy --strict src tests
make fix            # ruff check --fix && ruff format
make test           # full pytest
make redis-up       # docker compose up -d redis
make redis-down     # docker compose down
make clean          # remove caches and artifacts
```

## Anti-bias / banned methods

The following are forbidden across `src/afml/`. Reviewers and CI must catch them.

| Banned | Required replacement |
|---|---|
| Hard-coded numerics (`window_size = 14`, `stop_loss = 20`) | Derived from data (rolling EWM moments, statistical tests). Domain constants live in `src/afml/config/`. |
| `sklearn.model_selection.KFold` / `TimeSeriesSplit` | Purged + Embargoed CV (built from numpy). |
| `pd.Series.diff()` on price series | Fixed-Width Fractional Differencing (FFD). |
| MDI / Gini `feature_importances_` | Clustered MDA via Purged K-Fold. |
| Raw `accuracy_score` / `roc_auc_score` as primary metric | Brier score, Negative Log-Loss, Precision, F1. |
| Expanding-window indicators | Fixed-width rolling windows + `.shift(1)` to enforce causality. |
| Selecting hyperparameters at an isolated performance peak | Select from a **stable plateau**. |

## Anti-leakage gates (mandatory)

- **Truncation hash test** (Phases 1, 3): SHA-256 of a derived series computed on full vs truncated data must match exactly over the overlap. Proves zero future-data leakage.
- **Index intersection test** (Phase 5): `max(train_times) + embargo < min(test_times)` for every CV fold, millisecond precision.
- **Target shuffling test** (Phase 6): retrain with randomly shuffled labels; if the model retains predictive power → `DataLeakageError`, strategy permanently rejected.

## Cross-phase integration contracts (AFML 0-4 audit)

- **V1 — realized t1.** Triple-Barrier output exposes both `event_timestamp` (t0), `vertical_timestamp` (conservative upper bound) and `exit_timestamp` (the **realized** barrier-touch time). Phase 4 / Phase 5 purging schemes (`PurgedKFold`, `PurgedWalkForwardCV`) **must** be given `exit_timestamp` as t1 — never the vertical. Realized t1 yields tighter purging and recovers training data without leaking.
- **V2 — burn-in alignment.** Phase 3 rolling-window features carry an unavoidable `max(window) - 1` burn-in. Phase 2 events that fire inside the burn-in must be dropped from the labels frame, not forward-filled. Use `afml.data.align_labels_to_features(labels_df, features)` / `align_events_to_features(events, features)` at every Phase 2 → Phase 3 → Phase 4 handoff. The aligned labels frame and the features frame must always have equal row counts.
- **V3 — empty-MDA circuit breaker.** `select_features` accepts an optional `registry: AlphaRegistryRepository` plus `experiment_metadata`. On empty survivors, it sets `SelectionResult.halted_at_mda = True` and (when a registry is wired) logs a `FAILED_AT_MDA` row — preserving the DSR multiple-testing trial count without crashing Phase 5.

Integration gate: `make integration` runs `tests/integration/test_phase1_to_4.py` which exercises a synthetic raw-tick → Phase 4 end-to-end with all three contracts asserted.

## Phase 5 leakage / calibration contracts (AFML 0-5 audit)

- **V1 — calibration via PurgedKFold.** ``CalibratedClassifierCV`` defaults to a stratified KFold that shuffles non-IID labels. Two safe alternatives are exposed:
  - **SBRF path:** ``afml.modeling.fit_calibrated_sbrf_with_purged_cv`` runs a manual cross-fitting loop that *explicitly invokes* ``PurgedKFold.split(t0, t1)``. Tests spy on the call.
  - **Generic-classifier path (XGBoost / vanilla RF):** ``afml.modeling.fit_calibrated_classifier_with_purged_cv`` passes ``cv=PurgedKFoldSklearn(t0, t1)`` into ``CalibratedClassifierCV``.
- **V2 — strict embargoed outer evaluation.** ``train_brain_two`` uses ``PurgedWalkForwardCV`` for the OUTER train/holdout split with a per-fold ``max(train_t1) + embargo < min(holdout_t0)`` assertion via ``FoldDiagnostics.passes_index_intersection``.
- **V3 — XGBoost mirror + sample-weight propagation.** When ``compare_with_xgboost=True`` (default), both SBRF and XGBoost get fit with ``sample_weight = ū_i`` and calibrated via purged CV. Tests verify weight propagation by comparing skewed-weight vs uniform-weight predictions.
- **V2.1 — empty-MDA pass-through.** When Phase 4's circuit breaker fires (``X.shape[1] == 0``), ``train_brain_two`` returns a sentinel ``BrainTwoResult`` with ``halted_at_mda_upstream=True`` instead of crashing.
- **Weight normalisation (pre-Phase-6 patch).** Uniqueness weights ``ū_i ∈ (0,1]`` are normalised to sum to ``N`` (``ū_i × N/Σū_i``) before any estimator fit — ``afml.modeling.calibration._normalize_sample_weights``. Prevents XGBoost ``min_child_weight`` vanishing-gradient suppression. Scale-invariant for sklearn trees.

## Phase 8 monitoring contracts (Blueprint §10)

- **GSADF (primary).** ``afml.monitoring.detect_bubble`` — Phillips-Wu-Yu 2011 double-sup ADF over all sub-windows (numba-JIT inner ADF), Monte-Carlo critical value under the random-walk null. ``is_bubble`` (stat > 95% crit) → ``MarketRegimeBreak`` event → halt Agent 7.
- **Chow (secondary).** ``chow_break_test`` — F-test for a break in the DF/AR(1) regression at a candidate point. Confirming diagnostic; GSADF is the decision-maker.
- **SHAP drift.** ``compute_shap_importance`` (mean |SHAP| per feature) + ``spearman_rank_correlation``; ``detect_concept_drift`` fires when rank corr < 0.5 → ``ConceptDriftAlert``.
- **Monitor.** ``StructuralBreakMonitor.check_regime`` / ``check_drift`` produce the Phase 0 event objects (``MarketRegimeBreak`` / ``ConceptDriftAlert``) for the agent runtime to publish — transport-free, unit-testable.

## Phase 7 execution contracts (Blueprint §9)

- **Bet sizing.** ``calculate_bet_size(p)`` = ``2·Φ(z)-1`` with ``z=(p-0.5)/√(p(1-p))``; ``0`` when ``p ≤ 0.5``. Batch sizing (``bet_sizes_for_batch``) auto-switches to a 2-component Gaussian-mixture CDF when the active z-scores fail Shapiro-Wilk (``p<0.05``).
- **Risk engine.** ``RiskEngine`` applies (1) ``c_95`` concurrent-position scaling, (2) ESMA leverage caps (``src/afml/config/risk.py`` — FX 30:1, index/metal 20:1, crypto 2:1), and (3) a hard FTMO drawdown-buffer cap (10% of equity) so total committed margin never breaches even under a 50-signal burst.
- **Broker contract.** ``BrokerAdapter`` ABC; ``InMemoryMockBroker`` for tests/dry-runs; ``MT5Adapter`` scaffold (lazy ``MetaTrader5`` import → clear error off-terminal; live wiring deferred to the broker-integration milestone). Same MT5 socket feeds features + execution.
- **ExecutionEngine.** Signals → batch sizing → per-bet risk sizing (descending-confidence order) → broker dispatch; ``emergency_flatten`` closes all + resets the budget (wired to the Phase 9 control-plane ``/emergency/flatten``).
- **Risk constants live in ``src/afml/config/risk.py``** — never inline.

## Phase 6 validation contracts (AFML 0-6 audit)

- **V1 — PBO needs a cohort.** ``compute_pbo`` rejects single-strategy (n×1) matrices — PBO is a *relative* statistic. Build the multi-strategy matrix via ``afml.validation.build_cohort_performance_matrices`` (runs a registry cohort through CPCV) and size the cohort with ``count_cohort_trials(registry, asset, family)``.
- **V2 — DSR cold-start breaker.** ``deflated_sharpe_ratio`` rejects when ``n_trials < DSR_MIN_TRIALS`` (=30): returns ``dsr=0.0``, ``quarantined=True``, logs ``"Insufficient trials for DSR (K<30). Auto-Quarantine."``. ``ValidationResult.passes_phase6_dod`` hard-fails on quarantine.
- **V3 — non-contiguous CPCV embargo.** ``CombinatoriallyPurgedKFold`` applies purge + embargo to the right boundary of *each contiguous test block*, not the global ``max(t1)``. A combination testing groups {0, 2} embargoes both group-0's and group-2's right edges, protecting the middle training block (group 1).

## Phase 9 control-plane contracts (Blueprint §11)

The CEO Human-in-the-Loop governance layer. **Testable, type-checked logic lives in ``src/afml/control_plane/``** (covered by ``make phase9`` / mypy strict); ``apps/api/main.py`` is a thin production ASGI entry, and ``apps/web/`` is the React frontend (validated by its own Node toolchain, not the Python gate).

- **Two trust levels, matching the Phase 0 event contracts.**
  - **Approve** (``CEOApproval``) commits capital → requires a valid Ed25519 signature over ``afml:approve:<experiment_id>:<timestamp_ms>`` **AND** a live TOTP code (§11.2 mandatory 2FA).
  - **Flatten** (``EmergencyFlatten``) is a risk-*reducing* kill-switch → requires the signature over ``afml:flatten:<nonce>:<timestamp_ms>`` only (never blocked by a TOTP window), plus a per-nonce replay guard.
  - Both bind a ±60 s ``timestamp_ms`` into the signed message (audit V1 anti-replay) — see the hardening section below.
- **Keys.** The server holds only the CEO **public** key (verification) + the TOTP secret (from the Keychain). The **private key never enters the server or the browser** — the CEO signs on their own device and pastes the signature. ``CEOAuthenticator`` (``src/afml/control_plane/security.py``) is the single verification seam; failures raise ``CEOAuthError`` subclasses → HTTP 403.
- **Endpoints (``/api/v1``).** ``GET /registry/strategies`` → ``AlphaRegistryRepository.awaiting_signoff()`` (``completed`` status, ``pbo``/``dsr`` populated, not deployed). ``POST /execution/approve`` → verify → ``mark_deployed`` + publish ``CEOApproval``. ``POST /emergency/flatten`` → verify → ``ExecutionEngine.emergency_flatten`` (closes all + resets risk budget) + publish ``EmergencyFlatten``.
- **Registry extension.** ``Experiment`` gained nullable ``pbo``/``dsr`` columns; ``record_validation(experiment_id, *, pbo, dsr)`` populates them (Phase 6 → Phase 9 handoff). Backward-compatible (additive nullable columns).
- **DI + transport seam.** ``ControlPlaneDeps`` bundles repository + execution engine + authenticator + ``EventPublisher``. Default ``InMemoryEventPublisher`` records events for tests; production swaps a Redis-backed publisher. ``create_app(deps)`` stashes them on ``app.state`` — tests inject in-memory doubles, no Redis/Keychain/real broker.
- **Crypto primitives** live in ``src/afml/crypto/`` (``signing`` Ed25519, ``totp`` RFC 6238, ``keychain`` ``keyring`` façade), re-exported from ``afml.crypto``.
- **Frontend** (``apps/web/``): React 18 + Vite + TS + Tailwind + shadcn-style UI + Recharts (PBO/DSR ship-gate charts). ``npm run typecheck`` / ``npm run build`` / ``npm run e2e`` (Playwright) all green; the approval-flow E2E mocks the API and asserts the §11.1 request contract.

## AFML 0-9 final integration audit (production hardening)

Five boundary vulnerabilities patched before live clearance. Edge cases covered in ``tests/unit/phase9/test_final_integration_audit.py`` + ``tests/integration/test_phase9_control_plane.py``.

- **V1 — cryptographic anti-replay.** The signed message binds a millisecond ``timestamp_ms`` (``afml:approve:<id>:<ts>`` / ``afml:flatten:<nonce>:<ts>``). ``CEOAuthenticator`` rejects ``abs(now − ts) > 60_000`` (``StaleTimestampError`` → 403). Flatten keeps its single-use nonce guard, so a captured payload is replayable neither within the window (nonce consumed) nor after (timestamp stale). ``time_provider`` is injectable for tests. Frontend captures a fresh ``Date.now()`` per modal open.
- **V2 — no ASGI event-loop blocking.** Control-plane routes are ``async def`` and offload every blocking SQLite/broker call through ``fastapi.concurrency.run_in_threadpool`` (synchronous crypto verify stays on the loop — microseconds). Concurrent-read tests assert the loop never stalls.
- **V3 — numpy/pandas API serialization.** ``StrategyOut`` carries ``mode="before"`` field validators that coerce ``numpy.float64``/``numpy.int64`` → native ``float``/``int`` and ``pandas.Timestamp`` → ``datetime`` (ISO-8601 on dump), so the response can't crash with ``Object of type int64 is not JSON serializable``.
- **V4 — execution race condition.** ``ExecutionEngine`` gained an ``asyncio.Lock`` (lazily bound to the running loop) + ``execute_batch_async`` / ``emergency_flatten_async``. The lock serializes **fetch live margin/positions → size → dispatch** so two concurrently-arriving signals can't both size against a stale margin snapshot and over-leverage. Blocking broker round-trips run via ``asyncio.to_thread``.
- **V5 — GSADF stagnant-tick guard.** ``gsadf_statistic`` / ``detect_bubble`` short-circuit to ``0.0`` (no explosive root) when ``np.var(y) < 1e-10`` — a flatlined window can't drive a singular ADF design matrix (and the statistic is clamped finite, never ``-inf``).

## AFML 0-9 final polishing audit (runtime hardening)

Four runtime-killer API/framework/hardware bottlenecks patched before live. Edge cases in ``tests/unit/phase9/test_final_polishing_audit.py``.

- **P1 — GMM has no ``.cdf()``.** ``sklearn.mixture.GaussianMixture`` exposes only ``score_samples`` / ``predict_proba``. The MoG bet-sizing fallback (``afml.execution.bet_sizing._mixture_cdf_sizes``) computes the mixture CDF *manually* as ``Σ_k w_k·Φ((z−μ_k)/σ_k)`` via ``scipy.stats.norm.cdf`` (weights/means/√covariances from the fitted GMM) — never ``gmm.cdf``. A dedicated test validates it against an independent recompute + asserts the attribute absence.
- **P2 — GSADF off the event loop.** GSADF is ``O(T²)`` nested OLS; running it per raw tick pegs CPU and starves Agent 7. ``StructuralBreakMonitor.check_regime_async`` offloads the sweep via ``loop.run_in_executor`` (returns the identical ``RegimeCheck``), and Agent 8 binds **only** to ``BAR_GENERATED`` (information-bar) events — never ``NEW_TICK``.
- **P3 — persistent TOTP seed.** ``afml.crypto.get_or_create_ceo_totp_secret`` loads the seed from the Keychain or, on first run, generates + persists it and echoes the ``otpauth://`` provisioning URI once. ``apps/api/main.py`` uses it so the CEO's authenticator survives every restart (no ephemeral-seed lockout). ``echo`` is injectable for tests.
- **P4 — strict CORS.** ``create_app`` sets ``CORSMiddleware`` with explicit Vite origins (``localhost``/``127.0.0.1:5173``, never ``"*"``), ``allow_credentials=True``, ``allow_methods=["*"]`` (so the browser preflight is never blocked), ``allow_headers=["*"]``.

## Workflow — PR per milestone

Remote: **<https://github.com/denissalamanca/QLab>** (default branch `main`).

1. Phase work: branch `phase-{N}-{slug}` (e.g. `phase-1-data-engineering`).
2. Foundational chores: `chore/{slug}`.
3. **Never push directly to `main`** — the harness blocks it.
4. PR body must include: Summary, DoD evidence table, gate output (ruff / mypy / pytest counts), test plan, next-milestone preview.
5. Sign commits with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
6. Wait for explicit sign-off / merge before starting the next phase's branch.

## Repo map

```
QLab/
├── CLAUDE.md                 — this file (keep current)
├── docs/                     — binding specifications (PRD, Blueprint)
├── pyproject.toml            — uv workspace; ruff; mypy strict; pytest phase markers
├── Makefile                  — make phase{0..9}, make lint/type/test
├── docker-compose.yml        — Redis 7-alpine for the broker
├── .gitignore                — update when a phase adds new artifact patterns
├── src/afml/
│   ├── config/{assets, settings}.py
│   ├── core/{events, broker}.py, registry/
│   ├── data/                 — Phase 1 (ingest, bars, FFD, stationarity, causality)
│   ├── labeling/             — Phase 2 (CUSUM + Bollinger + Donchian plugins, Triple-Barrier)
│   ├── features/             — Phase 3 (50+ causal microstructure metrics)
│   ├── selection/            — Phase 4 (ONC + Clustered MDA)
│   ├── modeling/             — Phase 5 (Sequential bootstrap, SBRF, calibration)
│   ├── validation/           — Phase 6 (CPCV, PBO, DSR, FWER, target shuffling)
│   ├── execution/            — Phase 7 (bet sizing, BrokerAdapter, MT5)
│   ├── monitoring/           — Phase 8 (GSADF, Chow, SHAP drift)
│   ├── crypto/               — Ed25519 + macOS Keychain + TOTP
│   └── agents/               — agent process entry points
├── apps/
│   ├── api/                  — FastAPI Control Plane backend (Phase 9)
│   └── web/                  — React + Vite frontend (Phase 9)
└── tests/
    ├── unit/phase{0..9}/     — Blueprint DoD tests per phase
    ├── property/             — hypothesis-based AFML invariants
    ├── integration/          — end-to-end with real services
    └── benchmarks/           — perf tests (real Redis throughput, etc.)
```

## Asset universe (14, canonical)

Defined in [`src/afml/config/assets.py`](src/afml/config/assets.py). **Never hard-code asset lists elsewhere.**

| Class | Symbols |
|---|---|
| **FX (8)** | EURUSD (2016–25), GBPUSD (2016–25), USDJPY, AUDUSD, USDCHF, NZDUSD, EURGBP, EURJPY |
| **Indices (2)** | DAX, USA500 |
| **Metals (2)** | XAUUSD, XAGUSD |
| **Crypto (2)** | BTCUSD, ETHUSD |

**Tick data:** `/Users/dsalamanca/vs_env/Antigravity/Quant Lab/data/multi_year_consolidated/{ASSET}_{START}_{END}_DUKASCOPY.parquet`. Schema: `timestamp[ms,UTC], ask, bid, ask_volume, bid_volume`. The `hybrid/` subdirectory is **ignored** (raw Dukascopy only — we cannot trust legacy blended feeds).

**`Antigravity/` is a read-only data store.** Never import shared utilities from it. This is a clean-room build.

## Tech stack (locked)

| Layer | Choice |
|---|---|
| Runtime | Python 3.12 |
| Deps / build | uv (`pyproject.toml`, dep groups `core / data / ml / api / dev`, `default-groups = ["dev"]`) |
| Lint / format | ruff (target py312, line 100) |
| Type checking | mypy `--strict` + pydantic plugin |
| Tests | pytest + pytest-asyncio + hypothesis |
| Tabular | Polars + DuckDB (out-of-core for tick parquets) |
| Hot math | numpy + numba (JIT for FFD weights, ONC distance, GSADF) |
| ML | scikit-learn, xgboost, statsmodels, shap, scipy |
| Async / messaging | redis-py 7.x asyncio + fakeredis (tests) |
| Storage | SQLite (WAL mode) via SQLAlchemy 2.0 + Alembic |
| API | FastAPI + uvicorn |
| Frontend | React 18 + Vite + TypeScript + TanStack Query + Tailwind + shadcn/ui + Recharts + Plotly |
| Crypto | `cryptography` (Ed25519) + `keyring` (macOS Keychain) + `pyotp` (TOTP) |
| Broker (Phase 7) | Abstract `BrokerAdapter` + in-memory mock first → MetaTrader 5 via Wine/VM bridge |
| Live feed | Broker's own MT5 socket (same prices for features and execution) |
| Containers | Docker Compose for Redis + Alpha Registry volume + Control Plane; container-ready for later lift to AWS EC2/ECS |

**Per-asset modeling** — one Brain 1 + Brain 2 pair per instrument (14 pairs). Cross-asset coordination only through the shared `feature_registry` + Alpha Registry orthogonality checks. FX vs indices vs crypto have fundamentally different microstructure; one model per asset prevents regime interference.

**Brain 1 primary alpha plugins** (Phase 2): Symmetric CUSUM (volatility events), Bollinger Band Mean-Reversion (ranges), Donchian Channel Breakout (momentum). Agent 2 sweeps all three; Orthogonality Checks compare across the registry.

## How to ship a phase (checklist)

1. Read `docs/Master Engineering Blueprint_ AFML Implementation Spec.md` §{N+2} end-to-end.
2. Confirm previous phase is green: `make phase{N-1}`.
3. Branch off latest `main`: `git checkout main && git pull && git checkout -b phase-{N}-{slug}`.
4. Implement under `src/afml/{module}/`. **Every Blueprint DoD claim must have a matching test in `tests/unit/phase{N}/`.**
5. Add new module imports to mypy overrides only if a library lacks stubs.
6. Iterate `make phase{N}` until green (ruff + mypy + pytest all pass).
7. Update **this CLAUDE.md** — mark the phase shipped in the status table; record any new conventions.
8. Update `.gitignore` if the phase introduces new ignorable artifacts (e.g. local model pickles, generated parquets).
9. `git add -A && git commit` with detailed body + `Co-Authored-By` line.
10. `git push -u origin phase-{N}-{slug}` and `gh pr create --base main --head ... --title "Phase {N} — ..." --body "..."`.
11. Report PR URL to the user. **Wait for sign-off** before starting phase N+1.

## Glossary

- **AFML** — *Advances in Financial Machine Learning* (López de Prado, Wiley 2018).
- **Brain 1** — high-recall structural event filter (Phase 2). Detects setups; profitability is irrelevant.
- **Brain 2** — meta-model that predicts P(success | Brain 1 setup); sizes or vetoes bets (Phase 5).
- **CPCV** — Combinatorially Purged Cross-Validation. Generates synthetic OOS paths for backtest overfitting analysis.
- **CUSUM** — Cumulative Sum (Symmetric) filter that fires on cumulative price moves above a vol-scaled threshold.
- **DSR** — Deflated Sharpe Ratio. Penalizes a strategy's Sharpe against the expected max Sharpe under multiple testing (Bailey & López de Prado 2014).
- **FFD** — Fixed-Width Fractional Differencing. Stationary diff that preserves long memory (vs naive `.diff()`).
- **FWER** — Familywise Error Rate. Multiple-testing penalty.
- **GSADF** — Generalized Supremum Augmented Dickey-Fuller (Phillips, Wu, Yu 2011). Detects explosive-root regime breaks.
- **HITL** — Human-in-the-Loop. CEO cryptographically signs every capital deployment.
- **MDA** — Mean Decrease Accuracy (Clustered). Permutation-based feature importance via Purged K-Fold; replaces MDI.
- **ONC** — Optimal Number of Clusters (López de Prado). Hierarchical clustering with silhouette-maximized cut.
- **OFI** — Order Flow Imbalance. Microstructure feature, strictly causal.
- **PBO** — Probability of Backtest Overfitting. Must be < 5 % for a strategy to ship.
- **SBRF** — Sequentially Bootstrapped Random Forest. Sample-weighted RF respecting non-IID labels via the Average-Uniqueness scheme.
- **TIB / TRB** — Tick Imbalance Bar / Tick Run Bar. Information-driven sampling alternatives to time bars.
- **Triple-Barrier** — labeling method: profit-take / stop-loss / vertical-time barriers, all dynamic (EWM volatility).
