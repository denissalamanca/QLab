# CLAUDE.md — AFML Quant Lab

This file is automatically loaded by every Claude Code session in this repo. **Keep it current**: amend the phase status table, conventions, and decision log as the build progresses. Every phase PR must update this file when it ships.

## Mission

Build an institutional-grade autonomous quantitative trading system implementing Marcos López de Prado's *Advances in Financial Machine Learning* "Meta-Labeling" framework end-to-end across 14 liquid FX/Indices/Metals/Crypto instruments, under cryptographic Human-in-the-Loop (HITL) CEO governance. Targets ICMarkets EU ($100K, ESMA-regulated) and FTMO (5 % daily / 10 % max drawdown) prop accounts. Strategies must pass `PBO < 5 %` and `DSR > 1.0` before any capital deployment.

## Binding specifications

Two documents in `docs/` define the work and supersede any other source of truth:

- [AI-First Quant Lab — Architecture & PRD](<docs/AI-First Quant Lab_ AFML Architecture & PRD.md>) — strategic / KPIs / governance.
- [Master Engineering Blueprint — Implementation Spec](<docs/Master Engineering Blueprint_ AFML Implementation Spec.md>) — exact algorithms, schemas, and Definition-of-Done unit tests for each phase.

When the PRD and Blueprint disagree, the Blueprint wins (it's the engineering contract).

## Phase status

| # | Phase | Agent | Status | Blueprint § | PR |
|---|---|---|---|---|---|
| 0 | Orchestration & Alpha Registry | — | ✅ shipped | §2 | [#1](https://github.com/denissalamanca/QLab/pull/1) |
| 1 | Structural Data Engineering | Agent 1 | next | §3 | — |
| 2 | Primary Signals / Brain 1 | Agent 2 | pending | §4 | — |
| 3 | Microstructure Features | Agent 3 | pending | §5 | — |
| 4 | Feature Selection & ONC | Agent 4 | pending | §6 | — |
| 5 | Meta-Labeling / Brain 2 | Agent 5 | pending | §7 | — |
| 6 | Validation / CPCV / DSR | Agent 6 | pending | §8 | — |
| 7 | Bet Sizing & Execution | Agent 7 | pending | §9 | — |
| 8 | MLOps / Structural Breaks | Agent 8 | pending | §10 | — |
| 9 | Control Plane (React/FastAPI) | — | pending | §11 | — |

**Strict phase-by-phase build.** `make phase{N-1}` must be green before any code is written for phase N. No vertical-slice shortcuts. No relaxing of unit-test assertions to make them pass — fix the underlying code.

## Build commands

```bash
make install        # uv sync --all-groups
make sync           # uv sync (default groups only)
make phase0         # ruff + ruff format check + mypy --strict + pytest -m phase0
make phaseN         # same for phase N
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
│   ├── data/                 — Phase 1
│   ├── labeling/             — Phase 2 (CUSUM + Bollinger + Donchian plugins, Triple-Barrier)
│   ├── features/             — Phase 3
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
