"""Phase 9 — AFML 0-9 final integration audit edge cases (V2-V5).

- **V2** async routes never block the ASGI loop (concurrent reads via threadpool).
- **V3** numpy / pandas artifacts serialize over the API without crashing.
- **V4** the execution lock serializes fetch→size→dispatch (no stale-margin race).
- **V5** a stagnant (zero-variance) series can't crash GSADF.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd
import pytest

from afml.control_plane.schemas import StrategyOut
from afml.execution.brokers.base import OrderSide
from afml.execution.brokers.mock import InMemoryMockBroker
from afml.execution.pipeline import ExecutionEngine, Signal
from afml.execution.risk import RiskEngine
from afml.monitoring.gsadf import detect_bubble, gsadf_statistic
from tests.unit.phase9.conftest import CPHarness

pytestmark = pytest.mark.phase9


# ---------------------------------------------------------------- V2: async I/O
async def test_concurrent_strategy_reads_do_not_block(harness: CPHarness) -> None:
    """20 concurrent reads all succeed — the blocking SQLite read is offloaded
    to a worker thread (run_in_threadpool), so the event loop never stalls."""
    transport = httpx.ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://cp") as client:
        responses = await asyncio.gather(*[
            client.get("/api/v1/registry/strategies") for _ in range(20)
        ])
    assert all(r.status_code == 200 for r in responses)
    assert all(len(r.json()) == 1 for r in responses)


# ------------------------------------------------------- V3: numpy/pandas JSON
def test_strategy_out_scrubs_numpy_and_pandas_types() -> None:
    """numpy.float64 / numpy.int64 / pandas.Timestamp must serialize cleanly."""
    obj = SimpleNamespace(
        experiment_id=uuid4(),
        asset="EURUSD",
        algorithmic_family="cusum",
        agent_version="agent_6@audit",
        timestamp=pd.Timestamp("2026-05-20T00:00:00Z"),
        num_events_triggered=np.int64(750),
        orthogonality_score=np.float64(0.12),
        brain_1_recall=np.float64(0.82),
        brain_2_log_loss=np.float64(0.41),
        pbo=np.float64(0.03),
        dsr=np.float64(1.42),
        is_deployed=False,
        status="completed",
    )

    out = StrategyOut.model_validate(obj)

    # Native Python scalar types after validation.
    assert type(out.num_events_triggered) is int
    assert type(out.pbo) is float
    assert type(out.dsr) is float

    # The crucial assertion: JSON serialization must not raise
    # "Object of type int64/float64 is not JSON serializable".
    payload = out.model_dump_json()
    assert '"num_events_triggered":750' in payload
    assert "2026-05-20T00:00:00" in payload  # pandas Timestamp → ISO-8601


def test_strategy_out_handles_none_optionals() -> None:
    obj = SimpleNamespace(
        experiment_id=uuid4(),
        asset="BTCUSD",
        algorithmic_family="donchian",
        agent_version="agent_6@audit",
        timestamp=pd.Timestamp("2026-05-20T00:00:00Z"),
        num_events_triggered=np.int64(500),
        orthogonality_score=None,
        brain_1_recall=None,
        brain_2_log_loss=None,
        pbo=None,
        dsr=None,
        is_deployed=False,
        status="completed",
    )
    out = StrategyOut.model_validate(obj)
    assert out.pbo is None
    out.model_dump_json()  # must not raise


# ------------------------------------------------------- V4: execution race
async def test_async_execution_lock_prevents_stale_margin_overcommit() -> None:
    """Two signals arriving concurrently must not both size against an empty
    book. The lock serializes fetch→size→dispatch: the first claims the budget,
    the second sees it (via rehydrate) and is clamped to zero."""
    broker = InMemoryMockBroker(starting_equity=100_000.0)
    broker.connect()
    risk = RiskEngine(account_equity=100_000.0, c95=1.0)  # margin_budget = 10_000
    engine = ExecutionEngine(broker=broker, risk_engine=risk)

    # Two max-confidence crypto signals (0.5 margin fraction) on distinct
    # assets — each alone would demand 50k margin → clamped to the 10k budget.
    sig_btc = Signal(asset="BTCUSD", probability=0.99, side=OrderSide.BUY, reference_price=60_000.0)
    sig_eth = Signal(asset="ETHUSD", probability=0.99, side=OrderSide.SELL, reference_price=3_000.0)

    results = await asyncio.gather(
        engine.execute_batch_async([sig_btc]),
        engine.execute_batch_async([sig_eth]),
    )

    # Total committed never breaches the FTMO budget.
    assert risk.committed_margin <= risk.margin_budget + 1e-6
    assert risk.committed_margin == pytest.approx(risk.margin_budget)
    # Exactly one position opened — the second concurrent signal saw the first's
    # margin and was clamped to zero (skipped), proving no double-spend.
    assert len(broker.open_positions()) == 1
    assert sum(r.n_orders_submitted for r in results) == 1


async def test_async_flatten_shares_the_execution_lock() -> None:
    broker = InMemoryMockBroker(starting_equity=100_000.0)
    broker.connect()
    risk = RiskEngine(account_equity=100_000.0, c95=1.0)
    engine = ExecutionEngine(broker=broker, risk_engine=risk)
    await engine.execute_batch_async([
        Signal(asset="EURUSD", probability=0.9, side=OrderSide.BUY, reference_price=1.10)
    ])
    assert len(broker.open_positions()) == 1

    closes = await engine.emergency_flatten_async({"EURUSD": 1.10})
    assert len(closes) == 1
    assert broker.open_positions() == []
    assert risk.committed_margin == 0.0


# ------------------------------------------------------- V5: GSADF stagnation
def test_gsadf_flat_series_returns_zero_without_crashing() -> None:
    """A perfectly flatlined window → zero-variance design matrix. The guard
    returns 0.0 (no explosive root) instead of a singular regression."""
    flat = np.full(120, 1.2345, dtype=np.float64)
    assert gsadf_statistic(flat) == 0.0

    result = detect_bubble(flat, n_simulations=20)
    assert result.is_bubble is False
    assert result.gsadf_statistic == 0.0
    assert np.isfinite(result.gsadf_statistic)


def test_gsadf_constant_then_jump_is_finite() -> None:
    """A near-stagnant series with a single late move must stay finite (no -inf)."""
    y = np.concatenate([np.full(100, 5.0), np.linspace(5.0, 5.0001, 20)]).astype(np.float64)
    stat = gsadf_statistic(y)
    assert np.isfinite(stat)
