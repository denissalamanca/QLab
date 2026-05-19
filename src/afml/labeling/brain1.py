"""Brain 1 orchestrator — runs primary-alpha plugins and labels their events.

Pipeline per Blueprint §4:

1. For each ``PrimaryAlpha`` plugin in the sweep, call ``plugin.detect(bars)``.
2. Apply Triple-Barrier labeling to the events (dynamic EWM-vol barriers).
3. Log the experiment to the Alpha Registry (``(asset, family, hparams)`` →
   immutable UUID). Duplicates are silently skipped — Phase 6 reads the trial
   count to compute the Deflated Sharpe Ratio penalty.

The result list lets Phase 3 (feature engineering) and Phase 5 (Brain 2)
consume the labeled events without re-running anything upstream.

``merge_brain1_events`` (AFML audit §2.2 — Dual-Path Cleanliness) is the
canonical entry point for Phase 5 / Brain 2: it concatenates events from every
plugin, sorts chronologically, and dedupes by timestamp (keeping the first
plugin's verdict on collisions). The output is guaranteed to be unique and
monotonically increasing in time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import polars as pl

from afml.core.registry import AlphaRegistryRepository, DuplicateHypothesisError
from afml.labeling.primary_alphas.base import PrimaryAlpha
from afml.labeling.triple_barrier import TripleBarrierLabels, apply_triple_barrier


@dataclass(frozen=True, slots=True)
class Brain1Result:
    """One plugin's labeled output for a single asset."""

    asset: str
    plugin_family: str
    hyperparameters: dict[str, Any]
    events: pl.DataFrame
    labels: TripleBarrierLabels
    experiment_id: UUID | None  # ``None`` if duplicate hypothesis (skipped)

    @property
    def n_events(self) -> int:
        return int(self.events.height)

    @property
    def n_positive(self) -> int:
        return self.labels.n_positive

    @property
    def label_rate(self) -> float:
        return self.n_positive / self.n_events if self.n_events else 0.0


def run_brain1(
    bars: pl.DataFrame,
    *,
    asset: str,
    plugins: Sequence[PrimaryAlpha],
    triple_barrier_params: dict[str, Any] | None = None,
    registry: AlphaRegistryRepository | None = None,
    agent_version: str = "agent_2/0.1",
) -> list[Brain1Result]:
    """Run every plugin on the bar sequence and Triple-Barrier-label the events.

    Parameters
    ----------
    bars : Phase 1 bar DataFrame (any of time / TIB / TRB).
    asset : symbol the experiments are tagged with.
    plugins : list of instantiated ``PrimaryAlpha`` subclasses (CUSUM, Bollinger,
        Donchian, etc.). Each carries its own ``hyperparameter_vector`` via
        ``self.params``.
    triple_barrier_params : kwargs forwarded to ``apply_triple_barrier``.
    registry : optional Alpha Registry. When provided, each plugin run is logged
        as one experiment row (deduped on
        ``(asset, family, hparam_hash)``).
    agent_version : version string written to the registry.

    Returns
    -------
    One ``Brain1Result`` per plugin (skipped if it produced zero events).
    """
    tb_params = triple_barrier_params or {}
    results: list[Brain1Result] = []

    for plugin in plugins:
        events = plugin.detect(bars)
        if events.height == 0:
            continue

        labels = apply_triple_barrier(bars, events, **tb_params)

        exp_id: UUID | None = None
        if registry is not None:
            try:
                exp_id = registry.record_experiment(
                    agent_version=agent_version,
                    asset=asset,
                    algorithmic_family=plugin.algorithmic_family,
                    hyperparameter_vector=dict(plugin.params),
                    num_events_triggered=int(events.height),
                )
            except DuplicateHypothesisError:
                exp_id = None

        results.append(
            Brain1Result(
                asset=asset,
                plugin_family=plugin.algorithmic_family,
                hyperparameters=dict(plugin.params),
                events=events,
                labels=labels,
                experiment_id=exp_id,
            )
        )

    return results


def merge_brain1_events(results: Sequence[Brain1Result]) -> pl.DataFrame:
    """Merge events from multiple primary-alpha plugins into a single dedup'd set.

    AFML audit §2.2 (Event-Sampling Deduplication / Dual-Path Cleanliness):
    the array of event timestamps passed downstream to Phase 3 features and
    Phase 5 Brain 2 must contain only unique, monotonically-increasing
    datetime indices. Multiple plugins triggering on the same bar must
    collapse to a single event.

    Resolution rule on a timestamp collision: keep the first plugin's verdict
    in the iteration order of ``results``. Document the priority by ordering
    plugins deliberately when assembling the input list.

    Parameters
    ----------
    results : list of ``Brain1Result`` from ``run_brain1``.

    Returns
    -------
    Polars DataFrame with columns ``timestamp`` (unique, sorted ascending),
    ``side`` (``"long"`` / ``"short"``), and ``plugin_family`` (which plugin's
    verdict was kept). Empty schema-correct frame if ``results`` is empty.
    """
    empty_schema: dict[str, pl.DataType] = {
        "timestamp": pl.Datetime("ms", "UTC"),
        "side": pl.Utf8(),
        "plugin_family": pl.Utf8(),
    }
    if not results:
        return pl.DataFrame(schema=empty_schema)

    framed = [
        r.events.with_columns(pl.lit(r.plugin_family).alias("plugin_family"))
        for r in results
        if r.events.height > 0
    ]
    if not framed:
        return pl.DataFrame(schema=empty_schema)

    combined = pl.concat(framed, how="vertical_relaxed")
    deduped = (
        combined
        .sort("timestamp", maintain_order=True)
        .unique(subset=["timestamp"], keep="first", maintain_order=True)
        .sort("timestamp")
    )
    return deduped
