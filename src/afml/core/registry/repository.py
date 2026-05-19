"""Repository for the Historical Alpha Registry.

Responsibilities:
- Hash ``hyperparameter_vector`` deterministically and use it as the dedup key.
- Raise ``DuplicateHypothesisError`` on attempted re-insert of an identical trial.
- Expose the trial count → drives the Deflated Sharpe Ratio penalty (Phase 6).
- Expose deployed / per-asset / per-family subsets for the orthogonality check (Phase 2).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

import orjson
from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from afml.core.registry.exceptions import DuplicateHypothesisError
from afml.core.registry.schema import (
    ALLOWED_EXPERIMENT_STATUSES,
    EXPERIMENT_STATUS_COMPLETED,
    EXPERIMENT_STATUS_FAILED_AT_MDA,
    Experiment,
    metadata,
)


def _hash_hyperparameters(hparams: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a hyperparameter vector.

    Uses ``orjson`` with ``OPT_SORT_KEYS`` so ``{"a":1, "b":2}`` and ``{"b":2, "a":1}``
    produce the same hash — key ordering must not produce a "new" experiment.
    """
    payload = orjson.dumps(hparams, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


class AlphaRegistryRepository:
    """Repository over the SQLite-backed Alpha Registry."""

    def __init__(
        self,
        db_url: str,
        *,
        wal_mode: bool = True,
        echo: bool = False,
    ) -> None:
        self._engine: Engine = create_engine(db_url, echo=echo, future=True)

        # SQLite WAL mode → concurrent reads alongside writes; required for high-
        # throughput hypothesis logging during parameter sweeps.
        if wal_mode and db_url.startswith("sqlite"):

            @event.listens_for(self._engine, "connect")
            def _set_wal(dbapi_conn: Any, _conn_record: Any) -> None:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ------------------------------------------------------- schema management
    def create_all(self) -> None:
        metadata.create_all(self._engine)

    def drop_all(self) -> None:
        metadata.drop_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ------------------------------------------------------------------ writes
    def record_experiment(
        self,
        *,
        agent_version: str,
        asset: str,
        algorithmic_family: str,
        hyperparameter_vector: dict[str, Any],
        num_events_triggered: int,
        orthogonality_score: float | None = None,
        brain_1_recall: float | None = None,
        brain_2_log_loss: float | None = None,
        is_deployed: bool = False,
        status: str = EXPERIMENT_STATUS_COMPLETED,
    ) -> UUID:
        """Insert an immutable hypothesis record.

        Raises ``DuplicateHypothesisError`` if the same
        ``(asset, algorithmic_family, hyperparameter_vector)`` triple was already
        recorded — preserving the integrity of the DSR trial count.

        ``status`` defaults to :data:`EXPERIMENT_STATUS_COMPLETED`. Pass
        :data:`EXPERIMENT_STATUS_FAILED_AT_MDA` to record a circuit-breaker
        halt at Phase 4's Clustered MDA gate without ever attempting Phase 5
        / Brain 2.
        """
        if status not in ALLOWED_EXPERIMENT_STATUSES:
            raise ValueError(
                f"unknown status {status!r}; allowed: {sorted(ALLOWED_EXPERIMENT_STATUSES)}"
            )
        hparam_hash = _hash_hyperparameters(hyperparameter_vector)
        exp = Experiment(
            agent_version=agent_version,
            asset=asset,
            algorithmic_family=algorithmic_family,
            hyperparameter_vector=hyperparameter_vector,
            hyperparameter_hash=hparam_hash,
            orthogonality_score=orthogonality_score,
            num_events_triggered=num_events_triggered,
            brain_1_recall=brain_1_recall,
            brain_2_log_loss=brain_2_log_loss,
            is_deployed=is_deployed,
            status=status,
        )
        try:
            with self.session() as s:
                s.add(exp)
                s.flush()
                return exp.experiment_id
        except IntegrityError as e:
            raise DuplicateHypothesisError(
                f"Hypothesis already exists for asset={asset!r}, "
                f"family={algorithmic_family!r}, hparam_hash={hparam_hash[:12]}…"
            ) from e

    def record_failed_mda(
        self,
        *,
        agent_version: str,
        asset: str,
        algorithmic_family: str,
        hyperparameter_vector: dict[str, Any],
        num_events_triggered: int,
        orthogonality_score: float | None = None,
        brain_1_recall: float | None = None,
    ) -> UUID:
        """Log a Phase 4 ``FAILED_AT_MDA`` hypothesis without crashing.

        AFML 0-4 integration audit V3: Clustered MDA may correctly return zero
        surviving features on a pure-noise hypothesis. The pipeline must NOT
        crash — it must record the trial (preserving the DSR multiple-testing
        denominator) with ``status = FAILED_AT_MDA``, ``brain_2_log_loss =
        NULL``, and ``is_deployed = False``, so no further capital is committed
        to the dead branch.

        Returns the new ``experiment_id``.
        """
        return self.record_experiment(
            agent_version=agent_version,
            asset=asset,
            algorithmic_family=algorithmic_family,
            hyperparameter_vector=hyperparameter_vector,
            num_events_triggered=num_events_triggered,
            orthogonality_score=orthogonality_score,
            brain_1_recall=brain_1_recall,
            brain_2_log_loss=None,
            is_deployed=False,
            status=EXPERIMENT_STATUS_FAILED_AT_MDA,
        )

    def mark_deployed(self, experiment_id: UUID, deployed: bool = True) -> None:
        with self.session() as s:
            exp = s.get(Experiment, experiment_id)
            if exp is None:
                raise KeyError(f"No experiment with id={experiment_id}")
            exp.is_deployed = deployed

    # ------------------------------------------------------------------- reads
    def total_trials(self) -> int:
        """Total hypotheses tested — drives DSR penalty in Phase 6."""
        with self.session() as s:
            return s.scalar(select(func.count(Experiment.experiment_id))) or 0

    def trials_for(
        self,
        asset: str | None = None,
        family: str | None = None,
    ) -> int:
        with self.session() as s:
            stmt = select(func.count(Experiment.experiment_id))
            if asset is not None:
                stmt = stmt.where(Experiment.asset == asset)
            if family is not None:
                stmt = stmt.where(Experiment.algorithmic_family == family)
            return s.scalar(stmt) or 0

    def deployed(self) -> list[Experiment]:
        with self.session() as s:
            return list(s.scalars(select(Experiment).where(Experiment.is_deployed.is_(True))))

    def get(self, experiment_id: UUID) -> Experiment | None:
        with self.session() as s:
            return s.get(Experiment, experiment_id)
