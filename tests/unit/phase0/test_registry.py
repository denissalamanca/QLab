"""Phase 0 — Alpha Registry: schema integrity + dedup enforcement.

Anti-Lazy Constraint (Blueprint §2.3): inserting a duplicate
``(asset, algorithmic_family, hyperparameter_vector)`` MUST raise
``DuplicateHypothesisError`` so the trial count for the Deflated Sharpe Ratio
remains an honest measure of the multiple-testing penalty.
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

import orjson
import pytest

from afml.core.registry import (
    AlphaRegistryRepository,
    DuplicateHypothesisError,
)
from afml.core.registry.repository import _hash_hyperparameters


@pytest.fixture
def repo(tmp_db_url: str) -> AlphaRegistryRepository:
    r = AlphaRegistryRepository(tmp_db_url)
    r.create_all()
    return r


@pytest.mark.phase0
def test_record_and_lookup_roundtrip(repo: AlphaRegistryRepository) -> None:
    exp_id = repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"vol_span": 100, "h_multiplier": 2.0},
        num_events_triggered=812,
        brain_1_recall=0.74,
    )
    assert exp_id is not None
    assert repo.total_trials() == 1
    fetched = repo.get(exp_id)
    assert fetched is not None
    assert fetched.asset == "EURUSD"
    assert fetched.algorithmic_family == "cusum"
    assert fetched.hyperparameter_vector == {"vol_span": 100, "h_multiplier": 2.0}
    assert fetched.brain_1_recall == pytest.approx(0.74)
    assert fetched.is_deployed is False


@pytest.mark.phase0
def test_duplicate_hyperparameter_raises(repo: AlphaRegistryRepository) -> None:
    hparams = {"vol_span": 100, "h_multiplier": 2.0}
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector=hparams,
        num_events_triggered=812,
    )
    with pytest.raises(DuplicateHypothesisError):
        repo.record_experiment(
            agent_version="agent_2/0.1",
            asset="EURUSD",
            algorithmic_family="cusum",
            hyperparameter_vector=hparams,
            num_events_triggered=900,
        )


@pytest.mark.phase0
def test_dedup_key_is_order_independent(repo: AlphaRegistryRepository) -> None:
    """``{a:1, b:2}`` and ``{b:2, a:1}`` must hash identically — JSON key order
    is not semantic."""
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"a": 1, "b": 2},
        num_events_triggered=600,
    )
    with pytest.raises(DuplicateHypothesisError):
        repo.record_experiment(
            agent_version="agent_2/0.1",
            asset="EURUSD",
            algorithmic_family="cusum",
            hyperparameter_vector={"b": 2, "a": 1},
            num_events_triggered=600,
        )


@pytest.mark.phase0
def test_same_hparams_different_asset_is_separate_trial(
    repo: AlphaRegistryRepository,
) -> None:
    hp = {"vol_span": 100}
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector=hp,
        num_events_triggered=600,
    )
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="GBPUSD",
        algorithmic_family="cusum",
        hyperparameter_vector=hp,
        num_events_triggered=600,
    )
    assert repo.total_trials() == 2
    assert repo.trials_for(asset="EURUSD") == 1
    assert repo.trials_for(asset="GBPUSD") == 1


@pytest.mark.phase0
def test_same_hparams_different_family_is_separate_trial(
    repo: AlphaRegistryRepository,
) -> None:
    hp = {"span": 50}
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector=hp,
        num_events_triggered=600,
    )
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="bbands_meanrev",
        hyperparameter_vector=hp,
        num_events_triggered=600,
    )
    assert repo.total_trials() == 2
    assert repo.trials_for(family="cusum") == 1
    assert repo.trials_for(family="bbands_meanrev") == 1


@pytest.mark.phase0
def test_total_trials_counts_full_sweep(repo: AlphaRegistryRepository) -> None:
    """Phase 6 reads ``total_trials()`` to penalize the strategy Sharpe with the
    Deflated Sharpe Ratio formula. The count invariant must hold across many
    inserts.
    """
    n = 50
    for i in range(n):
        repo.record_experiment(
            agent_version="agent_2/0.1",
            asset="EURUSD",
            algorithmic_family="cusum",
            hyperparameter_vector={"vol_span": i},
            num_events_triggered=600,
        )
    assert repo.total_trials() == n


@pytest.mark.phase0
def test_deployed_filter(repo: AlphaRegistryRepository) -> None:
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"x": 1},
        num_events_triggered=600,
        is_deployed=True,
    )
    repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"x": 2},
        num_events_triggered=600,
        is_deployed=False,
    )
    deployed = repo.deployed()
    assert len(deployed) == 1
    assert deployed[0].hyperparameter_vector == {"x": 1}


@pytest.mark.phase0
def test_mark_deployed_promotes_existing(repo: AlphaRegistryRepository) -> None:
    exp_id = repo.record_experiment(
        agent_version="agent_2/0.1",
        asset="EURUSD",
        algorithmic_family="cusum",
        hyperparameter_vector={"x": 1},
        num_events_triggered=600,
        is_deployed=False,
    )
    assert repo.deployed() == []
    repo.mark_deployed(exp_id, deployed=True)
    deployed = repo.deployed()
    assert len(deployed) == 1
    assert deployed[0].experiment_id == exp_id


@pytest.mark.phase0
def test_mark_deployed_missing_raises(repo: AlphaRegistryRepository) -> None:
    with pytest.raises(KeyError):
        repo.mark_deployed(uuid4(), True)


@pytest.mark.phase0
def test_hash_algorithm_is_sha256_orjson_sorted_keys() -> None:
    """AFML audit §2.1 — Alpha Registry dedup is computed as SHA-256 over the
    orjson-serialized, sorted-key hyperparameter vector.

    Pin the exact hex digest for a known input so a future refactor that
    silently swaps the hash algorithm (or drops sorted-key serialization) is
    immediately caught.
    """
    hparams = {"vol_span": 100, "h_multiplier": 2.0, "asset": "EURUSD"}
    digest = _hash_hyperparameters(hparams)

    # Re-derive the expected digest exactly per the audit's required recipe.
    expected = hashlib.sha256(orjson.dumps(hparams, option=orjson.OPT_SORT_KEYS)).hexdigest()

    assert digest == expected
    # SHA-256 is a 64-hex-char digest.
    assert len(digest) == 64
    # And it must be order-independent (sorted-keys requirement).
    reordered = {"asset": "EURUSD", "h_multiplier": 2.0, "vol_span": 100}
    assert _hash_hyperparameters(reordered) == digest
