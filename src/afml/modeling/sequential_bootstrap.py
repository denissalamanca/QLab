"""Sequential Bootstrap (AFML Snippet 4.3).

The naïve bootstrap draws ``n_samples`` events with uniform probability and
replacement. On overlapping Triple-Barrier events this overweights clustered
regions: an event sharing time with five others "votes" five times when the
random draw lands on any of the six.

The sequential bootstrap fixes this by drawing each sample with a probability
proportional to its **conditional uniqueness** — uniqueness recomputed given
the samples already drawn. Once a region has been hit, draws from that region
are penalised.

Algorithm (AFML 4.5.2):

::

    phi := []
    while len(phi) < n_samples:
        for each candidate event i:
            avg_u_i := average uniqueness of i conditioning on phi (i.e.
                       concurrency contributed by phi already in the pool)
        prob := avg_u / sum(avg_u)
        draw one event with this distribution; append to phi

The inner loop is ``O(n_events × n_grid)``. For ``n_samples`` total draws the
naive recomputation is ``O(n_samples · n_events · n_grid)`` which is
prohibitive. We exploit the fact that picking event ``i`` only updates
concurrency at the bars where ``I[·, i] = 1`` — an incremental update of
shape ``O(n_grid)``. The resulting algorithm runs in ``O(n_samples ·
n_events · n_grid / n_events) = O(n_samples · n_grid)`` per tree, more than
fast enough for production tick datasets.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Smallest probability mass per candidate. Stops a long tail of zero-weight
# candidates from making the np.random.choice degenerate when MANY samples
# have been drawn from the same overlapping region.
MIN_PROB_EPSILON: float = 1e-12


def sequential_bootstrap(
    indicator_mat: npt.NDArray[np.integer],
    n_samples: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.int64]:
    """Draw ``n_samples`` event indices via the AFML sequential bootstrap.

    Parameters
    ----------
    indicator_mat
        ``(n_grid, n_events)`` matrix from :func:`indicator_matrix`.
    n_samples
        Number of events to draw. Defaults to the column count (one tree's
        worth, matching sklearn's ``bootstrap`` default).
    rng
        Reproducibility seed. ``None`` uses ``np.random.default_rng()``.

    Returns
    -------
    ``(n_samples,)`` int64 array of column indices into ``indicator_mat``.
    Duplicates are allowed (true bootstrap with replacement).
    """
    ind = np.asarray(indicator_mat, dtype=np.float64)
    if ind.ndim != 2:
        raise ValueError(f"indicator_mat must be 2-D, got shape {ind.shape}")
    n_grid, n_events = ind.shape
    if n_events == 0:
        return np.empty(0, dtype=np.int64)
    if n_samples is None:
        n_samples = n_events
    if n_samples < 0:
        raise ValueError(f"n_samples must be ≥ 0, got {n_samples}")
    if rng is None:
        rng = np.random.default_rng()

    # Running concurrency contributed by already-picked events.
    # Adding the candidate to this gives the augmented concurrency.
    picked_concurrency = np.zeros(n_grid, dtype=np.float64)
    # Per-event count of active bars — precomputed once.
    active_bars = ind.sum(axis=0)  # shape (n_events,)
    # Filter to a candidate pool of events with at least one active bar; the
    # rest can never be drawn (ū = 0 always).
    valid_mask = active_bars > 0.0
    valid_events = np.where(valid_mask)[0].astype(np.int64)
    if valid_events.size == 0:
        raise ValueError("no event has any active bar — every column of ind is zero")
    valid_active = active_bars[valid_events]  # shape (n_valid,)
    valid_cols = ind[:, valid_events]  # shape (n_grid, n_valid)

    phi = np.empty(n_samples, dtype=np.int64)
    for step in range(n_samples):
        # Augmented concurrency if event i were added: picked_concurrency + col_i
        # then uniqueness of i at each active bar t is 1 / (1 + picked_concurrency[t])
        # (because adding i raises c_t by 1 at the bars where I[t, i] = 1).
        # Mean over those active bars gives the conditional average uniqueness.
        denom = 1.0 + picked_concurrency  # broadcast: shape (n_grid,)
        # u_{t,i} | phi = I[t, i] / (1 + picked_concurrency[t])  if I[t,i] = 1
        per_bar_u = valid_cols / denom[:, None]  # shape (n_grid, n_valid)
        # Average over active bars only ⇒ sum / active_bars_per_event.
        # per_bar_u is already zero where I[t, i] = 0, so column-sum gives Σ u.
        sum_u = per_bar_u.sum(axis=0)  # shape (n_valid,)
        avg_u = sum_u / valid_active

        avg_u = np.maximum(avg_u, MIN_PROB_EPSILON)
        probs = avg_u / avg_u.sum()
        # Numerical safety — renormalize after epsilon flooring.
        idx_in_valid = int(rng.choice(valid_events.size, p=probs))
        chosen_event = int(valid_events[idx_in_valid])
        phi[step] = chosen_event
        # Update picked_concurrency with the chosen event's column.
        picked_concurrency += ind[:, chosen_event]

    return phi
