"""
B111 — temporal arc construction
================================

Initial temporal segmentation for B111.

Scientific rule
---------------
Temporal gaps are controlled explicitly by GAPARC, following the
parameter naming used by the supplied Bernese RNXSMT source.

No gap factor relative to an estimated sampling interval is introduced.

Cycle-slip detection is NOT performed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Arc:
    """One continuous temporal GNSS observation arc."""

    arc_id: int
    start_index: int
    end_index: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    n_epochs: int


def _as_datetime_index(times) -> pd.DatetimeIndex:
    """Convert input epochs to a validated DatetimeIndex."""

    idx = pd.DatetimeIndex(
        pd.to_datetime(times)
    )

    if len(idx) == 0:
        return idx

    if idx.hasnans:
        raise ValueError(
            "Epoch array contains invalid/NaT times."
        )

    if not idx.is_monotonic_increasing:
        raise ValueError(
            "Epochs must be sorted in increasing time."
        )

    if idx.has_duplicates:
        raise ValueError(
            "Epoch array contains duplicate times."
        )

    return idx


def valid_observation_mask(*observables) -> np.ndarray:
    """
    True where every required observable is finite.
    """

    if not observables:
        raise ValueError(
            "At least one observable is required."
        )

    arrays = [
        np.asarray(x, dtype=float)
        for x in observables
    ]

    n = arrays[0].size

    if any(
        x.size != n
        for x in arrays
    ):
        raise ValueError(
            "All observables must have identical length."
        )

    mask = np.ones(
        n,
        dtype=bool,
    )

    for x in arrays:
        mask &= np.isfinite(x)

    return mask


def temporal_breaks(
    times,
    valid_mask,
    *,
    gaparc: float,
):
    """
    Determine epochs beginning new temporal arcs.

    Parameters
    ----------
    times
        Observation epochs.

    valid_mask
        True where all required observations exist.

    gaparc
        Explicit maximum temporal gap accepted inside an arc.

        The value is supplied by B111 configuration. No default and
        no inferred multiplier of the nominal sampling interval are
        used here.

    Returns
    -------
    ndarray[bool]
        True at epochs beginning a new temporal arc.
    """

    if gaparc is None:
        raise ValueError(
            "GAPARC must be supplied explicitly."
        )

    gaparc = float(gaparc)

    if (
        not np.isfinite(gaparc)
        or gaparc <= 0.0
    ):
        raise ValueError(
            f"Invalid GAPARC: {gaparc}"
        )

    idx = _as_datetime_index(times)

    valid = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if len(idx) != valid.size:
        raise ValueError(
            "times and valid_mask have different lengths."
        )

    n = len(idx)

    out = np.zeros(
        n,
        dtype=bool,
    )

    previous_valid = None

    for i in range(n):

        if not valid[i]:
            previous_valid = None
            continue

        if previous_valid is None:

            out[i] = True

        else:

            dt = (
                idx[i]
                - idx[previous_valid]
            ).total_seconds()

            if dt > gaparc:
                out[i] = True

        previous_valid = i

    return out


def build_temporal_arcs(
    times,
    *required_observables,
    gaparc: float,
    min_epochs: int = 1,
):
    """
    Build temporal observation arcs using explicit GAPARC.

    These are temporal continuity arcs only.
    They are not yet ambiguity/cycle-slip arcs.
    """

    if min_epochs < 1:
        raise ValueError(
            "min_epochs must be >= 1."
        )

    idx = _as_datetime_index(times)

    valid = valid_observation_mask(
        *required_observables
    )

    if len(idx) != valid.size:
        raise ValueError(
            "Epoch and observation lengths differ."
        )

    starts = temporal_breaks(
        idx,
        valid,
        gaparc=gaparc,
    )

    arcs: list[Arc] = []

    current: list[int] = []

    def close_current():

        nonlocal current

        if len(current) >= min_epochs:

            arcs.append(
                Arc(
                    arc_id=len(arcs) + 1,
                    start_index=current[0],
                    end_index=current[-1],
                    start_time=idx[current[0]],
                    end_time=idx[current[-1]],
                    n_epochs=len(current),
                )
            )

        current = []

    for i in range(len(idx)):

        if not valid[i]:

            close_current()
            continue

        if starts[i] and current:
            close_current()

        current.append(i)

    close_current()

    return arcs


def arc_labels(
    n_epochs: int,
    arcs: list[Arc],
) -> np.ndarray:
    """
    Produce per-epoch temporal arc labels.

    Invalid/unassigned epochs receive zero.
    """

    labels = np.zeros(
        n_epochs,
        dtype=int,
    )

    for arc in arcs:

        labels[
            arc.start_index:
            arc.end_index + 1
        ] = arc.arc_id

    return labels
