"""
B111 — Bernese-derived alternative GNSS preprocessing pipeline
==============================================================

B111 is an independent alternative to D111.

It does NOT read or reuse:
    - D111 flags
    - D111 arc boundaries
    - D111 thresholds
    - D111 leveling results

Implemented Bernese-derived components currently available:
    - explicit GAPARC temporal segmentation
    - POLYAP
    - UPDARC
    - CHKWUB control logic
    - CHKL4P
    - CHKNOI
    - DEFREQ-equivalent WLGT/FACLIN quantities

Unavailable supplied Bernese dependencies remain explicit:
    - RSMTIN
    - LINCMB
    - OFFSET
    - DETSLP

No scientific numerical defaults are introduced for those dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arcs import (
    Arc,
    build_temporal_arcs,
    arc_labels,
)

from .combinations import (
    C,
    FrequencyPair,
    validate_pair,
)

from .config import (
    RnxSmtConfig,
    validate_no_missing,
)


@dataclass(frozen=True)
class BerneseFrequencySystem:
    """
    DEFREQ-equivalent quantities for one dual-frequency pair.
    """

    f1: float
    f2: float

    wlgt1: float
    wlgt2: float
    wlgt3: float
    wlgt4: float
    wlgt5: float

    faclin31: float
    faclin32: float


@dataclass
class B111Preparation:
    """
    Result of the currently supported B111 preparation stage.
    """

    frequency: BerneseFrequencySystem
    arcs: list[Arc]
    labels: np.ndarray


def bernese_frequency_system(
    pair: FrequencyPair,
) -> BerneseFrequencySystem:
    """
    Construct WLGT/FACLIN quantities following the supplied DEFREQ.f
    definitions.
    """

    validate_pair(pair)

    f1 = float(pair.f1)
    f2 = float(pair.f2)

    wlgt1 = C / f1
    wlgt2 = C / f2

    wlgt3 = (
        wlgt1
        * wlgt2
        / (wlgt1 + wlgt2)
    )

    wlgt4 = (
        wlgt1
        - wlgt2
    )

    wlgt5 = (
        wlgt1
        * wlgt2
        / (wlgt2 - wlgt1)
    )

    denominator = (
        f1**2
        - f2**2
    )

    faclin31 = (
        f1**2
        / denominator
    )

    faclin32 = (
        -f2**2
        / denominator
    )

    return BerneseFrequencySystem(
        f1=f1,
        f2=f2,
        wlgt1=wlgt1,
        wlgt2=wlgt2,
        wlgt3=wlgt3,
        wlgt4=wlgt4,
        wlgt5=wlgt5,
        faclin31=faclin31,
        faclin32=faclin32,
    )


def prepare_b111(
    times,
    *required_observables,
    pair: FrequencyPair,
    config: RnxSmtConfig,
) -> B111Preparation:
    """
    Perform the B111 preparation stage.

    Current responsibilities:
        1. validate the explicit RNXSMT configuration;
        2. construct Bernese frequency quantities;
        3. construct independent temporal arcs using GAPARC.

    This function deliberately does NOT yet synthesize LINCMB or OFFSET.
    """

    if config is None:
        raise ValueError(
            "RnxSmtConfig must be supplied explicitly."
        )

    validate_no_missing(
        config
    )

    frequency = bernese_frequency_system(
        pair
    )

    arcs = build_temporal_arcs(
        times,
        *required_observables,
        gaparc=config.gaparc,
    )

    labels = arc_labels(
        len(times),
        arcs,
    )

    return B111Preparation(
        frequency=frequency,
        arcs=arcs,
        labels=labels,
    )
