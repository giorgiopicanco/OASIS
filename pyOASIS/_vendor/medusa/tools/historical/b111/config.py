"""
B111 — Bernese-derived preprocessing configuration
===================================================

Parameter names follow the Bernese MAUPRP/PREPHA/RNXSMT source code
provided for development of B111.

IMPORTANT:
No arbitrary numerical defaults are introduced here.

Parameters whose numerical values are normally read from Bernese
input files remain unset until they are recovered from source/input
material or explicitly supplied.
"""

from dataclasses import dataclass


@dataclass
class MauprpConfig:

    # PREPHA / DETSLP
    rmsmax: float | None = None
    jmpopt: int | None = None
    toljmp: float | None = None

    # Observation gaps / continuity
    mxogap: int | None = None
    mxhole: int | None = None
    mncont: int | None = None

    # Ionospheric screening
    mxiond: float | None = None
    iiono: int | None = None

    # Carrier measurement noise
    sigl1: float | None = None
    sigl2: float | None = None

    # Cycle-slip acceptance
    mincyc: int | None = None

    # Observation screening
    maxzen: int | None = None
    omcmax: float | None = None

    # Frequency strategy
    nfrchk: int | None = None
    nfreq: int | None = None


@dataclass
class RnxSmtConfig:

    # Arc construction
    gaparc: float | None = None

    # L4 screening
    gapl4: float | None = None
    rmsl4: float | None = None
    minl4: int | None = None

    # Melbourne-Wubbena / L5
    rmsl5: float | None = None

    # L3/noise screening
    rmsl3: float | None = None

    # Slip criteria
    slpmin: float | None = None
    slpout: float | None = None

    # Minimum observations
    minobs: int | None = None

    # Cycle-slip fixing
    ifxslp: int | None = None


def validate_no_missing(config):
    missing = [
        name
        for name, value in vars(config).items()
        if value is None
    ]

    if missing:
        raise ValueError(
            "Bernese parameters still unknown: "
            + ", ".join(missing)
        )
