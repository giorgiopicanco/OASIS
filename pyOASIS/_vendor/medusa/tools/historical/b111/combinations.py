"""
B111 — Bernese-inspired GNSS phase preprocessing
=================================================

Independent implementation of GNSS dual-frequency combinations used
for cycle-slip screening.

IMPORTANT
---------
B111 must remain scientifically independent from D111.
No D111 flags, arc boundaries, thresholds, or leveling results are used.

The architecture is inspired by the processing strategy visible in the
Bernese MAUPRP / PREPHA / RNXSMT source code available to the project.

This is NOT claimed to reproduce the unavailable Bernese DETSLP
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------

C = 299_792_458.0  # speed of light [m/s]


@dataclass(frozen=True)
class FrequencyPair:
    """Two GNSS carrier frequencies."""

    f1: float
    f2: float

    @property
    def lambda1(self) -> float:
        return C / self.f1

    @property
    def lambda2(self) -> float:
        return C / self.f2

    @property
    def lambda_wl(self) -> float:
        """
        Signed wide-lane wavelength.

        Bernese DEFREQ convention:

            WLGT(5) = lambda1*lambda2/(lambda2-lambda1)
                    = C/(f1-f2)

        The sign therefore follows the ordering of the two frequencies.
        """
        return C / (self.f1 - self.f2)

    @property
    def lambda_wl_abs(self) -> float:
        """Absolute magnitude of the wide-lane wavelength."""
        return abs(self.lambda_wl)

    @property
    def lambda_nl(self) -> float:
        """Narrow-lane wavelength."""
        return C / (self.f1 + self.f2)


# ---------------------------------------------------------------------
# Nominal frequency pairs
# ---------------------------------------------------------------------

GPS_L1_L2 = FrequencyPair(
    f1=1575.42e6,
    f2=1227.60e6,
)

GALILEO_E1_E5A = FrequencyPair(
    f1=1575.42e6,
    f2=1176.45e6,
)

BEIDOU_B1C_B2A = FrequencyPair(
    f1=1575.42e6,
    f2=1176.45e6,
)


# ---------------------------------------------------------------------
# Basic carrier conversion
# ---------------------------------------------------------------------

def phase_meters(
    phase_cycles: np.ndarray | float,
    frequency_hz: float,
):
    """Carrier phase cycles -> meters."""
    return np.asarray(phase_cycles, dtype=float) * (C / frequency_hz)


# ---------------------------------------------------------------------
# Geometry-free carrier phase
# ---------------------------------------------------------------------

def geometry_free_phase(
    phase1_cycles,
    phase2_cycles,
    pair: FrequencyPair,
):
    """
    Geometry-free carrier-phase combination [m].

        L_GF = lambda1 * Phi1 - lambda2 * Phi2

    Common geometric range and receiver/satellite clock terms cancel
    to first order.

    Remaining terms contain mainly ionosphere, ambiguities, hardware
    effects and measurement noise.
    """

    L1 = phase_meters(phase1_cycles, pair.f1)
    L2 = phase_meters(phase2_cycles, pair.f2)

    return L1 - L2


# Bernese terminology commonly calls the geometry-free carrier
# combination L4.
l4_phase = geometry_free_phase


# ---------------------------------------------------------------------
# Melbourne-Wubbena
# ---------------------------------------------------------------------

def melbourne_wubbena(
    phase1_cycles,
    phase2_cycles,
    code1_m,
    code2_m,
    pair: FrequencyPair,
):
    """
    Melbourne-Wubbena combination expressed in wide-lane cycles.

    Carrier wide lane:

        L_W = (f1*L1 - f2*L2) / (f1-f2)

    Narrow-lane code:

        P_N = (f1*P1 + f2*P2) / (f1+f2)

    Melbourne-Wubbena:

        MW_m = L_W - P_N

        MW_cycles = MW_m / lambda_WL

    The combination is useful for detecting changes in the wide-lane
    ambiguity while strongly reducing geometry and ionosphere.
    """

    phi1 = np.asarray(phase1_cycles, dtype=float)
    phi2 = np.asarray(phase2_cycles, dtype=float)
    p1 = np.asarray(code1_m, dtype=float)
    p2 = np.asarray(code2_m, dtype=float)

    L1 = phi1 * pair.lambda1
    L2 = phi2 * pair.lambda2

    f1 = pair.f1
    f2 = pair.f2

    Lw = (f1 * L1 - f2 * L2) / (f1 - f2)
    Pn = (f1 * p1 + f2 * p2) / (f1 + f2)

    mw_m = Lw - Pn

    return mw_m / pair.lambda_wl


# ---------------------------------------------------------------------
# Epoch differences
# ---------------------------------------------------------------------

def epoch_difference(values):
    """
    First epoch difference.

    Output has the same length as input.
    First element is NaN.
    """

    x = np.asarray(values, dtype=float)

    out = np.full(x.shape, np.nan, dtype=float)

    if x.size > 1:
        out[1:] = np.diff(x)

    return out


def delta_l4(
    phase1_cycles,
    phase2_cycles,
    pair: FrequencyPair,
):
    """Epoch-to-epoch change of the L4/GF carrier combination [m]."""

    return epoch_difference(
        geometry_free_phase(
            phase1_cycles,
            phase2_cycles,
            pair,
        )
    )


def delta_mw(
    phase1_cycles,
    phase2_cycles,
    code1_m,
    code2_m,
    pair: FrequencyPair,
):
    """Epoch-to-epoch change of Melbourne-Wubbena [wide-lane cycles]."""

    return epoch_difference(
        melbourne_wubbena(
            phase1_cycles,
            phase2_cycles,
            code1_m,
            code2_m,
            pair,
        )
    )


# ---------------------------------------------------------------------
# Frequency sanity checks
# ---------------------------------------------------------------------

def describe_pair(pair: FrequencyPair) -> dict:
    """Return useful physical quantities for diagnostics."""

    return {
        "f1_MHz": pair.f1 / 1e6,
        "f2_MHz": pair.f2 / 1e6,
        "lambda1_m": pair.lambda1,
        "lambda2_m": pair.lambda2,
        "lambda_wl_m": pair.lambda_wl,
        "lambda_nl_m": pair.lambda_nl,
    }


def validate_pair(pair: FrequencyPair) -> None:
    if not (
        math.isfinite(pair.f1)
        and math.isfinite(pair.f2)
        and pair.f1 > 0.0
        and pair.f2 > 0.0
        and pair.f1 != pair.f2
    ):
        raise ValueError(
            f"Invalid GNSS frequency pair: {pair}"
        )


# ---------------------------------------------------------------------
# Multi-GNSS signal-frequency catalogue
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GnssSignal:
    """
    One GNSS carrier signal/band.

    frequency_hz is the carrier center frequency for CDMA signals.

    For GLONASS FDMA G1/G2, use glonass_signal() because the carrier
    frequency depends on the satellite frequency-channel number k.
    """

    constellation: str
    band: str
    frequency_hz: float

    @property
    def wavelength_m(self) -> float:
        return C / self.frequency_hz


# ---------------------------------------------------------------------
# GPS
# ---------------------------------------------------------------------

GPS_L1 = GnssSignal("GPS", "L1", 1575.42e6)
GPS_L2 = GnssSignal("GPS", "L2", 1227.60e6)
GPS_L5 = GnssSignal("GPS", "L5", 1176.45e6)


# ---------------------------------------------------------------------
# Galileo
# ---------------------------------------------------------------------

GALILEO_E1 = GnssSignal("GALILEO", "E1", 1575.42e6)
GALILEO_E5A = GnssSignal("GALILEO", "E5a", 1176.45e6)
GALILEO_E5B = GnssSignal("GALILEO", "E5b", 1207.14e6)

# AltBOC E5 center frequency.
GALILEO_E5 = GnssSignal("GALILEO", "E5", 1191.795e6)

GALILEO_E6 = GnssSignal("GALILEO", "E6", 1278.75e6)


# ---------------------------------------------------------------------
# BeiDou
# ---------------------------------------------------------------------

# BDS-2 / legacy-compatible signals.
BEIDOU_B1I = GnssSignal("BEIDOU", "B1I", 1561.098e6)
BEIDOU_B2I = GnssSignal("BEIDOU", "B2I", 1207.14e6)
BEIDOU_B3I = GnssSignal("BEIDOU", "B3I", 1268.52e6)

# BDS-3 modern signals.
BEIDOU_B1C = GnssSignal("BEIDOU", "B1C", 1575.42e6)
BEIDOU_B2A = GnssSignal("BEIDOU", "B2a", 1176.45e6)
BEIDOU_B2B = GnssSignal("BEIDOU", "B2b", 1207.14e6)


# ---------------------------------------------------------------------
# GLONASS
# ---------------------------------------------------------------------

# GLONASS FDMA:
#
#   G1(k) = 1602.000 MHz + k * 0.5625 MHz
#   G2(k) = 1246.000 MHz + k * 0.4375 MHz
#
# The channel number must therefore be supplied explicitly.
#
# G3 is CDMA and does not use the G1/G2 FDMA channel formula.

GLONASS_G1_BASE_HZ = 1602.000e6
GLONASS_G1_STEP_HZ = 0.5625e6

GLONASS_G2_BASE_HZ = 1246.000e6
GLONASS_G2_STEP_HZ = 0.4375e6

GLONASS_G3 = GnssSignal(
    "GLONASS",
    "G3",
    1202.025e6,
)


def glonass_signal(
    band: str,
    channel: int | None = None,
) -> GnssSignal:
    """
    Construct a GLONASS carrier signal.

    G1 and G2 are FDMA and require frequency-channel number k.
    G3 is returned directly because it does not use the G1/G2 FDMA law.
    """

    b = band.upper()

    if b == "G1":
        if channel is None:
            raise ValueError(
                "GLONASS G1 requires an explicit FDMA channel number k."
            )

        frequency = (
            GLONASS_G1_BASE_HZ
            + int(channel) * GLONASS_G1_STEP_HZ
        )

    elif b == "G2":
        if channel is None:
            raise ValueError(
                "GLONASS G2 requires an explicit FDMA channel number k."
            )

        frequency = (
            GLONASS_G2_BASE_HZ
            + int(channel) * GLONASS_G2_STEP_HZ
        )

    elif b == "G3":
        if channel is not None:
            raise ValueError(
                "GLONASS G3 does not use the G1/G2 FDMA channel law."
            )

        return GLONASS_G3

    else:
        raise ValueError(
            f"Unsupported GLONASS band: {band!r}"
        )

    return GnssSignal(
        constellation="GLONASS",
        band=b,
        frequency_hz=frequency,
    )


# ---------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------

def frequency_pair(
    signal1: GnssSignal,
    signal2: GnssSignal,
) -> FrequencyPair:
    """
    Construct the dual-frequency mathematical pair used by B111.

    The two signals must belong to the same GNSS constellation.
    """

    if signal1.constellation != signal2.constellation:
        raise ValueError(
            "A B111 dual-frequency pair must belong to one constellation: "
            f"{signal1.constellation} != {signal2.constellation}"
        )

    pair = FrequencyPair(
        f1=float(signal1.frequency_hz),
        f2=float(signal2.frequency_hz),
    )

    validate_pair(pair)

    return pair


# ---------------------------------------------------------------------
# Standard B111 pairs
# ---------------------------------------------------------------------

GPS_L1_L5 = frequency_pair(
    GPS_L1,
    GPS_L5,
)

GPS_L2_L5 = frequency_pair(
    GPS_L2,
    GPS_L5,
)

GALILEO_E1_E5B = frequency_pair(
    GALILEO_E1,
    GALILEO_E5B,
)

GALILEO_E1_E6 = frequency_pair(
    GALILEO_E1,
    GALILEO_E6,
)

GALILEO_E5A_E5B = frequency_pair(
    GALILEO_E5A,
    GALILEO_E5B,
)

BEIDOU_B1I_B2I = frequency_pair(
    BEIDOU_B1I,
    BEIDOU_B2I,
)

BEIDOU_B1I_B3I = frequency_pair(
    BEIDOU_B1I,
    BEIDOU_B3I,
)

BEIDOU_B1C_B2B = frequency_pair(
    BEIDOU_B1C,
    BEIDOU_B2B,
)

BEIDOU_B1C_B3I = frequency_pair(
    BEIDOU_B1C,
    BEIDOU_B3I,
)


def bernese_l5_mw_cycles(
    phase1_cycles,
    phase2_cycles,
    code1_m,
    code2_m,
    pair: FrequencyPair,
):
    """
    Construct the Bernese RNXSMT ITYP=10 observable used by CHKWUB.

    Output
    ------
    Melbourne-Wubbena / Bernese L5 combination in signed WLGT(5)
    cycles.

    Important
    ---------
    This reconstructs only the numerical ITYP=10 combination required
    by RNXSMT -> CHKWUB.  It is not a reimplementation of the complete
    Bernese LINCMB routine, whose source is not present in the supplied
    Bernese source set.

    Bernese convention:

        WLGT(5) = C / (f1 - f2)

    Therefore pair ordering controls the sign of the resulting L5
    cycles, while the corresponding meter-domain observable is
    invariant when both frequencies and their observations are swapped.
    """
    return melbourne_wubbena(
        phase1_cycles=phase1_cycles,
        phase2_cycles=phase2_cycles,
        code1_m=code1_m,
        code2_m=code2_m,
        pair=pair,
    )
