"""
B111 — physical dual-frequency pair inventory
=============================================

Builds all unique physical frequency pairs available for one satellite.

Important:
    - no preferred pair is selected;
    - no scientific thresholds are introduced;
    - tracking-code multiplicity does not create duplicate physical pairs;
    - original RINEX observables are preserved for later signal selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .combinations import (
    FrequencyPair,
    frequency_pair,
)

from .rinex_signals import (
    RinexHeaderSignals,
    carrier_signal,
)


@dataclass(frozen=True)
class PhysicalBand:
    """
    One distinct physical carrier frequency available for one satellite.
    """

    frequency_hz: float
    observables: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalPair:
    """
    One unique physical dual-frequency pair.

    observables1/2 retain all RINEX carrier-phase observables found on
    each physical carrier.  No tracking-code preference is imposed here.
    """

    f1: float
    f2: float

    observables1: tuple[str, ...]
    observables2: tuple[str, ...]

    @property
    def pair(self) -> FrequencyPair:
        return FrequencyPair(
            f1=self.f1,
            f2=self.f2,
        )


def physical_bands(
    system: str,
    sv: str,
    available_phase_observables,
    header: RinexHeaderSignals,
) -> list[PhysicalBand]:
    """
    Collapse RINEX carrier observables onto distinct physical frequencies.
    """

    grouped: dict[float, list[str]] = {}

    for obs in available_phase_observables:

        if not obs.startswith("L"):
            continue

        signal = carrier_signal(
            system,
            obs,
            sv=sv if system == "R" else None,
            glonass_channels=(
                header.glonass_channels
                if system == "R"
                else None
            ),
        )

        f = float(signal.frequency_hz)

        grouped.setdefault(
            f,
            [],
        ).append(obs)

    bands = []

    for f in sorted(
        grouped,
        reverse=True,
    ):
        bands.append(
            PhysicalBand(
                frequency_hz=f,
                observables=tuple(
                    sorted(
                        set(grouped[f])
                    )
                ),
            )
        )

    return bands


def physical_pairs(
    system: str,
    sv: str,
    available_phase_observables,
    header: RinexHeaderSignals,
) -> list[PhysicalPair]:
    """
    Return every unique two-frequency combination available for one SV.

    No pair is preferred or discarded.
    """

    bands = physical_bands(
        system,
        sv,
        available_phase_observables,
        header,
    )

    out = []

    for a, b in combinations(
        bands,
        2,
    ):

        # Validate through the same B111 frequency-pair constructor.
        from .combinations import GnssSignal

        sa = GnssSignal(
            constellation=system,
            band="",
            frequency_hz=a.frequency_hz,
        )

        sb = GnssSignal(
            constellation=system,
            band="",
            frequency_hz=b.frequency_hz,
        )

        fp = frequency_pair(
            sa,
            sb,
        )

        out.append(
            PhysicalPair(
                f1=fp.f1,
                f2=fp.f2,
                observables1=a.observables,
                observables2=b.observables,
            )
        )

    return out
