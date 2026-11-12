"""
B111 — RINEX phase/code observable association
==============================================

Associates carrier-phase observations with code observations on the
same RINEX signal.

No tracking-code preference is imposed.
No scientific thresholds are introduced.
No D111 information is used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseCodeSignal:
    """
    One RINEX signal for which both carrier phase and code are available.
    """

    phase: str
    code: str


@dataclass(frozen=True)
class MwObservableSet:
    """
    Four RINEX observables required to construct one two-frequency
    Melbourne-Wubbena combination.
    """

    phase1: str
    phase2: str
    code1: str
    code2: str


def matching_code_observable(
    phase_observable: str,
    available_observables,
) -> str | None:
    """
    Return the code observable on exactly the same RINEX signal.

    Examples:
        L1C -> C1C
        L2W -> C2W
        L5Q -> C5Q

    If the exact code observable is absent, return None.

    No fallback between tracking codes is performed.
    """

    if (
        len(phase_observable) < 3
        or not phase_observable.startswith("L")
    ):
        raise ValueError(
            "Expected a RINEX carrier-phase observable, "
            f"got {phase_observable!r}"
        )

    code = (
        "C"
        + phase_observable[1:]
    )

    available = set(
        available_observables
    )

    if code in available:
        return code

    return None


def phase_code_signals(
    phase_observables,
    available_observables,
) -> list[PhaseCodeSignal]:
    """
    Return all exact phase/code signal matches.
    """

    out = []

    for phase in phase_observables:

        code = matching_code_observable(
            phase,
            available_observables,
        )

        if code is None:
            continue

        out.append(
            PhaseCodeSignal(
                phase=phase,
                code=code,
            )
        )

    return out


def mw_observable_sets(
    phase_observables1,
    phase_observables2,
    available_observables,
) -> list[MwObservableSet]:
    """
    Build every exact-tracking-code observable set available for one
    physical dual-frequency pair.

    This does not rank or select among alternatives.
    """

    signals1 = phase_code_signals(
        phase_observables1,
        available_observables,
    )

    signals2 = phase_code_signals(
        phase_observables2,
        available_observables,
    )

    out = []

    for a in signals1:
        for b in signals2:

            out.append(
                MwObservableSet(
                    phase1=a.phase,
                    phase2=b.phase,
                    code1=a.code,
                    code2=b.code,
                )
            )

    return out
