"""
B111 — low-level Bernese-compatible primitives.

Implemented from the supplied Bernese source code.

Flag semantics used by RNXSMT:
    bit 0 = outlier / bad observation
    bit 1 = new ambiguity / uncorrected cycle slip
    bit 2 = corrected cycle slip
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Bernese flag primitives
# ----------------------------------------------------------------------

def tstflg(flag: int, bit: int) -> bool:
    """Equivalent logical operation to Bernese TSTFLG."""
    return bool(int(flag) & (1 << bit))


def setflg(flag: int, bit: int) -> int:
    """Equivalent logical operation to Bernese SETFLG."""
    return int(flag) | (1 << bit)


def clrflg(flag: int, bit: int) -> int:
    """Equivalent logical operation to Bernese CLRFLG."""
    return int(flag) & ~(1 << bit)


# ----------------------------------------------------------------------
# POLYAP
# ----------------------------------------------------------------------

@dataclass
class PolyapResult:
    coefficients: np.ndarray
    coefficient_rms: np.ndarray
    residuals: np.ndarray
    rms: float

    # Python index of Bernese IFOK.
    ifok: int


def polyap(
    x: np.ndarray,
    y: np.ndarray,
    flags: np.ndarray,
    ip: int,
) -> PolyapResult:
    """
    Python port of Bernese POLYAP.f.

    Bernese semantics:
      bit 0 -> observation not used at all
      bit 2 -> observation not used for approximation

    IP is the number of polynomial coefficients.
    Polynomial degree = IP - 1.

    The independent variable is normalized around the first usable
    observation exactly as in POLYAP.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    flags = np.asarray(flags, dtype=np.int64)

    n = len(x)

    if len(y) != n or len(flags) != n:
        raise ValueError("x, y and flags must have identical length")

    maxip = 10

    if ip > maxip:
        raise ValueError(
            f"POLYAP: too many polynomial coefficients: {ip} > {maxip}"
        )

    usable_fit = np.array(
        [
            (not tstflg(f, 0)) and (not tstflg(f, 2))
            for f in flags
        ],
        dtype=bool,
    )

    indices = np.flatnonzero(usable_fit)

    if len(indices) == 0:
        raise ValueError("POLYAP: no usable observations")

    ifok = int(indices[0])

    # Fortran:
    # XMAX = max(abs(X(I)-X(IFOK))) over usable observations.
    xmax = float(
        np.max(
            np.abs(
                x[indices] - x[ifok]
            )
        )
    )

    if xmax == 0.0:
        xmax = 1.0

    neff = int(len(indices))

    if neff < ip:
        raise ValueError(
            "POLYAP: not enough observations "
            f"(NEFF={neff}, IP={ip})"
        )

    # Construct the same normal equations as POLYAP.
    q = np.zeros((ip, ip), dtype=np.float64)
    rhs = np.zeros(ip, dtype=np.float64)

    for j in indices:

        xred = (x[j] - x[ifok]) / xmax

        a = np.ones(ip, dtype=np.float64)

        for k in range(1, ip):
            a[k] = a[k - 1] * xred

        q += np.outer(a, a)
        rhs += a * y[j]

    try:
        qinv = np.linalg.inv(q)
    except np.linalg.LinAlgError as exc:
        raise ValueError("POLYAP: normal matrix singular") from exc

    c_norm = qinv @ rhs

    # Residuals.
    residuals = np.zeros(n, dtype=np.float64)

    rms_sum = 0.0

    for j in range(n):

        # Bernese skips bit 0 completely.
        if tstflg(flags[j], 0):
            continue

        xred = (x[j] - x[ifok]) / xmax

        ys = 0.0
        xhlp = 1.0

        for k in range(ip):
            ys += c_norm[k] * xhlp
            xhlp *= xred

        # Bernese definition:
        # V(J) = approximation - observation
        residuals[j] = ys - y[j]

        if not tstflg(flags[j], 2):
            rms_sum += residuals[j] ** 2

    if (neff - ip) != 0 and rms_sum > 0.0:
        rms = float(
            np.sqrt(
                rms_sum / (neff - ip)
            )
        )
    else:
        rms = 0.0

    # RMS of normalized coefficients.
    crms = np.zeros(ip, dtype=np.float64)

    for k in range(ip):
        crms[k] = rms * np.sqrt(qinv[k, k])

    # Convert normalized polynomial coefficients back to the
    # Bernese representation.
    coefficients = c_norm.copy()

    xhlp = 1.0

    for k in range(ip):
        coefficients[k] /= xhlp
        crms[k] /= xhlp
        xhlp *= xmax

    return PolyapResult(
        coefficients=coefficients,
        coefficient_rms=crms,
        residuals=residuals,
        rms=rms,
        ifok=ifok,
    )


def evaluate_polyap(
    result: PolyapResult,
    dt: float,
) -> float:
    """
    Evaluate a POLYAP polynomial using the same expression used
    explicitly by CHKL4P:

        OFF = sum(C(I) * DT**(I-1))
    """

    value = 0.0

    for k, coefficient in enumerate(result.coefficients):
        value += coefficient * (dt ** k)

    return float(value)


# ----------------------------------------------------------------------
# UPDARC
# ----------------------------------------------------------------------

@dataclass
class BerneseArc:
    """
    Python representation of one RNXSMT/UPDARC arc.

    start/end are zero-based Python indices.
    """

    start: int
    end: int
    nobs: int
    nbad: int

    # Slip epochs are absolute zero-based indices in the satellite series.
    slip_indices: list[int]

    # Five Bernese cycle-slip components corresponding to SLPCYC(*,*,1:5).
    # Shape: (number of slips, 5).
    slip_cycles: np.ndarray | None = None


@dataclass
class UpdarcResult:
    arcs: list[BerneseArc]
    flags: np.ndarray
    observations: np.ndarray


def updarc(
    lintim: np.ndarray,
    linobs: np.ndarray,
    flags: np.ndarray,
    gaparc: float,
    minobs: int,
    *,
    event_times: np.ndarray | None = None,
    event_sizes: np.ndarray | None = None,
    event_systems: list[str] | None = None,
    satellite_system: str | None = None,
    dtsim: float = 0.0,
    previous_arcs: list[BerneseArc] | None = None,
) -> UpdarcResult:
    """
    Python port of the arc-construction logic in Bernese UPDARC.f.

    Parameters
    ----------
    lintim
        Observation times in the same units used by GAPARC.
        In RNXSMT/UPDARC these are hours.

    linobs
        Linear-combination observations.

    flags
        Bernese-style bit flags:
          bit 0 = bad/outlier
          bit 1 = new ambiguity
          bit 2 = corrected cycle slip

    gaparc
        Maximum allowed time gap between pieces of one arc.

    minobs
        Minimum number of valid observations required for an arc.

    Notes
    -----
    UPDARC starts a new arc when:

        TIMDIF > GAPARC

    or when an applicable unknown-size clock event occurs, or when

        bit 1 AND bit 2

    are simultaneously set at an epoch.

    The latter condition follows the supplied UPDARC source literally.
    """

    lintim = np.asarray(lintim, dtype=np.float64)
    observations = np.asarray(linobs, dtype=np.float64).copy()
    out_flags = np.asarray(flags, dtype=np.int64).copy()

    if not (
        len(lintim)
        == len(observations)
        == len(out_flags)
    ):
        raise ValueError(
            "lintim, linobs and flags must have identical length"
        )

    nobs_total = len(lintim)

    # --------------------------------------------------------------
    # SAVE ORIGINAL CYCLE SLIP INDICES / SIZES
    #
    # Literal structural equivalent of the first UPDARC block.
    # Bernese stores absolute slip epochs in SLPIND and the five
    # SLPCYC components in SLSIZE before rebuilding the arcs.
    # --------------------------------------------------------------

    saved_slip_cycles: dict[int, np.ndarray] = {}

    if previous_arcs is not None:

        for old_arc in previous_arcs:

            indices = list(old_arc.slip_indices)

            if old_arc.slip_cycles is None:

                # No known SLPCYC state to preserve.
                # Do not manufacture five zero-valued cycle corrections.
                continue

            cycles = np.asarray(
                old_arc.slip_cycles,
                dtype=np.float64,
            )

            if cycles.shape != (len(indices), 5):
                raise ValueError(
                    "BerneseArc.slip_cycles must have shape "
                    "(len(slip_indices), 5)"
                )

            for slip_index, slip_cycle in zip(
                indices,
                cycles,
            ):
                saved_slip_cycles[int(slip_index)] = (
                    np.asarray(
                        slip_cycle,
                        dtype=np.float64,
                    ).copy()
                )

    if nobs_total == 0:
        return UpdarcResult(
            arcs=[],
            flags=out_flags,
            observations=observations,
        )

    if event_times is None:
        event_times = np.empty(0, dtype=np.float64)

    if event_sizes is None:
        event_sizes = np.empty(0, dtype=np.float64)

    event_times = np.asarray(event_times, dtype=np.float64)
    event_sizes = np.asarray(event_sizes, dtype=np.float64)

    if len(event_times) != len(event_sizes):
        raise ValueError(
            "event_times and event_sizes must have identical length"
        )

    if event_systems is None:
        event_systems = ["*"] * len(event_times)

    if len(event_systems) != len(event_times):
        raise ValueError(
            "event_systems and event_times must have identical length"
        )

    # UPDARC initializes the first epoch as a new ambiguity.
    out_flags[0] = setflg(out_flags[0], 1)

    raw_arcs: list[tuple[int, int]] = []

    arc_start = 0
    endtim = lintim[0]

    for iobs in range(nobs_total):

        timdif = lintim[iobs] - endtim

        # --------------------------------------------------------------
        # Clock-event logic from UPDARC
        # --------------------------------------------------------------

        is_event = False

        for evt_time, evt_size, evt_sys in zip(
            event_times,
            event_sizes,
            event_systems,
        ):
            # Bernese:
            # IF (SIZEVT(IEVT).NE.1D20) CYCLE
            if evt_size != 1.0e20:
                continue

            system_matches = (
                evt_sys == "*"
                or satellite_system is None
                or evt_sys == satellite_system
            )

            if (
                system_matches
                and abs(evt_time - lintim[iobs]) <= dtsim
            ):
                is_event = True

            # Bernese exits because TIMEVT is ordered.
            if evt_time - lintim[iobs] > dtsim:
                break

        # Exact UPDARC condition:
        #
        # TIMDIF > GAPARC
        # OR clock event
        # OR (bit1 AND bit2)
        #
        split = (
            timdif > gaparc
            or is_event
            or (
                tstflg(out_flags[iobs], 1)
                and tstflg(out_flags[iobs], 2)
            )
        )

        # Do not create an empty arc before the first observation.
        if split and iobs > arc_start:

            raw_arcs.append(
                (arc_start, iobs - 1)
            )

            arc_start = iobs

            # Bernese marks first epoch of new arc with bit 1.
            out_flags[iobs] = setflg(
                out_flags[iobs],
                1,
            )

        endtim = lintim[iobs]

    raw_arcs.append(
        (arc_start, nobs_total - 1)
    )

    # --------------------------------------------------------------
    # Reproduce NOBA / NBAD / MINOBS treatment
    # --------------------------------------------------------------

    arcs: list[BerneseArc] = []

    for start, end in raw_arcs:

        nobs_arc = end - start + 1

        bad = 0
        good = 0
        slip_indices: list[int] = []

        for i in range(start, end + 1):

            if tstflg(out_flags[i], 2):
                slip_indices.append(i)

            if tstflg(out_flags[i], 0):
                bad += 1
            else:
                good += 1

        # Bernese UPDARC:
        # IF (NOK.LT.MINOBS) THEN
        #     mark complete arc bad
        #     LINOBS = 0
        # ENDIF
        if good < minobs:

            for i in range(start, end + 1):
                out_flags[i] = setflg(
                    out_flags[i],
                    0,
                )
                observations[i] = 0.0

            bad = nobs_arc

        # ----------------------------------------------------------
        # RESTORE ORIGINAL CYCLE SLIP SIZES
        #
        # UPDARC reconstructs NSLIP/SLPTIM from bit 2 and then
        # restores SLPCYC by matching the absolute slip epoch.
        #
        # A slip that did not exist in the previous arc state has
        # no saved SLPCYC yet, matching the fact that there is no
        # corresponding SLPIND/SLSIZE entry to restore.
        # ----------------------------------------------------------

        slip_cycles = np.zeros(
            (len(slip_indices), 5),
            dtype=np.float64,
        )

        for j, slip_index in enumerate(
            slip_indices
        ):
            if slip_index in saved_slip_cycles:
                slip_cycles[j, :] = (
                    saved_slip_cycles[slip_index]
                )

        arcs.append(
            BerneseArc(
                start=start,
                end=end,
                nobs=nobs_arc,
                nbad=bad,
                slip_indices=slip_indices,
                slip_cycles=slip_cycles,
            )
        )

    return UpdarcResult(
        arcs=arcs,
        flags=out_flags,
        observations=observations,
    )
