"""
B111 — Bernese RNXSMT screening routines.

CHKWUB is ported from the supplied Bernese CHKWUB.f.

IMPORTANT
---------
The Bernese routine OFFSET is not present in the supplied source set.
Therefore CHKWUB does NOT silently invent its implementation.

An offset_func must be supplied with signature:

    offset, rms = offset_func(times, observations, flags)

where flags use the Bernese-compatible bit representation from
bernese_core.py.

CHKWUB flag semantics:
    bit 0 = outlier / bad observation
    bit 1 = new ambiguity

Python arrays use zero-based indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math
import numpy as np

from .bernese_core import (
    tstflg,
    setflg,
    clrflg,
)


OffsetFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    tuple[float, float],
]


@dataclass
class ChkwubResult:
    observations: np.ndarray
    flags: np.ndarray

    nbad: int

    # Bernese ARCOFF after conversion with WLGT(5,SVN).
    arcoff: float

    # Preserved exactly as returned by OFFSET / CHKWUB logic.
    arcrms: float

    # Zero-based Python indices.
    slip_indices: list[int]

    # L5 / Melbourne-Wubbena slip sizes in cycles.
    slip_cycles: list[float]


def fortran_dnint(x: float) -> float:
    """
    Fortran DNINT semantics for ordinary finite values:
    nearest integer, halfway cases away from zero.

    Python round() is intentionally not used because it performs
    bankers rounding.
    """

    if x >= 0.0:
        return float(math.floor(x + 0.5))

    return float(math.ceil(x - 0.5))


def _count_bad(flags: np.ndarray) -> int:
    return sum(
        1
        for flag in flags
        if tstflg(flag, 0)
    )


def _call_offset(
    offset_func: OffsetFunction,
    times: np.ndarray,
    observations: np.ndarray,
    flags: np.ndarray,
) -> tuple[float, float]:

    off, rms = offset_func(
        times,
        observations,
        flags,
    )

    return float(off), float(rms)


def chkwub(
    lintim: np.ndarray,
    linobs: np.ndarray,
    linflg: np.ndarray,
    *,
    rmsmax: float,
    minobs: int,
    slpmin: float,
    slpout: float,
    wlgt5: float,
    offset_func: OffsetFunction,
    maxslp: int | None = None,
) -> ChkwubResult:
    """
    Port of Bernese CHKWUB.f.

    Parameters correspond to the scientific inputs used by CHKWUB.

    No numerical defaults are introduced for:
        RMSMAX
        MINOBS
        SLPMIN
        SLPOUT

    wlgt5 corresponds to Bernese WLGT(5,SVN).

    offset_func represents the unavailable Bernese s_offset routine.
    """

    times = np.asarray(
        lintim,
        dtype=np.float64,
    )

    obs = np.asarray(
        linobs,
        dtype=np.float64,
    ).copy()

    flags = np.asarray(
        linflg,
        dtype=np.int64,
    ).copy()

    nobs = len(obs)

    if (
        len(times) != nobs
        or len(flags) != nobs
    ):
        raise ValueError(
            "lintim, linobs and linflg must have identical length"
        )

    if nobs == 0:
        return ChkwubResult(
            observations=obs,
            flags=flags,
            nbad=0,
            arcoff=0.0,
            arcrms=0.0,
            slip_indices=[],
            slip_cycles=[],
        )

    slip_indices: list[int] = []
    slip_cycles: list[float] = []

    # ==============================================================
    # Bernese label 1:
    # ESTIMATE L5-P5 OFFSET
    # ==============================================================

    while True:

        nsold = len(slip_indices)

        arcoff, arcrms = _call_offset(
            offset_func,
            times,
            obs,
            flags,
        )

        # ==========================================================
        # HIGH RMS:
        # split arc into two pieces
        # ==========================================================

        if arcrms > rmsmax:

            # Fortran:
            # IOBS(1)=NOBS/2
            # IOBS(2)=NOBS-IOBS(1)
            #
            # Integer division in Fortran for positive NOBS.
            n1 = nobs // 2
            n2 = nobs - n1

            j1 = 0
            j2 = n1

            # Bernese rejects the complete arc if either half
            # contains fewer than MINOBS samples.
            if n1 < minobs or n2 < minobs:

                nbad = nobs

                for i in range(nobs):
                    flags[i] = setflg(
                        flags[i],
                        0,
                    )
                    obs[i] = 0.0

                arcoff = 0.0
                arcrms = 0.0

                break

            off1, rms1 = _call_offset(
                offset_func,
                times[j1:j1 + n1],
                obs[j1:j1 + n1],
                flags[j1:j1 + n1],
            )

            off2, rms2 = _call_offset(
                offset_func,
                times[j2:j2 + n2],
                obs[j2:j2 + n2],
                flags[j2:j2 + n2],
            )

            offs = [off1, off2]
            rmss = [rms1, rms2]
            starts = [j1, j2]
            counts = [n1, n2]

            # INDEX = half with smaller RMS
            # ISLIP = half with larger RMS
            #
            # Preserve the Fortran branch ordering:
            # RMS(1) < RMS(2) -> INDEX=1, ISLIP=2
            # otherwise       -> INDEX=2, ISLIP=1
            if rms1 < rms2:
                index = 0
                islip = 1
            else:
                index = 1
                islip = 0

            # ======================================================
            # Estimate approximate cycle-slip size.
            # ======================================================

            ndif = 0
            cycles = 0.0
            cycmax = 0.0

            start_bad_half = starts[islip]
            stop_bad_half = (
                start_bad_half
                + counts[islip]
            )

            for i in range(
                start_bad_half,
                stop_bad_half,
            ):

                diff = abs(
                    obs[i]
                    - offs[index]
                )

                if diff > cycmax:
                    cycmax = diff

                if diff > slpmin:
                    ndif += 1
                    cycles += diff

            if ndif > 1:
                cycles /= ndif

            # ------------------------------------------------------
            # Bernese:
            # high RMS but no sufficiently large cycle slip.
            # Proceed directly to final outlier screening.
            # ------------------------------------------------------

            if ndif == 0:
                pass

            else:

                # ==================================================
                # Determine sensitivity for locating the jump.
                # ==================================================

                if cycles < slpmin:
                    difmax = slpmin
                else:
                    difmax = min(
                        10.0,
                        cycles,
                    )

                jump: int | None = None

                # --------------------------------------------------
                # Search direction follows CHKWUB.
                #
                # Because JUMP is overwritten while scanning,
                # direction matters.
                # --------------------------------------------------

                if islip == 0:

                    scan = range(
                        starts[0],
                        starts[0] + counts[0],
                    )

                else:

                    scan = range(
                        starts[1] + counts[1] - 1,
                        starts[1] - 1,
                        -1,
                    )

                for i in scan:

                    diff = abs(
                        obs[i]
                        - offs[index]
                    )

                    # Fortran:
                    # IF (DIFF+1D-6.GE.DIFMAX) JUMP=I
                    if diff + 1.0e-6 >= difmax:
                        jump = i

                # ==================================================
                # Define two pieces around the detected jump.
                # ==================================================

                if jump is None:

                    # Fortran fallback:
                    # IFLAG(ISLIP)=IOBS(ISLIP)
                    if islip == 0:
                        jump = starts[0] + counts[0]
                    else:
                        jump = starts[1]

                else:

                    # Fortran:
                    # IF (ISLIP.EQ.1) JUMP=JUMP+1
                    #
                    # ISLIP=1 in Fortran corresponds to Python 0.
                    if islip == 0:
                        jump += 1

                # Convert the Fortran split:
                #
                # JP(1)=1
                # JP(2)=JUMP
                # IOBS(1)=JUMP-1
                # IOBS(2)=NOBS-JUMP+1
                #
                # into zero-based Python.
                split = jump

                if split <= 0:
                    split = 1

                if split >= nobs:
                    split = nobs - 1

                piece_starts = [
                    0,
                    split,
                ]

                piece_counts = [
                    split,
                    nobs - split,
                ]

                piece_off = [
                    0.0,
                    0.0,
                ]

                piece_rms = [
                    0.0,
                    0.0,
                ]

                piece_bad = [
                    False,
                    False,
                ]

                # ==================================================
                # Iterative OFFSET/outlier screening for each piece.
                # ==================================================

                for ipiece in range(2):

                    pstart = piece_starts[ipiece]
                    pcount = piece_counts[ipiece]
                    pend = pstart + pcount

                    valid_count = sum(
                        1
                        for i in range(
                            pstart,
                            pend,
                        )
                        if not tstflg(
                            flags[i],
                            0,
                        )
                    )

                    if valid_count < minobs:

                        piece_bad[ipiece] = True

                        for i in range(
                            pstart,
                            pend,
                        ):
                            flags[i] = setflg(
                                flags[i],
                                0,
                            )

                        continue

                    slpold = 9.0e20

                    while True:

                        poff, prms = _call_offset(
                            offset_func,
                            times[pstart:pend],
                            obs[pstart:pend],
                            flags[pstart:pend],
                        )

                        piece_off[ipiece] = poff
                        piece_rms[ipiece] = prms

                        if prms <= rmsmax:
                            break

                        slplvl = prms * 3.0

                        if (
                            slplvl
                            >= slpold - 1.0e-4
                        ):
                            slplvl = slpold / 2.0

                        slpold = slplvl

                        nvalid = 0

                        for i in range(
                            pstart,
                            pend,
                        ):

                            diff = abs(
                                obs[i]
                                - poff
                            )

                            if diff > slplvl:
                                flags[i] = setflg(
                                    flags[i],
                                    0,
                                )
                            else:
                                flags[i] = clrflg(
                                    flags[i],
                                    0,
                                )
                                nvalid += 1

                        if nvalid < minobs:

                            piece_bad[ipiece] = True
                            piece_off[ipiece] = 0.0
                            piece_rms[ipiece] = 0.0

                            for i in range(
                                pstart,
                                pend,
                            ):
                                flags[i] = setflg(
                                    flags[i],
                                    0,
                                )

                            break

                # ==================================================
                # Evaluate the two pieces.
                # ==================================================

                if (
                    piece_bad[0]
                    and piece_bad[1]
                ):

                    nbad = nobs

                    for i in range(nobs):
                        flags[i] = setflg(
                            flags[i],
                            0,
                        )
                        obs[i] = 0.0

                    arcoff = 0.0
                    arcrms = 0.0

                    break

                elif piece_bad[0]:

                    arcoff = piece_off[1]

                elif piece_bad[1]:

                    arcoff = piece_off[0]

                else:

                    arcoff = piece_off[0]

                    # Bernese:
                    # CYCLES=DNINT(OFF(2)-OFF(1))
                    cycles = fortran_dnint(
                        piece_off[1]
                        - piece_off[0]
                    )

                    if abs(cycles) >= slpmin:

                        if (
                            maxslp is not None
                            and len(slip_indices) >= maxslp
                        ):
                            raise RuntimeError(
                                "CHKWUB: MAXSLP exceeded"
                            )

                        slip_indices.append(
                            split
                        )

                        slip_cycles.append(
                            cycles
                        )

                        flags[split] = setflg(
                            flags[split],
                            1,
                        )

                        # Correct all observations in second piece.
                        # Bernese CHKWUB.f lines 303-305:
                        # apply CYCLES to every observation in piece 2.
                        for i in range(
                            split,
                            nobs,
                        ):
                            obs[i] -= cycles

                # Bernese clears all temporary outlier flags
                # before restarting after a newly detected slip.
                for i in range(nobs):
                    flags[i] = clrflg(
                        flags[i],
                        0,
                    )

                if len(slip_indices) > nsold:
                    continue

        # ==============================================================
        # FINAL OUTLIER SCREENING
        # ==============================================================

        slpold = 9.0e20
        slplvl = slpout

        while True:

            nbad = 0

            for i in range(nobs):

                diff = abs(
                    obs[i]
                    - arcoff
                )

                if diff > slplvl:
                    nbad += 1
                    flags[i] = setflg(
                        flags[i],
                        0,
                    )
                else:
                    flags[i] = clrflg(
                        flags[i],
                        0,
                    )

            if nobs - nbad < minobs:

                nbad = nobs

                for i in range(nobs):
                    flags[i] = setflg(
                        flags[i],
                        0,
                    )
                    obs[i] = 0.0

                arcoff = 0.0
                arcrms = 0.0

                break

            arcoff, arcrms = _call_offset(
                offset_func,
                times,
                obs,
                flags,
            )

            if arcrms >= rmsmax:

                slplvl = min(
                    slpout,
                    arcrms * 4.0,
                )

                if slplvl >= slpold:
                    slplvl = slpold / 2.0

                slpold = slplvl

                continue

            else:

                slpold = slplvl

                slplvl = min(
                    slpold,
                    arcrms * 4.0,
                )

                break

        if nbad == nobs:
            break

        # ==============================================================
        # FINAL CHECK
        # ==============================================================

        nbad = 0

        for i in range(nobs):

            diff = abs(
                obs[i]
                - arcoff
            )

            if (
                diff > slplvl
                or tstflg(flags[i], 0)
            ):

                nbad += 1

                flags[i] = setflg(
                    flags[i],
                    0,
                )

                obs[i] = 0.0

        if nobs - nbad < minobs:

            nbad = nobs

            for i in range(nobs):
                flags[i] = setflg(
                    flags[i],
                    0,
                )
                obs[i] = 0.0

            arcoff = 0.0
            arcrms = 0.0

            break

        arcoff, arcrms = _call_offset(
            offset_func,
            times,
            obs,
            flags,
        )

        # ==============================================================
        # REMOVE OFFSET FOR LATER PLOTTING
        # ==============================================================

        for i in range(nobs):

            if not tstflg(
                flags[i],
                0,
            ):
                obs[i] -= arcoff

        break

    # ==============================================================
    # CONVERT ARCOFF TO METERS
    #
    # Bernese:
    # ARCOFF = ARCOFF * WLGT(5,SVN)
    # ARCRMS = ARCRMS
    # ==============================================================

    arcoff *= wlgt5

    # ==============================================================
    # MERGE SLIPS AT IDENTICAL EPOCHS
    # ==============================================================

    if slip_indices:

        merged_indices: list[int] = []
        merged_cycles: list[float] = []

        for idx, cyc in zip(
            slip_indices,
            slip_cycles,
        ):

            if (
                merged_indices
                and idx == merged_indices[-1]
            ):
                merged_cycles[-1] += cyc

            else:
                merged_indices.append(idx)
                merged_cycles.append(cyc)

        slip_indices = merged_indices
        slip_cycles = merged_cycles

    nbad = _count_bad(flags)

    return ChkwubResult(
        observations=obs,
        flags=flags,
        nbad=nbad,
        arcoff=float(arcoff),
        arcrms=float(arcrms),
        slip_indices=slip_indices,
        slip_cycles=slip_cycles,
    )


# ======================================================================
# CHKNOI
# ======================================================================

@dataclass
class ChknoiResult:
    observations: np.ndarray
    flags: np.ndarray
    nbad: int
    arcoff: float
    arcrms: float


def chknoi(
    lintim: np.ndarray,
    linobs: np.ndarray,
    linflg: np.ndarray,
    *,
    rmsmax: float,
    minobs: int,
    offset_func: OffsetFunction,
    slip_indices: list[int] | None = None,
    slip_cycles: list[float] | None = None,
    wavelength: float = 1.0,
    icor: int = 1,
) -> ChknoiResult:
    """
    Port of the supplied Bernese CHKNOI.f.

    OFFSET is an external dependency because s_offset is not present
    in the supplied Bernese source set.

    No RMSMAX or MINOBS defaults are introduced.
    """

    times = np.asarray(
        lintim,
        dtype=np.float64,
    )

    obs = np.asarray(
        linobs,
        dtype=np.float64,
    ).copy()

    flags = np.asarray(
        linflg,
        dtype=np.int64,
    ).copy()

    nobs = len(obs)

    if len(times) != nobs or len(flags) != nobs:
        raise ValueError(
            "lintim, linobs and linflg must have identical length"
        )

    slips = (
        []
        if slip_indices is None
        else [int(x) for x in slip_indices]
    )

    cycles = (
        []
        if slip_cycles is None
        else [float(x) for x in slip_cycles]
    )

    if len(slips) != len(cycles):
        raise ValueError(
            "slip_indices and slip_cycles must have identical length"
        )

    # ==============================================================
    # Apply supplied cycle-slip corrections.
    # ==============================================================

    if icor == 1:

        for index, cycle in zip(
            slips,
            cycles,
        ):

            if index < 0 or index >= nobs:
                raise IndexError(
                    f"slip index {index} outside observation arc"
                )

            correction = (
                cycle
                * wavelength
            )

            for i in range(
                index,
                nobs,
            ):

                if not tstflg(
                    flags[i],
                    0,
                ):
                    obs[i] -= correction

    # ==============================================================
    # Initial OFFSET.
    # ==============================================================

    arcoff, arcrms = _call_offset(
        offset_func,
        times,
        obs,
        flags,
    )

    # ==============================================================
    # If RMS is already acceptable, skip screening.
    # ==============================================================

    if arcrms > rmsmax:

        # CHKNOI uses a temporary flag array while screening.
        tmpflg = flags.copy()

        slpold = 9.0e20

        # Source:
        # SLPOUT = 3D0 * ARCRMS
        slpout = (
            3.0
            * arcrms
        )

        slplvl = slpout

        while True:

            nvalid = 0

            # ------------------------------------------------------
            # Temporary screening.
            # ------------------------------------------------------

            for i in range(nobs):

                diff = abs(
                    obs[i]
                    - arcoff
                )

                if diff > slplvl:

                    tmpflg[i] = setflg(
                        tmpflg[i],
                        0,
                    )

                else:

                    tmpflg[i] = clrflg(
                        tmpflg[i],
                        0,
                    )

                    nvalid += 1

            # ------------------------------------------------------
            # Too few observations.
            # ------------------------------------------------------

            if nvalid < minobs:

                for i in range(nobs):

                    flags[i] = setflg(
                        flags[i],
                        0,
                    )

                    obs[i] = 0.0

                return ChknoiResult(
                    observations=obs,
                    flags=flags,
                    nbad=nobs,
                    arcoff=0.0,
                    arcrms=0.0,
                )

            # ------------------------------------------------------
            # Re-estimate OFFSET using temporary flags.
            # ------------------------------------------------------

            arcoff, arcrms = _call_offset(
                offset_func,
                times,
                obs,
                tmpflg,
            )

            # ------------------------------------------------------
            # Source iterative threshold logic.
            # ------------------------------------------------------

            if arcrms >= rmsmax:

                slplvl = min(
                    slpout,
                    arcrms * 4.0,
                )

                if slplvl >= slpold:
                    slplvl = (
                        slpold
                        / 2.0
                    )

                slpold = slplvl

                continue

            else:

                slpold = slplvl

                slplvl = min(
                    slpold,
                    arcrms * 4.0,
                )

                break

        # ==========================================================
        # Final screening is written into the real LINFLG.
        # ==============================================================

        nbad = 0

        for i in range(nobs):

            diff = abs(
                obs[i]
                - arcoff
            )

            if (
                diff > slplvl
                or tstflg(
                    tmpflg[i],
                    0,
                )
            ):

                flags[i] = setflg(
                    flags[i],
                    0,
                )

                # CHKNOI source uses 1D0 for rejected observations
                # at this stage.
                obs[i] = 1.0

                nbad += 1

            else:

                flags[i] = clrflg(
                    flags[i],
                    0,
                )

        arcoff, arcrms = _call_offset(
            offset_func,
            times,
            obs,
            flags,
        )

    # ==============================================================
    # Remove offset from all valid observations.
    # ==============================================================

    for i in range(nobs):

        if not tstflg(
            flags[i],
            0,
        ):
            obs[i] -= arcoff

    nbad = _count_bad(
        flags
    )

    # ==============================================================
    # Bernese CHKNOI:
    #
    # BADPRC = NBAD*100/NOBS
    # IF (BADPRC.GE.75D0) NBAD=NOBS
    # ==============================================================

    if nobs > 0:

        badprc = (
            nbad
            * 100.0
            / nobs
        )

        if badprc >= 75.0:
            nbad = nobs

    # ==============================================================
    # Final MINOBS rejection.
    # ==============================================================

    if (
        nobs - _count_bad(flags)
        < minobs
    ):

        nbad = nobs

        arcoff = 0.0
        arcrms = 0.0

        for i in range(nobs):

            flags[i] = setflg(
                flags[i],
                0,
            )

            obs[i] = 0.0

    return ChknoiResult(
        observations=obs,
        flags=flags,
        nbad=nbad,
        arcoff=float(arcoff),
        arcrms=float(arcrms),
    )


# ----------------------------------------------------------------------
# OFFSET dependency boundary
# ----------------------------------------------------------------------

class MissingBerneseOffsetError(RuntimeError):
    """
    Raised when a Bernese-derived screening routine requires OFFSET
    but no OFFSET implementation has been explicitly supplied.

    The original Bernese routines CHKWUB, CHKNOI and CHKOFF depend on
    module s_offset.  That module is absent from the supplied Bernese
    source set used for the B111 reconstruction.

    B111 therefore does not silently substitute a statistical estimator.
    """


def require_offset(offset_fn):
    """
    Return an explicitly supplied OFFSET implementation.

    No scientific default is provided because the implementation of
    Bernese s_offset is unavailable in the supplied source set.
    """
    if offset_fn is None:
        raise MissingBerneseOffsetError(
            "Bernese s_offset is absent from the supplied source set. "
            "An explicit offset_fn must be supplied. "
            "No scientific default is assumed by B111."
        )

    if not callable(offset_fn):
        raise TypeError(
            "offset_fn must be callable."
        )

    return offset_fn


def bernese_offset(
    lintim,
    linobs,
    linflg,
):
    """
    Literal Python implementation of Bernese GNSS Software 5.2
    LIB/FOR/OFFSET.f, module s_OFFSET.

    Original Bernese semantics:

        SUBROUTINE OFFSET(X,Y,L,N,YMEAN,YRMS)

    X is accepted for interface compatibility but is not used by
    OFFSET.f.

    Observations are excluded when either:

        bit 0 = 1
        bit 2 = 1

    YMEAN:
        arithmetic mean of usable observations.

    YRMS:
        sqrt(
            sum((Y - YMEAN)**2)
            /
            (NTOT - 1)
        )

    This is the exact statistical definition present in the supplied
    Bernese 5.2 OFFSET.f source.
    """

    import numpy as np

    y = np.asarray(
        linobs,
        dtype=np.float64,
    )

    flags = np.asarray(
        linflg,
        dtype=np.int64,
    )

    if len(y) != len(flags):
        raise ValueError(
            "linobs and linflg must have identical length"
        )

    # Bernese OFFSET.f:
    #
    # IF(TSTFLG(L(I),0).OR.TSTFLG(L(I),2)) GOTO ...
    good = (
        ((flags & (1 << 0)) == 0)
        &
        ((flags & (1 << 2)) == 0)
    )

    values = y[good]

    ntot = int(values.size)

    # OFFSET.f assumes that its caller supplies enough usable
    # observations for RMS estimation.
    #
    # Make failure explicit in Python rather than silently inventing
    # a value.
    if ntot < 2:
        raise ValueError(
            "Bernese OFFSET requires at least two usable observations "
            f"(NTOT={ntot})"
        )

    # Fortran:
    # SUM=SUM+Y(I)
    # YMEAN=SUM/(NTOT*1D0)
    ymean = float(
        np.sum(
            values,
            dtype=np.float64,
        )
        /
        float(ntot)
    )

    # Fortran:
    # SUM2=SUM2+(Y(I)-YMEAN)**2
    # YRMS=DSQRT(SUM2/(1D0*(NTOT-1)))
    residual = values - ymean

    yrms = float(
        np.sqrt(
            np.sum(
                residual * residual,
                dtype=np.float64,
            )
            /
            float(ntot - 1)
        )
    )

    return ymean, yrms



def test_only_mean_rms_offset(
    lintim,
    linobs,
    linflg,
):
    """
    TEST ONLY — NOT A BERNESE SCIENTIFIC IMPLEMENTATION.

    Arithmetic mean and population RMS of observations whose bit-0
    outlier flag is not set.

    This exists only for controlled unit/integration tests of the
    surrounding Bernese-derived control flow.

    It MUST NOT be interpreted as a reconstruction of s_offset.
    """

    import numpy as np

    nobs = len(linobs)

    x = np.asarray(
        linobs,
        dtype=float,
    )

    # Existing B111 flags are integer bit masks:
    # bit 0 == outlier.
    flags = np.asarray(
        linflg,
    )

    good = (
        np.isfinite(x)
        & ((flags.astype(int) & 1) == 0)
    )

    if not np.any(good):
        return 0.0, 0.0

    values = x[good]

    arcoff = float(
        np.mean(values)
    )

    arcrms = float(
        np.sqrt(
            np.mean(
                (values - arcoff) ** 2
            )
        )
    )

    return arcoff, arcrms
