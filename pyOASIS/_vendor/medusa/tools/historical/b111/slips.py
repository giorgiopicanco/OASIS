"""
B111 — Bernese RNXSMT cycle-slip routines.

CHKL4P is ported from the supplied Bernese CHKL4P.f.

Purpose
-------
Check L4 phase observation arcs around cycle-slip candidates and derive
the corresponding L1/L2/L3/L4 corrections using the supplied Bernese
POLYAP implementation.

Flag semantics
--------------
bit 0 = outlier / bad observation
bit 1 = new ambiguity
bit 2 = corrected cycle slip

Python uses zero-based indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np

from .bernese_core import (
    tstflg,
    setflg,
    clrflg,
    polyap,
)


@dataclass
class Chkl4pResult:
    observations: np.ndarray
    flags: np.ndarray

    nbad: int

    slip_indices: list[int]

    cycl1: list[float]
    cycl2: list[float]
    cycl3: list[float]
    cycl4: list[float]
    cycl5: list[float]


def fortran_dnint(x: float) -> float:
    """
    Fortran DNINT:
    nearest integer, halfway cases away from zero.
    """

    if x >= 0.0:
        return float(math.floor(x + 0.5))

    return float(math.ceil(x - 0.5))


def _poly_value(
    coefficients: np.ndarray,
    x: float,
) -> float:
    """
    Evaluate Bernese POLYAP coefficients:
        C(1) + C(2)*x + C(3)*x**2 + ...
    """

    value = 0.0

    for i, coefficient in enumerate(coefficients):
        value += float(coefficient) * x**i

    return value


def chkl4p(
    lintim: np.ndarray,
    linobs: np.ndarray,
    linflg: np.ndarray,
    slip_indices: list[int],
    cycl5: list[float],
    *,
    rmsl4: float,
    gapl4: float,
    minl4: int,
    ifxslp: int,
    ipoly: int,
    wlgt1: float,
    wlgt2: float,
    wlgt3: float,
    wlgt4: float,
    faclin31: float,
    faclin32: float,
) -> Chkl4pResult:
    """
    Port of Bernese CHKL4P.f.

    Parameters correspond directly to CHKL4P / COMFREQ quantities.

    No scientific threshold defaults are supplied.
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

    if len(slip_indices) != len(cycl5):
        raise ValueError(
            "slip_indices and cycl5 must have identical length"
        )

    if minl4 <= 0:
        raise ValueError("minl4 must be positive")

    if ipoly <= 0:
        raise ValueError("ipoly must be positive")

    if wlgt4 == 0.0:
        raise ValueError("wlgt4 must not be zero")

    slips = [
        int(index)
        for index in slip_indices
    ]

    c5 = [
        float(value)
        for value in cycl5
    ]

    c1 = [0.0] * len(slips)
    c2 = [0.0] * len(slips)
    c3 = [0.0] * len(slips)
    c4 = [0.0] * len(slips)

    # Bernese CHKL4P NBAD accounting.
    #
    # CHKL4P.f receives NBAD as IN/OUT. This Python API currently
    # has no incoming NBAD argument, so this counter records the
    # number of NBAD increments produced by this invocation.
    nbad = 0

    # Process every cycle-slip candidate.
    for islp in range(len(slips)):

        c1[islp] = 0.0
        c2[islp] = 0.0
        c3[islp] = 0.0
        c4[islp] = 0.0

        original_slip = slips[islp]

        if original_slip < 0 or original_slip >= nobs:
            raise IndexError(
                f"slip index {original_slip} outside observation arc"
            )

        # ==========================================================
        # Literal CHKL4P.f window-state translation.
        #
        # Fortran:
        #
        #   ISTART = SLPTIM(ISLP)-1
        #   IEND   = SLPTIM(ISLP)
        #
        # ISTART/IEND delimit contiguous array sections passed to
        # POLYAP. Bad observations inside those sections remain in
        # the section and are rejected by their bit-0 flags.
        # ==========================================================

        istart = original_slip - 1
        iend = original_slip

        rms1 = 0.0
        rms2 = 0.0

        ilast = None
        ifirst = None

        fit1 = None
        fit2 = None

        abandon_slip = False

        while True:

            # ======================================================
            # Label 20:
            # FIND "MINL4" OBSERVATIONS BEFORE CYCLE SLIP
            # ======================================================

            if rms1 == 0.0:

                iobs = 0

                while True:

                    # Fortran ISTART .LT. 1.
                    #
                    # Fortran indices are one-based; therefore
                    # Python ISTART < 0 is the equivalent condition.

                    if istart < 0:

                        flags[original_slip] = clrflg(
                            flags[original_slip],
                            1,
                        )

                        # Fortran:
                        # DO I=1,SLPTIM(ISLP)-1
                        #
                        # Python:
                        # [0, original_slip)

                        for i in range(
                            0,
                            original_slip,
                        ):
                            flags[i] = setflg(
                                flags[i],
                                0,
                            )
                            obs[i] = 0.0
                            nbad += 1

                        abandon_slip = True
                        break

                    if tstflg(
                        flags[istart],
                        0,
                    ):
                        istart -= 1
                        continue

                    iobs += 1

                    if iobs == 1:
                        ilast = istart

                    if iobs < minl4:
                        istart -= 1
                        continue

                    break

                if abandon_slip:
                    break

            # ======================================================
            # Label 30:
            # FIND "MINL4" OBSERVATIONS AFTER CYCLE SLIP
            # ======================================================

            if rms2 == 0.0:

                iobs = 0

                while True:

                    # Fortran IEND .GT. NOBS.
                    #
                    # With zero-based Python indexing this becomes
                    # IEND >= nobs.

                    if iend >= nobs:

                        flags[original_slip] = clrflg(
                            flags[original_slip],
                            1,
                        )

                        # Fortran:
                        # DO I=SLPTIM(ISLP),NOBS
                        #
                        # Python:
                        # [original_slip, nobs)

                        for i in range(
                            original_slip,
                            nobs,
                        ):
                            flags[i] = setflg(
                                flags[i],
                                0,
                            )
                            obs[i] = 0.0
                            nbad += 1

                        abandon_slip = True
                        break

                    if tstflg(
                        flags[iend],
                        0,
                    ):
                        iend += 1
                        continue

                    iobs += 1

                    if iobs == 1:
                        ifirst = iend

                    if iobs < minl4:
                        iend += 1
                        continue

                    break

                if abandon_slip:
                    break

            assert ilast is not None
            assert ifirst is not None

            # ======================================================
            # CHECK IF GAP IS STILL SMALL ENOUGH
            #
            # Preserve CHKL4P.f comparison literally:
            #
            # DT = LINTIM(IFIRST)-LINTIM(ILAST)
            # IF (DT.GE.GAPL4) GOTO 300
            # ======================================================

            dtgap = (
                times[ifirst]
                - times[ilast]
            )

            if dtgap >= gapl4:

                for i in range(
                    ilast + 1,
                    ifirst,
                ):
                    flags[i] = setflg(
                        flags[i],
                        0,
                    )
                    nbad += 1

                    # CHKL4P.f also clears ambiguity bit 1.
                    flags[i] = clrflg(
                        flags[i],
                        1,
                    )

                    obs[i] = 0.0

                # Equivalent of GOTO 300.
                #
                # No L4 correction is performed in this branch.
                fit1 = None
                fit2 = None
                break

            # ======================================================
            # FIT POLYNOMIAL THROUGH BOTH PARTS
            #
            # Important:
            #
            # The Fortran passes the CONTIGUOUS interval
            # ISTART:ILAST and IFIRST:IEND to POLYAP.
            # Flagged observations remain in these arrays.
            # ======================================================

            left_indices = np.arange(
                istart,
                ilast + 1,
                dtype=np.int64,
            )

            right_indices = np.arange(
                ifirst,
                iend + 1,
                dtype=np.int64,
            )

            fit1 = polyap(
                times[left_indices],
                obs[left_indices],
                flags[left_indices],
                ipoly,
            )

            fit2 = polyap(
                times[right_indices],
                obs[right_indices],
                flags[right_indices],
                ipoly,
            )

            rms1 = float(
                fit1.rms
            )

            rms2 = float(
                fit2.rms
            )

            # ======================================================
            # CHKL4P.f lines 189-205.
            #
            # IF BOTH RMS TOO LARGE:
            #
            #   ISTART = ILAST-1
            #   IEND   = IFIRST+1
            #   RMS(1) = 0
            #   RMS(2) = 0
            #   GOTO 20
            #
            # LEFT ONLY:
            #
            #   ISTART = ILAST-1
            #   RMS(1) = 0
            #   GOTO 20
            #
            # RIGHT ONLY:
            #
            #   IEND   = IFIRST+1
            #   RMS(2) = 0
            #   GOTO 30
            # ======================================================

            bad1 = rms1 > rmsl4
            bad2 = rms2 > rmsl4

            if bad1 and bad2:

                istart = ilast - 1
                iend = ifirst + 1

                rms1 = 0.0
                rms2 = 0.0

                continue

            if bad1:

                istart = ilast - 1
                rms1 = 0.0

                continue

            if bad2:

                iend = ifirst + 1
                rms2 = 0.0

                # Emulate GOTO 30:
                #
                # left RMS remains non-zero, so label 20 is skipped
                # on the next loop iteration.

                continue

            # Both fits accepted.
            break

        if abandon_slip:
            continue

        # ==========================================================
        # GAPL4 branch: equivalent of source GOTO 300.
        # ==========================================================

        if fit1 is None or fit2 is None:

            assert ilast is not None
            assert ifirst is not None

            flags[original_slip] = clrflg(
                flags[original_slip],
                1,
            )

            flags[ifirst] = setflg(
                flags[ifirst],
                2,
            )

            slips[islp] = ifirst

            # Source reaches label 300 with DT >= GAPL4,
            # therefore the final ambiguity condition is true.

            flags[ifirst] = setflg(
                flags[ifirst],
                1,
            )

            continue

        # Final contiguous POLYAP sections corresponding exactly to
        # the accepted Fortran ISTART/ILAST and IFIRST/IEND states.

        left_indices = np.arange(
            istart,
            ilast + 1,
            dtype=np.int64,
        )

        right_indices = np.arange(
            ifirst,
            iend + 1,
            dtype=np.int64,
        )

        # ==========================================================
        # Mark observations between the two polynomial pieces bad.
        # ==========================================================

        for i in range(
            ilast + 1,
            ifirst,
        ):
            flags[i] = setflg(
                flags[i],
                0,
            )
            obs[i] = 0.0
            nbad += 1

        flags[original_slip] = clrflg(
            flags[original_slip],
            1,
        )

        # ==========================================================
        # Extrapolate both polynomial fits to the original slip epoch.
        #
        # Bernese POLYAP returns coefficients referenced to X(IFOK).
        # After removal of the XMAX normalization, the polynomial
        # argument remains:
        #
        #     DT = X - X(IFOK)
        #
        # CHKL4P.f evaluates:
        #
        #   DT = LINTIM(SLPTIM) - LINTIM(ISTART + IFOK(1))
        #   DT = LINTIM(SLPTIM) - LINTIM(IFIRST + IFOK(2))
        #
        # Python IFOK is zero-based, so the corresponding absolute
        # reference epochs are obtained from the actual fit windows.
        # ==========================================================

        slip_time = times[original_slip]

        ref1 = times[
            left_indices[fit1.ifok]
        ]

        ref2 = times[
            right_indices[fit2.ifok]
        ]

        dt1 = slip_time - ref1
        dt2 = slip_time - ref2

        off1 = _poly_value(
            fit1.coefficients,
            dt1,
        )

        off2 = _poly_value(
            fit2.coefficients,
            dt2,
        )

        # ==========================================================
        # Bernese CHKL4P cycle-slip equations.
        # ==========================================================

        diff = (
            off2 - off1
        ) / wlgt4

        dl4 = fortran_dnint(
            diff
        )

        dl2 = (
            wlgt1 * c5[islp]
            - wlgt4 * dl4
        ) / (
            wlgt2 - wlgt1
        )

        dl1 = (
            dl2
            + c5[islp]
        )

        c1[islp] = fortran_dnint(
            dl1
        )

        c2[islp] = fortran_dnint(
            dl2
        )

        c4[islp] = dl4

        # ==========================================================
        # Recompute L4 from integer L1/L2 corrections.
        # ==========================================================

        dl4_check = (
            wlgt1 * c1[islp]
            - wlgt2 * c2[islp]
        )

        dl4_check = fortran_dnint(
            dl4_check / wlgt4
        )

        # ==========================================================
        # Recompute L5.
        # ==========================================================

        dl5_check = (
            c1[islp]
            - c2[islp]
        )

        diffl4 = abs(
            dl4_check
            - c4[islp]
        )

        diffl5 = abs(
            dl5_check
            - c5[islp]
        )

        # Preserve source calculation even though DIFFL5 is not
        # part of the final ambiguity condition shown in CHKL4P.
        _ = diffl5

        # ==========================================================
        # L3 cycle-slip correction.
        # ==========================================================

        dl3 = (
            faclin31
            * wlgt1
            * c1[islp]
            +
            faclin32
            * wlgt2
            * c2[islp]
        )

        c3[islp] = fortran_dnint(
            dl3 / wlgt3
        )

        # ==========================================================
        # Correct L4 observations after IFIRST.
        # ==========================================================

        for i in range(
            ifirst,
            nobs,
        ):

            if tstflg(
                flags[i],
                0,
            ):
                obs[i] = 0.0

            else:
                obs[i] -= (
                    c4[islp]
                    * wlgt4
                )

        # ==========================================================
        # Final flag logic.
        # ==========================================================

        dtgap = (
            times[ifirst]
            - times[ilast]
        )

        flags[original_slip] = clrflg(
            flags[original_slip],
            1,
        )

        flags[ifirst] = setflg(
            flags[ifirst],
            2,
        )

        slips[islp] = ifirst

        if (
            diffl4 >= rmsl4
            or dtgap >= gapl4
            or ifxslp == 0
        ):
            flags[ifirst] = setflg(
                flags[ifirst],
                1,
            )

    return Chkl4pResult(
        observations=obs,
        flags=flags,
        nbad=nbad,
        slip_indices=slips,
        cycl1=c1,
        cycl2=c2,
        cycl3=c3,
        cycl4=c4,
        cycl5=c5,
    )
