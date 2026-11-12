"""
B111 geometry-free phase-to-code leveling.

The leveling operation is taken directly from the existing MEDUSA D56
procedure:

    bias = mean(L_GF - P_GF)
    leveled_GF = L_GF - bias

The bias is estimated independently for every final B111 arc.

This module does not detect or repair cycle slips.  It operates on the
cycle-slip-corrected L4 observations produced by CHKL4P/UPDARC.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class B111LeveledArc:
    arc_id: int
    start: int
    end: int
    indices: np.ndarray

    bias_m: float

    phase_gf_corrected_m: np.ndarray
    code_gf_m: np.ndarray
    leveled_gf_m: np.ndarray
    residual_m: np.ndarray


@dataclass
class B111LevelingResult:
    leveled_gf_m: np.ndarray
    code_gf_m: np.ndarray
    arc_bias_m: np.ndarray
    arc_id: np.ndarray
    residual_m: np.ndarray
    arcs: list[B111LeveledArc]


def level_b111_geometry_free(
    corrected_l4_m: np.ndarray,
    code_gf_m: np.ndarray,
    flags: np.ndarray,
    arcs,
) -> B111LevelingResult:
    """
    Level cycle-slip-corrected geometry-free phase to geometry-free code.

    Per final B111 arc:

        bias = mean(L_GF - P_GF)
        L_GF_leveled = L_GF - bias

    Epochs with Bernese bit-0 set, or non-finite phase/code values,
    are excluded from the bias estimate and remain NaN in the leveled
    output.
    """

    phase = np.asarray(
        corrected_l4_m,
        dtype=np.float64,
    )

    code = np.asarray(
        code_gf_m,
        dtype=np.float64,
    )

    flg = np.asarray(
        flags,
        dtype=np.int64,
    )

    if not (
        len(phase)
        == len(code)
        == len(flg)
    ):
        raise ValueError(
            "phase, code and flags must have identical length"
        )

    n = len(phase)

    leveled = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    bias_array = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    arc_ids = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    residual = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    result_arcs = []

    for arc_id, arc in enumerate(
        arcs,
        start=1,
    ):

        start = int(arc.start)
        end = int(arc.end)

        if start < 0 or end >= n or end < start:
            raise ValueError(
                f"Invalid B111 arc bounds: {start}:{end}"
            )

        idx = np.arange(
            start,
            end + 1,
            dtype=np.int64,
        )

        good = (
            ((flg[idx] & 1) == 0)
            & np.isfinite(phase[idx])
            & np.isfinite(code[idx])
        )

        good_idx = idx[good]

        if good_idx.size == 0:
            continue

        bias = float(
            np.mean(
                phase[good_idx]
                -
                code[good_idx]
            )
        )

        arc_leveled = (
            phase[good_idx]
            -
            bias
        )

        arc_residual = (
            arc_leveled
            -
            code[good_idx]
        )

        leveled[good_idx] = arc_leveled
        residual[good_idx] = arc_residual
        bias_array[good_idx] = bias
        arc_ids[good_idx] = arc_id

        result_arcs.append(
            B111LeveledArc(
                arc_id=arc_id,
                start=start,
                end=end,
                indices=good_idx.copy(),
                bias_m=bias,
                phase_gf_corrected_m=phase[
                    good_idx
                ].copy(),
                code_gf_m=code[
                    good_idx
                ].copy(),
                leveled_gf_m=arc_leveled.copy(),
                residual_m=arc_residual.copy(),
            )
        )

    return B111LevelingResult(
        leveled_gf_m=leveled,
        code_gf_m=code.copy(),
        arc_bias_m=bias_array,
        arc_id=arc_ids,
        residual_m=residual,
        arcs=result_arcs,
    )
