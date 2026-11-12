"""
B111 RINEX driver.

Real RINEX -> exact available dual-frequency observable sets -> L4/L5.

No scientific thresholds are introduced here.
No reconstruction of RSMTIN, OFFSET, DETSLP or complete LINCMB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rinex_signals import (
    read_rinex_header,
)

from .rinex3_stream import (
    read_rinex3_stream,
)

from .pairs import (
    physical_pairs,
)

from .observables import (
    mw_observable_sets,
)

from .combinations import (
    FrequencyPair,
    geometry_free_phase,
    bernese_l5_mw_cycles,
)


@dataclass
class B111RinexCombination:
    satellite: str

    phase1: str
    phase2: str
    code1: str
    code2: str

    f1_hz: float
    f2_hz: float

    times: np.ndarray

    phase1_cycles: np.ndarray
    phase2_cycles: np.ndarray
    code1_m: np.ndarray
    code2_m: np.ndarray

    # Geometry-free observables in meters.
    #
    # Phase:
    #     L4 = L1 - L2
    #
    # Code:
    #     P_GF = P1 - P2
    #
    # code_gf_m is retained explicitly because the final B111
    # phase-to-code leveling uses:
    #
    #     bias = mean(L4_corrected - P_GF)
    #
    # independently for each final arc.
    l4_m: np.ndarray
    code_gf_m: np.ndarray

    l5_cycles: np.ndarray

    valid: np.ndarray


def build_rinex_combinations_from_stream(
    path,
    satellite: str,
    *,
    stream,
    header,
) -> list[B111RinexCombination]:

    if not satellite:
        raise ValueError(
            "Empty satellite identifier"
        )

    system = satellite[0]

    # ----------------------------------------------------------
    # Header
    # ----------------------------------------------------------

    if system not in header.observation_types:
        raise ValueError(
            f"GNSS system {system!r} "
            "is not present in RINEX header"
        )

    header_observables = set(
        header.observation_types[system]
    )

    # ----------------------------------------------------------
    # Observation body
    # ----------------------------------------------------------

    if satellite not in stream.satellites:
        raise ValueError(
            f"Satellite {satellite!r} "
            "is not present in RINEX data"
        )

    sat = stream.satellites[satellite]

    stream_observables = set(
        sat.keys()
    )

    # Only observables actually represented in both the RINEX
    # header and streaming decoder are candidates.
    available = (
        header_observables
        & stream_observables
    )

    available_phase = sorted(
        obs
        for obs in available
        if obs.startswith("L")
    )

    # ----------------------------------------------------------
    # Physical dual-frequency pairs.
    #
    # This existing B111 routine collapses tracking codes that
    # share the same physical carrier and returns every unique
    # two-frequency pair.
    # ----------------------------------------------------------

    pairs = physical_pairs(
        system,
        satellite,
        available_phase,
        header,
    )

    results: list[
        B111RinexCombination
    ] = []

    # ----------------------------------------------------------
    # Exact tracking-code observable sets for every physical pair.
    # ----------------------------------------------------------

    for physical_pair in pairs:

        observable_sets = mw_observable_sets(
            physical_pair.observables1,
            physical_pair.observables2,
            available,
        )

        pair = FrequencyPair(
            f1=float(physical_pair.f1),
            f2=float(physical_pair.f2),
        )

        for obsset in observable_sets:

            phase1 = np.asarray(
                sat[obsset.phase1],
                dtype=np.float64,
            )

            phase2 = np.asarray(
                sat[obsset.phase2],
                dtype=np.float64,
            )

            code1 = np.asarray(
                sat[obsset.code1],
                dtype=np.float64,
            )

            code2 = np.asarray(
                sat[obsset.code2],
                dtype=np.float64,
            )

            valid = (
                np.isfinite(phase1)
                & np.isfinite(phase2)
                & np.isfinite(code1)
                & np.isfinite(code2)
            )

            # Existing B111 mathematical implementations.
            l4 = geometry_free_phase(
                phase1,
                phase2,
                pair,
            )

            # Geometry-free code consistent with the ionospheric
            # sign of the phase geometry-free combination:
            #
            #     L4 = L1 - L2
            #     P4 = P2 - P1
            #
            # Carrier phase and pseudorange contain the first-order
            # ionospheric contribution with opposite signs. Therefore
            # the code combination must use the reverse order for
            # phase-code leveling.
            #
            # RINEX pseudoranges are already in meters.
            code_gf = (
                code2
                -
                code1
            )

            l5 = bernese_l5_mw_cycles(
                phase1,
                phase2,
                code1,
                code2,
                pair,
            )

            results.append(
                B111RinexCombination(
                    satellite=satellite,

                    phase1=obsset.phase1,
                    phase2=obsset.phase2,
                    code1=obsset.code1,
                    code2=obsset.code2,

                    f1_hz=pair.f1,
                    f2_hz=pair.f2,

                    times=np.asarray(
                        stream.times
                    ),

                    phase1_cycles=phase1,
                    phase2_cycles=phase2,
                    code1_m=code1,
                    code2_m=code2,

                    l4_m=np.asarray(
                        l4,
                        dtype=np.float64,
                    ),

                    code_gf_m=np.asarray(
                        code_gf,
                        dtype=np.float64,
                    ),

                    l5_cycles=np.asarray(
                        l5,
                        dtype=np.float64,
                    ),

                    valid=valid,
                )
            )

    return results


@dataclass
class B111RinexArcSet:
    """
    Temporal GAPARC segmentation for one real RINEX combination.

    These are temporal continuity arcs only.
    They are not ambiguity/cycle-slip arcs.
    """

    combination: B111RinexCombination
    arcs: list
    labels: np.ndarray




def build_rinex_combinations(
    path,
    satellite: str,
) -> list[B111RinexCombination]:
    """
    Backward-compatible single-satellite interface.

    This interface streams the RINEX file once for the requested
    GNSS system. Multi-satellite processing should decode the file
    once externally and use build_rinex_combinations_from_stream().
    """

    if not satellite:
        raise ValueError(
            "Empty satellite identifier"
        )

    system = satellite[0]

    header = read_rinex_header(
        path
    )

    stream = read_rinex3_stream(
        path,
        systems=system,
    )

    return build_rinex_combinations_from_stream(
        path,
        satellite,
        stream=stream,
        header=header,
    )

def build_rinex_temporal_arcs(
    combination: B111RinexCombination,
    *,
    gaparc: float,
    min_epochs: int = 1,
) -> B111RinexArcSet:
    """
    Apply the existing B111 GAPARC temporal segmentation to one
    real-RINEX observable combination.

    GAPARC must be supplied explicitly.  No scientific default is
    introduced here.
    """

    from .arcs import (
        build_temporal_arcs,
        arc_labels,
    )

    arcs = build_temporal_arcs(
        combination.times,
        combination.phase1_cycles,
        combination.phase2_cycles,
        combination.code1_m,
        combination.code2_m,
        gaparc=gaparc,
        min_epochs=min_epochs,
    )

    labels = arc_labels(
        len(combination.times),
        arcs,
    )

    return B111RinexArcSet(
        combination=combination,
        arcs=arcs,
        labels=labels,
    )


@dataclass
class B111RinexUpdarcResult:
    """
    Real-RINEX input prepared and processed by the B111 UPDARC port.

    lintim_hours follows the Bernese RNXSMT/UPDARC convention:
    elapsed hours relative to the first RINEX epoch.
    """

    combination: B111RinexCombination

    # Indices of the original RINEX epochs retained in the
    # Bernese/LINCMB-like compact observation series.
    source_indices: np.ndarray

    lintim_hours: np.ndarray
    updarc_result: object


def build_rinex_updarc(
    combination: B111RinexCombination,
    *,
    gaparc: float,
    minobs: int,
) -> B111RinexUpdarcResult:
    """
    Feed one real-RINEX MW combination into the B111 UPDARC port.

    The input is represented as a compact observation series,
    corresponding to the role of LINCMB before UPDARC in RNXSMT.

    Epochs for which the selected MW/L4/code combination does not
    exist are not observations and therefore do not enter NOBS.

    source_indices preserves the mapping back to the original
    RINEX epoch grid.

    Parameters are intentionally explicit:
      gaparc : hours
      minobs : Bernese MINOBS

    No RSMTIN defaults are synthesized.
    """

    from .bernese_core import updarc

    full_times = np.asarray(
        combination.times,
        dtype="datetime64[ns]",
    )

    full_mw = np.asarray(
        combination.l5_cycles,
        dtype=np.float64,
    )

    valid = np.asarray(
        combination.valid,
        dtype=bool,
    )

    if not (
        len(full_times)
        == len(full_mw)
        == len(valid)
    ):
        raise ValueError(
            "RINEX time, L5 and validity lengths differ."
        )

    source_indices = np.flatnonzero(
        valid
    ).astype(np.int64)

    times = full_times[source_indices]

    linobs = full_mw[
        source_indices
    ].copy()

    if times.size == 0:

        lintim_hours = np.empty(
            0,
            dtype=np.float64,
        )

    else:

        # Preserve the original RINEX time origin. Only the
        # observation series is compacted; absolute elapsed
        # times and therefore physical gaps are preserved.
        lintim_hours = (
            (
                times - full_times[0]
            )
            / np.timedelta64(1, "h")
        ).astype(np.float64)

    # Every entry is now an actual observation of this
    # satellite/combination. Bad/outlier flags are subsequently
    # generated by the Bernese-derived QC routines.
    flags = np.zeros(
        len(linobs),
        dtype=np.int64,
    )

    result = updarc(
        lintim_hours,
        linobs,
        flags,
        gaparc=gaparc,
        minobs=minobs,
    )

    return B111RinexUpdarcResult(
        combination=combination,
        source_indices=source_indices,
        lintim_hours=lintim_hours,
        updarc_result=result,
    )


@dataclass
class B111RinexQcResult:
    """Complete B111 RNXSMT-style QC result for one RINEX combination."""

    combination: B111RinexCombination

    # Mapping from compact Bernese observation indices back to
    # the original RINEX epoch grid.
    source_indices: np.ndarray

    lintim_hours: np.ndarray

    initial_updarc: object
    chkwub_result: object
    chkl4p_result: object
    final_updarc: object

    # Phase-to-code leveling after the final B111 UPDARC.
    leveling_result: object


def run_b111_rinex_qc(
    combination: B111RinexCombination,
    *,
    gaparc: float,
    minobs: int,
    rmsl5: float,
    slpmin: float,
    slpout: float,
    rmsl4: float,
    gapl4: float,
    minl4: int,
    ifxslp: int,
    offset_func,
) -> B111RinexQcResult:
    """
    Execute the implemented RNXSMT core on one real RINEX combination.

    Processing follows the RNXSMT arc structure:

        L5/MW
          -> UPDARC
          -> for each UPDARC arc:
                 CHKWUB
                 CHKL4P
          -> UPDARC
          -> phase/code leveling on final arcs

    CHKWUB and CHKL4P never operate across an UPDARC arc boundary.

    All scientific thresholds are explicit.
    No RSMTIN or OFFSET defaults are synthesized.
    """

    from .bernese_core import updarc
    from .screening import (
        chkwub,
        ChkwubResult,
    )
    from .slips import (
        chkl4p,
        Chkl4pResult,
    )
    from .combinations import FrequencyPair
    from .b111 import bernese_frequency_system
    from .leveling import level_b111_geometry_free

    # ----------------------------------------------------------
    # Initial RINEX -> UPDARC
    # ----------------------------------------------------------

    first = build_rinex_updarc(
        combination,
        gaparc=gaparc,
        minobs=minobs,
    )

    source_indices = np.asarray(
        first.source_indices,
        dtype=np.int64,
    )

    lintim = first.lintim_hours
    up1 = first.updarc_result

    pair = FrequencyPair(
        f1=float(combination.f1_hz),
        f2=float(combination.f2_hz),
    )

    freq = bernese_frequency_system(pair)

    n = len(lintim)

    # ----------------------------------------------------------
    # Global output buffers.
    #
    # CHKWUB / CHKL4P themselves receive only one UPDARC arc at
    # a time. Results are copied back to absolute RINEX indices.
    # ----------------------------------------------------------

    mw_observations = np.asarray(
        up1.observations,
        dtype=np.float64,
    ).copy()

    mw_flags = np.asarray(
        up1.flags,
        dtype=np.int64,
    ).copy()

    l4_observations = np.asarray(
        combination.l4_m,
        dtype=np.float64,
    )[source_indices].copy()

    l4_flags = mw_flags.copy()

    # CHKWUB and CHKL4P have different slip semantics.
    #
    # CHKWUB indices/cycles describe the MW-detected candidates.
    # CHKL4P may relocate those candidates while resolving the
    # L1/L2 cycle-slip components. Keep both result streams
    # independent.
    global_mw_slip_indices = []
    global_mw_slip_cycles = []

    global_slip_indices = []

    global_cycl1 = []
    global_cycl2 = []
    global_cycl3 = []
    global_cycl4 = []
    global_cycl5 = []

    total_mw_nbad = 0
    total_l4_nbad = 0

    # CHKWUB ARCOFF / ARCRMS are intrinsically arc-specific.
    # Preserve one value per initial UPDARC arc.
    mw_arcoff = []
    mw_arcrms = []

    # ----------------------------------------------------------
    # RNXSMT-style processing: ONE ARC AT A TIME
    # ----------------------------------------------------------

    for arc in up1.arcs:

        a = int(arc.start)
        b = int(arc.end) + 1

        if a < 0 or b > n or b <= a:
            raise ValueError(
                f"Invalid UPDARC bounds: {a}:{b}"
            )

        # ------------------------------------------------------
        # CHKWUB on this arc only.
        # Returned slip indices are LOCAL to this slice.
        # ------------------------------------------------------

        mw_arc = chkwub(
            lintim[a:b],
            mw_observations[a:b],
            mw_flags[a:b],
            rmsmax=rmsl5,
            minobs=minobs,
            slpmin=slpmin,
            slpout=slpout,
            wlgt5=freq.wlgt5,
            offset_func=offset_func,
        )

        mw_observations[a:b] = mw_arc.observations
        mw_flags[a:b] = mw_arc.flags

        total_mw_nbad += int(mw_arc.nbad)

        mw_arcoff.append(float(mw_arc.arcoff))
        mw_arcrms.append(float(mw_arc.arcrms))

        local_slips = [
            int(i)
            for i in mw_arc.slip_indices
        ]

        # Preserve CHKWUB candidate locations BEFORE CHKL4P.
        for j, local_index in enumerate(local_slips):

            absolute_index = (
                a
                + int(local_index)
            )

            if not (
                arc.start
                <= absolute_index
                <= arc.end
            ):
                raise RuntimeError(
                    "CHKWUB returned a slip outside "
                    "its UPDARC arc"
                )

            global_mw_slip_indices.append(
                absolute_index
            )

            global_mw_slip_cycles.append(
                float(mw_arc.slip_cycles[j])
            )

        # ------------------------------------------------------
        # CHKL4P on exactly the SAME arc.
        #
        # local_slips remain local here because CHKL4P receives
        # l4_observations[a:b].
        # ------------------------------------------------------

        l4_arc = chkl4p(
            lintim[a:b],
            l4_observations[a:b],
            mw_arc.flags,
            local_slips,
            list(mw_arc.slip_cycles),
            rmsl4=rmsl4,
            gapl4=gapl4,
            minl4=minl4,
            ifxslp=ifxslp,
            # RNXSMT sets IPOLY=2 before CHKL4P.
            ipoly=2,
            wlgt1=freq.wlgt1,
            wlgt2=freq.wlgt2,
            wlgt3=freq.wlgt3,
            wlgt4=freq.wlgt4,
            faclin31=freq.faclin31,
            faclin32=freq.faclin32,
        )

        l4_observations[a:b] = l4_arc.observations
        l4_flags[a:b] = l4_arc.flags

        total_l4_nbad += int(l4_arc.nbad)

        # CHKL4P may move the slip epoch (e.g. IFIRST), therefore
        # use ITS returned indices, not the original CHKWUB ones.
        for j, local_index in enumerate(
            l4_arc.slip_indices
        ):
            absolute_index = (
                a
                + int(local_index)
            )

            if not (
                arc.start
                <= absolute_index
                <= arc.end
            ):
                raise RuntimeError(
                    "CHKL4P returned a slip outside "
                    "its UPDARC arc"
                )

            global_slip_indices.append(
                absolute_index
            )

            global_cycl1.append(
                float(l4_arc.cycl1[j])
            )
            global_cycl2.append(
                float(l4_arc.cycl2[j])
            )
            global_cycl3.append(
                float(l4_arc.cycl3[j])
            )
            global_cycl4.append(
                float(l4_arc.cycl4[j])
            )
            global_cycl5.append(
                float(l4_arc.cycl5[j])
            )

    # ----------------------------------------------------------
    # Aggregate per-arc results into the existing result types.
    #
    # ARCOFF/ARCRMS are no longer scalar scientifically because
    # CHKWUB was executed separately for each arc. The aggregate
    # object uses NaN for those scalar legacy fields. Per-arc
    # values are attached explicitly below.
    # ----------------------------------------------------------

    mw = ChkwubResult(
        observations=mw_observations,
        flags=mw_flags,
        nbad=total_mw_nbad,
        arcoff=float("nan"),
        arcrms=float("nan"),
        slip_indices=global_mw_slip_indices.copy(),
        slip_cycles=global_mw_slip_cycles.copy(),
    )

    # Preserve arc-specific CHKWUB diagnostics without changing
    # the frozen ChkwubResult dataclass/API.
    mw.arc_arcoff = mw_arcoff
    mw.arc_arcrms = mw_arcrms

    l4_result = Chkl4pResult(
        observations=l4_observations,
        flags=l4_flags,
        nbad=total_l4_nbad,
        slip_indices=global_slip_indices.copy(),
        cycl1=global_cycl1,
        cycl2=global_cycl2,
        cycl3=global_cycl3,
        cycl4=global_cycl4,
        cycl5=global_cycl5,
    )

    # ----------------------------------------------------------
    # Preserve all five recovered SLPCYC components for UPDARC #2.
    # ----------------------------------------------------------

    if global_slip_indices:

        slip_cycles = np.column_stack(
            [
                global_cycl1,
                global_cycl2,
                global_cycl3,
                global_cycl4,
                global_cycl5,
            ]
        ).astype(np.float64)

    else:

        slip_cycles = np.empty(
            (0, 5),
            dtype=np.float64,
        )

    previous_arcs = []

    slip_row = {
        int(index): row
        for row, index
        in enumerate(global_slip_indices)
    }

    for arc in up1.arcs:

        indices = [
            int(i)
            for i in global_slip_indices
            if arc.start <= i <= arc.end
        ]

        if indices:

            rows = [
                slip_row[i]
                for i in indices
            ]

            cycles = slip_cycles[
                rows,
                :
            ].copy()

        else:

            cycles = np.empty(
                (0, 5),
                dtype=np.float64,
            )

        previous_arcs.append(
            type(arc)(
                start=arc.start,
                end=arc.end,
                nobs=arc.nobs,
                nbad=arc.nbad,
                slip_indices=indices,
                slip_cycles=cycles,
            )
        )

    # ----------------------------------------------------------
    # Second UPDARC.
    # ----------------------------------------------------------

    up2 = updarc(
        lintim,
        l4_result.observations,
        l4_result.flags,
        gaparc=gaparc,
        minobs=minobs,
        previous_arcs=previous_arcs,
    )

    # ----------------------------------------------------------
    # Final phase-to-code leveling, independently per final arc.
    # ----------------------------------------------------------

    leveling = level_b111_geometry_free(
        corrected_l4_m=np.asarray(
            up2.observations,
            dtype=np.float64,
        ),
        code_gf_m=np.asarray(
            combination.code_gf_m,
            dtype=np.float64,
        )[source_indices],
        flags=np.asarray(
            up2.flags,
            dtype=np.int64,
        ),
        arcs=up2.arcs,
    )

    return B111RinexQcResult(
        combination=combination,
        source_indices=source_indices.copy(),
        lintim_hours=lintim,
        initial_updarc=up1,
        chkwub_result=mw,
        chkl4p_result=l4_result,
        final_updarc=up2,
        leveling_result=leveling,
    )

