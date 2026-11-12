"""
EXPERIMENTAL MEDUSA/B111 — multi-satellite real-data runner.

IMPORTANT
---------
* Does not modify run_real_g08.py.
* RSMTIN parameters are unavailable.
* Bernese OFFSET implementation is unavailable.
* Uses TEST-ONLY mean/RMS OFFSET.
* Thresholds are experimental, not Bernese defaults.
* Plotting never connects independent UPDARC arcs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt

# MEDUSA: disable plot legends globally
# Scientific data/curves are unchanged; only legend rendering is disabled.
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Axes.legend = lambda self, *args, **kwargs: None
Figure.legend = lambda self, *args, **kwargs: None
import numpy as np
import pandas as pd

from tools.historical.b111.rinex_driver import (
    build_rinex_combinations_from_stream,
    run_b111_rinex_qc,
)

from tools.historical.b111.rinex_signals import (
    read_rinex_header,
)

from tools.historical.b111.rinex3_stream import (
    read_rinex3_stream,
)

from tools.historical.b111.screening import (
    bernese_offset,
)


def choose_combination(combinations, system):
    """
    MEDUSA/B111 deterministic multi-GNSS observable selection.

    Bernese scientific QC remains downstream:
        LINCMB -> UPDARC -> CHKWUB -> CHKL4P -> UPDARC

    This function modernizes only the RINEX signal-selection layer.

    IMPORTANT:
    - Never select a physical frequency pair merely because it has
      the largest number of valid epochs.
    - Physical frequency pairs have an explicit constellation-specific
      scientific priority.
    - Within one physical pair, tracking codes have explicit priorities.
    - The Bernese READGEOS2 0.6 availability principle is retained:
      a candidate tracking combination must contain more than 60% of
      the valid epochs of the best candidate for that same physical pair.
    """

    if not combinations:
        return None

    system = str(system).upper()

    # ----------------------------------------------------------
    # Ordered PHYSICAL frequency-pair policy.
    #
    # Digits are RINEX frequency identifiers.
    #
    # G:
    #   1/2 = GPS L1/L2
    #
    # R:
    #   1/2 = GLONASS G1/G2
    #
    # E:
    #   1/7 = Galileo E1/E5b
    #   This is also the historical Bernese DEFREQ default.
    #
    # C:
    #   1/5 = BDS-3 B1C/B2a
    #   2/7 = BDS-2 B1I/B2I
    #   remaining entries are explicit fallbacks.
    # ----------------------------------------------------------

    pair_policy = {
        "G": [
            ("1", "2"),
        ],

        "R": [
            ("1", "2"),
        ],

        "E": [
            ("1", "7"),  # E1 + E5b
            ("1", "5"),  # E1 + E5a
            ("1", "8"),  # E1 + E5 AltBOC
            ("1", "6"),  # E1 + E6
            ("5", "7"),  # E5a + E5b
        ],

        "C": [
            ("1", "5"),  # BDS-3: B1C + B2a
            ("2", "7"),  # BDS-2: B1I + B2I
            ("2", "6"),  # B1I + B3I
            ("1", "7"),  # B1C + B2b/B2I carrier
            ("1", "6"),  # B1C + B3I
            ("6", "7"),  # B3I + B2I/B2b
            ("5", "7"),  # B2a + B2b/B2I
            ("2", "5"),  # B1I + B2a
        ],
    }

    # ----------------------------------------------------------
    # Tracking-code priorities for the two carriers.
    #
    # These do NOT change the physical carrier frequency.
    # They only choose between signals sharing that carrier.
    # ----------------------------------------------------------

    def tracking_preferences(sys, pair):

        if sys == "G":
            return (
                ("P", "W", "C", "X", "L", "S", "Y", "M"),
                ("P", "W", "C", "D", "X", "L", "S", "Y", "M"),
            )

        if sys == "R":
            return (
                ("P", "C", "X"),
                ("P", "C", "X"),
            )

        if sys == "E":
            first = (
                "C", "X", "A", "B", "Z",
                "Q", "I",
            )

            if pair[1] in {"5", "7", "8"}:
                second = (
                    "Q", "X", "I",
                    "C", "A", "B", "Z",
                )
            else:
                second = (
                    "C", "X", "A", "B", "Z",
                    "Q", "I",
                )

            return first, second

        if sys == "C":

            # B1C + B2a
            if pair == ("1", "5"):
                return (
                    ("P", "X", "D"),
                    ("P", "X", "D"),
                )

            # B1I + B2I
            if pair == ("2", "7"):
                return (
                    ("I", "X", "Q"),
                    ("I", "X", "Q", "D", "P", "Z"),
                )

            # B1I + B3I
            if pair == ("2", "6"):
                return (
                    ("I", "X", "Q"),
                    ("I", "X", "Q"),
                )

            # B1C + B2b
            if pair == ("1", "7"):
                return (
                    ("P", "X", "D"),
                    ("D", "P", "Z", "X", "I", "Q"),
                )

            return (
                ("P", "I", "Q", "X", "D", "Z"),
                ("P", "I", "Q", "X", "D", "Z"),
            )

        return (
            tuple(),
            tuple(),
        )

    def frequency_digits(c):
        """
        Return RINEX frequency digits of phase1/phase2.
        Example:
            L1C/L7Q -> ('1','7')
        """
        if (
            len(c.phase1) < 2
            or len(c.phase2) < 2
        ):
            return ("", "")

        return (
            c.phase1[1],
            c.phase2[1],
        )

    def track_code(obs):
        if len(obs) >= 3:
            return obs[2].upper()
        return ""

    def rank_tracking(obs, preferences):
        code = track_code(obs)

        try:
            return preferences.index(code)
        except ValueError:
            return len(preferences) + 100

    policies = pair_policy.get(system)

    if not policies:
        return None

    # ----------------------------------------------------------
    # Evaluate physical pairs in SCIENTIFIC priority order.
    # We never compare valid-counts across different frequency
    # pairs to choose the constellation's primary combination.
    # ----------------------------------------------------------

    for wanted_pair in policies:

        pair_candidates = []

        for c in combinations:

            if frequency_digits(c) != wanted_pair:
                continue

            nvalid = int(
                np.count_nonzero(c.valid)
            )

            if nvalid <= 0:
                continue

            pair_candidates.append(
                (c, nvalid)
            )

        if not pair_candidates:
            continue

        max_valid = max(
            nvalid
            for _, nvalid in pair_candidates
        )

        # Bernese READGEOS2:
        #
        #     obsfac = 0.6
        #
        # Keep that principle here for tracking-code selection.
        eligible = [
            (c, nvalid)
            for c, nvalid in pair_candidates
            if nvalid > 0.6 * max_valid
        ]

        if not eligible:
            continue

        pref1, pref2 = tracking_preferences(
            system,
            wanted_pair,
        )

        def candidate_key(item):

            c, nvalid = item

            return (
                rank_tracking(
                    c.phase1,
                    pref1,
                ),
                rank_tracking(
                    c.code1,
                    pref1,
                ),
                rank_tracking(
                    c.phase2,
                    pref2,
                ),
                rank_tracking(
                    c.code2,
                    pref2,
                ),

                # Valid count is ONLY a final tie breaker inside
                # the already-selected physical frequency pair.
                -nvalid,

                c.phase1,
                c.phase2,
                c.code1,
                c.code2,
            )

        eligible.sort(
            key=candidate_key
        )

        return eligible[0][0]

    # No scientifically permitted pair exists.
    return None


def plot_arc(
    ax,
    times,
    values,
    arc,
    *,
    color,
    linestyle="-",
    linewidth=0.8,
    alpha=1.0,
    label=None,
):
    """
    Plot exactly one UPDARC arc.

    NaN and non-finite values remain absent.
    No line can cross from this arc to another arc.
    """

    a = int(arc.start)
    b = int(arc.end) + 1

    t = np.asarray(times[a:b])
    y = np.asarray(
        values[a:b],
        dtype=float,
    ).copy()

    good = np.isfinite(y)

    if not np.any(good):
        return

    y[~good] = np.nan

    ax.plot(
        t,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("rinex")

    parser.add_argument(
        "--systems",
        default="G",
        help=(
            "GNSS systems to process, e.g. G or GRE. "
            "Default: GPS only."
        ),
    )

    parser.add_argument(
        "--gaparc",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--minobs",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--rmsl5",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--slpmin",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--slpout",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--rmsl4",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--gapl4-seconds",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--minl4",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--ifxslp",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "data/historical/qc/b111/"
            "experimental/all_satellites"
        ),
    )

    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    print()
    print("=" * 70)
    print("EXPERIMENTAL MEDUSA/B111 — MULTI-SATELLITE")
    print("=" * 70)
    print("RSMTIN parameters unavailable.")
    print("Bernese OFFSET unavailable.")
    print("Using TEST-ONLY mean/RMS OFFSET.")
    print()

    # ----------------------------------------------------------
    # Discover satellites once.
    # ----------------------------------------------------------

    print("Reading RINEX header once...")

    header = read_rinex_header(
        args.rinex
    )

    requested_systems = set(
        args.systems
    )

    print(
        "Streaming mixed RINEX ONCE..."
    )

    stream = read_rinex3_stream(
        args.rinex,
        systems=args.systems,
    )

    satellites = sorted(
        satellite
        for satellite in stream.satellites
        if (
            satellite
            and satellite[0] in requested_systems
        )
    )

    print(
        "Systems:",
        " ".join(sorted(requested_systems)),
    )

    print(
        "Satellites:",
        len(satellites),
    )

    if not satellites:
        raise RuntimeError(
            "No requested satellites found."
        )

    print(
        f"Satellites ({len(satellites)}):",
        " ".join(satellites),
    )

    # ----------------------------------------------------------
    # Figures.
    #
    # One color per satellite.
    # Same satellite keeps the same color in both figures.
    # ----------------------------------------------------------

    fig_mw, ax_mw = plt.subplots(
        figsize=(17, 8)
    )

    fig_gf, ax_gf = plt.subplots(
        figsize=(17, 8)
    )

    # MEDUSA GF BEFORE/AFTER LEVELING FIGURE
    fig_gf_compare, (
        ax_gf_before,
        ax_gf_after,
    ) = plt.subplots(
        1,
        2,
        figsize=(20, 8),
        sharex=True,
    )

    ax_gf_before.set_title(
        "Geometry-Free Before Leveling"
    )

    ax_gf_after.set_title(
        "Geometry-Free After Leveling"
    )

    ax_gf_before.set_xlabel("Time")
    ax_gf_after.set_xlabel("Time")

    ax_gf_before.set_ylabel(
        "Geometry-Free Phase [m]"
    )

    ax_gf_after.set_ylabel(
        "Geometry-Free Phase [m]"
    )

    ax_gf_before.grid(
        True,
        alpha=0.25,
    )

    ax_gf_after.grid(
        True,
        alpha=0.25,
    )

    # Use Matplotlib's default color cycle.
    # We do not hard-code colors.
    color_cycle = (
        plt.rcParams[
            "axes.prop_cycle"
        ].by_key()["color"]
    )

    output_frames = []
    summaries = []

    # ----------------------------------------------------------
    # Process each satellite.
    # ----------------------------------------------------------

    for sat_number, satellite in enumerate(
        satellites,
        start=1,
    ):

        print()
        print(
            f"[{sat_number:02d}/{len(satellites):02d}] "
            f"{satellite}"
        )

        try:
            combinations = (
                build_rinex_combinations_from_stream(
                    args.rinex,
                    satellite,
                    stream=stream,
                    header=header,
                )
            )

            combo = choose_combination(
                combinations,
                satellite[0],
            )

            if combo is None:
                print("  SKIP: no usable dual-frequency combination")
                continue

            nvalid = int(
                np.count_nonzero(
                    combo.valid
                )
            )

            print(
                "  combination:",
                combo.phase1,
                combo.phase2,
                combo.code1,
                combo.code2,
            )

            print(
                "  valid epochs:",
                nvalid,
            )

            result = run_b111_rinex_qc(
                combo,

                gaparc=args.gaparc,
                minobs=args.minobs,

                rmsl5=args.rmsl5,
                slpmin=args.slpmin,
                slpout=args.slpout,

                rmsl4=args.rmsl4,

                # B111 LINTIM is hours.
                gapl4=(
                    args.gapl4_seconds
                    / 3600.0
                ),

                minl4=args.minl4,
                ifxslp=args.ifxslp,

                offset_func=(
                    bernese_offset
                ),
            )

        except Exception as exc:
            print(
                "  FAILED:",
                type(exc).__name__,
                str(exc),
            )
            continue

        up1 = result.initial_updarc
        mw = result.chkwub_result
        l4 = result.chkl4p_result
        up2 = result.final_updarc
        level = result.leveling_result

        # ------------------------------------------------------
        # Map compact Bernese observation series back onto the
        # original RINEX epoch grid.
        #
        # All B111 scientific processing remains compact.
        # Expansion happens ONLY here for output/plotting.
        # ------------------------------------------------------

        n = len(combo.times)

        source_indices = np.asarray(
            result.source_indices,
            dtype=np.int64,
        )

        ncompact = len(source_indices)

        if not (
            len(mw.observations)
            == len(mw.flags)
            == len(l4.observations)
            == len(l4.flags)
            == len(up2.observations)
            == len(up2.flags)
            == len(level.leveled_gf_m)
            == ncompact
        ):
            raise RuntimeError(
                "Compact B111 output lengths are inconsistent."
            )

        # ------------------------------------------------------
        # Epoch slip vectors on ORIGINAL RINEX grid.
        # ------------------------------------------------------

        slip = np.zeros(
            n,
            dtype=np.int8,
        )

        cycl1 = np.full(n, np.nan)
        cycl2 = np.full(n, np.nan)
        cycl3 = np.full(n, np.nan)
        cycl4 = np.full(n, np.nan)
        cycl5 = np.full(n, np.nan)

        for j, compact_index in enumerate(
            l4.slip_indices
        ):
            ci = int(compact_index)

            if not (0 <= ci < ncompact):
                raise RuntimeError(
                    "Compact slip index outside observation series."
                )

            i = int(source_indices[ci])

            slip[i] = 1

            cycl1[i] = l4.cycl1[j]
            cycl2[i] = l4.cycl2[j]
            cycl3[i] = l4.cycl3[j]
            cycl4[i] = l4.cycl4[j]
            cycl5[i] = l4.cycl5[j]

        # ------------------------------------------------------
        # Full-grid output arrays.
        #
        # NaN means there was no selected combination at that
        # original RINEX epoch.
        # ------------------------------------------------------

        raw_mw = np.asarray(
            combo.l5_cycles,
            dtype=float,
        ).copy()

        raw_l4 = np.asarray(
            combo.l4_m,
            dtype=float,
        ).copy()

        corrected_mw = np.full(
            n,
            np.nan,
            dtype=float,
        )

        corrected_l4 = np.full(
            n,
            np.nan,
            dtype=float,
        )

        leveled = np.full(
            n,
            np.nan,
            dtype=float,
        )

        corrected_mw[source_indices] = np.asarray(
            mw.observations,
            dtype=float,
        )

        corrected_l4[source_indices] = np.asarray(
            up2.observations,
            dtype=float,
        )

        leveled[source_indices] = np.asarray(
            level.leveled_gf_m,
            dtype=float,
        )

        invalid = ~np.asarray(
            combo.valid,
            dtype=bool,
        )

        raw_mw[invalid] = np.nan
        raw_l4[invalid] = np.nan

        # ------------------------------------------------------
        # Expand compact QC flags onto original RINEX grid.
        # ------------------------------------------------------

        mw_bad_compact = (
            np.asarray(
                mw.flags,
                dtype=np.int64,
            )
            & 1
        ) != 0

        final_bad_compact = (
            np.asarray(
                up2.flags,
                dtype=np.int64,
            )
            & 1
        ) != 0

        corrected_mw[
            source_indices[mw_bad_compact]
        ] = np.nan

        corrected_l4[
            source_indices[final_bad_compact]
        ] = np.nan

        leveled[
            source_indices[final_bad_compact]
        ] = np.nan

        # ------------------------------------------------------
        # Full-grid flag arrays for CSV output.
        #
        # Missing epochs retain bit 0 because they are absent
        # from the selected observation combination, but these
        # flags are OUTPUT metadata only. They never re-enter
        # the compact scientific processing.
        # ------------------------------------------------------

        flags_initial_full = np.ones(
            n,
            dtype=np.int64,
        )

        flags_chkwub_full = np.ones(
            n,
            dtype=np.int64,
        )

        flags_final_full = np.ones(
            n,
            dtype=np.int64,
        )

        flags_initial_full[source_indices] = np.asarray(
            up1.flags,
            dtype=np.int64,
        )

        flags_chkwub_full[source_indices] = np.asarray(
            mw.flags,
            dtype=np.int64,
        )

        flags_final_full[source_indices] = np.asarray(
            up2.flags,
            dtype=np.int64,
        )

        # ------------------------------------------------------
        # Convert compact UPDARC arc bounds to original RINEX
        # coordinates for OUTPUT/PLOTTING ONLY.
        #
        # Scientific processing remains entirely in compact
        # Bernese observation coordinates.
        # ------------------------------------------------------

        def map_arcs_to_rinex(compact_arcs):

            mapped = []

            for arc in compact_arcs:

                ca = int(arc.start)
                cb = int(arc.end)

                if not (
                    0 <= ca <= cb < ncompact
                ):
                    raise RuntimeError(
                        "Invalid compact UPDARC arc."
                    )

                mapped.append(
                    type(arc)(
                        start=int(source_indices[ca]),
                        end=int(source_indices[cb]),
                        nobs=arc.nobs,
                        nbad=arc.nbad,
                        slip_indices=[
                            int(source_indices[int(i)])
                            for i in arc.slip_indices
                            if 0 <= int(i) < ncompact
                        ],
                        slip_cycles=(
                            None
                            if arc.slip_cycles is None
                            else np.asarray(
                                arc.slip_cycles,
                                dtype=np.float64,
                            ).copy()
                        ),
                    )
                )

            return mapped


        plot_arcs_initial = map_arcs_to_rinex(
            up1.arcs
        )

        plot_arcs_final = map_arcs_to_rinex(
            up2.arcs
        )

        # ------------------------------------------------------
        # Satellite color
        # ------------------------------------------------------

        # Fixed MEDUSA constellation colors
        color = {
            "G": "red",      # GPS
            "R": "blue",     # GLONASS
            "E": "lime",     # Galileo
            "C": "purple",   # BeiDou
        }.get(
            satellite[0],
            "black",
        )

        # MEDUSA GF BEFORE/AFTER LEVELING PLOT
        #
        # LEFT:
        #   corrected L4/GF after Bernese-style QC and
        #   cycle-slip correction, BEFORE leveling.
        #
        # RIGHT:
        #   same geometry-free phase AFTER phase-to-code
        #   leveling.
        #
        # NaNs already present in these full-grid arrays ensure
        # that observation gaps are never connected.

        ax_gf_before.plot(
            combo.times,
            corrected_l4,
            color=color,
            linewidth=0.70,
            alpha=0.85,
        )

        ax_gf_after.plot(
            combo.times,
            leveled,
            color=color,
            linewidth=0.70,
            alpha=0.85,
        )

        # ------------------------------------------------------
        # MW figure.
        #
        # Raw    = thin / transparent
        # CHKWUB = solid
        #
        # Plot every arc separately -> NO gap connections.
        # ------------------------------------------------------

        raw_label_used = False
        corrected_label_used = False

        for arc in plot_arcs_initial:

            plot_arc(
                ax_mw,
                combo.times,
                raw_mw,
                arc,
                color=color,
                linestyle=":",
                linewidth=0.65,
                alpha=0.35,
                label=(
                    f"{satellite} raw"
                    if not raw_label_used
                    else None
                ),
            )

            raw_label_used = True

            plot_arc(
                ax_mw,
                combo.times,
                corrected_mw,
                arc,
                color=color,
                linestyle="-",
                linewidth=0.9,
                alpha=0.95,
                label=(
                    f"{satellite} CHKWUB"
                    if not corrected_label_used
                    else None
                ),
            )

            corrected_label_used = True

        # ------------------------------------------------------
        # GF figure.
        #
        # Raw       = dotted
        # Corrected = dashed
        # Leveled   = solid
        # ------------------------------------------------------

        raw_label_used = False
        corrected_label_used = False
        leveled_label_used = False

        for arc in plot_arcs_final:

            plot_arc(
                ax_gf,
                combo.times,
                raw_l4,
                arc,
                color=color,
                linestyle=":",
                linewidth=0.6,
                alpha=0.30,
                label=(
                    f"{satellite} raw"
                    if not raw_label_used
                    else None
                ),
            )

            raw_label_used = True

            plot_arc(
                ax_gf,
                combo.times,
                corrected_l4,
                arc,
                color=color,
                linestyle="--",
                linewidth=0.7,
                alpha=0.55,
                label=(
                    f"{satellite} corrected"
                    if not corrected_label_used
                    else None
                ),
            )

            corrected_label_used = True

            plot_arc(
                ax_gf,
                combo.times,
                leveled,
                arc,
                color=color,
                linestyle="-",
                linewidth=1.0,
                alpha=0.95,
                label=(
                    f"{satellite} leveled"
                    if not leveled_label_used
                    else None
                ),
            )

            leveled_label_used = True

        # ------------------------------------------------------
        # Expand remaining compact scientific products onto the
        # original RINEX grid for CSV/output ONLY.
        # ------------------------------------------------------

        l4_corrected_full = np.full(
            n, np.nan, dtype=float
        )

        mw_screened_full = np.full(
            n, np.nan, dtype=float
        )

        arc_id_full = np.full(
            n, np.nan, dtype=float
        )

        arc_bias_full = np.full(
            n, np.nan, dtype=float
        )

        leveled_full = np.full(
            n, np.nan, dtype=float
        )

        residual_full = np.full(
            n, np.nan, dtype=float
        )

        l4_corrected_full[source_indices] = np.asarray(
            l4.observations,
            dtype=float,
        )

        mw_screened_full[source_indices] = np.asarray(
            mw.observations,
            dtype=float,
        )

        arc_id_full[source_indices] = np.asarray(
            level.arc_id,
            dtype=float,
        )

        arc_bias_full[source_indices] = np.asarray(
            level.arc_bias_m,
            dtype=float,
        )

        leveled_full[source_indices] = np.asarray(
            level.leveled_gf_m,
            dtype=float,
        )

        residual_full[source_indices] = np.asarray(
            level.residual_m,
            dtype=float,
        )

        # ------------------------------------------------------
        # CSV dataframe
        # ------------------------------------------------------

        df = pd.DataFrame(
            {
                "time": combo.times,

                "satellite":
                    combo.satellite,

                "phase1":
                    combo.phase1,

                "phase2":
                    combo.phase2,

                "code1":
                    combo.code1,

                "code2":
                    combo.code2,

                "valid":
                    combo.valid,

                "l4_raw_m":
                    combo.l4_m,

                "l4_corrected_m":
                    l4_corrected_full,

                "code_gf_m":
                    combo.code_gf_m,

                "arc_id":
                    arc_id_full,

                "arc_bias_m":
                    arc_bias_full,

                "leveled_gf_m":
                    leveled_full,

                "leveled_minus_code_gf_m":
                    residual_full,

                "mw_raw_cycles":
                    combo.l5_cycles,

                "mw_screened_cycles":
                    mw_screened_full,

                "flags_initial":
                    flags_initial_full,

                "flags_chkwub":
                    flags_chkwub_full,

                "flags_final":
                    flags_final_full,

                "slip":
                    slip,

                "cycl1":
                    cycl1,

                "cycl2":
                    cycl2,

                "cycl3":
                    cycl3,

                "cycl4":
                    cycl4,

                "cycl5":
                    cycl5,
            }
        )

        output_frames.append(df)

        residual_good = np.isfinite(
            level.residual_m
        )

        residual_rms = (
            float(
                np.sqrt(
                    np.mean(
                        np.asarray(
                            level.residual_m
                        )[residual_good] ** 2
                    )
                )
            )
            if residual_good.any()
            else np.nan
        )

        summaries.append(
            {
                "satellite":
                    satellite,

                "combination":
                    (
                        f"{combo.phase1}/"
                        f"{combo.phase2} + "
                        f"{combo.code1}/"
                        f"{combo.code2}"
                    ),

                "valid_epochs":
                    nvalid,

                "initial_arcs":
                    len(up1.arcs),

                "final_arcs":
                    len(up2.arcs),

                "slips":
                    len(
                        l4.slip_indices
                    ),

                "mw_nbad":
                    mw.nbad,

                "l4_nbad":
                    l4.nbad,

                "residual_rms_m":
                    residual_rms,
            }
        )

        print(
            "  arcs:",
            len(up1.arcs),
            "->",
            len(up2.arcs),
        )

        print(
            "  slips:",
            len(l4.slip_indices),
        )

    # ----------------------------------------------------------
    # Final figure formatting
    # ----------------------------------------------------------

    if not summaries:
        raise RuntimeError(
            "No satellite completed processing."
        )

    ax_mw.set_title(
        "B111 — Multi-satellite Melbourne-Wübbena / CHKWUB\n"
        "EXPERIMENTAL — mock OFFSET + experimental thresholds"
    )

    ax_mw.set_xlabel("UTC")
    ax_mw.set_ylabel("MW/L5 [cycles]")
    ax_mw.grid(
        True,
        alpha=0.25,
    )

    # With many PRNs the legend can become huge.
    ax_mw.legend(
        fontsize=7,
        ncol=4,
    )

    fig_mw.autofmt_xdate()
    fig_mw.tight_layout()

    ax_gf.set_title(
        "B111 — Multi-satellite geometry-free processing\n"
        "Raw / CHKL4P corrected / phase-code leveled"
    )

    ax_gf.set_xlabel("UTC")
    ax_gf.set_ylabel(
        "Geometry-free [m]"
    )

    ax_gf.grid(
        True,
        alpha=0.25,
    )

    ax_gf.legend(
        fontsize=7,
        ncol=4,
    )

    fig_gf.autofmt_xdate()
    # MEDUSA GF BEFORE/AFTER LEVELING SAVE
    fig_gf_compare.tight_layout()

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gf_compare_path = (
        output_dir
        / "b111_geometry_free_before_after_leveling.png"
    )

    fig_gf_compare.savefig(
        gf_compare_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        "GF before/after figure:",
        gf_compare_path,
    )

    fig_gf.tight_layout()

    # ----------------------------------------------------------
    # Summary before display
    # ----------------------------------------------------------

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Displaying figures BEFORE saving."
    )
    print(
        "Close BOTH Matplotlib windows "
        "when you finish inspecting them."
    )

    # ----------------------------------------------------------
    # DISPLAY FIRST
    # ----------------------------------------------------------

    plt.show()

    # ----------------------------------------------------------
    # SAVE ONLY AFTER USER CONFIRMATION
    # ----------------------------------------------------------

    answer = input(
        "\nSave CSVs and figures? [y/N]: "
    ).strip().lower()

    if answer not in {
        "y",
        "yes",
    }:
        print(
            "Nothing saved."
        )
        return

    outdir = Path(
        args.output
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_df = pd.concat(
        output_frames,
        ignore_index=True,
    )

    csv_path = (
        outdir
        / "b111_all_satellites_experimental.csv"
    )

    summary_path = (
        outdir
        / "b111_all_satellites_summary.csv"
    )

    mw_path = (
        outdir
        / "b111_all_satellites_mw.png"
    )

    gf_path = (
        outdir
        / "b111_all_satellites_gf.png"
    )

    all_df.to_csv(
        csv_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    fig_mw.savefig(
        mw_path,
        dpi=220,
        bbox_inches="tight",
    )

    fig_gf.savefig(
        gf_path,
        dpi=220,
        bbox_inches="tight",
    )

    print()
    print("Saved:")
    print(" ", csv_path)
    print(" ", summary_path)
    print(" ", mw_path)
    print(" ", gf_path)


if __name__ == "__main__":
    main()
