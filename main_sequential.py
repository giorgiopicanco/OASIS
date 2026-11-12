"""Editable pyOASIS sequential processing entry point."""

from pathlib import Path
import sys

from pyOASIS.api import OASISConfig, process
from pyOASIS.cli import main as cli_main

ROOT = Path(__file__).resolve().parent


def main():
    # Preserve CLI commands: process, plot-hdf5, inspect-hdf5.
    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])

    rinex_files = (
        ROOT.parent / (
            "INPUT/RINEX_TESTS/EARTHSCOPE/"
            "CHPI00BRA_R_20251200000_01D_30S_MO.rnx"
        ),
        ROOT / "INPUT/RINEX/boav0491.23o",
    )

    # Validate both input paths before processing.
    for rinex_file in rinex_files:
        if not rinex_file.is_file():
            raise FileNotFoundError(f"RINEX file not found: {rinex_file}")

    results = []
    for index, rinex_file in enumerate(rinex_files, start=1):
        print(f"\n[{index}/{len(rinex_files)}] Processing {rinex_file.name}", flush=True)
        config = OASISConfig(
            # INPUT AND OUTPUT
            rinex=rinex_file,
            medusa_root=ROOT / "references/MEDUSA",
            orbit_dir=ROOT.parent / "INPUT/ORBITS",
            output_dir=ROOT / "OUTPUT",
            offline=False,

            # GNSS CONSTELLATIONS: G=GPS, R=GLONASS, E=Galileo, C=BeiDou
            systems="GREC",

            # SCIENTIFIC PRODUCTS
            products=("ALL",),

            # DATA EXPORT
            save_hdf5=True,
            overwrite_output=False,

            # FIGURES
            figures=True,
            figure_products=(
                "LGF", "ROTI", "DELTATEC", "SIDX",
                "STEC", "VTEC", "dTEC", "IPP",
            ),
            figure_dpi=300,
            plot_min_elevation=20.0,
            product_figure_color_mode="blue_red",  # Or "constellation".

            # SINGLE ARITHMETIC MEAN: VTEC, DELTATEC and dTEC only
            mean_line=True,
            mean_min_satellites=4,

            # OPTIONAL RNX_QC FIGURES
            qc_figures=False,
            qc_zoom=True,
            qc_elevation=False,
        )
        results.append(process(config))

    return results


if __name__ == "__main__":
    main()
