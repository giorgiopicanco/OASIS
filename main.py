"""Editable pyOASIS daily processing entry point."""

from pathlib import Path
import sys

from pyOASIS.api import OASISConfig, process
from pyOASIS.cli import main as cli_main

ROOT = Path(__file__).resolve().parent


def main():

    # Preserve command-line interface.
    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])

    config = OASISConfig(

        # ==========================================
        # INPUT AND OUTPUT
        # ==========================================

        rinex=ROOT.parent / (
            "INPUT/RINEX_TESTS/EARTHSCOPE/"
            "CHPI00BRA_R_20251200000_01D_30S_MO.rnx"
        ),

        medusa_root=ROOT / "references/MEDUSA",
        orbit_dir=ROOT.parent / "INPUT/ORBITS",
        output_dir=ROOT / "OUTPUT",
        offline=True,

        # ==========================================
        # GNSS CONSTELLATIONS
        # ==========================================

        systems="GREC",

        # G: GPS
        # R: GLONASS
        # E: Galileo
        # C: BeiDou

        # ==========================================
        # SCIENTIFIC PRODUCTS
        # ==========================================

        products=("ALL",),

        # ==========================================
        # DATA EXPORT
        # ==========================================

        save_hdf5=True,
        overwrite_output=False,

        # ==========================================
        # PRODUCT FIGURES
        # ==========================================

        figures=True,

        figure_products=(
            "LGF",
            "ROTI",
            "DELTATEC",
            "SIDX",
            "STEC",
            "VTEC",
            "dTEC",
            "IPP",
        ),

        figure_dpi=300,
        plot_min_elevation=20.0,

        product_figure_color_mode="blue_red",

        # Color modes:
        # "constellation"
        # "blue_red"

        # ==========================================
        # STATION MEAN
        # VTEC, DELTATEC and dTEC only
        # ==========================================

        mean_line=True,
        mean_min_satellites=4,

        # ==========================================
        # OPTIONAL RNX_QC FIGURES
        # ==========================================

        qc_figures=False,
        qc_zoom=True,
        qc_elevation=False,

    )

    return process(config)


if __name__ == "__main__":
    main()