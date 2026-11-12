"""Editable Stage 24 example. Run from the OASIS_FINAL_WORKSPACE root.

    python -m examples.process_day

Adjust paths and switches below for other stations and dates.
"""
from pathlib import Path
from pyOASIS.api import OASISConfig, process


def main():
    root = Path(__file__).resolve().parents[1]
    config = OASISConfig(
        rinex=root / 'INPUT/RINEX/CHPI00BRA_R_20251200000_01D_30S_MO.rnx',
        medusa_root=root / 'references/MEDUSA',
        orbit_dir=root / 'INPUT/ORBITS',
        output_dir=root / 'OUTPUT',
        systems='GREC',
        products=('LGF', 'ROTI', 'DELTATEC', 'SIDX', 'STEC', 'VTEC', 'dTEC'),
        save_hdf5=True,
        figures=True,
        overview_figures=True,
        qc_figures=False,
        figure_products=('LGF', 'ROTI', 'DELTATEC', 'SIDX', 'STEC', 'VTEC', 'dTEC', 'IPP'),
        qc_elevation=False,
        qc_zoom=True,
        max_qc_figures=None,  # None: all satellites; 0: no QC panels.
        figure_dpi=300,
        plot_min_elevation=20.0,
        mean_line=True,
        mean_min_satellites=4,
        mean_method="mean",  # or "median"
        product_figure_color_mode="blue_red",  # or "constellation"
        mean_smooth_minutes=0.,
        offline=True,
        overwrite_output=False,
        require_complete_geometry=False,
    )
    result = process(config)
    print('Run status:', result['run_status'])
    print('Products:', result['products'])
    print('HDF5:', result.get('hdf5_file', 'Disabled'))


if __name__ == '__main__':
    main()
