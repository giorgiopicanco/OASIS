"""pyOASIS - RBMC RINEX 2 Batch Processing.
Date: 2025-11-12 (DOY 316).
"""

import time
import traceback
from pathlib import Path
from pyOASIS.api import OASISConfig, process

# DIRECTORIES
ROOT = Path(__file__).resolve().parent
OASIS_ROOT = ROOT.parent
YEAR = 2025
DOY = "316"
DATE = "2025-11-12"
RINEX_DIR = OASIS_ROOT / "RBMC" / str(YEAR) / DOY / "RINEX2"
MEDUSA_ROOT = ROOT / "references" / "MEDUSA"
ORBIT_DIR = OASIS_ROOT / "INPUT" / "ORBITS"
GLONASS_CHANNELS = ORBIT_DIR / f"GLONASS_CHANNELS_{YEAR}_{DOY}.json"
OUTPUT_DIR = ROOT / "OUTPUT" / "RBMC" / "RINEX2"

# PROCESSING CONFIGURATION
SYSTEMS = "GREC"
PRODUCTS = ("ALL",)
SAVE_HDF5 = True
OVERWRITE = False
OFFLINE = False

# SCIENTIFIC FIGURES
FIGURES = True
FIGURE_PRODUCTS = (
    "LGF", "ROTI", "DELTATEC", "SIDX", "STEC", "VTEC", "dTEC", "IPP",
)
FIGURE_DPI = 300
MIN_ELEVATION = 20.0
# Options: "constellation" or "blue_red"
COLOR_MODE = "constellation"
# Station mean line (VTEC, DELTATEC and dTEC only)
MEAN_LINE = False
MEAN_MIN_SATELLITES = 4
# RNX_QC figures only; internal QC remains enabled.
QC_FIGURES = False
QC_ZOOM = False
QC_ELEVATION = False


def process_station(rinex):
    config = OASISConfig(
        rinex=rinex,
        medusa_root=MEDUSA_ROOT,
        orbit_dir=ORBIT_DIR,
        glonass_channels=GLONASS_CHANNELS,
        offline=OFFLINE,
        output_dir=OUTPUT_DIR,
        save_hdf5=SAVE_HDF5,
        overwrite_output=OVERWRITE,
        systems=SYSTEMS,
        products=PRODUCTS,
        figures=FIGURES,
        figure_products=FIGURE_PRODUCTS,
        figure_dpi=FIGURE_DPI,
        plot_min_elevation=MIN_ELEVATION,
        product_figure_color_mode=COLOR_MODE,
        mean_line=MEAN_LINE,
        mean_min_satellites=MEAN_MIN_SATELLITES,
        qc_figures=QC_FIGURES,
        qc_zoom=QC_ZOOM,
        qc_elevation=QC_ELEVATION,
    )
    return process(config)


def main():
    if not RINEX_DIR.is_dir():
        raise FileNotFoundError(RINEX_DIR)
    if not MEDUSA_ROOT.is_dir():
        raise FileNotFoundError(MEDUSA_ROOT)
    if not GLONASS_CHANNELS.is_file():
        raise FileNotFoundError(GLONASS_CHANNELS)
    if COLOR_MODE not in ("constellation", "blue_red"):
        raise ValueError(f"Invalid color mode: {COLOR_MODE}")
    if set(p.upper() for p in PRODUCTS) & {"ALL", "TEC", "STEC", "VTEC", "DTEC"}:
        try:
            import scipy.sparse
        except ImportError as error:
            raise RuntimeError("SciPy is required. Run: python -m pip install scipy") from error
    ORBIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rinex_files = sorted(
        path for path in RINEX_DIR.iterdir()
        if path.is_file() and path.name.lower().endswith(f".{YEAR % 100:02d}o")
    )
    total = len(rinex_files)
    if total == 0:
        raise RuntimeError(f"No RINEX 2 files found in {RINEX_DIR}")
    successful = failed = skipped = 0
    failed_stations = []
    batch_start = time.perf_counter()
    print("=" * 80)
    print("pyOASIS - RBMC RINEX 2 BATCH")
    print(f"Date:          {DATE}")
    print(f"DOY:           {DOY}")
    print(f"Stations:      {total}")
    print(f"Systems:       {SYSTEMS}")
    print(f"Products:      {PRODUCTS}")
    print(f"Figures:       {FIGURES}")
    print(f"Color mode:    {COLOR_MODE}")
    print(f"Mean line:     {MEAN_LINE}")
    print(f"Figure DPI:    {FIGURE_DPI}")
    print(f"Min elevation: {MIN_ELEVATION}")
    print(f"QC figures:    {QC_FIGURES}")
    print(f"Offline:       {OFFLINE}")
    print(f"RINEX dir:     {RINEX_DIR}")
    print(f"Orbit cache:   {ORBIT_DIR}")
    print(f"GLONASS map:   {GLONASS_CHANNELS}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print("=" * 80)
    for index, rinex in enumerate(rinex_files, start=1):
        station = rinex.name[:4].upper()
        start = time.perf_counter()
        print(f"\n[{index:03d}/{total:03d}] STATION: {station} | {rinex.name}", flush=True)
        # Fast resume for standard daily RBMC filenames and output layout.
        expected = OUTPUT_DIR / station / str(YEAR) / DOY / f"{station}_{YEAR}_{DOY}_ION.h5"
        if (SAVE_HDF5 and not OVERWRITE and len(rinex.name) == 12
                and rinex.name[4:8] == DOY + "1" and expected.is_file()):
            skipped += 1
            print(f"[SKIPPED] {station}: HDF5 exists: {expected}")
            print(f"Progress: {index}/{total} | Success: {successful} | Skipped: {skipped} | Failed: {failed}", flush=True)
            continue
        try:
            result = process_station(rinex)
            successful += 1
            print(f"[{result.get('run_status', 'SUCCESS')}] {station} ({time.perf_counter() - start:.2f} s)")
        except FileExistsError as error:
            if not OVERWRITE and str(error).startswith("HDF5 exists:"):
                skipped += 1
                print(f"[SKIPPED] {station}: {error}")
            else:
                failed += 1
                failed_stations.append(station)
                print(f"[FAILED] {station}: {error}")
                traceback.print_exc()
        except Exception as error:
            failed += 1
            failed_stations.append(station)
            print(f"[FAILED] {station}: {error}")
            traceback.print_exc()
        print(f"Progress: {index}/{total} | Success: {successful} | Skipped: {skipped} | Failed: {failed}", flush=True)
    elapsed = time.perf_counter() - batch_start
    print("\n" + "=" * 80)
    print("FINAL BATCH SUMMARY")
    print(f"Total stations: {total}")
    print(f"Successful:     {successful}")
    print(f"Skipped:        {skipped}")
    print(f"Failed:         {failed}")
    print(f"Elapsed hours:  {elapsed / 3600:.2f}")
    print(f"Orbit cache:    {ORBIT_DIR}")
    if failed_stations:
        print("Failed stations: " + ", ".join(failed_stations))
    print("=" * 80)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
