import os
import subprocess
from pathlib import Path
import pyOASIS
import sys

# Define the arguments directly
sta = "BOAV"  # Converted to lowercase manually as per "${sta,,}"
doy = "049"  # Day of the year
year = "2023"  # Year

# Directory input for rinex and MGEX orbits
input_rinex = "/home/debian-giorgio/pyOASIS/pyOASIS/INPUT/RINEX"
input_orbits = "/home/debian-giorgio/pyOASIS/pyOASIS/INPUT/ORBITS"

output_base_dir = "/home/debian-giorgio/pyOASIS/pyOASIS/OUTPUT"
sta_output = Path(output_base_dir) / year / doy / sta  # Construct the station directory path
orbit_output = Path(output_base_dir) / year / doy / "ORBITS"  # Construct the station directory path

# Create the output directory (and parent directories if needed)
Path(sta_output).mkdir(parents=True, exist_ok=True)
Path(orbit_output).mkdir(parents=True, exist_ok=True)

# Interpolate satellite orbits using SP3 files for the given day of year and year
pyOASIS.SP3intp(year,doy,input_orbits,orbit_output)

# Convert raw RINEX observation files (.yyo) to internal GNSS-clean format, performing
# initial detection of cycle slips, outliers, and identifying data gaps (arcs). (Output: .RNX1 files).
pyOASIS.RNXclean(sta,doy,year,input_rinex,orbit_output,sta_output)

# Screen GNSS observations by eliminating corrupt arcs and performing a refined second-pass
# detection of cycle slips and outliers (mini-arcs). (Output: .RNX2 files).
pyOASIS.RNXScreening(sta_output)

# Apply geometry-free leveling to remove satellite and receiver biases, performing a third-pass
# detection of outliers and cycle slips. (Output: .RNX3 files).
pyOASIS.RNXlevelling(sta,sta_output,show_plot=True)

# Compute the Rate of TEC Index (ROTI) using leveled geometry-free data from .RNX3 files.
pyOASIS.ROTIcalc(sta, doy, year, sta_output, sta_output,show_plot=True)

# Compute the Delta TEC index using leveled geometry-free data from .RNX3 files.
pyOASIS.DTECcalc(sta, doy, year, sta_output, sta_output, show_plot=True)

# Compute the SIDX index using leveled geometry-free data from .RNX3 files.
pyOASIS.SIDXcalc(sta, doy, year, sta_output, sta_output, show_plot=True)

#%%
