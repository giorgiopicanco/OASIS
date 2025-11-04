#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyOASIS Automated Processing Pipeline - Example Script
Author: Giorgio Picanço
Date: November 2025

Pipeline overview:
- Steps (1)–(3) are sequential and mandatory to generate the leveled .RNX3 files.
- Steps (4)–(7) are independent and can be executed in any order,
  as they all use the leveled .RNX3 data as input.

1) SP3 interpolation
2) RNXclean
3) RNXlevelling
4) ROTIcalc
5) DTECcalc
6) SIDXcalc
7) TECcalc
"""

import os
from pathlib import Path
import pyOASIS

# ===============================
# USER PARAMETERS
# ===============================
sta  = "BOAV"   # GNSS station code
doy  = "049"    # Day of Year (DOY)
year = "2023"   # Year (YYYY)

# ===============================
# DIRECTORY DEFINITIONS
# ===============================
base_dir        = Path(__file__).resolve().parent
rinex_dir       = base_dir / "INPUT"  / "RINEX"   # directory to place the RINEX observation files
orbit_input_dir = base_dir / "INPUT"  / "ORBITS"  # directory to place the orbit files
sta_output      = base_dir / "OUTPUT" / "RINEX"  / year / doy / sta  # directory for RINEX processed outputs (e.g. RNX3, ROTI, VTEC, SIDX)
orbit_output    = base_dir / "OUTPUT" / "ORBITS" / year / doy  # directory for tabulated orbit outputs

# Create directories if they do not exist
orbit_output.mkdir(parents=True, exist_ok=True)
sta_output.mkdir(parents=True, exist_ok=True)

# ===============================
# 1) SP3 interpolation
# ===============================
# Interpolate satellite orbits using SP3 files for the given day of year and year
pyOASIS.SP3intp(year, doy, orbit_input_dir, orbit_output)

# ===============================
# 2) RNXclean
# ===============================
# Convert raw RINEX observation files (.yyo) to internal GNSS-clean format,
# performing the initial detection of cycle slips, outliers, and identifying
# data gaps (arcs). (Output: .RNX1, .RNX2 files)
pyOASIS.RNXclean(sta, doy, year, rinex_dir, orbit_output, sta_output)

# ===============================
# 3) RNXlevelling
# ===============================
# Apply geometry-free leveling to remove satellite and receiver biases,
# performing the final detection of outliers and cycle slips. (Output: .RNX3 files)
pyOASIS.RNXlevelling(sta, sta_output, show_plot=True)

# ===============================
# 4) ROTIcalc
# ===============================
# Compute the Rate of TEC Index (ROTI) using leveled geometry-free data from .RNX3 files
pyOASIS.ROTIcalc(sta, doy, year, sta_output, sta_output, show_plot=True)

# ===============================
# 5) DTECcalc
# ===============================
# Compute the Delta TEC index using leveled geometry-free data from .RNX3 files
pyOASIS.DTECcalc(sta, doy, year, sta_output, sta_output, show_plot=True)

# ===============================
# 6) SIDXcalc
# ===============================
# Compute the SIDX index using leveled geometry-free data from .RNX3 files
pyOASIS.SIDXcalc(sta, doy, year, sta_output, sta_output, show_plot=True)

# ===============================
# 7) TECcalc
# ===============================
# Compute absolute Total Electron Content (TEC) from leveled geometry-free combinations
pyOASIS.TECcalc(sta, doy, year, sta_output, sta_output, show_plot=True)
