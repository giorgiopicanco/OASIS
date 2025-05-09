import os
import subprocess
from pathlib import Path
import pyOASIS
import sys

# Define the arguments directly
sta = "BOAV"  # Converted to lowercase manually as per "${sta,,}"
doy = "049"  # Day of the year
year = "2023"  # Year

input_dir = "/home/debian-giorgio/pyOASIS/pyOASIS/INPUT"
output_dir = "/home/debian-giorgio/pyOASIS/pyOASIS/OUTPUT"
sta_dir = Path(output_dir) / sta  # Construct the station directory path
print(sta_dir)

# Create the directory if it does not exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

#pyOASIS.SP3intp(year,doy,input_dir,output_dir)

#pyOASIS.RNXclean(sta,doy,year,output_dir,input_dir,output_dir)

#pyOASIS.RNXScreening(sta_dir)

#pyOASIS.RNXlevelling(sta,sta_dir,show_plot=True)

pyOASIS.ROTIcalc(sta, doy, year, output_dir, output_dir,show_plot=True)

#pyOASIS.DTECcalc(sta, doy, year, output_dir, output_dir, show_plot=True)

pyOASIS.SIDXcalc(sta, doy, year, output_dir, output_dir)

#%%
