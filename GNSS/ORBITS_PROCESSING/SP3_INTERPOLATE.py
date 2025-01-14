#!/usr/bin/env python3
import sys
import re
from datetime import datetime
import pandas as pd
import os

# Arguments from the management program
year = sys.argv[1]
doy = sys.argv[2]
input_folder = sys.argv[3]
output_folder = sys.argv[4]

# Regular expressions for data analysis
pattern_datetime = r"\*\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}\.\d{8})"
pattern_data = r"([A-Z]{2}\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"

# Constructing the full path to the data folder
path_ = input_folder

# Final list for all the data (without interpolation)
satellite_data = []

# Defining the key date in dd-mm-yyyy format
key_date = datetime.strptime(f"{doy}/{year}", "%j/%Y").strftime("%d-%m-%Y")

# Check directory existence and process files if it exists
if os.path.exists(path_):
    for file in os.listdir(path_):
        if file.endswith(".SP3"):
            file_path = os.path.join(path_, file)
            print()
            print(f"Processing file: {file_path}")

            # Opens the file and reads its lines
            with open(file_path, "r") as file:
                for line in file:
                    # Extracts date and time if the line matches the pattern
                    if re.match(pattern_datetime, line):
                        match = re.match(pattern_datetime, line)
                        file_year, month, day, hour, minute = (int(match.group(i)) for i in range(1, 6))
                        second = float(match.group(6))
                        last_date_time = datetime(file_year, month, day, hour, minute, int(second))

                        # Skips the line if the date is greater than the key date
                        if last_date_time.date() > datetime.strptime(key_date, "%d-%m-%Y").date():
                            continue

                    # Checks if the line has satellite data
                    elif re.match(pattern_data, line):
                        match = re.match(pattern_data, line)
                        satellite_name, x, y, z = match.groups()

                        # Adds the current record to the satellite data (without interpolation)
                        satellite_data.append({
                            'Date': last_date_time.strftime("%d-%m-%Y"),
                            'Time': last_date_time.strftime("%H:%M:%S"),
                            'Satellite': satellite_name,
                            'X': x,
                            'Y': y,
                            'Z': z
                        })

    # Creating a DataFrame with the original data (without interpolation)
    df_satellite = pd.DataFrame(satellite_data)
    df_satellite = df_satellite.sort_values(by=['Satellite', 'Date', 'Time'])

    # Removes rows with dates different from the key date
    df_satellite = df_satellite[df_satellite['Date'] == key_date]

    # Saves the data to the file
    output_file_name = os.path.join(output_folder, f'ORBITS_{year}_{doy}.SP3')
    df_satellite.to_csv(output_file_name, sep='\t', index=False)
    print()
    print(f"Original data file saved to: {output_file_name}")
