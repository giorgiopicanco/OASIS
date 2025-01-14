#!/usr/bin/env python3
import sys
import re
from datetime import datetime, timedelta
import pandas as pd
import os
import requests

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

# Final list for all interpolated data
interpolated_data = []

# Defining the key date in dd-mm-yyyy format
key_date = datetime.strptime(f"{doy}/{year}", "%j/%Y").strftime("%d-%m-%Y")

# Defining RINEX sampling rates
rate = 15


# Check directory existence and process files if it exists
if os.path.exists(path_):
    for file in os.listdir(path_):
        if file.endswith(".SP3"):
            file_path = os.path.join(path_, file)
            print()
            print(f"Processing file for {rate}-sec rate: {file_path}")

            # Stores the last known satellite data
            last_data_by_satellite = {}

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

                    # Checks if the line has satellite data and interpolates if necessary
                    elif re.match(pattern_data, line):
                        match = re.match(pattern_data, line)
                        satellite_name, x, y, z = match.groups()

                        # Updates interpolated data every 15 seconds until the next known time
                        if satellite_name in last_data_by_satellite:
                            last_known_time = last_data_by_satellite[satellite_name]['date_time']
                            while last_known_time + timedelta(seconds=rate) < last_date_time:
                                last_known_time += timedelta(seconds=rate)
                                interpolated_data.append({
                                    'Date': last_known_time.strftime("%d-%m-%Y"),
                                    'Time': last_known_time.strftime("%H:%M:%S"),
                                    'Satellite': satellite_name,
                                    'X': last_data_by_satellite[satellite_name]['x'],
                                    'Y': last_data_by_satellite[satellite_name]['y'],
                                    'Z': last_data_by_satellite[satellite_name]['z']
                                })

                        # Updates the last known data for this satellite
                        last_data_by_satellite[satellite_name] = {
                            'date_time': last_date_time,
                            'x': x,
                            'y': y,
                            'z': z
                        }
                        # Also adds the current record to the interpolated data
                        interpolated_data.append({
                            'Date': last_date_time.strftime("%d-%m-%Y"),
                            'Time': last_date_time.strftime("%H:%M:%S"),
                            'Satellite': satellite_name,
                            'X': x,
                            'Y': y,
                            'Z': z
                        })

    # After processing all files:
    # Sets the end time for extrapolation
    final_hour = datetime(file_year, month, day, 23, 59, 45)

    # Iterates over each satellite to complete the data until the end of the day
    for satellite_name, data in last_data_by_satellite.items():
        last_record = data  # Last record of the satellite
        while last_record['date_time'] < final_hour:
            last_record['date_time'] += timedelta(seconds=rate)  # Increments 15 seconds
            # Creates and adds the new extrapolated record
            interpolated_data.append({
                'Date': last_record['date_time'].strftime("%d-%m-%Y"),
                'Time': last_record['date_time'].strftime("%H:%M:%S"),
                'Satellite': satellite_name,
                'X': last_record['x'],
                'Y': last_record['y'],
                'Z': last_record['z']
            })

    # Creating a DataFrame with all interpolated data
    df_interpolated = pd.DataFrame(interpolated_data)
    df_interpolated = df_interpolated.sort_values(by=['Satellite', 'Date', 'Time'])

    # print(len(df_interpolated))

    # Removes rows with dates different from the key date
    df_interpolated = df_interpolated[df_interpolated['Date'] == key_date]

    # Saves the interpolated data to the file
    interpolated_file_name = os.path.join(output_folder, f'ORBITS_{year}_{doy}_{rate}S.SP3')

    df_interpolated.to_csv(interpolated_file_name, sep='\t', index=False)
    print()
    print(f"Interpolated file saved to: {interpolated_file_name}")

