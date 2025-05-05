import datetime as dt
import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from astropy.time import Time
from scipy.constants import speed_of_light
from scipy.ndimage import uniform_filter1d
from numpy.polynomial import Polynomial
from matplotlib import font_manager
import georinex as gr
import sys
import time
from pyOASIS import settings
from pyOASIS import linear_combinations
from pyOASIS import gnss_freqs
import warnings
import pyOASIS

def RNXclean(station_name,day_of_year,year,output_folder,input_folder,orbit_folder):

    # Accessing the frequencies of the GPS system
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]
    f2 = gps_freqs[2]
    f5 = gps_freqs[5]
    
    # Calculating the frequencies of the GLONASS system (glonass_channels.dat)
    # To obtain the GLONASS frequencies (code 'R'):
    file_name = os.path.join(pyOASIS.__path__[0], 'glonass_channels.dat')
    
    # Read the file into a DataFrame with column names defined
    df_slots = pd.read_csv(file_name, sep=' ', header=None, names=['Slot', 'Channel'])
    glonass_frequencies = gnss_freqs.FREQUENCY[gnss_freqs.GLO]
    
    # List to store the data
    data = []
    
    # Iterate over each row of the DataFrame
    for index, row in df_slots.iterrows():
        satellite = row['Slot']
        k = row['Channel']
        row_data = [satellite]
    
        for channel, frequency in glonass_frequencies.items():
            if callable(frequency):  # Check if it is a lambda function
                freq_value = frequency(k)
            else:
                freq_value = frequency
            formatted_freq = f"{freq_value:.1f}"
            row_data.append(formatted_freq)
        data.append(row_data)
    
    # Convert the list of lists into a pandas DataFrame
    glo_freqs = pd.DataFrame(data, columns=['Satellite', 'fr1', 'fr2', 'fr3'])
    
    # Initial time, number of hours, and xticks interval on the graph.
    h1 = 0
    n_horas = 24
    int1 = 120
    
    # Minimum number of observations per arc
    ARCL = 15
    
    constellations = ['G', 'R']  # G: GPS, R: GLONASS, E: Galileo (future), C: BeiDou (future)
    
    for c in constellations:
        sat_class = c
    
        # RINEX OPENING:
        # ------------------------------------
        # Selection of version for RINEX files
        version_number = '1'
        print()
        year_format = f"{year[-2:]}o"
        rinex_file_path = f"{input_folder}/{station_name.lower()}{day_of_year}{version_number}.{year_format}"
        if not os.path.exists(rinex_file_path):
            version_number = '0' if version_number == '1' else '1'
            rinex_file_path = f"{input_folder}/{station_name.lower()}{day_of_year}{version_number}.{year_format}"
            if not os.path.exists(rinex_file_path):
                print(f"File {rinex_file_path} not found either. Please check the path or other issues.")
            else:
                print(f"File {rinex_file_path} found with opposite version of {version_number}.")
        else:
            print(f"File {rinex_file_path} found with version_number = {version_number}")
    
        # Supressing warning about timedelta
        warnings.filterwarnings("ignore", message="Converting non-nanosecond precision datetime values to nanosecond precision.", category=UserWarning)
        obs_data = gr.load(rinex_file_path)

        print()
    
        # ORBIT FILE OPENING:
        # -------------------------
        # Path and time definitions
    
        freq = int(obs_data.interval)
        file_path = os.path.join(orbit_folder, f'ORBITS_{year}_{day_of_year}.SP3')
        column_names = ["Date", "Time", "Satélite", "X", "Y", "Z"]
        df2 = pd.read_csv(file_path, sep="\t", header=0, names=column_names)
    
        # List to store IPP coordinates
        result = {"Date": [], "Time": [], "SAT": [], "lon": [], "lat": [], "el": []}
        index = []
    
        # Filtering satellites by class
        satellites_to_plot = [sv for sv in np.unique(obs_data.sv.values) if sv.startswith(sat_class)]
    
        # List to store DataFrames for each satellite
        dfs = []
    
        # Remove the first character of each element in the 'Satellite' column
        df2['Satélite'] = df2['Satélite'].str[1:]
    
        IPP_UNIQ = np.unique(df2['Satélite'])
    
        # Converting lists to sets
        set_IPP_UNIQ = set(IPP_UNIQ)
        set_satellites_to_plot = set(satellites_to_plot)
    
        # Finding the intersection between the two sets
        common_elements = set_IPP_UNIQ.intersection(set_satellites_to_plot)
    
        # Converting the set of common elements back into a list, if necessary
        common_elements_list = list(common_elements)
    
        # Defining the desired order for the initial letters
        order = {'G': 1, 'R': 2, 'E': 3, 'C': 4}
    
        # Sorting the list based on the initial letter and the last two digits
        sorted_common_elements_list = sorted(common_elements_list, key=lambda x: (order[x[0]], int(x[-2:])))

    
        for sat in sorted_common_elements_list:
            print()
            print("Processing satellite:", sat)
            print()
    
            # Checking the satellite class and adjusting the values of f1, f2, f5
            if sat_class == 'G':
                f1 = f1
                f2 = f2
                f5 = f5
            elif sat_class == 'R':
                # Locating the row where 'Satellite' matches 'sat'
                sat_row = glo_freqs.loc[glo_freqs['Satellite'] == sat]
    
                if not sat_row.empty:
                    f1 = float(sat_row['fr1'].values[0])
                    f2 = float(sat_row['fr2'].values[0])
                    f5 = float(sat_row['fr3'].values[0])
                else:
                    f1 = f2 = f5 = None  # Or default values

            L1 = np.array(obs_data['L1'].sel(sv=sat))
            L2 = np.array(obs_data['L2'].sel(sv=sat))
    
            # Adding the L5 frequency to L1 and L2
            if 'L5' in obs_data:
                L5 = np.array(obs_data['L5'].sel(sv=sat))
            else:
                L5 = np.full_like(L1, np.nan)  # Fill with NaN if L5 data is not available
    
            if 'P1' in obs_data:
                P1 = np.array(obs_data['P1'].sel(sv=sat))
                code_obs1 = 'P1'
            else:
                P1 = np.array(obs_data['C1'].sel(sv=sat))
                code_obs1 = 'C1'
            if np.all(np.isnan(P1)):
                P1 = np.array(obs_data['C1'].sel(sv=sat))
                code_obs1 = 'C1'
            else:
                code_obs1 = 'P1'
            if 'P2' in obs_data:
                P2 = np.array(obs_data['P2'].sel(sv=sat))
                code_obs2 = 'P2'
            else:
                P2 = np.array(obs_data['C2'].sel(sv=sat))
                code_obs2 = 'C2'
            if np.all(np.isnan(P2)):
                P2 = np.array(obs_data['C2'].sel(sv=sat))
                code_obs2 = 'C2'
            else:
                code_obs2 = 'P2'
    
            # Checking for the presence of carrier phase (P5) or pseudorange (C5) observations
            if 'P5' in obs_data:
                P5 = np.array(obs_data['P5'].sel(sv=sat))
                code_obs5 = 'P5'
                if np.all(np.isnan(P5)):
                    P5 = np.array(obs_data['C5'].sel(sv=sat))
                    code_obs5 = 'C5'
                    if np.all(np.isnan(P5)):
                        # print(f"WARNING: All values in {code_obs5} are NaNs for satellite {sat}")
                        print()
            elif 'C5' in obs_data:
                P5 = np.array(obs_data['C5'].sel(sv=sat))
                code_obs5 = 'C5'
                if np.all(np.isnan(P5)):
                    # print(f"WARNING: All values in {code_obs5} are NaNs for satellite {sat}")
                    print()
            else:
                # If neither P5 nor C5 are present, set P5 as a column of NaNs
                P5 = np.full_like(L5, np.nan)
                code_obs5 = "None"  # Or any other value to indicate the absence of data, as you prefer
    
            if np.all(np.isnan(P5)):
                print(f"WARNING: All values in L5 data", f"{code_obs5} are NaNs for satellite {sat}")
    
            df = pd.DataFrame({'time': obs_data.time})
            df.set_index('time', inplace=True)
            df['index'] = df.index
            df['mjd'] = Time(df.index).mjd
            mjd = Time(df.index).mjd
    
            df['timestamp'] = pd.to_datetime(df.index)
    
            # Extracting only the time from 'timestamp' and storing it in a new 'time' column
            df['time'] = df['timestamp'].dt.time
            df['date'] = df['timestamp'].dt.date
    
            df['L1'] = L1
            df['L2'] = L2
            df['L5'] = L5
            df['P1'] = P1
            df['P2'] = P2
            df['P5'] = P5
    
            df['satellite'] = [sat] * len(L1)
            df['station'] = [station_name.upper()] * len(L1)
            df['position'] = [obs_data.position] * len(L1)
            df['mjd'] = ["{:.12f}".format(valor) for valor in mjd]
    
            timep = df['time']
    
            L1, L2, L5, P1, P2, P5 = df['L1'] , df['L2'] , df['L5'] , df['P1'] , df['P2'], df['P5']
    
            mjd, date, time = df['mjd'] , df['date'] , df['time']
    
            df['LMW12'] = linear_combinations.melbourne_wubbena(f1, f2, L1, L2, P1, P2)
    
            df['LMW15'] = linear_combinations.melbourne_wubbena(f1, f5, L1, L5, P1, P5)
    
            station_coords = df['position'].iloc[0]
    
            # Converting each string in the list into a float (or int, if appropriate)
            coords_list = [float(coord) for coord in station_coords]
    
            # Converting the list of coordinates into a list containing a single tuple
            obs_x, obs_y, obs_z = coords_list[0], coords_list[1], coords_list[2]
    
            # Initializing empty lists to collect data
            all_data = []
            all_time = []
            all_sat = []
            all_longitude = []
            all_latitude = []
            all_elevation = []
    
            # for ipp_sat in sorted_common_elements_list:
            indices = np.where(df2['Satélite'] == sat)[0]
            df_filtrado = df2.iloc[indices]
    
            import time

            # Getting the current time in seconds since the Unix Epoch
            current_time = time.time()
    
            # Converting the time into a local time structure
            local_time = time.localtime(current_time)
    
            # Formatting the time in hh:mm format
            formatted_time = time.strftime("%H:%M:%S", local_time)

            from datetime import datetime, timedelta
    
            # Defining the interpolation rate in seconds (e.g., 15 seconds)
            rate = freq

            # Lists to store the original and interpolated data
            all_data = []
            all_time = []
            all_sat = []
            all_longitude = []
            all_latitude = []
            all_elevation = []
    
            # Variables to store the last known data for each satellite
            last_data_by_satellite = {}
    
            for _, row in df_filtrado.iterrows():
                date = row["Date"]
                sat = row["Satélite"]
                sx = row["X"]
                sy = row["Y"]
                sz = row["Z"]
                time = row["Time"]
    
                # Assuming that convert_coords and IonosphericPiercingPoint are functions defined earlier
                lon, lat, alt = settings.convert_coords(obs_x, obs_y, obs_z, to_radians=True)
                ip = settings.IonosphericPiercingPoint(sx, sy, sz, obs_x, obs_y, obs_z)
                elevation = ip.elevation(lat, lon)
                lat_ip, lon_ip = ip.coordinates(lat, lon)
    
                # Converting the current date and time into a datetime object to calculate time intervals
                current_time = datetime.strptime(f"{date} {time}", "%d-%m-%Y %H:%M:%S")
    
                # If we already have previous data for the satellite, we check whether interpolation is needed
                if sat in last_data_by_satellite:
                    last_known_time = last_data_by_satellite[sat]['date_time']
    
                    # If the difference between the last known time and the current time is greater than the rate, interpolate
                    while last_known_time + timedelta(seconds=rate) < current_time:
                        last_known_time += timedelta(seconds=rate)
    
                        # Add the interpolated data to the lists (using the last known values)
                        all_data.append(last_known_time.strftime("%d-%m-%Y"))
                        all_time.append(last_known_time.strftime("%H:%M:%S"))
                        all_sat.append(sat)
                        all_longitude.append(last_data_by_satellite[sat]['lon'])
                        all_latitude.append(last_data_by_satellite[sat]['lat'])
                        all_elevation.append(last_data_by_satellite[sat]['elevation'])
    
                # Update the satellite data for the next cycle
                last_data_by_satellite[sat] = {
                    'date_time': current_time,
                    'lon': lon_ip,
                    'lat': lat_ip,
                    'elevation': elevation
                }
    
                # Add the original data (without interpolation) to the lists
                all_data.append(date)
                all_time.append(time)
                all_sat.append(sat)
                all_longitude.append(lon_ip)
                all_latitude.append(lat_ip)
                all_elevation.append(elevation)
    
            # After processing all satellites:
            # Set the final time as 23:59:45
            final_hour = datetime.strptime(f"{date} 23:59:45", "%d-%m-%Y %H:%M:%S")
    
            # Iterate over each satellite to extrapolate the data until the end of the day
            for satellite_name, data in last_data_by_satellite.items():
                last_record = data  # Last recorded data for the satellite
                while last_record['date_time'] < final_hour:
                    last_record['date_time'] += timedelta(seconds=rate)  # Increment by 15 seconds
    
                    # Add the extrapolated data (the last known values) to the lists
                    all_data.append(last_record['date_time'].strftime("%d-%m-%Y"))
                    all_time.append(last_record['date_time'].strftime("%H:%M:%S"))
                    all_sat.append(satellite_name)
                    all_longitude.append(last_record['lon'])
                    all_latitude.append(last_record['lat'])
                    all_elevation.append(last_record['elevation'])
    
            # After exiting the loop, convert the lists to NumPy arrays (if necessary)
            all_date = pd.Series(all_data)
            all_time = pd.Series(all_time)
            all_sat = pd.Series(all_sat)
            all_longitude = pd.Series(all_longitude)
            all_latitude = pd.Series(all_latitude)
            all_elevation = pd.Series(all_elevation)
    
            # Create a new combined DataFrame with the original, interpolated, and extrapolated data
            combined_df = pd.DataFrame({
                "Date": all_date,
                "Time": all_time,
                "SAT": all_sat,
                "Longitude": all_longitude,
                "Latitude": all_latitude,
                "Elevation": all_elevation
            })
    
            # Converting the time data to datetime type
            combined_df['Time'] = pd.to_datetime(combined_df['Time'], format='%H:%M:%S')
    
            df['time'] = df['time'].astype(str)
    
            # Converting the times into datetime objects
            combined_df['Time'] = pd.to_datetime(combined_df['Time']).dt.time
    
            # Specify the format when converting to datetime
            df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    
            # Finding the times that are present in both dataframes
            horas_em_common = set(combined_df['Time']).intersection(set(df['time']))
    
            # Defining the reference length as the smaller length between the two dataframes
            comprimento_referencia = min(len(combined_df), len(df))
    
            # Filtering the dataframes to keep only the common rows based on the reference length
            combined_df = combined_df[combined_df['Time'].isin(horas_em_common)].iloc[:comprimento_referencia]
            df = df[df['time'].isin(horas_em_common)].iloc[:comprimento_referencia]
    
            colunas_desejadas_df1 = ['Elevation','Longitude','Latitude']
    
            combined_df_selecionado = combined_df[colunas_desejadas_df1]
    
            L1, L2, L5, P1, P2, P5 = df['L1'] , df['L2'] , df['L5'], df['P1'] , df['P2'] , df['P5']
    
            mjd, date, time = df['mjd'] , df['date'] , df['time']
    
            # Assuming that df_filtered is your DataFrame
            # Split the 'position' column into three new columns and expand the result into separate columns
            df['pos_x'], df['pos_y'], df['pos_z'] = obs_x, obs_y, obs_z
    
            df['height'] = np.full(len(L1), 450.0)
    
            # Now df_filtered has the columns 'pos_x', 'pos_y', and 'pos_z'
            # You can then select these new columns for inclusion in the DataFrame to be saved
            colunas_desejadas_df2 = ['date','time', 'mjd', 'pos_x', 'pos_y', 'pos_z', 'L1', 'L2', 'L5', 'P1', 'P2', 'P5', 'satellite','station','height']
            df_filtered_selecionado = df[colunas_desejadas_df2]
    
            # Make sure that df_filtered and combined_df have the same length
            if len(df_filtered_selecionado) == len(combined_df_selecionado):
                # Merging the DataFrames side by side
                df_final = pd.concat([df_filtered_selecionado, combined_df_selecionado], axis=1)
            else:
                print("Error: df_filtered and combined_df do not have the same length!")
                # Exit the execution if the DataFrames do not have the same size
                sys.exit()
    
            LMW2 = df['LMW12']
            LMW3 = df['LMW15']
    
            abs_elevation = abs(combined_df_selecionado['Elevation'])
    
            df['LMW2'] = LMW2
            df['LMW3'] = LMW3
    
            combined_df_selecionado.index = df.index
    
            indices_low_elevation = combined_df_selecionado.index[abs_elevation < 10]
    
            # List of columns that should be set to NaN
            cols_nan = ['LMW2', 'LMW3']
    
            # Assign NaN to the specified columns only for the rows in indices_low_elevation
            df.loc[indices_low_elevation, cols_nan] = np.nan
    
            # Assuming that LMW is a NumPy array and time is a corresponding time vector
            LMW = np.array(df['LMW2'])  # Ensure that LMW is a NumPy array to facilitate manipulation
            LMW15 = np.array(df['LMW3'])  # Ensure that LMW is a NumPy array to facilitate manipulation
    
            arcs = []  # List to store the observation arcs
            current_arc = []  # Temporary list to store the current observation arc
    
            # Iterate over all elements of LMW
            for idx, value in enumerate(LMW):
                if np.isnan(value):
                    # If the current value is NaN, check if the current arc is empty
                    # This avoids adding empty arcs in case of consecutive NaNs
                    if current_arc:
                        arcs.append(current_arc)
                        current_arc = []
                else:
                    # If the value is not NaN, add the index to the current arc
                    current_arc.append(idx)
    
            # Add the last arc if it is not empty
            if current_arc:
                arcs.append(current_arc)
    
            print()
            print('Melbourne-Wubbena combination for L1-L2')
            print()
    
            # Print information about each arc and classify them
            for i, arc in enumerate(arcs):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
    
                print(f"Arc {i + 1}: Start index = {start_index}, End index = {end_index}, "
                    f"Number of observations = {num_observations}, Status = {status}")
    
            arcs15 = []  # List to store the observation arcs
            current_arc15 = []  # Temporary list to store the current observation arc
    
            # Iterate over all elements of LMW15
            for idx, value in enumerate(LMW15):
                if np.isnan(value):
                    # If the current value is NaN, check if the current arc is empty
                    # This avoids adding empty arcs in case of consecutive NaNs
                    if current_arc15:
                        arcs15.append(current_arc15)
                        current_arc15 = []
                else:
                    # If the value is not NaN, add the index to the current arc
                    current_arc15.append(idx)
    
            # Add the last arc if it is not empty
            if current_arc15:
                arcs15.append(current_arc15)
    
            print()
            print()
    
            print('Melbourne-Wubbena combination for L1-L5')
    
            print()
    
            # Print information about each arc and classify them
            for i, arc in enumerate(arcs15):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i + 1}: Start index = {start_index}, End index = {end_index}, "
                    f"Number of observations = {num_observations}, Status = {status}")
    
            LMW = np.array(LMW)  # Still assuming that LMW is your data array
            LMW15 = np.array(LMW15)  # Still assuming that LMW is your data array
    
            print()
            print()
    
            # Function to rescale the data to the range [-10, 10]
            def rescale_data(data):
                min_val = np.min(data)
                max_val = np.max(data)
                # Rescale the data to the range [0, 1]
                scaled_data = (data - min_val) / (max_val - min_val)
                # Adjust it to the range [-10, 10]
                final_data = scaled_data * 20 - 10
                return final_data
    
    
            # -- [L1 - L2]
            idx_total = []
    
            for arc in arcs:
                arc_data = df.iloc[arc]
                time = df.index[arc]
    
                if len(arc_data) < ARCL:  # Check if there are enough points for the fitting
                    continue
    
                x = arc_data.index.astype(np.int64) // 10**9  # Converting to seconds
                xx = arc_data['time']
                y = arc_data['LMW2'].values
    
                y_rescaled = rescale_data(y)
                delta_y = np.diff(y_rescaled, prepend=np.nan)
    
                # Fit a polynomial only to the valid values (excluding np.nan)
                p = Polynomial.fit(x[1:], delta_y[1:], 3)
    
                delta_y_fit = p(x)  # Polynomial fitted values
                residuals = abs(delta_y - delta_y_fit)  # Calculate residuals
    
                # Calculation of customized quartiles and IQR for outlier identification
                Q1 = np.nanpercentile(residuals, 15)
                Q3 = np.nanpercentile(residuals, 85)
                IQR = Q3 - Q1
                outlier_mask = (residuals < Q1 - 4 * IQR) | (residuals > Q3 + 4 * IQR)
                high_residuals_mask = residuals > 1  # Máscara para resíduos altos
                other_residuals_mask = ~(outlier_mask | high_residuals_mask)  # Máscara para os demais resíduos

                # Suppose that 'residuals' is the variable containing the calculated residuals
                limiar_outlier = 4  # Threshold for identifying outliers (e.g., 4 times the IQR)
                limiar_residuo_alto = 1  # Threshold for identifying high residuals
    
                # Identify indices of outliers and high residuals
                indices_outliers =  arc[0] + np.where(np.abs(residuals) > limiar_outlier * IQR)[0]
                indices_residuos_altos = arc[0] + np.where(np.abs(residuals) > limiar_residuo_alto)[0]
    
                # Merge the indices without duplicates
                indices_combinados = np.union1d(indices_outliers, indices_residuos_altos)
    
                # Add the combined_indices values to the list
                idx_total.append(indices_combinados)
    
            # Concatenate all values in the list into a single NumPy array
            if idx_total:
                idx_total = np.concatenate(idx_total)
    
            # Initialize 'flags' with 'C' for all indices in LMW
            flags = ['C'] * len(LMW)
    
            # Check where LMW is NaN and replace the corresponding flags with 'S'
            nan_indices = np.where(np.isnan(LMW))[0]
            flags = list(flags)  # Convert flags to a list
            for idx in nan_indices:
                flags[idx] = 'S'
    
            # Use idx_total as indices to change the corresponding flags to 'S'
            for idx in idx_total:
                if idx < len(flags):
                    flags[idx] = 'S'
    
            for idx in idx_total:
                print("L1-L2 (+)", idx, timep.iloc[idx])
    
    
            # ---- [L1 - L5]
            idx_total15 = []
    
            for arc in arcs15:
                arc_data = df.iloc[arc]
                time = df.index[arc]
    
                if len(arc_data) < ARCL:  # Check if there are enough points for fitting
                    continue
    
                x = arc_data.index.astype(np.int64) // 10**9  # Convert to seconds
                xx = arc_data['time']
                y = arc_data['LMW3'].values
    
                y_rescaled = rescale_data(y)
                delta_y = np.diff(y_rescaled, prepend=np.nan)
    
                # Fit a polynomial only to valid values (excluding np.nan)
                p = Polynomial.fit(x[1:], delta_y[1:], 3)

                delta_y_fit = p(x)  # Polynomial fitted values
                residuals = abs(delta_y - delta_y_fit)  # Calculate residuals
    
                # Calculation of customized quartiles and IQR for outlier identification
                Q1 = np.nanpercentile(residuals, 15)
                Q3 = np.nanpercentile(residuals, 85)
                IQR = Q3 - Q1
                outlier_mask = (residuals < Q1 - 4 * IQR) | (residuals > Q3 + 4 * IQR)
                high_residuals_mask = residuals > 1  # Mask for high residuals
                other_residuals_mask = ~(outlier_mask | high_residuals_mask)  # Mask for normal residuals
    
                # Assume that 'residuals' is the variable containing the calculated residuals
                limiar_outlier = 4  # Threshold to identify outliers (e.g., 4 times the IQR)
                limiar_residuo_alto = 1  # Threshold to identify high residuals
    
                # Identify indices of outliers and high residuals
                indices_outliers15 =  arc[0] + np.where(np.abs(residuals) > limiar_outlier * IQR)[0]
                indices_residuos_altos15 = arc[0] + np.where(np.abs(residuals) > limiar_residuo_alto)[0]
    
                # Combine the indices without duplication
                indices_combinados15 = np.union1d(indices_outliers15, indices_residuos_altos15)
    
                # Add the combined indices to the list
                idx_total15.append(indices_combinados15)
    
                print("Combined indices (L1 - L5):", indices_combinados15)
    
            # Concatenate all values from the list into a single NumPy array
            if idx_total15:
                idx_total15 = np.concatenate(idx_total15)
    
            # Check if LMW15 is not entirely composed of NaN
            if not np.all(np.isnan(LMW15)):
                # Identify where LMW15 is NaN and assign 'S' to the corresponding flags
                nan_indices15 = np.where(np.isnan(LMW15))[0]
                for idx in nan_indices15:
                    flags[idx] = 'S'
    
            # Use idx_total15 as indices to set the corresponding flags to 'S'
            for idx in idx_total15:
                if idx < len(flags):
                    flags[idx] = 'S'
    
            for idx in idx_total15:
                print("L1-L5 (+)", idx, timep.iloc[idx])
    
            satellite_values = [sat] * len(L1)
            station = [station_name.upper()] * len(L1)
            position = [obs_data.position] * len(L1)
    
            L1 = [value if pd.notna(value) else -999999.999 for value in L1]
            L2 = [value if pd.notna(value) else -999999.999 for value in L2]
            L5 = [value if pd.notna(value) else -999999.999 for value in L5]
    
            P1 = [value if pd.notna(value) else -999999.999 for value in P1]
            P2 = [value if pd.notna(value) else -999999.999 for value in P2]
            P5 = [value if pd.notna(value) else -999999.999 for value in P5]
    
            L1 = ["{:.3f}".format(valor) for valor in L1]
            L2 = ["{:.3f}".format(valor) for valor in L2]
            L5 = ["{:.3f}".format(valor) for valor in L5]
    
            P1 = ["{:.3f}".format(valor) for valor in P1]
            P2 = ["{:.3f}".format(valor) for valor in P2]
            P5 = ["{:.3f}".format(valor) for valor in P5]
    
            df.reset_index(drop=True, inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
    
            export_df = pd.DataFrame({
                'date': df['date'],
                'time': df['time'],
                'mjd': df['mjd'],
                'pos_x': df['pos_x'],
                'pos_y': df['pos_y'],
                'pos_z': df['pos_z'],
                'L1': df['L1'],
                'L2': df['L2'],
                'L5': df['L5'],
                'P1': df['P1'],
                'P2': df['P2'],
                'P5': df['P5'],
                'cs_flags': flags,
                'satellite': satellite_values,
                'sta': station,
                'hght': df['height'],
                'El': abs(combined_df['Elevation'].round(2)),
                'Lon': combined_df['Longitude'],
                'Lat': combined_df['Latitude'],
            })
    
            export_df.rename(columns={'P1': code_obs1}, inplace=True)
            export_df.rename(columns={'P2': code_obs2}, inplace=True)
            export_df.rename(columns={'P5': code_obs5}, inplace=True)
    
            file_name = f"{station_name.upper()}_{sat}_{day_of_year}_{year}.RNX1"
    
            output_directory = os.path.join(output_folder, station_name.upper())
            full_path = output_directory
    
    
            os.makedirs(full_path, exist_ok=True)
            output_file_path = os.path.join(full_path, file_name)
            export_df.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')
