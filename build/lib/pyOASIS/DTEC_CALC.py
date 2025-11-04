import os
import sys
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pyOASIS import gnss_freqs

def DTECcalc(station, doy, year, input_folder, destination_directory, show_plot=True):
    """
    Compute DTEC index from GNSS data for a given station and day-of-year.

    Parameters:
        station (str): GNSS station code
        doy (int or str): Day of year
        year (int): Year of observation
        input_folder (str): Directory containing input data files
        destination_directory (str): Directory to store outputs
    """

    # Initial hour and total number of hours to process (full day)
    h1 = 0
    n_hours = 24  # hours

    # Time interval (in minutes) used in processing or averaging
    int1 = 320  # minutes

    # Get GPS frequencies from pyOASIS configuration
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]  # L1 frequency
    f2 = gps_freqs[2]  # L2 frequency
    f5 = gps_freqs[5]  # L5 frequency

    # TEC conversion constants for L1-L2 and L1-L5 combinations (in TECU)
    akl = 40.3e16 * ((1 / f2**2) - (1 / f1**2))     # For L1-L2
    akl15 = 40.3e16 * ((1 / f5**2) - (1 / f1**2))   # For L1-L5

    # Configuration parameters for DTEC calculation

    WINDOW = 60.0 * 2.5  # Window duration in seconds (2.5 minutes)
    TDTEC = 60.0         # Time unit for expressing DTEC in TECU/min
    GAP = 1.5 * WINDOW   # Time threshold for gap detection (seconds)
    H = 450000           # Ionospheric shell height in meters
    SIGMA = 5            # Threshold multiplier for outlier removal
    DE = 3               # Degree of polynomial fit (e.g., cubic)
    GAP2 = 3600          # Minimum gap between observation arcs (seconds)
    elev_angle = 30      # Elevation cutoff angle in degrees

    # Build the full path to the station-specific subdirectory
    path_ = os.path.join(input_folder)

    # Check if the directory for the given station exists
    if os.path.exists(path_):
        # List all files in the station directory
        content_ = os.listdir(path_)

        # Filter only the relevant .RNX3 files that match the station name
        files = [file for file in content_ if file.startswith(station) and file.endswith(".RNX3")]

        # Sort the files by satellite number (assuming file name format: STATION_SAT.RNX3)
        ord_files = sorted(files, key=lambda x: int(x.split("_")[1][1:]))

        # Print the total number of valid files found
        print()
        number_of_files = len(ord_files)
        print("Number of RINEX_LEVELLING (.RNX3) files in the directory:", number_of_files)

    else:
        # If the station directory does not exist, notify the user
        print("The specified directory does not exist.")

    print()

    # Initialize empty lists to store data parsed from all .RNX3 files

    # Temporal information
    date = []                # Date string
    time2 = []               # Time string
    mjd = []                 # Modified Julian Date

    # Receiver coordinates (in ECEF, meters)
    pos_x = []               # X-coordinate
    pos_y = []               # Y-coordinate
    pos_z = []               # Z-coordinate

    # Geometry-Free combinations (used for ionospheric analysis)
    LGF_combination = []     # Geometry-Free (L1-L2) combination
    LGF_combination15 = []   # Geometry-Free (L1-L5) combination

    # Metadata
    satellites = []          # Satellite identifier (e.g., G01, R22)
    sta = []                 # GNSS station code (e.g., BOAV, LPGS)
    hght = []                # Ionospheric pierce point height (in meters)
    el = []                  # Satellite elevation angle (degrees)

    # Ionospheric pierce point (IPP) geolocation
    lonn = []                # IPP longitude (degrees)
    latt = []                # IPP latitude (degrees)

    # Raw GNSS observations
    obs_La = []              # L1 phase observation (carrier)
    obs_Lb = []              # L2 phase observation (carrier)
    obs_Lc = []              # L5 phase observation (carrier)
    obs_Ca = []              # C1 code observation (pseudorange)
    obs_Cb = []              # C2 code observation (pseudorange)
    obs_Cc = []              # C5 code observation (pseudorange)

    # Loop through each RNX3 file from the sorted list
    for file_ in ord_files:
        path_file_ = os.path.join(path_, file_)

        # Open and read the file
        with open(path_file_, 'r') as f:
            # Read the header line (tab-separated column names)
            header = f.readline().strip().split('\t')

            # Read and process each data line
            for line in f:
                # Split the line into values based on tab separator
                columns = line.strip().split('\t')

                # Build a dictionary associating column names with values
                record = {
                    'date': columns[0],
                    'time2': columns[1],
                    'mjd': columns[2],
                    'pos_x': columns[3],
                    'pos_y': columns[4],
                    'pos_z': columns[5],
                    'LGF_combination': columns[6],
                    'LGF_combination15': columns[7],
                    'satellite': columns[8],
                    'sta': columns[9],
                    'hght': columns[10],
                    'el': columns[11],
                    'lonn': columns[12],
                    'latt': columns[13],
                    'obs_La': columns[14],
                    'obs_Lb': columns[15],
                    'obs_Lc': columns[16],
                    'obs_Ca': columns[17],
                    'obs_Cb': columns[18],
                    'obs_Cc': columns[19]
                }

                # Append each field to its corresponding list
                date.append(record['date'])
                time2.append(record['time2'])
                mjd.append(record['mjd'])
                pos_x.append(record['pos_x'])
                pos_y.append(record['pos_y'])
                pos_z.append(record['pos_z'])
                LGF_combination.append(record['LGF_combination'])
                LGF_combination15.append(record['LGF_combination15'])
                satellites.append(record['satellite'])
                sta.append(record['sta'])
                hght.append(record['hght'])
                el.append(record['el'])
                lonn.append(record['lonn'])
                latt.append(record['latt'])
                obs_La.append(record['obs_La'])
                obs_Lb.append(record['obs_Lb'])
                obs_Lc.append(record['obs_Lc'])
                obs_Ca.append(record['obs_Ca'])
                obs_Cb.append(record['obs_Cb'])
                obs_Cc.append(record['obs_Cc'])


    # Create a single figure for plotting results
    plt.figure(figsize=(12, 6))  # Set the figure size (width, height in inches)

    # Define the color palette to use for different satellites
    palette = plt.get_cmap('tab10')

    # GNSS satellite system classes to analyze: 'G' for GPS, 'R' for GLONASS
    sat_classes = ['G', 'R']

    # Loop through each GNSS class (e.g., GPS, GLONASS)
    for sat in sat_classes:
        satx = sat
        print()
        print(f"Calculating DTEC for {station.upper()}  |  Year: {year}  |  DOY: {doy}")
        print()
        # Filter list of satellites that belong to the current GNSS system
        if satx:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(satx)]
        else:
            satellites_to_plot = np.unique(satellites)

        # Initialize list to store per-satellite DTEC outputs or analysis results
        satellites_data = []

        # Process each satellite individually
        #for sat1 in satellites_to_plot:
        for idx, sat1 in enumerate(satellites_to_plot, start=1):
            print(f"Processing satellite {sat1} ({idx} of {len(satellites_to_plot)} in {satx} system)...")
            print()
            # Find all indices where the satellite ID matches sat1
            indices = np.where(np.array(satellites) == sat1)[0]

            # Initialize filtered lists to collect only the data related to this satellite
            date_filtered = []
            time2_filtered = []
            mjd_filtered = []
            pos_x_filtered = []
            pos_y_filtered = []
            pos_z_filtered = []
            LGF_combination_filtered = []
            LGF_combination15_filtered = []
            satellites_list_filtered = []
            sta_filtered = []
            hght_filtered = []
            el_filtered = []
            lonn_filtered = []
            latt_filtered = []
            obs_La_filtered = []
            obs_Lb_filtered = []
            obs_Lc_filtered = []
            obs_Ca_filtered = []
            obs_Cb_filtered = []
            obs_Cc_filtered = []

            # Loop through each selected index and extract the corresponding values
            for idx in indices:
                date_filtered.append(date[idx])
                time2_filtered.append(time2[idx])
                mjd_filtered.append(mjd[idx])
                pos_x_filtered.append(pos_x[idx])
                pos_y_filtered.append(pos_y[idx])
                pos_z_filtered.append(pos_z[idx])
                LGF_combination_filtered.append(LGF_combination[idx])
                LGF_combination15_filtered.append(LGF_combination15[idx])
                satellites_list_filtered.append(satellites[idx])
                sta_filtered.append(sta[idx])
                hght_filtered.append(hght[idx])
                el_filtered.append(el[idx])
                lonn_filtered.append(lonn[idx])
                latt_filtered.append(latt[idx])
                obs_La_filtered.append(obs_La[idx])
                obs_Lb_filtered.append(obs_Lb[idx])
                obs_Lc_filtered.append(obs_Lc[idx])
                obs_Ca_filtered.append(obs_Ca[idx])
                obs_Cb_filtered.append(obs_Cb[idx])
                obs_Cc_filtered.append(obs_Cc[idx])

            # Create a dictionary from the filtered data lists
            data = {
                'date': date_filtered,
                'time': time2_filtered,
                'mjd': mjd_filtered,
                'pos_x': pos_x_filtered,
                'pos_y': pos_y_filtered,
                'pos_z': pos_z_filtered,
                'LGF': LGF_combination_filtered,
                'LGF15': LGF_combination15_filtered,
                'satellites': satellites_list_filtered,
                'sta': sta_filtered,
                'hh': hght_filtered,
                'elev': el_filtered,
                'lonn': lonn_filtered,
                'latt': latt_filtered,
                'obs_La': obs_La_filtered,
                'obs_Lb': obs_Lb_filtered,
                'obs_Lc': obs_Lc_filtered,
                'obs_Ca': obs_Ca_filtered,
                'obs_Cb': obs_Cb_filtered,
                'obs_Cc': obs_Cc_filtered
            }

            # Convert the dictionary into a pandas DataFrame for further processing
            df = pd.DataFrame(data)

            # Create a new column combining date and time for timestamp
            df['timestamp'] = df['date'] + ' ' + df['time']

            # Convert the 'timestamp' column to pandas datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Convert key columns to float type for numerical processing
            columns_to_convert = ['LGF', 'LGF15', 'mjd', 'lonn', 'latt', 'hh', 'elev']
            df[columns_to_convert] = df[columns_to_convert].astype(float)

            # Replace invalid values (-999999.999) with NaN
            df.replace(-999999.999, np.nan, inplace=True)

            # Assign variables from DataFrame for ease of manipulation
            t = df['mjd']                          # Time in Modified Julian Date
            stec = df['LGF'] / akl                # Convert L1-L2 geometry-free to STEC (TECU)
            stec15 = df['LGF15'] / akl15          # Convert L1-L5 geometry-free to STEC
            lat = df['latt']
            lon = df['lonn']
            elev = df['elev']
            hh = df['hh']

            # Compute time differences in seconds between consecutive epochs
            d = 86400.0 * np.diff(t)

            # Determine sampling interval (e.g., 15 or 30 seconds)
            freq = int(np.ceil(np.min(d)))

            # Define time windows (T15 and T60) depending on data frequency
            if freq == 30:
                T15 = 5   # 5 samples = 2.5 minutes at 30s rate
                T60 = 20  # 20 samples = 10 minutes
            elif freq == 15:
                T15 = 15  # 15 samples = 3.75 minutes at 15s rate
                T60 = 60
            else:
                # Unexpected frequency value – set to None and print warning
                T15 = None
                T60 = None
                print(f"Unexpected frequency: {freq}")

            #print(f"T15 = {T15}, T60 = {T60}")

            # Mask zero time intervals (repeated epochs)
            d = ma.masked_values(d, 0.0)

            # Correct small STEC discontinuities (inter-frequency biases or IFB jumps)
            i = np.where(np.append(d.mask, False) == True)[0]
            for j in range(i.size):
                dIFB = stec[i[j] + 1] - stec[i[j]]
                stec[i[j] + 1:] -= dIFB

                dIFB15 = stec15[i[j] + 1] - stec15[i[j]]
                stec15[i[j] + 1:] -= dIFB15

            # Remove corresponding data points from all arrays if gaps exist
            if d.mask.any():
                mask = np.append(~d.mask, True)  # Restore size consistency
                lat = lat[mask]
                lon = lon[mask]
                hh = hh[mask]
                elev = elev[mask]
                t = t[mask]
                stec = stec[mask]
                stec15 = stec15[mask]

            # Define the starting time
            t0 = t[0]

            # Assign each epoch to a window index
            i = np.floor(np.round(86400 * (t - t0)) / WINDOW)

            # Get unique window identifiers
            j = np.unique(i)

            # Initialize accumulators for each time window
            alon = []       # Longitude
            alat = []       # Latitude
            at = []         # Window center time
            ahh = []        # Average IPP height
            DTECf = []      # DTEC L1-L2 result
            DTEC15f = []    # DTEC L1-L5 result
            elev1 = []      # Average elevation
            ROTf = []       # ROT-like parameter (optional or unused)

            # Loop over each unique window index
            for k in range(j.size):
                # Mask to select only data belonging to the current window
                l = ma.masked_values(i, j[k])

                # Only proceed if the window contains more than one observation
                if lon[l.mask].size > 1:
                    # Store averaged values for location, height, time, and elevation
                    alon.append(np.mean(lon[l.mask]))       # Average longitude
                    alat.append(np.mean(lat[l.mask]))       # Average latitude
                    ahh.append(np.mean(hh[l.mask]))         # Average IPP height
                    at.append(np.mean(t[l.mask]))           # Average time (MJD)
                    elev1.append(np.mean(elev[l.mask]))     # Average elevation angle

                    # Convert time from MJD to seconds for rolling operations
                    t_seconds = t[l.mask] * 86400.0

                    # Create a DataFrame with STEC and STEC15 time series
                    df_stec = pd.DataFrame({
                        'timestamp': t_seconds,
                        'stec': stec[l.mask],
                        'stec15': stec15[l.mask]
                    })

                    # Set time (in seconds) as the index for rolling calculations
                    df_stec.set_index('timestamp', inplace=True)

                    # Apply centered rolling average with short (T15) and long (T60) windows for L1-L2
                    stec_15min = df_stec['stec'].rolling(window=T15, min_periods=1, center=True).mean()
                    stec_60min = df_stec['stec'].rolling(window=T60, min_periods=1, center=True).mean()

                    # Apply centered rolling average for L1-L5
                    stec15_15min = df_stec['stec15'].rolling(window=T15, min_periods=1, center=True).mean()
                    stec15_60min = df_stec['stec15'].rolling(window=T60, min_periods=1, center=True).mean()

                    # Compute ΔTEC as the difference between short-term and long-term averages
                    df_stec['dtec'] = stec_15min - stec_60min
                    df_stec['dtec15'] = stec15_15min - stec15_60min

                    # Extract the most recent value from the ΔTEC series as representative
                    DTEC = df_stec['dtec'].values[-1]        # Last ΔTEC value for L1-L2
                    DTEC15 = df_stec['dtec15'].values[-1]    # Last ΔTEC value for L1-L5

                    # Append the results to the respective lists
                    DTECf.append(DTEC)
                    DTEC15f.append(DTEC15)

            # Final assignment of DTEC vectors after processing all windows
            DTEC = DTECf
            DTEC15 = DTEC15f

            ## Convert all accumulators to NumPy arrays
            alon = np.array(alon)
            alat = np.array(alat)
            ahh = np.array(ahh)
            at = np.array(at)           # Timestamps (MJD)
            DTEC = np.array(DTEC)       # ΔTEC L1-L2
            DTEC15 = np.array(DTEC15)   # ΔTEC L1-L5
            elev = np.array(elev1)      # Elevation angles

            # Compute time differences between consecutive windows (in seconds)
            d = 86400.0 * np.diff(at)

            # Mask time gaps greater than GAP2 (e.g., 3600 seconds) to identify arcs
            d = ma.masked_greater_equal(d, GAP2)

            # Indices where time gaps occur
            i = np.where(np.append(d.mask, False) == True)[0]

            # Determine start and end indices of arcs
            i1 = np.append(0, i)                         # Arc start indices
            i2 = np.append(i - 1, alon.size - 1)         # Arc end indices

            # Prepare matrices for fitted values and bounds
            y = np.empty(DTEC.size)      # Fitted ΔTEC L1-L2
            yup = np.empty(DTEC.size)    # Upper bound
            ydown = np.empty(DTEC.size)  # Lower bound

            y15 = np.empty(DTEC15.size)
            yup15 = np.empty(DTEC15.size)
            ydown15 = np.empty(DTEC15.size)

            # Polynomial fitting for each arc to smooth and detect outliers
            for j in range(i1.size):
                # Compute center time for normalization
                tm = np.mean(at[i1[j]:i2[j]+1])

                if (at[i2[j]] - at[i1[j]]) != 0.0:
                    # Normalize time over current arc
                    x = (at[i1[j]:i2[j]+1] - tm) / (at[i2[j]] - at[i1[j]])

                    # Fit polynomial of degree DE (e.g., 3)
                    c = np.polyfit(x, DTEC[i1[j]:i2[j]+1], DE)
                    c15 = np.polyfit(x, DTEC15[i1[j]:i2[j]+1], DE)

                    # Evaluate polynomial fit
                    y[i1[j]:i2[j]+1] = np.polyval(c, x)
                    y15[i1[j]:i2[j]+1] = np.polyval(c15, x)

                    # Compute RMS of residuals
                    rms = np.std(DTEC[i1[j]:i2[j]+1] - y[i1[j]:i2[j]+1])
                    rms15 = np.std(DTEC15[i1[j]:i2[j]+1] - y15[i1[j]:i2[j]+1])
                else:
                    # If arc contains only one epoch, fit is just the value itself
                    y[i1[j]:i2[j]+1] = DTEC[i1[j]:i2[j]+1]
                    y15[i1[j]:i2[j]+1] = DTEC15[i1[j]:i2[j]+1]
                    rms = 0.0
                    rms15 = 0.0

                # Define outlier detection bounds
                yup[i1[j]:i2[j]+1] = y[i1[j]:i2[j]+1] + SIGMA * rms
                ydown[i1[j]:i2[j]+1] = y[i1[j]:i2[j]+1] - SIGMA * rms

                yup15[i1[j]:i2[j]+1] = y15[i1[j]:i2[j]+1] + SIGMA * rms15
                ydown15[i1[j]:i2[j]+1] = y15[i1[j]:i2[j]+1] - SIGMA * rms15

            # Create mask for values exceeding the fit bounds (outliers)
            mask = np.abs(DTEC - y) > (yup - ydown) / 2.0

            # Remove outliers from all arrays
            alatm = alat[~mask]
            alonm = alon[~mask]
            ahhm = ahh[~mask]
            atm = at[~mask]
            DTECm = DTEC[~mask]
            DTECm15 = DTEC15[~mask]
            elevm = elev[~mask]

            # Filter out data with low elevation (below threshold)
            cutoff = np.where(elevm >= elev_angle)

            # Final filtered values (only good elevation and no outliers)
            alat = alatm[cutoff]
            alon = alonm[cutoff]
            ahh = ahhm[cutoff]
            at = atm[cutoff]
            DTEC = DTECm[cutoff]
            DTEC15 = DTECm15[cutoff]
            elev = elevm[cutoff]

            # Prepare dictionary for the current satellite's DTEC results
            satellite_data = {
                'MJD': at,
                'Longitude': alon,
                'Latitude': alat,
                'Height': ahh,
                'Elevation': elev,
                'DTEC': 10 * DTEC,       # Convert to TECU/hour
                'DTEC15': 10 * DTEC15,   # Convert to TECU/hour
                'STA': station,
                'SAT': sat1
            }

            # Store in the list of all satellites processed
            satellites_data.append(satellite_data)

        # Skip system if no data found
        if not satellites_data:
            print(f"No data found for {satx} system. Skipping...")
            continue

        # Concatenate all satellite data dictionaries into a single DataFrame
        concatenated_df = pd.concat([pd.DataFrame(data) for data in satellites_data], ignore_index=True)


        # Compute the mean values of each parameter grouped by MJD (i.e., per epoch)
        df_mean = concatenated_df.groupby('MJD').agg({
            'Longitude': 'mean',
            'Latitude': 'mean',
            'Height': 'mean',
            'Elevation': 'mean',
            'DTEC': 'mean',
            'DTEC15': 'mean',
            'STA': 'first',  # Use first occurrence for station name (constant)
            'SAT': 'first'   # Use first occurrence for satellite (may vary)
        }).reset_index()

        # Build the output file path for saving data
        output_directory = os.path.join(destination_directory)
        file_name = f"{station}_{doy}_{year}_{satx}_DTEC.txt"
        output_file_path = os.path.join(output_directory, file_name)

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Save the full dataset (tab-separated, using NaN placeholder)
        concatenated_df.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')

        # Convert MJD to datetime for plotting
        base_date = datetime(1858, 11, 17)
        concatenated_df['datetime'] = concatenated_df['MJD'].astype(float).apply(lambda x: base_date + timedelta(days=x))

        # Choose color and label depending on satellite system and frequency combination
        if satx == 'G':
            color_dtec = 'navy'
            label_dtec = 'DTEC: L1-L2 (GPS)'
            color_dtec15 = 'blue'
            label_dtec15 = 'DTEC: L1-L5 (GPS)'
        elif satx == 'R':
            color_dtec = 'red'
            label_dtec = 'DTEC: L1-L2 (GLONASS)'
            color_dtec15 = 'orange'
            label_dtec15 = 'DTEC: L2-L3 (GLONASS)'
        else:
            color_dtec = 'gray'
            label_dtec = f'DTEC - {satx}'
            color_dtec15 = 'darkgray'
            label_dtec15 = f'DTEC15 - {satx}'

        # Plot L1-L2 (DTEC)
        plt.scatter(concatenated_df['datetime'], concatenated_df['DTEC'], marker='o', s=30, color=color_dtec, label=label_dtec)

        # Plot L1-L5 or L2-L3 (DTEC15)
        plt.scatter(concatenated_df['datetime'], concatenated_df['DTEC15'], marker='o', s=30, color=color_dtec15, label=label_dtec15)

        # Smoothed line with gap detection (optional)
        xx = df_mean['MJD'].values
        yy = df_mean['DTEC'].values
        threshold = 0.01
        mask = np.diff(xx) > threshold
        xx_gap = np.insert(xx, np.where(mask)[0] + 1, np.nan)
        yy_gap = np.insert(yy, np.where(mask)[0] + 1, np.nan)
        datetime_gap = [base_date + timedelta(days=val) if not np.isnan(val) else np.nan for val in xx_gap]
        #plt.plot(datetime_gap, yy_gap, color='red', linewidth=2, label='Smoothed DTEC')
        # Plot smoothed DTEC line separately for GPS and GLONASS

        # Base date for datetime conversion
        base_date = datetime(1858, 11, 17)

        # Smoothed DTEC (GPS)
        df_gps = df_mean[df_mean['SAT'].str.startswith('G')]
        xx_gps = df_gps['MJD'].values
        yy_gps = df_gps['DTEC'].values
        mask_gps = np.diff(xx_gps) > 0.01
        xx_gap_gps = np.insert(xx_gps, np.where(mask_gps)[0] + 1, np.nan)
        yy_gap_gps = np.insert(yy_gps, np.where(mask_gps)[0] + 1, np.nan)
        datetime_gap_gps = [base_date + timedelta(days=val) if not np.isnan(val) else np.nan for val in xx_gap_gps]
        plt.plot(datetime_gap_gps, yy_gap_gps, color='magenta', linewidth=2, label='Smoothed DTEC (GPS)', zorder=11)

        # Smoothed DTEC (GLONASS)
        df_glonass = df_mean[df_mean['SAT'].str.startswith('R')]
        xx_glonass = df_glonass['MJD'].values
        yy_glonass = df_glonass['DTEC'].values
        mask_glonass = np.diff(xx_glonass) > 0.01
        xx_gap_glonass = np.insert(xx_glonass, np.where(mask_glonass)[0] + 1, np.nan)
        yy_gap_glonass = np.insert(yy_glonass, np.where(mask_glonass)[0] + 1, np.nan)
        datetime_gap_glonass = [base_date + timedelta(days=val) if not np.isnan(val) else np.nan for val in xx_gap_glonass]
        plt.plot(datetime_gap_glonass, yy_gap_glonass, color='gold', linewidth=2, label='Smoothed DTEC (GLONASS)', zorder=12)


        # Configure the x-axis
        hours_fmt = mdates.DateFormatter('%H')
        hour_locator = mdates.HourLocator(interval=2)
        plt.gca().xaxis.set_major_formatter(hours_fmt)
        plt.gca().xaxis.set_major_locator(hour_locator)

        # Labels and appearance
        plt.xlabel('Time (UT)', fontsize=16)
        plt.ylabel('DTEC (TECU/hour)', fontsize=16)
        plt.title(f"Station: {station.upper()}  |  Year: {year}  |  DOY: {doy}", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', linewidth=1, color='gray')
        # Remover duplicatas na legenda
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.tight_layout()

        # Save the figure
        file_name_png = f"{station}_{doy}_{year}_DTEC.png"
        output_file_path_png = os.path.join(output_directory, file_name_png)
        plt.savefig(output_file_path_png, dpi=300)
    if show_plot:
        plt.show()
