import os
import sys
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pyOASIS import gnss_freqs

def SIDXcalc(station, doy, year, input_folder, destination_directory, show_plot=True):
    """
    Compute SIDX (Slant Ionospheric Index) for a given GNSS station and day-of-year.

    Parameters:
        station (str): GNSS station code (e.g., BOAV)
        doy (int or str): Day of year
        year (int): Observation year
        input_folder (str): Directory containing input files
        destination_directory (str): Output directory for results
    """

    # Basic parameters
    h1 = 0                      # Starting hour
    n_hours = 24                # Number of hours to process
    int1 = 320                  # Internal processing interval in minutes

    # Access GPS frequencies from the pyOASIS configuration
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]           # L1 frequency
    f2 = gps_freqs[2]           # L2 frequency
    f5 = gps_freqs[5]           # L5 frequency

    # Geometry-Free scaling factors (in mTECU)
    akl = 40.3e16 * ((1 / f2**2) - (1 / f1**2))      # For G:L1-L2 or R:L1-L2
    akl15 = 40.3e16 * ((1 / f5**2) - (1 / f1**2))    # For G:L1-L5 or R:L2-L3

    WINDOW = 60.0   # Processing window in seconds (1-min for SIDX)
    TSIDX = 60.0 # Normalization time unit to keep SIDX in mTECU/sec
    GAP = 1.5 * WINDOW  # Maximum gap duration (s) allowed between valid observations
    H = 450000 # Ionospheric shell height for IPP (meters)
    SIGMA = 5 # Sigma threshold for outlier removal (in units of standard deviation)
    DE = 3 # Polynomial degree used for smoothing and fitting
    GAP2 = 3600 # Minimum time separation between arcs (in seconds)
    elev_angle = 20 # Elevation angle cutoff (in degrees) to filter low-angle observations

    # Build the full path to the station directory
    path_ = os.path.join(input_folder, station)
    print(path_)

    # Check if the station directory exists
    if os.path.exists(path_):
        # List all files in the directory
        content_ = os.listdir(path_)

        # Filter files: keep only those that start with the station name and end with ".RNX3"
        files = [file for file in content_ if file.startswith(station) and file.endswith(".RNX3")]

        # Sort files by satellite number (e.g., G01, G02, R01...)
        ord_files = sorted(files, key=lambda x: int(x.split("_")[1][1:]))

        # Print the number of valid RINEX3 files found
        print()
        number_of_files = len(ord_files)
        print("Number of RINEX_SCREENED files in the directory:", number_of_files)
    else:
        print("The specified directory does not exist.")
    print()

    # Initialize empty lists to store the extracted GNSS data
    date = []                # Date (YYYY-MM-DD)
    time2 = []               # Time (HH:MM:SS)
    mjd = []                 # Modified Julian Date
    pos_x = []               # Receiver X position (meters)
    pos_y = []               # Receiver Y position (meters)
    pos_z = []               # Receiver Z position (meters)
    LGF_combination = []     # Geometry-Free combination G:L1-L2 (or R:L1-L2 ...)
    LGF_combination15 = []   # Geometry-Free combination G:L1-L5 (or R:L2-L3 ...)
    satellites = []          # Satellite ID (e.g., G01, R08)
    sta = []                 # Station code
    hght = []                # IPP height
    el = []                  # Elevation angle (degrees)
    lonn = []                # IPP longitude
    latt = []                # IPP latitude
    obs_La = []              # GL1/RL1 carrier phase
    obs_Lb = []              # GL2/RL2 carrier phase
    obs_Lc = []              # GL5/RL3 carrier phase
    obs_Ca = []              # GC1/RC1 pseudorange
    obs_Cb = []              # GC2/RC2 pseudorange
    obs_Cc = []              # GC5/RC3 pseudorange

    # Loop through each .RNX3 file
    for file in ord_files:
        path_file = os.path.join(path_, file)

        with open(path_file, 'r') as f:
            # Read the header line to get column names
            header = f.readline().strip().split('\t')

            # Read each subsequent data line
            for line in f:
                # Split line by tab separator
                columns = line.strip().split('\t')

                # Map each column to its corresponding variable
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

    # Create a figure for plotting the results
    plt.figure(figsize=(12, 6))

    # Define the color palette for plotting
    palette = plt.get_cmap('tab10')

    # GNSS constellations to process: G = GPS, R = GLONASS
    sat_classes = ['G', 'R']

    # Loop through each GNSS constellation class
    for sat in sat_classes:
        label_plotted = False  # Used to avoid repeating legend entries
        satx = sat

        # Filter satellites belonging to the current class (e.g., all GPS satellites)
        if satx:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        else:
            satellites_to_plot = np.unique(satellites)

        # Initialize a list to store data for all satellites in this class
        satellites_data = []

        # Loop through each individual satellite (e.g., G01, G02, R01...)
        for sat1 in satellites_to_plot:
            # Get the indices in the full list where this satellite appears
            indices = np.where(np.array(satellites) == sat1)[0]

            # Initialize filtered lists for this satellite
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

            # Append the data for the current satellite
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

            # Organize the filtered data into a dictionary for DataFrame construction
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

            # Convert the dictionary to a pandas DataFrame for analysis
            df = pd.DataFrame(data)
    
            # Combine 'date' and 'time' columns to form full timestamps
            df['timestamp'] = df['date'] + ' ' + df['time']

            # Convert the 'timestamp' column to datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Convert selected columns to float type for numerical processing
            columns_to_convert = ['LGF', 'LGF15', 'mjd', 'lonn', 'latt', 'hh', 'elev']
            df[columns_to_convert] = df[columns_to_convert].astype(float)

            # Replace invalid values with NaN
            df.replace(-999999.999, np.nan, inplace=True)

            # Assign relevant columns for further processing
            t = df['mjd']                           # Time in Modified Julian Date
            stec = df['LGF'] / akl                  # Convert LGF to Slant TEC (L1-L2)
            stec15 = df['LGF15'] / akl15            # Convert LGF15 to Slant TEC (L1-L5)
            lat = df['latt']                        # Latitude of IPP
            lon = df['lonn']                        # Longitude of IPP
            elev = df['elev']                       # Elevation angle
            hh = df['hh']                           # IPP height

            # Compute time differences between epochs (in seconds)
            d = 86400.0 * np.diff(t)

            # Mask repeated epochs (common at interval boundaries like 15 min)
            d = ma.masked_values(d, 0.0)

            # Detect and correct small STEC discontinuities (IFB "jumps")
            i = np.where(np.append(d.mask, False))[0]
            for j in range(i.size):
                dIFB = stec[i[j] + 1] - stec[i[j]]
                stec[i[j] + 1:] -= dIFB

                dIFB15 = stec15[i[j] + 1] - stec15[i[j]]
                stec15[i[j] + 1:] -= dIFB15

            # If any masked values (repeated epochs) exist, remove them
            if d.mask.any():
                mask = np.append(~d.mask, True)  # Restore full length
                lat = lat[mask]
                lon = lon[mask]
                hh = hh[mask]
                t = t[mask]
                stec = stec[mask]
                stec15 = stec15[mask]
                elev = elev[mask]

            # Reference time (first valid epoch)
            t0 = t[0]

            # Assign each epoch to a processing window (based on WINDOW size)
            i = np.floor(np.round(86400 * (t - t0)) / WINDOW)

            # Unique windows that contain data
            j = np.unique(i)

            # Accumulators for results across time windows
            alon = []       # Mean longitude per window
            alat = []       # Mean latitude per window
            at = []         # Central time (MJD) per window
            ahh = []        # Mean IPP height
            SIDXf = []      # SIDX (L1-L2) result per window
            SIDX15f = []    # SIDX (L1-L5) result per window
            elev1 = []      # Mean elevation angle per window

            # Compute ROT and SIDX within each window
            for k in range(j.size):
                l = ma.masked_values(i, j[k])

                # Proceed only if there are at least two observations in the window
                if lon[l.mask].size > 1:
                    # Store average geolocation and elevation per window
                    alon.append(np.mean(lon[l.mask]))
                    alat.append(np.mean(lat[l.mask]))
                    ahh.append(np.mean(hh[l.mask]))
                    at.append(np.mean(t[l.mask]))
                    elev1.append(np.mean(elev[l.mask]))

                    # Compute Rate of TEC (ROT) in TECU/sec
                    ROT = np.divide(np.diff(stec[l.mask]), 86400.0 * np.diff(t[l.mask]) / TSIDX)

                    # Compute SIDX: mean absolute ROT in the window
                    SIDX = np.mean(np.abs(ROT))

                    # Store SIDX for this window
                    SIDXf.append(SIDX)

            # Convert accumulators to NumPy arrays
            alon = np.array(alon)
            alat = np.array(alat)
            ahh = np.array(ahh)
            at = np.array(at)              # Mean MJD per window
            SIDX = np.array(SIDXf)         # L1-L2 SIDX
            SIDX15 = np.array(SIDXf)       # L1-L5 SIDX
            elev = np.array(elev1)

            # Compute time differences between windows (seconds)
            d = 86400.0 * np.diff(at)

            # Identify independent arcs separated by gaps larger than GAP2
            d = ma.masked_greater_equal(d, GAP2)

            # Indices where time gaps occur
            i = np.where(np.append(d.mask, False))[0]

            # Start and end indices of each arc
            i1 = np.append(0, i)
            i2 = np.append(i - 1, alon.size - 1)

            # Prepare matrices for polynomial fits and outlier thresholds
            y = np.empty(SIDX.size)
            yup = np.empty(SIDX.size)
            ydown = np.empty(SIDX.size)

            y15 = np.empty(SIDX15.size)
            yup15 = np.empty(SIDX15.size)
            ydown15 = np.empty(SIDX15.size)

            # Loop over each arc to fit a polynomial and compute residuals
            for j in range(i1.size):
                # Center time of current arc
                tm = np.mean(at[i1[j]:i2[j]+1])

                # Check if arc contains more than one time step
                if (at[i2[j]] - at[i1[j]]) != 0.0:
                    # Normalize time variable for polynomial fitting
                    x = (at[i1[j]:i2[j]+1] - tm) / (at[i2[j]] - at[i1[j]])

                    # Fit polynomials of degree DE
                    c = np.polyfit(x, SIDX[i1[j]:i2[j]+1], DE)
                    c15 = np.polyfit(x, SIDX15[i1[j]:i2[j]+1], DE)

                    # Evaluate polynomial fit
                    y[i1[j]:i2[j]+1] = np.polyval(c, x)
                    y15[i1[j]:i2[j]+1] = np.polyval(c15, x)

                    # Compute RMS of residuals
                    rms = np.std(SIDX[i1[j]:i2[j]+1] - y[i1[j]:i2[j]+1])
                    rms15 = np.std(SIDX15[i1[j]:i2[j]+1] - y15[i1[j]:i2[j]+1])
                else:
                    # For single-point arcs, no fitting is performed
                    y[i1[j]:i2[j]+1] = SIDX[i1[j]:i2[j]+1]
                    rms = 0.0
                    y15[i1[j]:i2[j]+1] = SIDX15[i1[j]:i2[j]+1]
                    rms15 = 0.0

                # Define upper and lower bounds for outlier detection
                yup[i1[j]:i2[j]+1] = y[i1[j]:i2[j]+1] + SIGMA * rms
                ydown[i1[j]:i2[j]+1] = y[i1[j]:i2[j]+1] - SIGMA * rms

                yup15[i1[j]:i2[j]+1] = y15[i1[j]:i2[j]+1] + SIGMA * rms15
                ydown15[i1[j]:i2[j]+1] = y15[i1[j]:i2[j]+1] - SIGMA * rms15

             # Create a mask to identify outliers based on polynomial fit
            mask = np.abs(SIDX - y) > (yup - ydown) / 2.0

            # Remove masked values (outliers)
            alatm = alat[~mask]
            alonm = alon[~mask]
            ahhm = ahh[~mask]
            atm = at[~mask]
            SIDXm = SIDX[~mask]
            SIDXm15 = SIDX15[~mask]
            elevm = elev[~mask]

            # Apply elevation angle cutoff
            cutoff = np.where(elevm >= elev_angle)

            # Final filtered data
            alat = alatm[cutoff]
            alon = alonm[cutoff]
            ahh = ahhm[cutoff]
            at = atm[cutoff]
            SIDX = SIDXm[cutoff]
            SIDX15 = SIDXm15[cutoff]
            elev = elevm[cutoff]

            # Optional: print SIDX results to stdout (commented out)
            # for i in range(SIDX.size):
            #     print("%13.8f %13.8f %13.8f %13.8f %16.10f %6.3f %s %s" %
            #           (alon[i], alat[i], ahh[i], at[i], elev[i], SIDX[i], station, sat1))

            # Store the cleaned and processed results for the current satellite
            satellite_data = {
                'MJD': at,
                'Longitude': alon,
                'Latitude': alat,
                'Height': ahh,
                'Elevation': elev,
                'SIDX': SIDX,
                'SIDX15': SIDX15,
                'STA': station,
                'SAT': sat1
            }

            # Append this satellite's data to the full list
            satellites_data.append(satellite_data)

        # Combine the data from all satellites into a single DataFrame
        concatenated_df = pd.concat([pd.DataFrame(dados) for dados in satellites_data], ignore_index=True)

        # Group data by MJD to compute mean values across all satellites for each epoch
        df_mean = concatenated_df.groupby('MJD').agg({
            'Longitude': 'mean',
            'Latitude': 'mean',
            'Height': 'mean',
            'Elevation': 'mean',
            'SIDX': 'mean',
            'SIDX15': 'mean',
            'STA': 'first',  # Use the first non-numeric entry (station)
            'SAT': 'first'   # Use the first satellite ID
        }).reset_index()

        # Define output file path
        output_directory = os.path.join(destination_directory, station.upper())
        full_path = output_directory
        file_name = f"{station}_{doy}_{year}_{satx}_SIDX.txt"
        output_file_path = os.path.join(full_path, file_name)

        # Create output directory if it doesn't exist
        os.makedirs(full_path, exist_ok=True)

        # Save the full dataset to a .txt file (tab-separated, using -999999.999 for missing values)
        concatenated_df.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')

        # Sort data by MJD to prepare for plotting
        concatenated_df = concatenated_df.sort_values(by='MJD')

        # Define fixed color mapping for satellite systems
        color_map = {'G': 'blue', 'R': 'red'}

        # Convert MJD to datetime for plotting
        base_date = datetime(1858, 11, 17)
        concatenated_df['datetime'] = concatenated_df['MJD'].astype(float).apply(lambda x: base_date + timedelta(days=x))

        # Define colors and labels for each GNSS system
        if satx == 'G':
            color_sidx = 'navy'
            label_sidx = 'SIDX: L1-L2 (GPS)'
            color_sidx15 = 'blue'
            label_sidx15 = 'SIDX: L1-L5 (GPS)'
        elif satx == 'R':
            color_sidx = 'red'
            label_sidx = 'SIDX: L1-L2 (GLONASS)'
            color_sidx15 = 'orange'
            label_sidx15 = 'SIDX: L2-L3 (GLONASS)'
        else:
            color_sidx = 'gray'
            label_sidx = f'SIDX - {satx}'
            color_sidx15 = 'darkgray'
            label_sidx15 = f'SIDX15 - {satx}'

        # Plot raw SIDX values
        #plt.scatter(concatenated_df['datetime'], concatenated_df['SIDX'], marker='o', s=30, color=color_sidx, label=label_sidx)
        #plt.scatter(concatenated_df['datetime'], concatenated_df['SIDX15'], marker='o', s=30, color=color_sidx15, label=label_sidx15)

        # SIDX from GPS or GLONASS is averaged only when data from multiple satellites are available at the same time epochs, minimizing noise and inter-satellite biases.
        base_date = datetime(1858, 11, 17)

        # GPS
        df_gps = df_mean[df_mean['SAT'].str.startswith('G')]
        xx_gps = df_gps['MJD'].values
        yy_gps = df_gps['SIDX'].values
        mask_gps = np.diff(xx_gps) > 0.01
        xx_gap_gps = np.insert(xx_gps, np.where(mask_gps)[0] + 1, np.nan)
        yy_gap_gps = np.insert(yy_gps, np.where(mask_gps)[0] + 1, np.nan)
        datetime_gap_gps = [base_date + timedelta(days=val) if not np.isnan(val) else np.nan for val in xx_gap_gps]
        plt.plot(datetime_gap_gps, yy_gap_gps * 10, color='blue', linewidth=2, label='SIDX (GPS)', zorder=11)

        # GLONASS
        df_glonass = df_mean[df_mean['SAT'].str.startswith('R')]
        xx_glonass = df_glonass['MJD'].values
        yy_glonass = df_glonass['SIDX'].values
        mask_glonass = np.diff(xx_glonass) > 0.01
        xx_gap_glonass = np.insert(xx_glonass, np.where(mask_glonass)[0] + 1, np.nan)
        yy_gap_glonass = np.insert(yy_glonass, np.where(mask_glonass)[0] + 1, np.nan)
        datetime_gap_glonass = [base_date + timedelta(days=val) if not np.isnan(val) else np.nan for val in xx_gap_glonass]
        plt.plot(datetime_gap_glonass, yy_gap_glonass * 10, color='red', linewidth=2, label='SIDX (GLONASS)', zorder=12)

        # Configure time axis (x-axis)
        hours_fmt = mdates.DateFormatter('%H')
        hour_locator = mdates.HourLocator(interval=2)
        plt.gca().xaxis.set_major_formatter(hours_fmt)
        plt.gca().xaxis.set_major_locator(hour_locator)

        # Plot styling
        plt.xlabel('Time (UT)', fontsize=16)
        plt.ylabel('SIDX (mTECU/sec)', fontsize=16)
        plt.title(f"GNSS Station: {station.upper()} | Year: {year} | DOY: {doy}", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', linewidth=1, color='gray')

        # Remove duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=13)
        plt.tight_layout()

        # Save figure
        file_name_png = f"{station}_{doy}_{year}_SIDX.png"
        output_file_path_png = os.path.join(full_path, file_name_png)
        plt.savefig(output_file_path_png, dpi=300)
    if show_plot:
        plt.show()

