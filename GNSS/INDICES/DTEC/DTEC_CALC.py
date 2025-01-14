#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import numpy.ma as ma
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import gnss_freqs

# Arguments from the management program
station = sys.argv[1]
doy = sys.argv[2]
year = sys.argv[3]
input_folder = sys.argv[4]
destination_directory = sys.argv[5]

# Variables and Parameters
h1 = 0
n_hours = 24  # horas
int1 = 320  # minutos

# Accessing the frequencies of the GPS system
gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
f1 = gps_freqs[1]
f2 = gps_freqs[2]
f5 = gps_freqs[5]

akl = 40.3 * 10 ** 16 * ((1 / f2 ** 2) - (1 / f1 ** 2))
akl15 = 40.3 * 10 ** 16 * ((1 / f5 ** 2) - (1 / f1 ** 2))

# Adjustable window to calculate averages, deviations, etc., in seconds...
# Defines the sampling frequency of the calculated index...
# Here it is defined as 2.5 minutes (following Giorgio's method)...
WINDOW = 60.0 * 2.5

# Standard unit of time so that DTEC remains in TECU/min
TDTEC = 60.0

# Missing data due to successive passes or prolonged outages caused by obstructions, etc...
GAP = 1.5 * WINDOW

# Layer height
H = 450000  # m

# Tolerance for outlier detection...
SIGMA = 5

# Degree of polynomials
DE = 3

# Separation between arcs (i.e., passes)
GAP2 = 3600  # s

# Elevation angle (cutoff):
elev_angle=30

# Building the full path to the folder
path_ = os.path.join(input_folder, station)
# print(path_)
# Checking if the directory exists
if os.path.exists(path_):
    # Listing the content of the folder
    content_ = os.listdir(path_)
    # print("Text files in the folder:")

    # Defining the files variable according to the content of the folder
    files = [file for file in content_ if file.startswith(station) and file.endswith(".RNX3")]

    # Sorting the files by satellite number
    ord_files = sorted(files, key=lambda x: int(x.split("_")[1][1:]))

    print()
    # Counting the number of files
    number_of_files = len(ord_files)
    print("Number of RINEX_SCREENED files in the directory:", number_of_files)

else:
    print("The specified directory does not exist.")

print()

date = []
time2 = []
mjd = []
pos_x = []
pos_y = []
pos_z = []
LGF_combination = []
LGF_combination15 = []
satellites = []
sta = []
hght = []
el = []
lonn = []
latt = []
obs_La = []
obs_Lb = []
obs_Lc = []
obs_Ca = []
obs_Cb = []
obs_Cc = []

for file_ in ord_files:
    path_file_ = os.path.join(path_, file_)

    with open(path_file_, 'r') as f:
        # Reading the file header
        header = f.readline().strip().split('\t')

        # Reading each data line from the file
        for line in f:
            # Splitting the line into columns
            columns = line.strip().split('\t')  # Assuming the columns are separated by tabs (\t)
            # Associating each column with the corresponding header

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

            # Adding the values of each variable to the respective lists
            # timestamp.append(record['timestamp'])
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



# Creates a single figure
plt.figure(figsize=(12, 6))

# Defining the color palette
palette = plt.get_cmap('tab10')

#G: GPS, R: GLONASS
sat_classes = ['G','R']

for sat in sat_classes:
    satx = sat
    # Filtering the satellites
    if satx:
        satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
    else:
        satellites_to_plot = np.unique(satellites)

    # Initializing a list to store the DTEC values of all satellites
    # List to store the data dictionaries of each satellite
    satellites_data = []

    for sat1 in satellites_to_plot:
        # print(sat1)
        indices = np.where(np.array(satellites) == sat1)[0]

        # Initializing filtered lists for each satellite
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


        df = pd.DataFrame(data)



        # Converting the 'date' and 'time' columns to datetime type and then concatenating
        df['timestamp'] = df['date'] + ' ' + df['time']

        # Converting the 'timestamp' column to datetime type, if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Converting relevant columns to float
        columns_to_convert = ['LGF', 'LGF15', 'mjd', 'lonn', 'latt', 'hh', 'elev']
        df[columns_to_convert] = df[columns_to_convert].astype(float)

        df.replace(-999999.999, np.nan, inplace=True)


        # L1-L2
        t = df['mjd']
        stec = df['LGF'] / akl
        stec15 = df['LGF15'] / akl15
        lat = df['latt']
        lon = df['lonn']
        elev = df['elev']
        hh = df['hh']


        # difference between consecutive epochs...
        d = 86400.0 * np.diff(t)

        # Getting the minimum value of d and rounding it up
        freq = int(np.ceil(np.min(d)))

        # Defining time windows (in seconds) based on the value of freq
        if freq == 30:
            T15 = 5   # 15 minutes to 15 seconds, 5 for 30
            T60 = 20  # 60 minutes
        elif freq == 15:
            T15 = 15  # 15 minutes
            T60 = 60  # 60 minutes
        else:
            # In case freq has an unexpected value, you can define a default behavior or raise an error
            T15 = None
            T60 = None
            print(f"Unexpected frequency: {freq}")


        print(f"T15 = {T15}, T60 = {T60}")

        # create mask because at the 15-minute boundary repeated epochs appear...
        d = ma.masked_values(d, 0.0)

        # search for and correct residual IFB "jumps" (between adjustments)...
        i = np.where(np.append(d.mask, False) == True)[0]
        for j in range(i.size):
            dIFB = stec[i[j] + 1] - stec[i[j]]
            stec[i[j] + 1:] = stec[i[j] + 1:] - dIFB

            dIFB15 = stec15[i[j] + 1] - stec15[i[j]]
            stec15[i[j] + 1:] = stec15[i[j] + 1:] - dIFB15


        # If at least one data point had to be masked (i.e., at least one repeated epoch)...
        if (d.mask.any()):
            lat = lat[np.append(~ d.mask, True)]
            lon = lon[np.append(~ d.mask, True)]
            hh = hh[np.append(~ d.mask, True)]
            # az = az[np.append(~ d.mask, True)]
            t = t[np.append(~ d.mask, True)]
            stec = stec[np.append(~ d.mask, True)]
            stec15 = stec15[np.append(~ d.mask, True)]
            elev = elev[np.append(~ d.mask, True)]


        # First obvervation...
        t0 = t[0]

        # index of data in each window...
        i = np.floor(np.round(86400*(t - t0)) / WINDOW)

        # windows with data...
        j = np.unique(i)

        # accumulators...
        alon = []
        alat = []
        at = []
        ahh = []
        # aaz = []
        DTECf = []
        DTEC15f = []
        elev1 = []
        ROTf = []


        for k in range(j.size):
            l = ma.masked_values(i, j[k])

            # Only calculate if there are at least two observations in the window
            if lon[l.mask].size > 1:
                alon.append(np.mean(lon[l.mask]))
                alat.append(np.mean(lat[l.mask]))
                ahh.append(np.mean(hh[l.mask]))
                at.append(np.mean(t[l.mask]))
                elev1.append(np.mean(elev[l.mask]))

                # Converting time to seconds (since t is MJD)
                t_seconds = t[l.mask] * 86400.0  # Converting MJD to seconds

                # Create a DataFrame with time and STEC
                df_stec = pd.DataFrame({
                    'timestamp': t_seconds,
                    'stec': stec[l.mask],
                    'stec15': stec15[l.mask]
                })

                # Set the timestamp as the index
                df_stec.set_index('timestamp', inplace=True)

                # Calculate the moving average of STEC with 15-minute and 60-minute windows
                stec_15min = df_stec['stec'].rolling(window=T15, min_periods=1, center=True).mean()
                stec_60min = df_stec['stec'].rolling(window=T60, min_periods=1, center=True).mean()

                # Calculate the moving average of STEC15 with 15-minute and 60-minute windows
                stec15_15min = df_stec['stec15'].rolling(window=T15, min_periods=1, center=True).mean()
                stec15_60min = df_stec['stec15'].rolling(window=T60, min_periods=1, center=True).mean()

                # Calculate DTEC as the difference between the 15-minute and 60-minute moving averages
                df_stec['dtec'] = stec_15min - stec_60min
                df_stec['dtec15'] = stec15_15min - stec15_60min

                # Store the DTEC value in the array (replacing DTEC)
                DTEC = df_stec['dtec'].values[-1]  # Last calculated ΔTEC value
                DTEC15 = df_stec['dtec15'].values[-1]  # Last calculated ΔTEC value
                DTECf.append(DTEC)  # Replacing DTEC with DTEC
                DTEC15f.append(DTEC15)  # Replacing DTEC with DTEC


        DTEC = DTECf
        DTEC15 = DTEC15f


        ## Matrices..
        alon = np.array(alon)
        alat = np.array(alat)
        ahh = np.array(ahh)
        # aaz = np.array(aaz)
        at = np.array(at)
        DTEC = np.array(DTEC)
        DTEC15 = np.array(DTEC15)
        elev = np.array(elev1)

        # Difference between consecutive epochs...
        d = 86400.0 * np.diff(at)

        # Mask to find independent arcs...
        d = ma.masked_greater_equal(d, GAP2)

        # Where do the time gaps occur?
        i = np.where(np.append(d.mask, False) == True)[0]

        # Where do the arcs start?
        i1 = np.append(0, np.where(np.append(d.mask, False) == True)[0])
        # Where do the arcs end?
        i2 = np.append(np.where(np.append(d.mask, False) == True)[0] - 1, alon.size)

        # Matrices for the fitted polynomials and the upper and lower limits...
        y = np.empty((DTEC.size,))
        yup = np.empty((DTEC.size,))
        ydown = np.empty((DTEC.size,))

        y15 = np.empty((DTEC15.size,))
        yup15 = np.empty((DTEC15.size,))
        ydown15 = np.empty((DTEC15.size,))

        # Filtering for "spikes" due to different "intervals" leveled with constant ambiguity, whose limits are not yet exposed by OASIS...
        for j in range(i1.size):
            # Mean epoch
            tm = np.mean(at[i1[j]:i2[j]])

            if (at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0]) != 0.0:
                # Normalized independent variable
                x = (at[i1[j]:i2[j]] - tm)/(at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0])

                # Polynomial fitting
                c = np.polyfit(x,DTEC[i1[j]:i2[j]],DE)
                c15 = np.polyfit(x,DTEC15[i1[j]:i2[j]],DE)

                # Polynomial evaluation
                y[i1[j]:i2[j]] = np.polyval(c,x)
                y15[i1[j]:i2[j]] = np.polyval(c15,x)

                # RMS of the residuals
                rms = np.std(DTEC[i1[j]:i2[j]] - y[i1[j]:i2[j]])
                rms15 = np.std(DTEC15[i1[j]:i2[j]] - y15[i1[j]:i2[j]])
            else:
                y[i1[j]:i2[j]] = DTEC[i1[j]:i2[j]]
                rms = 0.0

                y15[i1[j]:i2[j]] = DTEC15[i1[j]:i2[j]]
                rms15 = 0.0

            # Upper and lower limits
            yup[i1[j]:i2[j]] = y[i1[j]:i2[j]] + SIGMA*rms
            ydown[i1[j]:i2[j]] = y[i1[j]:i2[j]] - SIGMA*rms

            yup15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] + SIGMA*rms15
            ydown15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] - SIGMA*rms15


        # Create mask
        mask = np.abs(DTEC - y) > (yup - ydown)/2.0

        # Discard the masked values (the outliers)...
        alatm = alat[~ mask]
        alonm = alon[~ mask]
        ahhm = ahh[~ mask]
        # aazm = aaz[~ mask]
        atm = at[~ mask]
        DTECm = DTEC[~ mask]
        DTECm15 = DTEC15[~ mask]
        elevm = elev[~ mask]


        cutoff = np.where(elevm>=elev_angle)

        alat = alatm[cutoff]
        alon = alonm[cutoff]
        ahh = ahhm[cutoff]
        # aaz = aazm[cutoff]
        at = atm[cutoff]
        DTEC = DTECm[cutoff]
        DTEC15 = DTECm15[cutoff]
        elev = elevm[cutoff]


        # Initialize a dictionary to store the data for this satellite
        satellite_data = {
            'MJD': at,
            'Longitude': alon,
            'Latitude': alat,
            'Height': ahh,
            'Elevation': elev,
            'DTEC': 10*DTEC,
            'DTEC15': 10*DTEC15,
            'STA': station,
            'SAT': sat1
        }

        # Add the satellite data dictionary to the list
        satellites_data.append(satellite_data)

    # Concatenate all the data dictionaries into a DataFrame
    concatenated_df = pd.concat([pd.DataFrame(data) for data in satellites_data], ignore_index=True)


    # Group by MJD and calculate the mean of DTEC and other numeric columns
    df_mean = concatenated_df.groupby('MJD').agg({
        'Longitude': 'mean',
        'Latitude': 'mean',
        'Height': 'mean',
        'Elevation': 'mean',
        'DTEC': 'mean',
        'DTEC15': 'mean',
        'STA': 'first',  # For non-numeric columns, take the first value (if desired)
        'SAT': 'first'   # Take the first value for each epoch
    }).reset_index()

    output_directory = os.path.join(destination_directory, station.upper())
    full_path = output_directory
    file_name = f"{station}_{doy}_{year}_{satx}_DTEC.txt"
    output_file_path = os.path.join(full_path, file_name)

    # Ensure the directory exists
    os.makedirs(full_path, exist_ok=True)

    # Save the selected DataFrame to a tab-separated text file
    concatenated_df.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')

    # Plotting DTEC for the current satellite with a color from the palette
    color = palette(idx / len(satellites_to_plot))
    plt.scatter(concatenated_df['MJD'], concatenated_df['DTEC'], marker='s', s=30, color='blue')
    plt.scatter(concatenated_df['MJD'], concatenated_df['DTEC15'], marker='s', s=30, color='blue')

    # Set the data
    xx = df_mean['MJD'].values
    yy = df_mean['DTEC'].values

    # Find gaps (e.g., where the deltas between points are larger than a threshold value)
    # We'll consider a gap if the difference is greater than a certain threshold
    threshold = 0.01  # Adjust as needed
    mask = np.diff(xx) > threshold

    # Create a new variable for the gap points, with gap values as NaN
    xx_gap = np.insert(xx, np.where(mask)[0] + 1, np.nan)
    yy_gap = np.insert(yy, np.where(mask)[0] + 1, np.nan)

    # Plot the points and lines connecting only where there is continuous data
    plt.plot(xx_gap, yy_gap, color='red', label='DTEC', linewidth=2)  # Line connecting the points

    # # Converting the concatenated DTEC list to a numpy array
    # DTEC_concatenated = np.array(DTEC_concatenated)

    # Assuming 'mjd' is a list of MJD values in string format
    # Convert MJD values to datetime objects
    start_time_mjd = min(map(float, mjd))
    start_time_datetime = datetime(1858, 11, 17) + timedelta(days=start_time_mjd)
    datetimes = [start_time_datetime + timedelta(days=float(at_val)) for at_val in mjd]

    # Set the x-axis format to display only hours and minutes
    hours_fmt = mdates.DateFormatter('%H')

    # Set the hour locator to a 2-hour interval
    hour_locator = mdates.HourLocator(interval=2)

    # Configure the plot
    plt.gca().xaxis.set_major_formatter(hours_fmt)
    plt.gca().xaxis.set_major_locator(hour_locator)

    # Increase the size of x-axis labels
    plt.xticks(fontsize=14)

    # Set y-axis label
    plt.ylabel('DTEC (TECU)', fontsize=16)

    # Set x-axis label
    plt.xlabel('Time (UT)', fontsize=16)

    # Increase the size of x-axis labels
    plt.xticks(fontsize=14)

    # Increase the size of y-axis labels
    plt.yticks(fontsize=14)

    # Set plot title
    plt.title(rf"GNSS Station: {station} ({doy}/{year})", fontsize=18)


    #plt.ylim(0,5)
    plt.grid(True)
    plt.tight_layout()

    file_name_png = f"{station}_{doy}_{year}_DTEC.png"
    output_file_path_png = os.path.join(full_path, file_name_png)

    # # Show the plot
    # plt.show()

    plt.gca().xaxis.set_major_formatter(hours_fmt)

    # Save the figure with 300 DPI
    plt.savefig(output_file_path_png, dpi=300)




