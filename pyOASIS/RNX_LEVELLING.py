import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
from scipy.constants import speed_of_light
from pyOASIS import gnss_freqs
from pyOASIS import levelling_settings
import pyOASIS

def RNXlevelling(estacao, diretorio_principal, show_plot=True):

    # Suppress all error and warning messages by redirecting stderr to the system null device
    sys.stderr = open(os.devnull, 'w')

    # Variables and Parameters
    h1 = 0
    n_horas = 24  # hours
    int1 = 120  # minutes

    # Accessing the frequencies of the GPS system
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]
    f2 = gps_freqs[2]
    f5 = gps_freqs[5]

    # Define the file name
    file_name = pyOASIS.__path__[0] + '/glonass_channels.dat'  # option without slashes

    # Read the file into a DataFrame with defined column names
    df_slots = pd.read_csv(file_name, sep=' ', header=None, names=['Slot', 'Channel'])
    glonass_frequencies = gnss_freqs.FREQUENCY[gnss_freqs.GLO]

    # List to store the data
    data = []

    # Iterate over each row in the DataFrame
    for index, row in df_slots.iterrows():
        satellite = row['Slot']
        k = row['Channel']
        row_data = [satellite]

        for channel, frequency in glonass_frequencies.items():
            if callable(frequency):  # Check if it's a lambda function
                freq_value = frequency(k)
            else:
                freq_value = frequency
            formatted_freq = f"{freq_value:.1f}"
            row_data.append(formatted_freq)

        # Add row data to the data list
        data.append(row_data)

    # Convert the list of lists into a pandas DataFrame
    glo_freqs = pd.DataFrame(data, columns=['Satellite', 'fr1', 'fr2', 'fr3'])

    # Building the full path to the folder
    caminho_ = os.path.join(diretorio_principal)

    # Checking if the directory exists
    if os.path.exists(caminho_):
        conteudo_ = os.listdir(caminho_)

        # Defining the 'arquivos' variable to include only files ending with .RNX2
        arquivos = [arquivo for arquivo in conteudo_ if arquivo.endswith(".RNX2")]

        first = arquivos[0]
        doy = first[9:12]
        ano = first[13:17]

        # Sorting files by satellite number
        arquivos_ordenados = sorted(arquivos, key=lambda x: int(x.split("_")[1][1:]))

        # Printing the found files
        for arquivo in arquivos_ordenados:
            print("File:", arquivo)

        print()
        # Counting the number of files
        numero_de_arquivos = len(arquivos_ordenados)
        print("Number of RINEX_SCREENED (.RNX2) files in the directory:", numero_de_arquivos)

    else:
        print("The specified directory does not exist.")

    print()

    # Initializing lists to store values from each variable of all files
    date = []
    time = []
    mjd = []
    pos_x = []
    pos_y = []
    pos_z = []
    L1 = []
    L2 = []
    L5 = []
    P1 = []
    P2 = []
    P5 = []
    cs_flag = []
    outlier_flag = []
    satellites = []
    sta = []
    hght = []
    El = []
    Lon = []
    Lat = []
    obs_La = []
    obs_Lb = []
    obs_Lc = []
    obs_Ca = []
    obs_Cb = []
    obs_Cc = []

    for arquivo in arquivos:
        caminho_arquivo = os.path.join(caminho_, arquivo)

        with open(caminho_arquivo, 'r') as f:
            header = f.readline().strip().split('\t')

            for linha in f:
                colunas = linha.strip().split('\t')
                registro = {
                    'date': colunas[0],
                    'time': colunas[1],
                    'mjd': colunas[2],
                    'pos_x': colunas[3],
                    'pos_y': colunas[4],
                    'pos_z': colunas[5],
                    'L1': colunas[6],
                    'L2': colunas[7],
                    'L5': colunas[8],
                    'P1': colunas[9],
                    'P2': colunas[10],
                    'P5': colunas[11],
                    'cs_flag': colunas[12],
                    'outlier_flag': colunas[13],
                    'satellite': colunas[14],
                    'sta': colunas[15],
                    'hght': colunas[16],
                    'El': colunas[17],
                    'Lon': colunas[18],
                    'Lat': colunas[19],
                    'obs_La': colunas[20],
                    'obs_Lb': colunas[21],
                    'obs_Lc': colunas[22],
                    'obs_Ca': colunas[23],
                    'obs_Cb': colunas[24],
                    'obs_Cc': colunas[25]
                }

                # Appending each variable to its respective list
                date.append(registro['date'])
                time.append(registro['time'])
                mjd.append(registro['mjd'])
                pos_x.append(registro['pos_x'])
                pos_y.append(registro['pos_y'])
                pos_z.append(registro['pos_z'])
                L1.append(registro['L1'])
                L2.append(registro['L2'])
                L5.append(registro['L5'])
                P1.append(registro['P1'])
                P2.append(registro['P2'])
                P5.append(registro['P5'])
                cs_flag.append(registro['cs_flag'])
                outlier_flag.append(registro['outlier_flag'])
                satellites.append(registro['satellite'])
                sta.append(registro['sta'])
                hght.append(registro['hght'])
                El.append(registro['El'])
                Lon.append(registro['Lon'])
                Lat.append(registro['Lat'])
                obs_La.append(registro['obs_La'])
                obs_Lb.append(registro['obs_Lb'])
                obs_Lc.append(registro['obs_Lc'])
                obs_Ca.append(registro['obs_Ca'])
                obs_Cb.append(registro['obs_Cb'])
                obs_Cc.append(registro['obs_Cc'])

    sat_classes = ['G', 'R']

    # Create a single figure
    plt.figure(figsize=(12, 6))

    # Iterating over sat_class values
    for sat_class in sat_classes:
        sat = sat_class

        if sat:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        else:
            satellites_to_plot = np.unique(satellites)

        # Initializing lists to store the values of each variable from all files
        date_filtered = []
        time_filtered = []
        mjd_filtered = []
        pos_x_filtered = []
        pos_y_filtered = []
        pos_z_filtered = []
        L1_filtered = []
        L2_filtered = []
        L5_filtered = []
        P1_filtered = []
        P2_filtered = []
        P5_filtered = []
        cs_flag_filtered = []
        outlier_flag_filtered = []
        satellite_filtered = []
        sta_filtered = []
        hght_filtered = []
        El_filtered = []
        Lon_filtered = []
        Lat_filtered = []
        obs_La_filtered = []
        obs_Lb_filtered = []
        obs_Lc_filtered = []
        obs_Ca_filtered = []
        obs_Cb_filtered = []
        obs_Cc_filtered = []

        for satellite in satellites_to_plot:
            print()
            print(f"Processing {satellite} satellite...")
            sat = satellite

            # Checking satellite class and adjusting f1, f2, f5 values
            if sat.startswith('G'):
                f1 = f1
                f2 = f2
                f5 = f5
            elif sat.startswith('R'):
                # Locate the row where 'Satellite' matches 'sat'
                sat_row = glo_freqs.loc[glo_freqs['Satellite'] == sat]

                if not sat_row.empty:
                    f1 = float(sat_row['fr1'].values[0])
                    f2 = float(sat_row['fr2'].values[0])
                    f5 = float(sat_row['fr3'].values[0])
            else:
                f1 = f2 = f5 = None  # Or default values

            akl = 40.3 * 10 ** 16 * ((1 / f2 ** 2) - (1 / f1 ** 2))

            lambda1 = (speed_of_light / f1)
            lambda2 = (speed_of_light / f2)
            lambda5 = (speed_of_light / f5)

            indices = np.where(np.array(satellites) == satellite)[0]

            # Initializing filtered lists for each satellite
            date_filtered = []
            time_filtered = []
            mjd_filtered = []
            pos_x_filtered = []
            pos_y_filtered = []
            pos_z_filtered = []
            L1_filtered = []
            L2_filtered = []
            L5_filtered = []
            P1_filtered = []
            P2_filtered = []
            P5_filtered = []
            cs_flag_filtered = []
            outlier_flag_filtered = []
            satellite_filtered = []
            sta_filtered = []
            hght_filtered = []
            El_filtered = []
            Lon_filtered = []
            Lat_filtered = []
            obs_La_filtered = []
            obs_Lb_filtered = []
            obs_Lc_filtered = []
            obs_Ca_filtered = []
            obs_Cb_filtered = []
            obs_Cc_filtered = []

            for idx in indices:
                date_filtered.append(date[idx])
                time_filtered.append(time[idx])
                mjd_filtered.append(mjd[idx])
                pos_x_filtered.append(pos_x[idx])
                pos_y_filtered.append(pos_y[idx])
                pos_z_filtered.append(pos_z[idx])
                L1_filtered.append(L1[idx])
                L2_filtered.append(L2[idx])
                L5_filtered.append(L5[idx])
                P1_filtered.append(P1[idx])
                P2_filtered.append(P2[idx])
                P5_filtered.append(P5[idx])
                cs_flag_filtered.append(cs_flag[idx])
                outlier_flag_filtered.append(outlier_flag[idx])
                satellite_filtered.append(satellites[idx])
                sta_filtered.append(sta[idx])
                hght_filtered.append(hght[idx])
                El_filtered.append(El[idx])
                Lon_filtered.append(Lon[idx])
                Lat_filtered.append(Lat[idx])
                obs_La_filtered.append(obs_La[idx])
                obs_Lb_filtered.append(obs_Lb[idx])
                obs_Lc_filtered.append(obs_Lc[idx])
                obs_Ca_filtered.append(obs_Ca[idx])
                obs_Cb_filtered.append(obs_Cb[idx])
                obs_Cc_filtered.append(obs_Cc[idx])

            # Constructing a DataFrame with the filtered lists
            data = {
                'date': date_filtered,
                'time2': time_filtered,
                'mjd': mjd_filtered,
                'pos_x': pos_x_filtered,
                'pos_y': pos_y_filtered,
                'pos_z': pos_z_filtered,
                'L1': L1_filtered,
                'L2': L2_filtered,
                'L5': L5_filtered,
                'P1': P1_filtered,
                'P2': P2_filtered,
                'P5': P5_filtered,
                'cs_flag': cs_flag_filtered,
                'outlier_flag': outlier_flag_filtered,
                'satellite': satellite_filtered,
                'sta': sta_filtered,
                'hght': hght_filtered,
                'El': El_filtered,
                'Lon': Lon_filtered,
                'Lat': Lat_filtered,
                'obs_La': obs_La_filtered,
                'obs_Lb': obs_Lb_filtered,
                'obs_Lc': obs_Lc_filtered,
                'obs_Ca': obs_Ca_filtered,
                'obs_Cb': obs_Cb_filtered,
                'obs_Cc': obs_Cc_filtered
            }

            df = pd.DataFrame(data)

            # Converting relevant columns to float
            columns_to_convert = ['L1', 'L2', 'L5', 'P1', 'P2', 'P5']
            df[columns_to_convert] = df[columns_to_convert].astype(float)

            # Replace -999999.999 with NaN in relevant columns
            df.replace(-999999.999, np.nan, inplace=True)

            # Converting 'date' and 'time' columns to datetime and concatenating
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time2'])

            # Converting lists to numpy arrays and ensuring float64 data types
            L1_array = np.nan_to_num(np.array(df['L1'].tolist(), dtype=np.float64), nan=-999999.999)
            L2_array = np.nan_to_num(np.array(df['L2'].tolist(), dtype=np.float64), nan=-999999.999)
            L5_array = np.nan_to_num(np.array(df['L5'].tolist(), dtype=np.float64), nan=-999999.999)

            P1_array = np.nan_to_num(np.array(df['P1'].tolist(), dtype=np.float64), nan=-999999.999)
            P2_array = np.nan_to_num(np.array(df['P2'].tolist(), dtype=np.float64), nan=-999999.999)
            P5_array = np.nan_to_num(np.array(df['P5'].tolist(), dtype=np.float64), nan=-999999.999)

            # Replace -999999.999 with NaN in P arrays
            L1_array[L1_array == -999999.999] = np.nan
            L2_array[L2_array == -999999.999] = np.nan
            L5_array[L5_array == -999999.999] = np.nan

            P1_array[P1_array == -999999.999] = np.nan
            P2_array[P2_array == -999999.999] = np.nan
            P5_array[P5_array == -999999.999] = np.nan

            # Find indices where df['outlier_flag'] is 'Y'
            outlier_indices = df.index[df['outlier_flag'] == 'Y'].tolist()

            # Convert these positions to NaN in all arrays
            for index in outlier_indices:
                L1_array[index] = np.nan
                L2_array[index] = np.nan
                L5_array[index] = np.nan
                P1_array[index] = np.nan
                P2_array[index] = np.nan
                P5_array[index] = np.nan

            L_GF = levelling_settings.geometry_free_combination_L(lambda1, lambda2, L1_array, L2_array)
            P_GF = levelling_settings.geometry_free_combination_C(P1_array, P2_array)

            L_GF15 = levelling_settings.geometry_free_combination_L(lambda1, lambda5, L1_array, L5_array)
            P_GF15 = levelling_settings.geometry_free_combination_C(P1_array, P5_array)

            L_GF25 = levelling_settings.geometry_free_combination_L(lambda2, lambda5, L2_array, L5_array)
            P_GF25 = levelling_settings.geometry_free_combination_C(P2_array, P5_array)

            DE = 3
            Thr = DE + 1
            arc_len = 15

            # Get indices where df['outlier_flag'] is equal to 'Y'
            indices_outliers = df.index[df['outlier_flag'] == 'Y']

            # Replace values in df['cs_flag'] with 'S' at the outlier indices
            df.loc[indices_outliers, 'cs_flag'] = 'S'

            # Assuming df['cs_flag'] is a pandas Series
            arcos = []  # List to store observation arcs
            arc_atual = []  # Temporary list to store current observation arc

            # Creating new column 'mini_flag' filled with 'N'
            df['mini_flag'] = 'N'

            # Identifying where 'outlier_flag' or 'cs_flag' are 'S' or 'Y' and replacing 'N' with 'Y' in 'mini_flag'
            mask = (df['outlier_flag'] == 'S') | (df['outlier_flag'] == 'Y') | (df['cs_flag'] == 'S') | (df['cs_flag'] == 'Y')
            df.loc[mask, 'mini_flag'] = 'Y'

            # Iterate over all elements in df['cs_flag']
            for idx, value in enumerate(df['mini_flag']):
                if value == 'Y':
                    # If current value is 'S', check if arc is not empty
                    # This avoids adding empty arcs in case of consecutive 'S'
                    if arc_atual:
                        arcos.append(arc_atual)
                        arc_atual = []  # Reset current arc list
                else:
                    # If value is not 'S', add index to current arc
                    arc_atual.append(idx)

            # Add the last arc if not empty
            if arc_atual:
                arcos.append(arc_atual)

            print()

            # Print info of each arc and classify them
            for i, arc in enumerate(arcos):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                      f"Obs. = {num_observations}, Status = {status}")

            # Filter arcs that meet the length criterion
            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1]+1]) >= 15]

            # Step 1: Create a copy of L_GF
            L_GF2 = np.copy(L_GF)

            # Step 2: Initialize a boolean array with True
            marcador_fora_arco = np.ones_like(L_GF, dtype=bool)

            # Step 3: Mark arc indices as False (not to convert to NaN)
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco[start:end] = False

            # Step 4: Set values outside arcs to NaN
            L_GF2[marcador_fora_arco] = np.nan

            # Update L_GF to be the modified version
            L_GF = L_GF2

            # Reinitialize L_GF2 as a copy of L_GF
            L_GF2 = np.copy(L_GF)

            # Iterate over segments defined in 'arcos'
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Select current segment of L_GF
                segmento_data = L_GF[start:end]

                # Calculate first and third quartiles
                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1

                # Define limits for outliers
                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR

                # Find indices of outliers in the segment
                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))

                # Replace outliers with NaN in L_GF2, not in segmento_data
                L_GF2[start:end][outlier_indices] = np.nan

            # Update L_GF to be the version without outliers
            L_GF = L_GF2

            # First, create a copy of P_GF to preserve original values
            P_GF_adjusted = np.array(P_GF, copy=True)  # Assuming P_GF is a NumPy array

            # Initialize all values as NaN
            P_GF_adjusted[:] = np.nan

            # Now, use a loop to assign non-NaN values within arcs
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Copy values from P_GF to P_GF_adjusted for the current arc
                P_GF_adjusted[start:end] = P_GF[start:end]

            # Now, P_GF_adjusted will have NaN outside the arcs and preserve original values inside
            L_GF_adjusted = list(L_GF)  # Initialize with original values

            limiar = 2  # Define the threshold

            # List to store subarcs
            subarcos = []

            # Plot original arcs in black (currently commented/placeholder)
            for i, arc in enumerate(arcos):
                L_GF_adjusted_arc = [L_GF_adjusted[i] for i in arc]
                P_GF_adjusted_arc = [P_GF[i] for i in arc]

            # Iterate over arcs
            for arc in arcos:
                # Filter L_GF_adjusted values for the current arc
                L_GF_adjusted_arc = [L_GF_adjusted[i] for i in arc]

                # Calculate difference between consecutive points
                valor = abs(np.diff(L_GF_adjusted_arc, prepend=np.nan))

                # Find indices where difference exceeds threshold
                outliers = np.where(valor > limiar)[0]

                # If there are outliers, split arc into subarcs
                if len(outliers) > 0:
                    subarco_indices = [arc[0]]  # Add arc start index

                    # Add outlier indices as subarc dividers
                    for outlier in outliers:
                        subarco_indices.append(arc[outlier])        # Divider
                        subarco_indices.append(arc[outlier] + 1)    # Next index after divider

                    subarco_indices.append(arc[-1])  # Add arc end index

                    # Create subarcs with calculated indices
                    for i in range(0, len(subarco_indices), 2):
                        subarcos.append(list(range(subarco_indices[i], subarco_indices[i+1] + 1)))
                else:
                    # If no outliers, keep the original arc
                    subarcos.append(arc)

            # Update arc list to use subarcs
            arcos = subarcos

            # Print information for each arc and classify them
            for i, arc in enumerate(arcos):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                      f"Obs. = {num_observations}, Status = {status}")

            # Filter arcs that meet the minimum length criterion
            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1]+1]) >= 15]

            # Step 1: Create a copy of L_GF
            L_GF2 = np.copy(L_GF)

            # Step 2: Initialize a boolean array with True
            marcador_fora_arco = np.ones_like(L_GF, dtype=bool)

            # Step 3: Mark the indices within arcs as False (do not convert to NaN)
            # Iterate over each valid arc
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco[start:end] = False

            # Step 4: Set values outside arcs to NaN
            L_GF2[marcador_fora_arco] = np.nan

            # Update L_GF to be the modified version
            L_GF = L_GF2

            # Reinitialize L_GF2 as a copy of L_GF
            L_GF2 = np.copy(L_GF)

            # Iterate over the segments defined in 'arcos'
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Select current segment of L_GF
                segmento_data = L_GF[start:end]

                # Calculate first and third quartiles
                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1

                # Define outlier limits
                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR

                # Find indices of outliers within the segment
                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))

                # Replace outliers with NaN in L_GF2
                L_GF2[start:end][outlier_indices] = np.nan

            # Update L_GF to be the version without outliers
            L_GF = L_GF2

            # First, create a copy of P_GF to preserve original values
            P_GF_adjusted = np.array(P_GF, copy=True)  # Assuming P_GF is a NumPy array

            # Initialize all values as NaN
            P_GF_adjusted[:] = np.nan

            # Assign values from P_GF to P_GF_adjusted inside the arcs
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                P_GF_adjusted[start:end] = P_GF[start:end]

            # Now, P_GF_adjusted has NaNs outside arcs and original values inside

            L_GF_adjusted = list(L_GF)  # Initialize with original values

            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                # Calculate the difference and mean difference for the current arc
                diff = P_GF_adjusted[start:end] - L_GF_adjusted[start:end]
                mean_diff = np.nanmean(diff)

                # Adjust L_GF values inside this arc
                for j in range(start, end):
                    L_GF_adjusted[j] += mean_diff

            L_GF_adjusted_old = L_GF_adjusted

            # Define function to remove outliers based on the quartile method
            def remove_outliers_quartil(data):
                q1 = np.nanpercentile(data, 15)
                q3 = np.nanpercentile(data, 85)
                iqr = q3 - q1
                lower_bound = q1 - 1.3 * iqr
                upper_bound = q3 + 1.3 * iqr
                return [x for x in data if lower_bound <= x <= upper_bound]

            # Remove outliers individually for each arc
            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                L_GF_adjusted[start:end+1] = remove_outliers_quartil(L_GF_adjusted[start:end+1])

            L_GF_adjusted2 = L_GF_adjusted
            P_GF_adjusted2 = P_GF_adjusted

            # Assume L_GF and P_GF are your original data series.
            # Initialize adjusted versions with NaNs
            L_GF_adjusted = np.full_like(L_GF, np.nan, dtype=np.float64)
            P_GF_adjusted = np.full_like(P_GF, np.nan, dtype=np.float64)

            polynomial_fits = []
            polynomial_fits2 = []
            arc_data = []
            arc_data2 = []

            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Update values in L_GF_adjusted and P_GF_adjusted only in defined arc intervals
                L_GF_adjusted[start:end] = L_GF[start:end]
                P_GF_adjusted[start:end] = P_GF[start:end]

                arc_values = L_GF[start:end]  # Use original data for fitting
                arc_values2 = P_GF[start:end]  # Use original data for fitting
                arc_timestamps = df['timestamp'][start:end]

                # Fit a polynomial of degree 3 to each set of values
                x_values = np.arange(len(arc_values))
                polynomial_fit = levelling_settings.fit_polynomial(x_values, arc_values, 3)

                x_values2 = np.arange(len(arc_values2))
                polynomial_fit2 = levelling_settings.fit_polynomial(x_values2, arc_values2, 3)

                # Store the fits and arc data
                polynomial_fits.append(polynomial_fit)
                polynomial_fits2.append(polynomial_fit2)
                arc_data.append(arc_values)
                arc_data2.append(arc_values2)

            L_GF_adjusted = list(L_GF_adjusted)  # Initialize with original values

            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Calculate difference and mean difference for the current arc
                diff = P_GF_adjusted[start:end] - L_GF_adjusted[start:end]
                mean_diff = np.nanmean(diff)

                # Adjust L_GF values inside this arc
                for i in range(start, end):
                    L_GF_adjusted[i] += mean_diff

            # Configure pandas to display all columns and rows
            pd.set_option('display.max_columns', None)  # Use None or set a number if preferred
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', None)

            L_GF_adjusted2 = L_GF_adjusted

            # Calculate quartiles for all values in L_GF_adjusted
            q1 = np.nanpercentile(L_GF_adjusted, 25)
            q3 = np.nanpercentile(L_GF_adjusted, 75)
            iqr = q3 - q1

            # Compute lower and upper bounds to identify outliers
            lower_bound = q1 - 2.5 * iqr
            upper_bound = q3 + 7.5 * iqr

            # Replace outliers with NaN across the entire dataset
            L_GF_adjusted = [x if lower_bound <= x <= upper_bound else np.nan for x in L_GF_adjusted]

            #plt.scatter(df['timestamp'], L_GF_adjusted, marker='o', s=20, color='blue', label='GF: L1-L2', zorder=0)

            # Escolhe cor e legenda com base na classe do satélite
            cor = 'navy' if sat_class == 'G' else 'red'
            label = 'GF: L1-L2 (GPS)' if sat_class == 'G' else 'GF: L1-L2 (GLONASS)'

            # Plota apenas UMA VEZ por classe para a legenda
            if satellite == satellites_to_plot[0]:
                plt.scatter(df['timestamp'], L_GF_adjusted, marker='o', s=20, color=cor, label=label, zorder=1)
            else:
                plt.scatter(df['timestamp'], L_GF_adjusted, marker='o', s=20, color=cor, zorder=1)


            # Ensure L_GF_adjusted and df have the same length
            if len(L_GF_adjusted) == len(df):
                df['LGF_combination'] = L_GF_adjusted
            else:
                print("Error: L_GF_adjusted and df do not have the same length!")

            # ---------------------- [L1-L5]

            # Step 1: Create a copy of L_GF15
            L_GF3 = np.copy(L_GF15)

            # Step 2: Initialize a boolean array with True
            marcador_fora_arco15 = np.ones_like(L_GF15, dtype=bool)

            # Step 3: Mark indices inside arcs as False (not to convert to NaN)
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco15[start:end] = False

            # Step 4: Convert values outside arcs to NaN
            L_GF3[marcador_fora_arco15] = np.nan

            # Update L_GF15 to the modified version
            L_GF15 = L_GF3

            # Reinitialize L_GF3 as a copy of L_GF15
            L_GF3 = np.copy(L_GF15)

            # Iterate over segments defined by arcs
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                segmento_data = L_GF3[start:end]

                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR

                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))
                L_GF3[start:end][outlier_indices] = np.nan

            L_GF15 = L_GF3

            # First, create a copy of P_GF15 to preserve original values
            P_GF_adjusted15 = np.array(P_GF15, copy=True)

            # Initialize all values as NaN
            P_GF_adjusted15[:] = np.nan

            # Assign values within arcs
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                P_GF_adjusted15[start:end] = P_GF15[start:end]

            # Initialize with original values
            L_GF_adjusted15 = list(L_GF15)

            limiar = 2  # Set appropriate threshold
            subarcos = []

            # Plot original arcs in black (placeholder loop)
            for i, arc in enumerate(arcos):
                L_GF_adjusted_arc15 = [L_GF_adjusted15[i] for i in arc]
                P_GF_adjusted_arc15 = [P_GF15[i] for i in arc]

            # Iterate over arcs
            for arc in arcos:
                L_GF_adjusted_arc15 = [L_GF_adjusted15[i] for i in arc]
                valor = abs(np.diff(L_GF_adjusted_arc15, prepend=np.nan))
                outliers = np.where(valor > limiar)[0]

                if len(outliers) > 0:
                    subarco_indices = [arc[0]]
                    for outlier in outliers:
                        subarco_indices.append(arc[outlier])
                        subarco_indices.append(arc[outlier] + 1)
                    subarco_indices.append(arc[-1])
                    for i in range(0, len(subarco_indices), 2):
                        subarcos.append(list(range(subarco_indices[i], subarco_indices[i+1] + 1)))
                else:
                    subarcos.append(arc)

            arcos = subarcos

            for i, arc in enumerate(arcos):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                      f"Obs. = {num_observations}, Status = {status}")

            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1]+1]) >= 15]

            # Step 1: Create a copy of L_GF
            L_GF3 = np.copy(L_GF15)

            # Step 2: Initialize a boolean array with True
            marcador_fora_arco15 = np.ones_like(L_GF15, dtype=bool)

            # Step 3: Mark indices inside arcs as False
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco15[start:end] = False

            # Step 4: Convert values outside arcs to NaN
            L_GF3[marcador_fora_arco15] = np.nan

            L_GF15 = L_GF3
            L_GF3 = np.copy(L_GF15)

            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                segmento_data = L_GF15[start:end]

                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR

                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))
                L_GF3[start:end][outlier_indices] = np.nan


            # Update L_GF to be the version without outliers
            L_GF15 = L_GF3

            # First, create a copy of P_GF to preserve the original values
            P_GF_adjusted15 = np.array(P_GF15, copy=True)  # Assuming P_GF is a list or NumPy array

            # Initialize all values as NaN
            P_GF_adjusted15[:] = np.nan

            # Assign non-NaN values inside arcs
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                P_GF_adjusted15[start:end] = P_GF15[start:end]

            # Now P_GF_adjusted has NaN outside arcs and original values inside
            L_GF_adjusted15 = list(L_GF_adjusted15)  # Initialize with original values

            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                diff15 = P_GF_adjusted15[start:end] - L_GF_adjusted15[start:end]
                mean_diff15 = np.nanmean(diff15)
                for j in range(start, end):
                    L_GF_adjusted15[j] += mean_diff15

            L_GF_adjusted_old15 = L_GF_adjusted15

            # Define function to remove outliers using the quartile method
            def remove_outliers_quartil(data):
                q1 = np.nanpercentile(data, 15)
                q3 = np.nanpercentile(data, 85)
                iqr = q3 - q1
                lower_bound = q1 - 1.3 * iqr
                upper_bound = q3 + 1.3 * iqr
                return [x for x in data if lower_bound <= x <= upper_bound]

            # Remove outliers individually for each arc
            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                L_GF_adjusted15[start:end+1] = remove_outliers_quartil(L_GF_adjusted15[start:end+1])

            L_GF_adjusted3 = L_GF_adjusted15
            P_GF_adjusted3 = P_GF_adjusted15

            # Assume L_GF and P_GF are the original data series
            # Initialize adjusted versions with NaNs
            L_GF_adjusted15 = np.full_like(L_GF15, np.nan, dtype=np.float64)
            P_GF_adjusted15 = np.full_like(P_GF15, np.nan, dtype=np.float64)

            polynomial_fits = []
            polynomial_fits2 = []
            arc_data = []
            arc_data2 = []

            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                L_GF_adjusted15[start:end] = L_GF15[start:end]
                P_GF_adjusted15[start:end] = P_GF15[start:end]

                arc_values = L_GF15[start:end]  # Use original data for fitting
                arc_values2 = P_GF15[start:end]
                arc_timestamps = df['timestamp'][start:end]

                # Fit a 3rd-degree polynomial to each set
                x_values = np.arange(len(arc_values))
                polynomial_fit = levelling_settings.fit_polynomial(x_values, arc_values, 3)

                x_values2 = np.arange(len(arc_values2))
                polynomial_fit2 = levelling_settings.fit_polynomial(x_values2, arc_values2, 3)

                polynomial_fits.append(polynomial_fit)
                polynomial_fits2.append(polynomial_fit2)
                arc_data.append(arc_values)
                arc_data2.append(arc_values2)

            L_GF_adjusted15 = list(L_GF_adjusted15)  # Initialize with original values

            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                diff15 = P_GF_adjusted15[start:end] - L_GF_adjusted15[start:end]
                mean_diff15 = np.nanmean(diff15)
                for i in range(start, end):
                    L_GF_adjusted15[i] += mean_diff15

            # Configure pandas to display all columns and rows
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', None)

            L_GF_adjusted3 = L_GF_adjusted15

            # Calculate quartiles for all L_GF_adjusted values
            q1 = np.nanpercentile(L_GF_adjusted15, 25)
            q3 = np.nanpercentile(L_GF_adjusted15, 75)
            iqr = q3 - q1

            # Compute lower and upper bounds to identify outliers
            lower_bound = q1 - 2.5 * iqr
            upper_bound = q3 + 7.5 * iqr

            # Replace outliers with NaN
            L_GF_adjusted15 = [x if lower_bound <= x <= upper_bound else np.nan for x in L_GF_adjusted15]

            #plt.scatter(df['timestamp'], L_GF_adjusted15, marker='o', s=20, color='red', label='GF: L1-L5', zorder=1)

            # Escolhe cor e legenda com base na classe do satélite
            cor = 'blue' if sat_class == 'G' else 'orange'
            label = 'GF: L1-L5 (GPS)' if sat_class == 'G' else 'GF: L2-L3 (GLONASS)'

            if satellite == satellites_to_plot[0]:
                plt.scatter(df['timestamp'], L_GF_adjusted15, marker='s', s=5, color=cor, label=label, zorder=2)
            else:
                plt.scatter(df['timestamp'], L_GF_adjusted15, marker='s', s=5, color=cor, zorder=2)


            # Ensure L_GF_adjusted has the same length as df
            if len(L_GF_adjusted15) == len(df):
                df['LGF_combination15'] = L_GF_adjusted15
            else:
                print("Error: L_GF_adjusted15 and df do not have the same length!")

            ########################################

            # Now df contains the LGF_combination column
            # List of desired columns
            colunas_desejadas = [
                'date', 'time2', 'mjd', 'pos_x', 'pos_y', 'pos_z',
                'LGF_combination', 'LGF_combination15', 'satellite',
                'sta', 'hght', 'El', 'Lon', 'Lat', 'obs_La', 'obs_Lb',
                'obs_Lc', 'obs_Ca', 'obs_Cb', 'obs_Cc'
            ]
            df_selecionado = df[colunas_desejadas]

            # Path and file name
            destination_directory = "/media/debian-giorgio/DATA/GNSS_DATA/RINEX/PROCESSED"
            output_directory = os.path.join(str(ano), str(doy), estacao.upper())
            full_path = os.path.join(diretorio_principal)
            file_name = f"{estacao}_{satellite}_{doy}_{ano}.RNX3"
            fig_name = f"{estacao}_{doy}_{ano}.png"
            output_file_path = os.path.join(full_path, file_name)
            output_fig_path = os.path.join(full_path, fig_name)

            # Ensure directory exists
            os.makedirs(full_path, exist_ok=True)

            # Save selected DataFrame to tab-separated text file
            df_selecionado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')

            print(f"Data exported to {output_file_path}.")

        plt.title(f"Station: {estacao.upper()}  |  Year: {ano}  |  DOY: {doy}", fontsize=16)
        plt.xlabel('Time (UT)', fontsize=16)
        plt.ylabel('Levelled Geometry-Free', fontsize=16)
        hours_fmt = mdates.DateFormatter('%H')
        plt.gca().xaxis.set_major_formatter(hours_fmt)
        minute_locator = mdates.MinuteLocator(interval=int1)
        plt.gca().xaxis.set_major_locator(minute_locator)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.grid(axis='both', linestyle='--', color='gray', linewidth=1)
        plt.tight_layout()
    if show_plot:
        plt.show()
