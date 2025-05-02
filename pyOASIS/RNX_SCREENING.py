from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import itertools
import warnings
from numpy.polynomial import Polynomial
from pyOASIS import gnss_freqs
from pyOASIS import screening_settings
from pyOASIS import settings
import pyOASIS

def RNXScreening(destination_directory):    
    # List of files in the .RNX1 directory
    filess = os.listdir(destination_directory)

    # Filter only the .RNX1 files
    files = [file_ for file_ in filess if file_.endswith("RNX1")]
    
    for file in files:
        f = os.path.join(destination_directory, file)
    
        # Redirect stderr to /dev/null
        sys.stderr = open(os.devnull, 'w')    
        
        # Get the file name
        g = os.path.basename(f)
        
        ano = g[13:17]
        doy = g[9:12]
        estacao = g[0:4]
        sat = g[5:8]
        
        # Variables and Parameters
        h1 = 0
        n_horas = 24 # hours
        int1 = 120 # minutes
        
        # Accessing the frequencies of the GPS system
        gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
        f1 = gps_freqs[1]
        f2 = gps_freqs[2]
        f5 = gps_freqs[5]    
              
        # Calculating the frequencies of the GLONASS system (glonass_channels.dat)
        # To obtain the GLONASS frequencies (code 'R'):
        file_name = os.path.join(pyOASIS.__path__[0], 'glonass_channels.dat')
        
        # Read the file into a DataFrame with predefined column names
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
                if callable(frequency):  # Check if it is a lambda function
                    freq_value = frequency(k)
                else:
                    freq_value = frequency
                formatted_freq = f"{freq_value:.1f}"
                row_data.append(formatted_freq)
        
            # Add the row data to the data list
            data.append(row_data)
        
        # Convert the list of lists into a pandas DataFrame
        glo_freqs = pd.DataFrame(data, columns=['Satellite', 'fr1', 'fr2', 'fr3'])
        
        # Check the satellite class and adjust f1, f2, f5 accordingly
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
        
        # Initialize lists to store values of each variable from all files
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
        cs_flags = []
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
        
        caminho_arquivo = f

        with open(caminho_arquivo, 'r') as f:
            # Reading the file header
            header = f.readline().strip().split('\t')
            obs_La_header = header[6]
            obs_Lb_header = header[7]
            obs_Lc_header = header[8]
            obs_Ca_header = header[9]
            obs_Cb_header = header[10]
            obs_Cc_header = header[11]

            # Reading each data line from the file
            for linha in f:
                # Splitting the line into columns
                colunas = linha.strip().split('\t')  # Assuming tab-separated columns
                # Mapping each column to its corresponding header
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
                    'cs_flags': colunas[12],
                    'satellite': colunas[13],
                    'sta': colunas[14],
                    'hght': colunas[15],
                    'El': colunas[16],
                    'Lon': colunas[17],
                    'Lat': colunas[18],
                    'obs_La': obs_La_header,
                    'obs_Lb': obs_Lb_header,
                    'obs_Lc': obs_Lc_header,
                    'obs_Ca': obs_Ca_header,
                    'obs_Cb': obs_Cb_header,
                    'obs_Cc': obs_Cc_header
                }
                
                # Appending each variable to its corresponding list
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
                cs_flags.append(registro['cs_flags'])
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
        
        # Initializing lists to store filtered values for each variable
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
        cs_flags_filtered = []
        satellites_filtered = []
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
        
        satellite = sat
        print(satellite)
        indices = np.where(np.array(satellites) == satellite)[0]
        
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
        cs_flags_filtered = []
        satellites_filtered = []
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
               
        # Filtering the values for the selected satellite
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
            cs_flags_filtered.append(cs_flags[idx])
            satellites_filtered.append(satellites[idx])
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
                
        # Building a DataFrame with the filtered data for the selected satellite
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
            'P5': P2_filtered,
            'cs_flag': cs_flags_filtered,
            'satellite': satellites_filtered,
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

        # Convert relevant columns to float
        columns_to_convert = ['L1', 'L2', 'L5', 'P1', 'P2', 'P5']
        df[columns_to_convert] = df[columns_to_convert].astype(float)
        
        # Replace -999999.999 with NaN in relevant columns
        df.replace(-999999.999, np.nan, inplace=True)        
        
        # Convert 'date' and 'time' columns to datetime and combine them
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time2'])
        
        # Extract only the time from 'timestamp' and store in a new column 'time'
        df['time'] = df['timestamp'].dt.time
        
        # Convert lists to numpy arrays and ensure they are float64
        L1_array = np.nan_to_num(np.array(df['L1'].tolist(), dtype=np.float64), nan=-999999.999)
        L2_array = np.nan_to_num(np.array(df['L2'].tolist(), dtype=np.float64), nan=-999999.999)
        L5_array = np.nan_to_num(np.array(df['L5'].tolist(), dtype=np.float64), nan=-999999.999)
        
        P1_array = np.nan_to_num(np.array(df['P1'].tolist(), dtype=np.float64), nan=-999999.999)
        P2_array = np.nan_to_num(np.array(df['P2'].tolist(), dtype=np.float64), nan=-999999.999)
        P5_array = np.nan_to_num(np.array(df['P5'].tolist(), dtype=np.float64), nan=-999999.999)
        
        # Replace -999999.999 with NaN in the arrays
        L1_array[L1_array == -999999.999] = np.nan
        L2_array[L2_array == -999999.999] = np.nan
        L5_array[L5_array == -999999.999] = np.nan
        
        P1_array[P1_array == -999999.999] = np.nan
        P2_array[P2_array == -999999.999] = np.nan
        P5_array[P5_array == -999999.999] = np.nan
        
        # Compute the Melbourne-Wübbena combination for the current satellite
        MW_combination = screening_settings.melbourne_wubbena_combination(f1, f2, L1_array, L2_array, P1_array, P2_array)
        MW_combination2 = screening_settings.melbourne_wubbena_combination(f1, f5, L1_array, L5_array, P1_array, P5_array)
              
        IFL_combination = screening_settings.iono_free_phase_combination(f1, f2, L1_array, L2_array)
        IFP_combination = screening_settings.iono_free_range_combination(f1, f2, P1_array, P2_array)

        # Add MW, ionosphere-free phase, and ionosphere-free code combinations to the DataFrame
        df['MW'] = MW_combination
        df['MW2'] = MW_combination2        
        
        # Assuming df['cs_flag'] is a pandas Series
        arcos = []  # List to store observation arcs
        arc_atual = []  # Temporary list to store current observation arc
        
        # Iterate over all elements in df['cs_flag']
        for idx, value in enumerate(df['cs_flag']):
            if value == 'S':
                # If the current value is 'S', check if the current arc is not empty
                # This avoids appending empty arcs in case of consecutive 'S'
                if arc_atual:
                    arcos.append(arc_atual)
                    arc_atual = []  # Reseta a lista do arco atual
            else:
                # If the value is not 'S', add index to the current arc
                arc_atual.append(idx)
        
        # Append the last arc if it's not empty
        if arc_atual:
            arcos.append(arc_atual)
        
        print()
        
        # Print information about each arc and classify them
        for i, arc in enumerate(arcos):
            start_index = arc[0]
            end_index = arc[-1]
            num_observations = len(arc)
            status = "Kept" if num_observations >= 15 else "Discarded"
            print(f"Arc {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                f"Obs. = {num_observations}, Status = {status}")

        # Separate MW_combination data for each arc
        arc_data = []
        arc_idx = []
        polynomial_fits = []
        
        print()
        
        for i, arc in enumerate(arcos):
            start = arc[0]
            end = arc[-1]
        
            arc_values = MW_combination[start:end+1]
            arc_timestamps = df['timestamp'][start:end+1]
        
            if len(arc_values) < 15:
                continue
        
            # Fit a 3rd-degree polynomial
            x_values = np.arange(len(arc_values))
            polynomial_fit = screening_settings.fit_polynomial(x_values, arc_values, 3)
        
            # Store arc values and the polynomial fit
            arc_data.append(arc_values)
            arc_idx.append(arc_timestamps)
            polynomial_fits.append(polynomial_fit)
        
            # Print arc information
            num_observations = len(arc_values)
            num_points_fit = len(polynomial_fit)
        
            print(f"Arc {i + 1}: Start index = {start}, End index = {end}, "
                f"Number of observations = {num_observations}, Number of fit points = {num_points_fit}")

        # Filter arcs that meet the minimum length criterion
        arcos_validos = [arc for arc in arcos if len(MW_combination[arc[0]:arc[-1]+1]) >= 15]
        
        # If there's only one valid arc, duplicate it to ensure at least two
        if len(arcos_validos) == 1:
            arcos_validos.append(arcos_validos[0])
        
        # Set the maximum number of columns per row
        max_colunas_por_linha = 2
        
        # Calculate the number of rows and columns needed
        num_arcos_validos = len(arcos_validos)
        
        num_linhas = (num_arcos_validos - 1) // max_colunas_por_linha + 1
        num_colunas = min(num_arcos_validos, max_colunas_por_linha)
        
        # Create the subplot grid
        fig, axes = plt.subplots(num_linhas, num_colunas, figsize=(6*num_colunas, 5*num_linhas))
        
        all_all = []        
        all_index = []
        
        # Iterate over each valid arc and plot the data
        for i, (arc, ax) in enumerate(zip(arcos_validos, axes.flatten()), start=1):
        
            start = arc[0]
            end = arc[-1]
        
            arc_data = df.iloc[arc]
            time = df.index[arc]
        
            arc_values = MW_combination[start:end+1]
            arc_timestamps = df['timestamp'][start:end+1]
            arc_values2 = arc_values
        
            # Calculate elapsed time in seconds from the first timestamp in the arc
            x = (arc_timestamps - arc_timestamps.iloc[0]).dt.total_seconds()
        
            y_rescaled = screening_settings.rescale_data(arc_values)
            delta_y = np.diff(y_rescaled, prepend=np.nan)
        
            # Fit a polynomial only to valid values (excluding NaNs)
            p = Polynomial.fit(x[1:], delta_y[1:], 3)
        
            delta_y_fit = p(x)  # Polynomial-fitted values
            residuals = delta_y - delta_y_fit  # Compute residuals
        
            mini_arcos = []  # List to store mini-arcs
            mini_arcos_mantidos = []
            mini_arc_atual = []  # Temporary list to store the current mini-arc
            signo_anterior = None
        
            # Iterate over all residual values
            for idx, value in enumerate(residuals):
                if signo_anterior is None:  # If it's the first value, initialize the sign
                    signo_anterior = np.sign(value)
        
                # Check if the sign has changed
                if np.sign(value) != signo_anterior:
                    # If the current mini-arc is not empty, add it to the list
                    if mini_arc_atual:
                        mini_arcos.append(mini_arc_atual)
                    mini_arc_atual = []  # Start a new mini-arc
        
                # Add the index to the current mini-arc
                mini_arc_atual.append(idx)
        
                # Update the previous sign
                signo_anterior = np.sign(value)
        
            # List to store mini-arcs that meet the minimum observation requirement
            mini_arcos_mantidos = []
        
            # Flag to control early termination of the loop
            should_break = False
        
            print()
            print("Looking for mini cycle-slips in L1-L2 pair:")
            print()
        
            # Outer loop
            for mini_i, mini_arc in enumerate(mini_arcos):
                mini_start_index = mini_arc[0]
                mini_end_index = mini_arc[-1]
                num_observations = len(mini_arc)
                status = "Kept" if num_observations >= 4 else "Discarded"
                print(f"Mini-arc {mini_i + 1}: Start = {mini_start_index}, End = {mini_end_index}, Obs. = {num_observations}, Status = {status}")
        
                # If the number of observations is less than 4, set the flag and continue
                if num_observations <= 4:
                    should_break = True
                    continue
        
                # If the mini-arc has at least 4 observations, keep it
                mini_arcos_mantidos.append(mini_arc)

            # If the flag indicates we should exit the loops, break from the outer loop as well
            if should_break:
                continue
        
            # Print the kept mini-arcs
            print("Kept mini-arcs:")
            for i, mini_arc in enumerate(mini_arcos_mantidos):
                print(f"Mini-arc {i + 1}: {mini_arc}")

            # Quartile and IQR calculation for identifying outliers
            Q1 = np.nanpercentile(residuals, 15)
            Q3 = np.nanpercentile(residuals, 85)
            IQR = Q3 - Q1
        
            threshold2 = 2
        
            outlier_mask = (residuals < Q1 - threshold2 * IQR) | (residuals > Q3 + threshold2 * IQR)
            high_residuals_mask = residuals > 1  # Mask for high residuals
            other_residuals_mask = ~(outlier_mask | high_residuals_mask)  # Mask for remaining residuals
        
            # Fit a third-degree polynomial
            x_values = np.arange(len(arc_values))
            polynomial_fit = screening_settings.fit_polynomial(x_values, arc_values, 3)
        
            # Plot arc data
            ax.scatter(arc_timestamps, arc_values, label='Dados', color='blue', s=15)
        
            # Plot polynomial fit
            ax.plot(arc_timestamps, polynomial_fit, label='Ajuste', color='red')        
        
            # Format x-axis as hour labels
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))       
        
            # Get the corresponding arc number
            num_arc_valido = arcos.index(arc) + 1
        
            # Add arc number to subplot title
            ax.set_title(f"Arco {num_arc_valido}")
        
            # Add inset subplots to the main subplot
            ax_fit = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            ax_residuals = ax.inset_axes([0.6, 0.15, 0.35, 0.35])
        
            # Colors for each mini-arc
            cores = ['lightblue', 'yellow', 'lime', 'green', 'purple', 'navy', 'blue','lime', 'green', 'navy']

            all_out = []  # Initialize 'all_out' list before the loop
            all__all = []
            all_indices = []    # List to store indices of high residuals and outliers
            todos_indices = []  # List to store all indices     

        
            for i, (mini_start, mini_end) in enumerate(mini_arcos_mantidos):
                mini_residuals = residuals[mini_start:mini_end]
                mini_time = time[mini_start:mini_end]

                try:
                    mini_fit = screening_settings.fit_polynomial(mini_time, mini_residuals, 3)        
                    new_mini_residuals = abs(mini_residuals-mini_fit)

                    # Quartile and IQR calculation for identifying outliers
                    mini_Q1 = np.nanpercentile(new_mini_residuals, 15)
                    mini_Q3 = np.nanpercentile(new_mini_residuals, 85)
                    mini_IQR = mini_Q3 - mini_Q1
        
                    mini_threshold2 = 1.3
        
                    mini_outlier_mask = (new_mini_residuals < mini_Q1 - mini_threshold2 * mini_IQR) | (new_mini_residuals > mini_Q3 + mini_threshold2 * mini_IQR)
                    mini_high_residuals_mask = new_mini_residuals > 0.0002  # Mask for high residuals
                    mini_other_residuals_mask = ~(mini_outlier_mask | mini_high_residuals_mask)  # Mask for remaining residuals
        
                    micro_residuals = abs(np.diff(new_mini_residuals, prepend=np.nan))
        
                    # Compute quartiles
                    q1_micro = np.nanpercentile(micro_residuals, 15)
                    q3_micro = np.nanpercentile(micro_residuals, 85)
        
                    # Compute interquartile range
                    iqr_micro = q3_micro - q1_micro
        
                    micro_threshold2 = 1.3
        
                    # Define the bounds for identifying outliers
                    lower_bound_micro = q1_micro - micro_threshold2 * iqr_micro
                    upper_bound_micro = q3_micro + micro_threshold2 * iqr_micro
        
                    # Identify outliers
                    outliers_micro = (micro_residuals < lower_bound_micro) | (micro_residuals > upper_bound_micro)

                    micro_high_residuals_mask = micro_residuals > 0.00002  # Mask for high residuals

                    ax_fit.scatter(mini_time, abs(micro_residuals), color=cores[i], s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[micro_high_residuals_mask], micro_residuals[micro_high_residuals_mask], color='red', s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[outliers_micro], abs(micro_residuals[outliers_micro]), color='red', s=2, label='Out.', marker='o')

                    # Convert local indices to global indices – mini
                    global_high_residuals_indices = np.where(mini_high_residuals_mask)[0] + mini_start
                    global_outlier_indices = np.where(mini_outlier_mask)[0] + mini_start
        
                    # Convert local indices to global indices – micro
                    global_high_residuals_indices_micro = np.where(micro_high_residuals_mask)[0] + mini_start
                    global_outlier_indices_micro = np.where(outliers_micro)[0] + mini_start
    
                    ax_residuals.scatter(mini_time[mini_outlier_mask], new_mini_residuals[mini_outlier_mask], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[outliers_micro], new_mini_residuals[outliers_micro], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[micro_high_residuals_mask], new_mini_residuals[micro_high_residuals_mask], edgecolor='pink', facecolor='none', s=2, label='Out.', marker='o')
        
                    # Lists to store the indices of high residuals and outliers
                    indices_residuos_altos_mini = start + global_high_residuals_indices
                    indices_outliers_mini = start + global_outlier_indices
                    indices_residuos_altos_micro = start + global_high_residuals_indices_micro
                    indices_outliers_micro = start + global_outlier_indices_micro
        
                    # Add the indices to the general list
                    todos_indices.append({
                        'residuos_altos_mini': indices_residuos_altos_mini,
                        'outliers_mini': indices_outliers_mini,
                        'residuos_altos_micro': indices_residuos_altos_micro,
                        'outliers_micro': indices_outliers_micro
                    })
        
                except:
                    pass  # Ignore and continue the loop
        
            # Remove unused subplots
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])

            # Create a copy of the all_indexes list before extending it to all_index_vertical
            todos_indices_copy = todos_indices.copy()
        
            # Access all indices outside the loop
            for indices in todos_indices_copy:
                print(f"Índices para iteração {i}:")
                print("Resíduos Altos para Mini-Arco:", indices['residuos_altos_mini'])
                print("Outliers para Mini-Arco:", indices['outliers_mini'])
                print("Resíduos Altos para Micro-Arco:", indices['residuos_altos_micro'])
                print("Outliers para Micro-Arco:", indices['outliers_micro'])
                print()
        
            # Concatenate all indices into a vertical list
            todos_indices_vertical = []
            for indices in todos_indices_copy:
                todos_indices_vertical.extend(indices['residuos_altos_mini'])
                todos_indices_vertical.extend(indices['outliers_mini'])
                todos_indices_vertical.extend(indices['residuos_altos_micro'])
                todos_indices_vertical.extend(indices['outliers_micro'])

            from collections import OrderedDict
        
            # Convert all_indices_vertical to a set to remove duplicates, then back to a list maintaining original order
            todos_indices_vertical = list(OrderedDict.fromkeys(todos_indices_vertical))        
        
            # Remove unused subplots
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])
        
            all_index.extend(todos_indices_vertical)
        
            # Remove empty entries from the list
            all_index = list(filter(None, all_index))

        print()
        print(satellite)
        
        df['outlier_flag'] = 'N'
        
        # Replace 'N' with 'Y' at the specified indices
        df.loc[all_index, 'outlier_flag'] = 'Y'
        
        # ----------------- [L1 - L5]        
        all_index15 = []
        
        # Iterate over each valid arc and plot the data
        for i, (arc, ax) in enumerate(zip(arcos_validos, axes.flatten()), start=1):        
            start = arc[0]
            end = arc[-1]        
            arc_data = df.iloc[arc]
            time = df.index[arc]
        
            arc_values = MW_combination2[start:end+1]
            arc_timestamps = df['timestamp'][start:end+1]
            arc_values2 = arc_values
        
            # Compute elapsed time in seconds from the first timestamp in the arc
            x = (arc_timestamps - arc_timestamps.iloc[0]).dt.total_seconds()
        
            y_rescaled = screening_settings.rescale_data(arc_values)
            delta_y = np.diff(y_rescaled, prepend=np.nan)
        
            # Fit a polynomial only to valid values (excluding NaN)
            p = Polynomial.fit(x[1:], delta_y[1:], 3)
        
            delta_y_fit = p(x)  # Fitted values
            residuals = delta_y - delta_y_fit  # Compute residuals
        
            mini_arcos = []  # List to store observation mini-arcs
            mini_arcos_mantidos = []
            mini_arc_atual = []  # Temporary list for current mini-arc
            signo_anterior = None
        
            # Iterate through all residual values
            for idx, value in enumerate(residuals):
                if signo_anterior is None:  # First value, initialize previous_sign
                    signo_anterior = np.sign(value)
        
                # Check if the sign changed
                if np.sign(value) != signo_anterior:
                    if mini_arc_atual:
                        mini_arcos.append(mini_arc_atual)
                    mini_arc_atual = []
                mini_arc_atual.append(idx)
                signo_anterior = np.sign(value)
        
            # List to store mini-arcs that pass the minimum criteria
            mini_arcos_mantidos = []
            should_break = False
        
            print()
            print("Looking for mini cycle-slips in L1-L5 pair:")
            print()
        
            # Loop externo
            for mini_i, mini_arc in enumerate(mini_arcos):
                mini_start_index = mini_arc[0]
                mini_end_index = mini_arc[-1]
                num_observations = len(mini_arc)
                status = "Kept" if num_observations >= 4 else "Discarded"
                print(f"Mini-arc {mini_i + 1}: Start = {mini_start_index}, End = {mini_end_index}, Obs. = {num_observations}, Status = {status}")

                if num_observations <= 4:
                    should_break = True
                    continue

                mini_arcos_mantidos.append(mini_arc)

            if should_break:
                continue

            print("Kept mini-arcs:")
            for i, mini_arc in enumerate(mini_arcos_mantidos):
                print(f"Mini-arc {i + 1}: {mini_arc}")
        
            # Compute quartiles and IQR to identify outliers
            Q1 = np.nanpercentile(residuals, 15)
            Q3 = np.nanpercentile(residuals, 85)
            IQR = Q3 - Q1
        
            threshold2 = 2
        
            outlier_mask = (residuals < Q1 - threshold2 * IQR) | (residuals > Q3 + threshold2 * IQR)
            high_residuals_mask = residuals > 1  # Máscara para resíduos altos
            other_residuals_mask = ~(outlier_mask | high_residuals_mask)  # Máscara para os demais resíduos
        
            # Fit a 3rd-degree polynomial
            x_values = np.arange(len(arc_values))
            polynomial_fit = screening_settings.fit_polynomial(x_values, arc_values, 3)
        
            # Plot arc data
            ax.scatter(arc_timestamps, arc_values, label='Dados', color='blue', s=15)
            ax.plot(arc_timestamps, polynomial_fit, label='Ajuste', color='red')        

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))        
            num_arc_valido = arcos.index(arc) + 1

            ax.set_title(f"Arc {num_arc_valido}")
        
            # Create inset subplots
            ax_fit = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            ax_residuals = ax.inset_axes([0.6, 0.15, 0.35, 0.35])
        
            # Colors for each mini-arc
            cores = ['lightblue', 'yellow', 'lime', 'green', 'purple', 'navy', 'blue','lime', 'green', 'navy']
        
            all_out = []
            all__all = []
            all_indices = []
            todos_indices = []
        
            for i, (mini_start, mini_end) in enumerate(mini_arcos_mantidos):        
                mini_residuals = residuals[mini_start:mini_end]
                mini_time = time[mini_start:mini_end]
        
                try:
                    mini_fit = screening_settings.fit_polynomial(mini_time, mini_residuals, 3)        
                    new_mini_residuals = abs(mini_residuals-mini_fit)
        
                    # Quartile and IQR for outlier detection
                    mini_Q1 = np.nanpercentile(new_mini_residuals, 15)
                    mini_Q3 = np.nanpercentile(new_mini_residuals, 85)
                    mini_IQR = mini_Q3 - mini_Q1
        
                    mini_threshold2 = 1.3
        
                    mini_outlier_mask = (new_mini_residuals < mini_Q1 - mini_threshold2 * mini_IQR) | (new_mini_residuals > mini_Q3 + mini_threshold2 * mini_IQR)
                    mini_high_residuals_mask = new_mini_residuals > 0.0002  # Máscara para resíduos altos
                    mini_other_residuals_mask = ~(mini_outlier_mask | mini_high_residuals_mask)  # Máscara para os demais resíduos
        
                    micro_residuals = abs(np.diff(new_mini_residuals, prepend=np.nan))
        
                    # Calculate Q1 and Q3
                    q1_micro = np.nanpercentile(micro_residuals, 15)
                    q3_micro = np.nanpercentile(micro_residuals, 85)
        
                    # Calculate IQR
                    iqr_micro = q3_micro - q1_micro        
                    micro_threshold2 = 1.3
        
                    lower_bound_micro = q1_micro - micro_threshold2 * iqr_micro
                    upper_bound_micro = q3_micro + micro_threshold2 * iqr_micro
        
                    # Find outliers
                    outliers_micro = (micro_residuals < lower_bound_micro) | (micro_residuals > upper_bound_micro)        
                    micro_high_residuals_mask = micro_residuals > 0.00002
        
                    ax_fit.scatter(mini_time, abs(micro_residuals), color=cores[i], s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[micro_high_residuals_mask], micro_residuals[micro_high_residuals_mask], color='red', s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[outliers_micro], abs(micro_residuals[outliers_micro]), color='red', s=2, label='Out.', marker='o')
        
                    # Convert local to global indices – mini
                    global_high_residuals_indices = np.where(mini_high_residuals_mask)[0] + mini_start
                    global_outlier_indices = np.where(mini_outlier_mask)[0] + mini_start
        
                    # Convert local to global indices – micro
                    global_high_residuals_indices_micro = np.where(micro_high_residuals_mask)[0] + mini_start
                    global_outlier_indices_micro = np.where(outliers_micro)[0] + mini_start

                    ax_residuals.scatter(mini_time[mini_outlier_mask], new_mini_residuals[mini_outlier_mask], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[outliers_micro], new_mini_residuals[outliers_micro], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[micro_high_residuals_mask], new_mini_residuals[micro_high_residuals_mask], edgecolor='pink', facecolor='none', s=2, label='Out.', marker='o')
        
                    # Lists to store indices of high residuals and outliers
                    indices_residuos_altos_mini = start + global_high_residuals_indices
                    indices_outliers_mini = start + global_outlier_indices
                    indices_residuos_altos_micro = start + global_high_residuals_indices_micro
                    indices_outliers_micro = start + global_outlier_indices_micro

                    indices_residuos_altos_mini = start + global_high_residuals_indices
                    indices_outliers_mini = start + global_outlier_indices
                    indices_residuos_altos_micro = start + global_high_residuals_indices_micro
                    indices_outliers_micro = start + global_outlier_indices_micro

                    todos_indices.append({
                        'residuos_altos_mini': indices_residuos_altos_mini,
                        'outliers_mini': indices_outliers_mini,
                        'residuos_altos_micro': indices_residuos_altos_micro,
                        'outliers_micro': indices_outliers_micro
                    })               
        
                except:
                    pass  # Ignore and continue
        
            # Remove unused subplots
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])
        
            # Create a copy of the todos_indices list before extending todos_indices_vertical
            todos_indices_copy = todos_indices.copy()
        
            # Now you can access all indices outside the loop
            for indices in todos_indices_copy:
                print(f"Índices para iteração {i}:")
                print("Resíduos Altos para Mini-Arco:", indices['residuos_altos_mini'])
                print("Outliers para Mini-Arco:", indices['outliers_mini'])
                print("Resíduos Altos para Micro-Arco:", indices['residuos_altos_micro'])
                print("Outliers para Micro-Arco:", indices['outliers_micro'])
                print()
        
            # Concatenate all indices into a vertical list
            todos_indices_vertical = []
            for indices in todos_indices_copy:
                todos_indices_vertical.extend(indices['residuos_altos_mini'])
                todos_indices_vertical.extend(indices['outliers_mini'])
                todos_indices_vertical.extend(indices['residuos_altos_micro'])
                todos_indices_vertical.extend(indices['outliers_micro'])
        
            from collections import OrderedDict
        
            # Convert todos_indices_vertical into a set to remove duplicates and then back to a list preserving original order
            todos_indices_vertical = list(OrderedDict.fromkeys(todos_indices_vertical))
        
            # Remove unused subplots
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])
        
            all_index15.extend(todos_indices_vertical)
        
            # Remove empty entries from the list
            all_index15 = list(filter(None, all_index15))
        
        print()
        print(satellite)        
        
        # Replace 'N' with 'Y' at the specified indices
        df.loc[all_index15, 'outlier_flag'] = 'Y'

        # Output directory and desired path
        output_directory = os.path.join(str(ano), str(doy), estacao.upper())

        # Full output directory path within the destination directoryo
        full_path = os.path.join(destination_directory)
        
        # Ensure the directory exists or create it if not
        os.makedirs(full_path, exist_ok=True)
        
        # Define the file name
        file_name = f"{estacao}_{satellite}_{doy}_{ano}.RNX2"
      
        # Full path of the output file
        output_file_path = os.path.join(full_path, file_name)
        
        # Select only the desired columns
        colunas_desejadas = ['date', 'time', 'mjd', 'pos_x', 'pos_y', 'pos_z', 'L1', 'L2', 'L5', 'P1', 'P2', 'P5', 'cs_flag', 'outlier_flag', 'satellite', 'sta', 'hght', 'El', 'Lon', 'Lat', 'obs_La', 'obs_Lb', 'obs_Lc', 'obs_Ca', 'obs_Cb', 'obs_Cc']
        
        df_selecionado = df[colunas_desejadas]

        # Replace NaN with -999999.999
        df_selecionado = df_selecionado.fillna(-999999.999)
        
        # Save the selected DataFrame to a tab-delimited text file
        df_selecionado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')
        
        # Read the file to verify it is correct
        with open(output_file_path, 'r') as f:
            file_content = f.read()
        
        print(f"Data exported to {output_file_path}.")