import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import numpy.ma as ma
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pyOASIS import gnss_freqs

def ROTIcalc(estacao,doy,ano,diretorio_principal,destination_directory, show_plot=True):

    # === Configuration Parameters ===

    h1 = 0
    n_horas = 24  # Number of hours to process
    int1 = 320  # Number of minutes (used elsewhere at this moment)
    
    # Access GNSS frequencies (GPS system)
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]
    f2 = gps_freqs[2]
    f5 = gps_freqs[5]
    
    akl = 40.3 * 10 ** 16 * ((1 / f2 ** 2) - (1 / f1 ** 2))
    akl15 = 40.3 * 10 ** 16 * ((1 / f5 ** 2) - (1 / f1 ** 2))
    
    # ROTI calculation parameters
    WINDOW = 60.0 * 2.5     # Time window in seconds (2.5 min) for ROTI computation
    TROTI = 60.0            # ROTI time normalization factor (TECU/min)
    GAP = 1.5 * WINDOW      # Maximum time gap allowed between observations within a window
    H = 450000              # Ionospheric shell height [m]
    SIGMA = 5               # Outlier detection threshold (in standard deviations)
    DE = 3                  # Degree of polynomial fit used for smoothing
    GAP2 = 3600             # Time threshold to define independent arcs (seconds)
    elev_angle = 30         # Minimum elevation angle (degrees) for valid data

    # === Directory and file handling ===

    # Construct the full path to the station directory
    caminho_ = os.path.join(diretorio_principal, estacao)

    # Check if the directory exists
    if os.path.exists(caminho_):
        # Listando o conteúdo da pasta
        conteudo_ = os.listdir(caminho_)
        print("Files found in directory:")

        # Filter only .RNX3 files that match the station prefix
        arquivos = [arquivo for arquivo in conteudo_ if arquivo.startswith(estacao) and arquivo.endswith(".RNX3")]
    
        # Sort files by satellite PRN (from filename)
        arquivos_ordenados = sorted(arquivos, key=lambda x: int(x.split("_")[1][1:]))
    
        # Print sorted file names
        for arquivo in arquivos_ordenados:
            print("File found:", arquivo)
    
        print()
        numero_de_arquivos = len(arquivos_ordenados)
        print("Number of screened RINEX files (.RNX3) in the directory:", numero_de_arquivos)
    else:
        print("Specified directory does not exist.")
    print()
    
    # === Data structure initialization ===

    # Lists to store parsed data from RINEX files

    date, time2, mjd = [], [], []
    pos_x, pos_y, pos_z = [], [], []
    LGF_combination, LGF_combination15 = [], []
    satellites, sta, hght, el = [], [], [], []
    lonn, latt = [], []
    obs_La, obs_Lb, obs_Lc = [], [], []
    obs_Ca, obs_Cb, obs_Cc = [], [], []

    # === Read and parse RINEX data files ===
    
    for arquivo in arquivos_ordenados:
        caminho_arquivo = os.path.join(caminho_, arquivo)
        with open(caminho_arquivo, 'r') as f:
            header = f.readline().strip().split('\t')
            for linha in f:
                colunas = linha.strip().split('\t')
                # Append each column value to the corresponding list
                registro = {
                    'date': colunas[0],
                    'time2': colunas[1],
                    'mjd': colunas[2],
                    'pos_x': colunas[3],
                    'pos_y': colunas[4],
                    'pos_z': colunas[5],
                    'LGF_combination': colunas[6],
                    'LGF_combination15': colunas[7],
                    'satellite': colunas[8],
                    'sta': colunas[9],
                    'hght': colunas[10],
                    'el': colunas[11],
                    'lonn': colunas[12],
                    'latt': colunas[13],
                    'obs_La': colunas[14],
                    'obs_Lb': colunas[15],
                    'obs_Lc': colunas[16],
                    'obs_Ca': colunas[17],
                    'obs_Cb': colunas[18],
                    'obs_Cc': colunas[19]
                }
    
                date.append(registro['date'])
                time2.append(registro['time2'])
                mjd.append(registro['mjd'])
                pos_x.append(registro['pos_x'])
                pos_y.append(registro['pos_y'])
                pos_z.append(registro['pos_z'])
                LGF_combination.append(registro['LGF_combination'])
                LGF_combination15.append(registro['LGF_combination15'])
                satellites.append(registro['satellite'])
                sta.append(registro['sta'])
                hght.append(registro['hght'])
                el.append(registro['el'])
                lonn.append(registro['lonn'])
                latt.append(registro['latt'])
                obs_La.append(registro['obs_La'])
                obs_Lb.append(registro['obs_Lb'])
                obs_Lc.append(registro['obs_Lc'])
                obs_Ca.append(registro['obs_Ca'])
                obs_Cb.append(registro['obs_Cb'])
                obs_Cc.append(registro['obs_Cc'])
    
    # === ROTI Computation Loop ===
    
    sat_classes = ['G','R'] # Currently supported: GPS (G) and GLONASS (R)
    palette = plt.get_cmap('tab10')
    plt.figure(figsize=(12, 6)) # Prepare one figure for all satellites
    
    for sat in sat_classes:
        satx=sat
        # Filter satellite PRNs by class
        if satx:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        else:
            satellites_to_plot = np.unique(satellites)
    
        # Initialize list to collect data
        dados_satelites = []
    
        for sat1 in satellites_to_plot:
            print()
            print(f"Processing {sat1} satellite...")
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
    
            # Creating a timestamp column by combining 'date' and 'time' columns
            df['timestamp'] = df['date'] + ' ' + df['time']
    
            # Converting the 'timestamp' column to datetime format, if not already
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
            # Converting relevant columns to float type
            columns_to_convert = ['LGF', 'LGF15', 'mjd','lonn','latt','hh','elev']
            df[columns_to_convert] = df[columns_to_convert].astype(float)
    
            # Replacing missing data values with NaN
            df.replace(-999999.999, np.nan, inplace=True)

            # --------- L1-L2 (GNSS frequency combination for ROTI computation)
    
            t = df['mjd']
            stec = df['LGF'] / akl
            stec15 = df['LGF15'] / akl15
            lat = df['latt']
            lon = df['lonn']
            elev = df['elev']
            hh = df['hh']
    
            # Difference between consecutive epochs (converted from days to seconds)
            d = 86400.0*np.diff(t)
    
            # Mask repeated epochs at the edges of 15-minute intervals
            d = ma.masked_values(d,0.0)
    
            # Search and correct residual IFB jumps at masked epochs (from leveling steps)
            i = np.where(np.append(d.mask, False) == True)[0]
            for j in range(i.size):
                dIFB = stec[i[j] + 1] - stec[i[j]]
                stec[i[j] + 1:] = stec[i[j] + 1:] - dIFB
    
                dIFB15 = stec15[i[j] + 1] - stec15[i[j]]
                stec15[i[j] + 1:] = stec15[i[j] + 1:] - dIFB15
    
            # If at least one value was masked (i.e., repeated epoch exists)
            if (d.mask.any()):
                lat = lat[np.append(~ d.mask, True)]
                lon = lon[np.append(~ d.mask, True)]
                hh = hh[np.append(~ d.mask, True)]
                t = t[np.append(~ d.mask, True)]
                stec = stec[np.append(~ d.mask, True)]
                stec15 = stec15[np.append(~ d.mask, True)]
                elev = elev[np.append(~ d.mask, True)]
    
            # First observation time
            t0 = t[0]
    
            # Index of data within each ROTI window
            i = np.floor(np.round(86400*(t - t0))/WINDOW)
    
            # Unique window indices with valid data
            j = np.unique(i)
    
            # Accumulators for ROTI calculation
            alon = []
            alat = []
            at = []
            ahh = []
            ROTI = []
            ROTI15 = []
            elev1 = []
    
            # Compute ROTI for each valid window
            for k in range(j.size):
                l = ma.masked_values(i,j[k])
    
                # Proceed only if there are at least two valid observations in the window
                if lon[l.mask].size > 1:
                    alon.append(np.mean(lon[l.mask]))
                    alat.append(np.mean(lat[l.mask]))
                    ahh.append(np.mean(hh[l.mask]))
                    at.append(np.mean(t[l.mask]))
                    elev1.append(np.mean(elev[l.mask]))
    
                    ROT = np.divide(np.diff(stec[l.mask]),86400.0*np.diff(t[l.mask])/TROTI)
                    ROT15 = np.divide(np.diff(stec15[l.mask]),86400.0*np.diff(t[l.mask])/TROTI)
    
                    # np.abs() ensures stability when ROTI is very small (e.g., ~1e-19)
                    ROTI.append(np.sqrt(np.abs(np.mean(ROT*ROT) - np.mean(ROT)**2)))
                    ROTI15.append(np.sqrt(np.abs(np.mean(ROT15*ROT15) - np.mean(ROT15)**2)))
    
            # Convert accumulators to numpy arrays
            alon = np.array(alon)
            alat = np.array(alat)
            ahh = np.array(ahh)
            at = np.array(at)
            ROTI = np.array(ROTI)
            ROTI15 = np.array(ROTI15)
            elev = np.array(elev1)
    
            # Identify time gaps to define independent arcs
            d = 86400.0*np.diff(at)
            d = ma.masked_greater_equal(d,GAP2)
    
            # Indices where time gaps occur
            i = np.where(np.append(d.mask, False) == True)[0]
    
            # Start and end indices for each arc
            i1 = np.append(0,np.where(np.append(d.mask, False) == True)[0])
            i2 = np.append(np.where(np.append(d.mask, False) == True)[0] - 1,alon.size)
    
            # Arrays for polynomial fits and bounds
            y = np.empty((ROTI.size,))
            yup = np.empty((ROTI.size,))
            ydown = np.empty((ROTI.size,))
            y15 = np.empty((ROTI15.size,))
            yup15 = np.empty((ROTI15.size,))
            ydown15 = np.empty((ROTI15.size,))
    
            # Filter out spikes from constant ambiguity intervals (e.g., multi-arc leveling)
            for j in range(i1.size):
                # Mean time of current arc
                tm = np.mean(at[i1[j]:i2[j]])
    
                if (at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0]) != 0.0:
                    # Normalized time for polynomial fitting
                    x = (at[i1[j]:i2[j]] - tm)/(at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0])
    
                    # Fit polynomials for ROTI and ROTI15
                    c = np.polyfit(x,ROTI[i1[j]:i2[j]],DE)
                    c15 = np.polyfit(x,ROTI15[i1[j]:i2[j]],DE)
    
                    # Evaluate fitted polynomials
                    y[i1[j]:i2[j]] = np.polyval(c,x)
                    y15[i1[j]:i2[j]] = np.polyval(c15,x)
    
                    # Compute residual standard deviations
                    rms = np.std(ROTI[i1[j]:i2[j]] - y[i1[j]:i2[j]])
                    rms15 = np.std(ROTI15[i1[j]:i2[j]] - y15[i1[j]:i2[j]])
                else:
                    y[i1[j]:i2[j]] = ROTI[i1[j]:i2[j]]
                    rms = 0.0
                    y15[i1[j]:i2[j]] = ROTI15[i1[j]:i2[j]]
                    rms15 = 0.0
    
                # Compute upper and lower bounds for outlier rejection
                yup[i1[j]:i2[j]] = y[i1[j]:i2[j]] + SIGMA*rms
                ydown[i1[j]:i2[j]] = y[i1[j]:i2[j]] - SIGMA*rms
                yup15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] + SIGMA*rms15
                ydown15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] - SIGMA*rms15
    
            # Mask outliers based on threshold bounds
            mask = np.abs(ROTI - y) > (yup - ydown)/2.0
    
            # Discard masked (outlier) values
            alatm = alat[~ mask]
            alonm = alon[~ mask]
            ahhm = ahh[~ mask]
            atm = at[~ mask]
            ROTIm = ROTI[~ mask]
            ROTIm15 = ROTI15[~ mask]
            elevm = elev[~ mask]

            # Apply elevation mask
            cutoff = np.where(elevm>=elev_angle)
            alat = alatm[cutoff]
            alon = alonm[cutoff]
            ahh = ahhm[cutoff]
            at = atm[cutoff]
            ROTI = ROTIm[cutoff]
            ROTI15 = ROTIm15[cutoff]
            elev = elevm[cutoff]
    
            # Discard extreme ROTI values (optional upper limit)
            cut_out = np.where(ROTI<=10)
            alat = alat[cut_out]
            alon = alon[cut_out]
            ahh = ahh[cut_out]
            at = at[cut_out]
            ROTI = ROTI[cut_out]
            ROTI15 = ROTI15[cut_out]
            elev = elev[cut_out]
    
            # Initialize dictionary to store results for this satellite
            dados_satelite = {
                'MJD': at,
                'Longitude': alon,
                'Latitude': alat,
                'Height': ahh,
                'Elevation': elev,
                'ROTI': ROTI,
                'ROTI15': ROTI15,
                'STA': estacao,
                'SAT': sat1
            }
    
            # Append this satellite's data to the list
            dados_satelites.append(dados_satelite)
    
        # Concatenate all satellite data dictionaries into a single DataFrame
        df_concatenado = pd.concat([pd.DataFrame(dados) for dados in dados_satelites], ignore_index=True)
    
        # Define the output directory and file path
        output_directory = os.path.join(destination_directory, estacao.upper())
        full_path = output_directory
        file_name = f"{estacao}_{doy}_{ano}_{satx}_ROTI.txt"
        output_file_path = os.path.join(full_path, file_name)
    
        # Ensure that the output directory exists
        os.makedirs(full_path, exist_ok=True)
    
        # Save the DataFrame to a tab-delimited text file
        df_concatenado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')

        # Choose color and label depending on satellite system
        if satx == 'G':
            color = 'blue'
            label = 'ROTI - GPS'
        elif satx == 'R':
            color = 'red'
            label = 'ROTI - GLONASS'
        else:
            color = 'gray'
            label = f'ROTI - {satx}'

        # Only plot one label per GNSS class to avoid duplicate legend entries
        plt.scatter(df_concatenado['MJD'], df_concatenado['ROTI'], marker='o', color=color, label=label)


        # Convert MJD values (as strings) to datetime objects
        start_time_mjd = min(map(float, mjd))
        start_time_datetime = datetime(1858, 11, 17) + timedelta(days=start_time_mjd)
        datetimes = [start_time_datetime + timedelta(days=float(at_val)) for at_val in mjd]

        # Configure the x-axis to show time in hours
        hours_fmt = mdates.DateFormatter('%H')
        hour_locator = mdates.HourLocator(interval=2)

        # Apply formatting and tick size adjustments
        plt.gca().xaxis.set_major_formatter(hours_fmt)
        plt.gca().xaxis.set_major_locator(hour_locator)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Set axis labels and plot title
        plt.title(f"Station: {estacao.upper()}  |  Year: {ano}  |  DOY: {doy}", fontsize=16)
        plt.ylabel('ROTI (TECU/min)', fontsize=16)
        plt.xlabel('Time (UT)', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.ylim(0, 5)
        plt.grid(True, linestyle='--', linewidth=1, color='gray')
        plt.legend(fontsize=14)
        plt.tight_layout()

        # Define the PNG file name and path
        file_name_png = f"{estacao}_{doy}_{ano}_ROTI.png"
        output_file_path_png = os.path.join(full_path, file_name_png)

        # Save the plot as a high-resolution PNG image
        plt.savefig(output_file_path_png, dpi=300)
    if show_plot:
        plt.show()
