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
    caminho_ = os.path.join(diretorio_principal)

    # Check if the directory exists
    if os.path.exists(caminho_):
        conteudo_ = os.listdir(caminho_)
        print("Files found in directory:")

        arquivos = [arquivo for arquivo in conteudo_ if arquivo.startswith(estacao) and arquivo.endswith(".RNX3")]
        arquivos_ordenados = sorted(arquivos, key=lambda x: int(x.split("_")[1][1:]))

        for arquivo in arquivos_ordenados:
            print("File found:", arquivo)

        print()
        numero_de_arquivos = len(arquivos_ordenados)
        print("Number of RINEX_LEVELLING (.RNX3) files in the directory:", numero_de_arquivos)
    else:
        print("Specified directory does not exist.")
    print()

    # === Data structure initialization ===

    date, time2, mjd = [], [], []
    pos_x, pos_y, pos_z = [], [], []
    LGF_combination, LGF_combination15 = [], []
    satellites, sta, hght, el = [], [], [], []
    lonn, latt = [], []
    obs_La, obs_Lb, obs_Lc = [], [], []
    obs_Ca, obs_Cb, obs_Cc = [], [], []

    # === Read and parse RINEX data files ===

    all_data = []

    for arquivo in arquivos_ordenados:
        caminho_arquivo = os.path.join(caminho_, arquivo)
        try:
            df = pd.read_csv(caminho_arquivo, sep='\t', engine='python')
        except Exception as e:
            print(f"[WARN] Could not read file {arquivo}: {e}")
            continue

        # Create datetime column (UTC)
        if 'date' in df.columns and 'time2' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time2'], errors='coerce')
        else:
            df['datetime'] = pd.NaT


        # === Masking using mini_flag ===
        if 'mini_flag' in df.columns:
            # Any record where mini_flag != 'N' is masked as NaN in LGF_combination
            mask_invalid = df['mini_flag'].astype(str) != 'N'
            n_masked = mask_invalid.sum()
            if n_masked > 0:
                print(f"  [INFO] Masking {n_masked} samples in {arquivo} where mini_flag != 'N'")
            df.loc[mask_invalid, 'LGF_combination'] = np.nan

        all_data.append(df)

    if not all_data:
        print("[ERROR] No RNX3 files could be read.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # Replace invalid placeholders
    df_all.replace([-999999.999, 999999.999], np.nan, inplace=True)

    # Ensure numeric conversion for alignment
    numeric_cols = [
        'mjd', 'pos_x', 'pos_y', 'pos_z', 'LGF_combination',
        'hght', 'El', 'Lon', 'Lat'
    ]
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    # Extract columns to lists (preserving your original variables)
    date               = df_all['date'].astype(str).tolist()
    time2              = df_all['time2'].astype(str).tolist()
    mjd                = df_all['mjd'].astype(str).tolist()
    pos_x              = df_all['pos_x'].astype(str).tolist()
    pos_y              = df_all['pos_y'].astype(str).tolist()
    pos_z              = df_all['pos_z'].astype(str).tolist()
    LGF_combination    = df_all['LGF_combination'].astype(str).tolist()
    #LGF_combination15  = df_all['LGF_combination15'].astype(str).tolist() if 'LGF_combination15' in df_all else [''] * len(df_all)
    satellites          = df_all['satellite'].astype(str).tolist()
    sta                 = df_all['sta'].astype(str).tolist()
    hght                = df_all['hght'].astype(str).tolist()
    el                  = df_all['El'].astype(str).tolist()
    lonn                = df_all['Lon'].astype(str).tolist()
    latt                = df_all['Lat'].astype(str).tolist()
    obs_La              = df_all['obs_La'].astype(str).tolist() if 'obs_La' in df_all else [''] * len(df_all)
    obs_Lb              = df_all['obs_Lb'].astype(str).tolist() if 'obs_Lb' in df_all else [''] * len(df_all)
    obs_Lc              = df_all['obs_Lc'].astype(str).tolist() if 'obs_Lc' in df_all else [''] * len(df_all)
    obs_Ca              = df_all['obs_Ca'].astype(str).tolist() if 'obs_Ca' in df_all else [''] * len(df_all)
    obs_Cb              = df_all['obs_Cb'].astype(str).tolist() if 'obs_Cb' in df_all else [''] * len(df_all)
    obs_Cc              = df_all['obs_Cc'].astype(str).tolist() if 'obs_Cc' in df_all else [''] * len(df_all)

    # === Save aligned merged table for inspection ===
    os.makedirs(destination_directory, exist_ok=True)
    merged_output = os.path.join(destination_directory, f"{estacao}_{doy}_{ano}_RNX3_merged.txt")

    df_all.to_csv(
        merged_output,
        sep='\t',
        index=False,
        na_rep='-999999.999',
        float_format='%11.5f',
        columns=[
            'datetime','date','time2','mjd','pos_x','pos_y','pos_z',
            'LGF_combination','satellite','sta',
            'hght','El','Lon','Lat','obs_La','obs_Lb','obs_Lc','obs_Ca','obs_Cb','obs_Cc'
        ]
    )
    print(f"[OUT] Merged and aligned RNX3 data saved to: {merged_output}")

    # === ROTI Computation Loop ===

    sat_classes = ['G','R'] # Currently supported: GPS (G) and GLONASS (R)
    palette = plt.get_cmap('tab10')
    plt.figure(figsize=(12, 6)) # Prepare one figure for all satellites

    for sat in sat_classes:
        satx=sat
        print()
        print(f"Calculating ROTI for {estacao.upper()}  |  Year: {ano}  |  DOY: {doy}")
        print()
        if satx:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        else:
            satellites_to_plot = np.unique(satellites)

        dados_satelites = []

        for idx, sat1 in enumerate(satellites_to_plot, start=1):
            print(f"Processing satellite {sat1} ({idx} of {len(satellites_to_plot)} in {satx} system)...")
            print()
            indices = np.where(np.array(satellites) == sat1)[0]

            date_filtered = []
            time2_filtered = []
            mjd_filtered = []
            pos_x_filtered = []
            pos_y_filtered = []
            pos_z_filtered = []
            LGF_combination_filtered = []
            #LGF_combination15_filtered = []
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
                #LGF_combination15_filtered.append(LGF_combination15[idx])
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
                #'LGF15': LGF_combination15_filtered,
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

            df['timestamp'] = df['date'] + ' ' + df['time']
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            columns_to_convert = ['LGF', 'mjd','lonn','latt','hh','elev']
            df[columns_to_convert] = df[columns_to_convert].astype(float)

            df.replace(-999999.999, np.nan, inplace=True)

            t = df['mjd']
            stec = df['LGF'] / akl
            #stec15 = df['LGF15'] / akl15
            lat = df['latt']
            lon = df['lonn']
            elev = df['elev']
            hh = df['hh']

            d = 86400.0*np.diff(t)
            d = ma.masked_values(d,0.0)

            i = np.where(np.append(d.mask, False) == True)[0]
            for j in range(i.size):
                dIFB = stec[i[j] + 1] - stec[i[j]]
                stec[i[j] + 1:] = stec[i[j] + 1:] - dIFB

                dIFB15 = stec15[i[j] + 1] - stec15[i[j]]
                stec15[i[j] + 1:] = stec15[i[j] + 1:] - dIFB15

            if (d.mask.any()):
                lat = lat[np.append(~ d.mask, True)]
                lon = lon[np.append(~ d.mask, True)]
                hh = hh[np.append(~ d.mask, True)]
                t = t[np.append(~ d.mask, True)]
                stec = stec[np.append(~ d.mask, True)]
                stec15 = stec15[np.append(~ d.mask, True)]
                elev = elev[np.append(~ d.mask, True)]

            t0 = t[0]
            i = np.floor(np.round(86400*(t - t0))/WINDOW)
            j = np.unique(i)

            alon = []
            alat = []
            at = []
            ahh = []
            ROTI = []
            #ROTI15 = []
            elev1 = []

            for k in range(j.size):
                l = ma.masked_values(i,j[k])
                if lon[l.mask].size > 1:
                    alon.append(np.mean(lon[l.mask]))
                    alat.append(np.mean(lat[l.mask]))
                    ahh.append(np.mean(hh[l.mask]))
                    at.append(np.mean(t[l.mask]))
                    elev1.append(np.mean(elev[l.mask]))
                    ROT = np.divide(np.diff(stec[l.mask]),86400.0*np.diff(t[l.mask])/TROTI)
                    #ROT15 = np.divide(np.diff(stec15[l.mask]),86400.0*np.diff(t[l.mask])/TROTI)
                    ROTI.append(np.sqrt(np.abs(np.mean(ROT*ROT) - np.mean(ROT)**2)))
                    #ROTI15.append(np.sqrt(np.abs(np.mean(ROT15*ROT15) - np.mean(ROT15)**2)))

            alon = np.array(alon)
            alat = np.array(alat)
            ahh = np.array(ahh)
            at = np.array(at)
            ROTI = np.array(ROTI)
            #ROTI15 = np.array(ROTI15)
            elev = np.array(elev1)

            d = 86400.0*np.diff(at)
            d = ma.masked_greater_equal(d,GAP2)

            i = np.where(np.append(d.mask, False) == True)[0]
            i1 = np.append(0,np.where(np.append(d.mask, False) == True)[0])
            i2 = np.append(np.where(np.append(d.mask, False) == True)[0] - 1,alon.size)

            y = np.empty((ROTI.size,))
            yup = np.empty((ROTI.size,))
            ydown = np.empty((ROTI.size,))
            #y15 = np.empty((ROTI15.size,))
            #yup15 = np.empty((ROTI15.size,))
            #ydown15 = np.empty((ROTI15.size,))

            for j in range(i1.size):
                tm = np.mean(at[i1[j]:i2[j]])
                if (at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0]) != 0.0:
                    x = (at[i1[j]:i2[j]] - tm)/(at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0])
                    c = np.polyfit(x,ROTI[i1[j]:i2[j]],DE)
                    #c15 = np.polyfit(x,ROTI15[i1[j]:i2[j]],DE)
                    y[i1[j]:i2[j]] = np.polyval(c,x)
                    #y15[i1[j]:i2[j]] = np.polyval(c15,x)
                    rms = np.std(ROTI[i1[j]:i2[j]] - y[i1[j]:i2[j]])
                    #rms15 = np.std(ROTI15[i1[j]:i2[j]] - y15[i1[j]:i2[j]])
                else:
                    y[i1[j]:i2[j]] = ROTI[i1[j]:i2[j]]
                    rms = 0.0
                    #y15[i1[j]:i2[j]] = ROTI15[i1[j]:i2[j]]
                    #rms15 = 0.0
                yup[i1[j]:i2[j]] = y[i1[j]:i2[j]] + SIGMA*rms
                ydown[i1[j]:i2[j]] = y[i1[j]:i2[j]] - SIGMA*rms
                #yup15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] + SIGMA*rms15
                #ydown15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] - SIGMA*rms15

            mask = np.abs(ROTI - y) > (yup - ydown)/2.0
            alatm = alat[~ mask]
            alonm = alon[~ mask]
            ahhm = ahh[~ mask]
            atm = at[~ mask]
            ROTIm = ROTI[~ mask]
            #ROTIm15 = ROTI15[~ mask]
            elevm = elev[~ mask]

            cutoff = np.where(elevm>=elev_angle)
            alat = alatm[cutoff]
            alon = alonm[cutoff]
            ahh = ahhm[cutoff]
            at = atm[cutoff]
            ROTI = ROTIm[cutoff]
            #ROTI15 = ROTIm15[cutoff]
            elev = elevm[cutoff]

            cut_out = np.where(ROTI<=10)
            alat = alat[cut_out]
            alon = alon[cut_out]
            ahh = ahh[cut_out]
            at = at[cut_out]
            ROTI = ROTI[cut_out]
            #ROTI15 = ROTI15[cut_out]
            elev = elev[cut_out]

            dados_satelite = {
                'MJD': at,
                'Longitude': alon,
                'Latitude': alat,
                'Height': ahh,
                'Elevation': elev,
                'ROTI': ROTI,
                #'ROTI15': ROTI15,
                'STA': estacao,
                'SAT': sat1
            }

            dados_satelites.append(dados_satelite)

        if not dados_satelites:
            print(f"No data found for {satx} system. Skipping...")
            continue

        df_concatenado = pd.concat([pd.DataFrame(dados) for dados in dados_satelites], ignore_index=True)

        output_directory = os.path.join(destination_directory)
        full_path = output_directory
        file_name = f"{estacao}_{doy}_{ano}_{satx}_ROTI.txt"
        output_file_path = os.path.join(full_path, file_name)

        os.makedirs(full_path, exist_ok=True)

        df_concatenado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999', float_format='%11.5f')

        if satx == 'G':
            color = 'navy'
            label = 'ROTI: L1-L2 (GPS)'
        elif satx == 'R':
            color = 'red'
            label = 'ROTI: L1-L2 (GLONASS)'
        else:
            color = 'magenta'
            label = f'ROTI - {satx}'

        base_date = datetime(1858, 11, 17)
        df_concatenado['datetime'] = df_concatenado['MJD'].astype(float).apply(lambda x: base_date + timedelta(days=x))

        plt.scatter(df_concatenado['datetime'], df_concatenado['ROTI'], marker='o', color=color, label=label)

        if satx == 'G':
            color15 = 'blue'
            label15 = 'ROTI: L1-L5 (GPS)'
        elif satx == 'R':
            color15 = 'orange'
            label15 = 'ROTI: L2-L3 (GLONASS)'
        else:
            color15 = 'darkgray'
            label15 = f'ROTI15 - {satx}'

        #plt.scatter(df_concatenado['datetime'], df_concatenado['ROTI15'], marker='o', color=color15, label=label15)

        start_time_mjd = min(map(float, mjd))
        start_time_datetime = datetime(1858, 11, 17) + timedelta(days=start_time_mjd)
        datetimes = [start_time_datetime + timedelta(days=float(at_val)) for at_val in mjd]

        hours_fmt = mdates.DateFormatter('%H')
        hour_locator = mdates.HourLocator(interval=2)
        plt.gca().xaxis.set_major_formatter(hours_fmt)
        plt.gca().xaxis.set_major_locator(hour_locator)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"Station: {estacao.upper()}  |  Year: {ano}  |  DOY: {doy}", fontsize=16)
        plt.ylabel('ROTI (TECU/min)', fontsize=16)
        plt.xlabel('Time (UT)', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.ylim(0, 5)
        plt.grid(True, linestyle='--', linewidth=1, color='gray')
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.tight_layout()

        file_name_png = f"{estacao}_{doy}_{ano}_ROTI.png"
        output_file_path_png = os.path.join(full_path, file_name_png)
        plt.savefig(output_file_path_png, dpi=300)
    if show_plot:
        plt.show()
