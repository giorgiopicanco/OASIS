#!/usr/bin/env python3
"""
======================================================================
 RNX_LEVELLING.py
 Author: Giorgio Picanço
 Date: October 2025
======================================================================
 Description:
     This script performs geometry-free leveling (GF) of GNSS observables
     derived from pre-processed RINEX data (.RNX2). It implements robust
     polynomial fitting, offset correction, arc-based segmentation, and
     interpolation of short temporal gaps (< 60 minutes), ensuring the
     leveled observables preserve both physical continuity and
     statistical robustness.

     The pipeline operates per satellite and per geometry-free
     combination (L1-L2 and L1-L5). It ensures all flagged samples
     (CYCLE SLIP, OUTLIER, GAP) remain in the final leveled arrays,
     applying the same offset corrections as valid samples while
     preventing them from affecting the regression itself.

 Dependencies:
     - Python ≥ 3.8
     - numpy
     - pandas
     - matplotlib
     - scipy
     - astropy
     - pyOASIS
======================================================================
"""

# ==============================================================
# Library Imports
# ==============================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
from scipy.constants import speed_of_light
from scipy.signal import savgol_filter
from pyOASIS import gnss_freqs
from pyOASIS import levelling_settings
import pyOASIS


# ==============================================================
# Function: remove_outliers_quartil
# ==============================================================
def remove_outliers_quartil(data):
    """
    Replace statistical outliers in a numeric sequence with NaN values
    while preserving its original length and alignment.

    This function applies an interquartile range (IQR)-based mask
    computed from the 15th and 85th percentiles. Any values lying
    outside 1.3 × IQR from these thresholds are considered outliers
    and replaced with NaN.

    Parameters
    ----------
    data : array-like
        Input numeric array (e.g., L_GF, P_GF) possibly containing NaNs.

    Returns
    -------
    numpy.ndarray
        Array of identical length, where outliers are replaced with NaN.
    """
    data = np.array(data, dtype=float)
    q1 = np.nanpercentile(data, 15)
    q3 = np.nanpercentile(data, 85)
    iqr = q3 - q1
    lower_bound = q1 - 1.3 * iqr
    upper_bound = q3 + 1.3 * iqr
    mask = (data < lower_bound) | (data > upper_bound)
    data[mask] = np.nan
    return data


# ==============================================================
# Function: process_gf_combination
# ==============================================================
def process_gf_combination(L_GF, P_GF, arcos, df, sat_class, satellite,
                           satellites_to_plot, remove_outliers_quartil,
                           label_tag="L1-L2"):
    """
    Perform geometry-free (GF) leveling between carrier-phase (L_GF)
    and code (P_GF) combinations for a given GNSS satellite.

    This function executes the full leveling pipeline:
        1. Mask data outside identified arcs.
        2. Remove mild outliers within each arc.
        3. Smooth the P_GF (code) combination.
        4. Apply robust offset leveling using Huber weights.
        5. Mask extreme residual outliers (NaN-preserved).
        6. Fit 3rd-order polynomials per arc.
        7. Align mean offsets between L-GF and P-GF.
        8. Reconnect consecutive arcs separated by < 1 h.
        9. Apply a second robust Huber correction.
       10. Interpolate short temporal gaps (≤ 30 min).
       11. Visualize leveled observables.
       12. Write results back to the main DataFrame.

    Notes
    -----
    • All flagged samples (GAP, SLIP, OUTLIER) remain in the arrays
      and receive the same offset corrections as valid data.
    • They never influence statistical fitting or weighting.
    • The procedure ensures physical continuity and numerical
      stability for subsequent TEC-related computations.

    Parameters
    ----------
    L_GF : array-like
        Geometry-free carrier-phase combination.
    P_GF : array-like
        Geometry-free pseudorange combination.
    arcos : list[list[int]]
        List of contiguous index segments (arcs) free from major
        anomalies.
    df : pandas.DataFrame
        DataFrame containing timestamps and observables.
    sat_class : str
        GNSS constellation identifier (“G”, “R”, “E”, “C”…).
    satellite : str
        Satellite PRN label (e.g., “G12”).
    satellites_to_plot : list
        List of satellites currently processed/visualized.
    remove_outliers_quartil : callable
        Function to mask outliers (IQR-based).
    label_tag : str, optional
        String identifying the GF pair (“L1-L2”, “L1-L5”, …).

    Returns
    -------
    dict
        {
            "L_adjusted": np.ndarray,   # leveled carrier phase
            "P_adjusted": np.ndarray,   # smoothed code
            "polynomial_fits": list,    # 3rd-order fits per arc
            "df": pandas.DataFrame      # DataFrame with leveled columns
        }
    """

    # ----------------------------------------------------------
    # 1. Mask samples outside valid arcs
    # ----------------------------------------------------------
    L_GF_copy = np.copy(L_GF)
    marcador_fora_arco = np.ones_like(L_GF, dtype=bool)
    for arco in arcos:
        start, end = arco[0], arco[-1]
        marcador_fora_arco[start:end] = False
    L_GF_copy[marcador_fora_arco] = np.nan
    L_GF = np.copy(L_GF_copy)

    # ----------------------------------------------------------
    # 2. Remove mild outliers within each arc (IQR rule)
    # ----------------------------------------------------------
    L_GF2 = np.copy(L_GF)
    for arco in arcos:
        start, end = arco[0], arco[-1]
        segmento = L_GF[start:end]
        if np.all(np.isnan(segmento)):
            continue
        Q1 = np.nanpercentile(segmento, 25)
        Q3 = np.nanpercentile(segmento, 75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 8 * IQR, Q3 + 8 * IQR
        mask_out = (segmento < lower) | (segmento > upper)
        L_GF2[start:end][mask_out] = np.nan
    L_GF = L_GF2

    # ----------------------------------------------------------
    # 3. Smooth P_GF by median → exponential mean → Savitzky-Golay
    # ----------------------------------------------------------
    P_GF_adjusted = np.full_like(P_GF, np.nan, dtype=float)
    for arco in arcos:
        start, end = arco[0], arco[-1]
        seg = P_GF[start:end]
        if np.all(np.isnan(seg)):
            continue
        s = pd.Series(seg)
        seg_med = s.rolling(window=7, center=True, min_periods=1).median()
        seg_smooth = seg_med.ewm(alpha=0.15, adjust=False).mean()
        try:
            wlen = min(11, len(seg_smooth))
            if wlen % 2 == 0:
                wlen -= 1
            seg_final = savgol_filter(seg_smooth.to_numpy(),
                                      window_length=wlen, polyorder=2)
        except Exception:
            seg_final = seg_smooth.to_numpy()
        P_GF_adjusted[start:end] = seg_final

    # ----------------------------------------------------------
    # 4. Robust Huber weighting to align L_GF with P_GF
    # ----------------------------------------------------------
    L_GF_adjusted = list(L_GF)
    for arco in arcos:
        start, end = arco[0], arco[-1]
        d = (P_GF_adjusted[start:end] -
             np.array(L_GF_adjusted[start:end], float))
        if np.all(np.isnan(d)):
            continue
        sigma = np.nanstd(d)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = np.nanmedian(np.abs(d - np.nanmedian(d))) * 1.4826 or 1.0
        c = 2.5 * sigma
        w = np.minimum(1.0, c / np.clip(np.abs(d), 1e-12, None))
        mean_diff = np.nansum(w * d) / np.nansum(w)
        for j in range(start, end):
            L_GF_adjusted[j] += mean_diff

    # ----------------------------------------------------------
    # 5. Mask extreme outliers again using quartile filter
    # ----------------------------------------------------------
    for arco in arcos:
        start, end = arco[0], arco[-1]
        L_GF_adjusted[start:end + 1] = remove_outliers_quartil(
            L_GF_adjusted[start:end + 1]
        )

    # ----------------------------------------------------------
    # 6. Polynomial fitting (3rd order) for each arc
    # ----------------------------------------------------------
    L_GF_adjusted_np = np.full_like(L_GF, np.nan, dtype=np.float64)
    P_GF_adjusted_np = np.full_like(P_GF, np.nan, dtype=np.float64)
    polynomial_fits = []
    polynomial_fits2 = []

    for arco in arcos:
        start, end = arco[0], arco[-1]
        if end - start < 4:
            continue
        L_GF_adjusted_np[start:end] = L_GF[start:end]
        P_GF_adjusted_np[start:end] = P_GF[start:end]
        x = np.arange(len(L_GF[start:end]), dtype=float)
        polynomial_fit = levelling_settings.fit_polynomial(x, L_GF[start:end], 3)
        polynomial_fit2 = levelling_settings.fit_polynomial(x, P_GF[start:end], 3)
        polynomial_fits.append(polynomial_fit)
        polynomial_fits2.append(polynomial_fit2)

    L_GF_adjusted_final = np.asarray(L_GF_adjusted_np, float).copy()

    # ----------------------------------------------------------
    # 7. Mean-difference alignment between L_GF and P_GF
    # ----------------------------------------------------------
    for arco in arcos:
        start, end = arco[0], arco[-1]
        diff = P_GF_adjusted_np[start:end] - L_GF_adjusted_final[start:end]
        mean_diff = np.nanmean(diff)
        if np.isfinite(mean_diff):
            L_GF_adjusted_final[start:end] += mean_diff

    # ----------------------------------------------------------
    # 8. Smooth reconnection of arcs separated by < 1 hour
    # ----------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"]).to_numpy()
    win = np.timedelta64(1, "m")
    ONE_HOUR = 3600.0
    L_arr = np.asarray(L_GF_adjusted_final, dtype=float)

    if len(arcos) > 0:
        clusters = [[arcos[0]]]
        for i in range(1, len(arcos)):
            prev = arcos[i - 1]
            curr = arcos[i]
            dt_sec = (ts[curr[0]] - ts[prev[-1]]).astype("timedelta64[s]").astype(float)
            if dt_sec > ONE_HOUR:
                clusters.append([curr])
            else:
                clusters[-1].append(curr)

        for cluster_id, cluster in enumerate(clusters, start=1):
            print(f"[CLUSTER {cluster_id}] contains {len(cluster)} arcs")
            for i in range(1, len(cluster)):
                prev = cluster[i - 1]
                curr = cluster[i]
                s_prev, e_prev = prev[0], prev[-1] + 1
                s_curr, e_curr = curr[0], curr[-1] + 1

                t_prev_end = ts[e_prev - 1]
                t_curr_start = ts[s_curr]

                idx_prev = np.where((ts >= t_prev_end - win) & (ts <= t_prev_end))[0]
                idx_prev = idx_prev[(idx_prev >= s_prev) & (idx_prev < e_prev)]
                idx_curr = np.where((ts >= t_curr_start) &
                                    (ts <= t_curr_start + win))[0]
                idx_curr = idx_curr[(idx_curr >= s_curr) & (idx_curr < e_curr)]

                m_prev = np.nanmean(L_arr[idx_prev]) if idx_prev.size else np.nan
                m_curr = np.nanmean(L_arr[idx_curr]) if idx_curr.size else np.nan
                if not np.isfinite(m_prev):
                    m_prev = L_arr[e_prev - 1]
                if not np.isfinite(m_curr):
                    m_curr = L_arr[s_curr]

                slide = m_prev - m_curr
                L_arr[s_curr:e_curr] += slide
                print(f"  → reconnected arc {i+1} with slide {slide:+.3f}")

        L_GF_adjusted_final = L_arr.tolist()
    else:
        L_GF_adjusted_final = L_arr.tolist()

    # ----------------------------------------------------------
    # 9. Final robust Huber adjustment after reconnection
    # ----------------------------------------------------------
    L_corr = np.asarray(L_GF_adjusted_final, float).copy()
    P_corr = np.asarray(P_GF_adjusted_np, float).copy()
    for arco in arcos:
        start, end = arco[0], arco[-1]
        d = (P_corr[start:end] - L_corr[start:end])
        if np.all(np.isnan(d)):
            continue
        sigma = np.nanstd(d)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = np.nanmedian(np.abs(d - np.nanmedian(d))) * 1.4826 or 1.0
        c = 2.5 * sigma
        w = np.minimum(1.0, c / np.clip(np.abs(d), 1e-12, None))
        mean_diff = np.nansum(w * d) / np.nansum(w)
        L_corr[start:end] += mean_diff
    L_GF_adjusted_final = L_corr.tolist()

    # ----------------------------------------------------------
    # 10. Interpolate gaps ≤ 30 min using time-aware linear fill
    # ----------------------------------------------------------
    df_interp = pd.DataFrame({
        "timestamp": df["timestamp"],
        "L_GF": L_GF_adjusted_final
    }).sort_values("timestamp").set_index("timestamp")

    ts = df_interp.index.values.astype("datetime64[s]").astype("int64")
    y = df_interp["L_GF"].to_numpy(dtype=float, copy=True)
    MAX_GAP_SECONDS = 30 * 60
    n = len(y)
    i = 0
    while i < n:
        if np.isfinite(y[i]):
            i += 1
            continue
        start_nan = i
        while i < n and not np.isfinite(y[i]):
            i += 1
        end_nan = i - 1
        left_idx = start_nan - 1
        right_idx = end_nan + 1
        if left_idx < 0 or right_idx >= n:
            continue
        left_val = y[left_idx]
        right_val = y[right_idx]
        if not np.isfinite(left_val) or not np.isfinite(right_val):
            continue
        gap_seconds = ts[right_idx] - ts[left_idx]
        if gap_seconds <= 0 or gap_seconds > MAX_GAP_SECONDS:
            continue
        t_left = ts[left_idx]
        t_right = ts[right_idx]
        for k in range(start_nan, end_nan + 1):
            w = (ts[k] - t_left) / (t_right - t_left)
            y[k] = (1.0 - w) * left_val + w * right_val
    L_GF_adjusted_final = y.tolist()

    # ----------------------------------------------------------
    # 11. Visualization of leveled series
    # ----------------------------------------------------------
    if satellite.startswith("G"):
        const = "GPS"
    elif satellite.startswith("R"):
        const = "GLONASS"
    elif satellite.startswith("E"):
        const = "Galileo"
    elif satellite.startswith("C"):
        const = "BeiDou"
    else:
        const = "Unknown"

    if label_tag == "L1-L2":
        cor = {"GPS": "blue", "GLONASS": "red",
               "Galileo": "purple", "BeiDou": "green"}.get(const, "gray")
        label = f"GF: L1-L2 ({const})"
    elif label_tag == "L1-L5":
        cor = {"GPS": "navy", "GLONASS": "orange",
               "Galileo": "magenta", "BeiDou": "teal"}.get(const, "gray")
        freq_tag = "L3" if const == "GLONASS" else "L5"
        label = f"GF: L1-{freq_tag} ({const})"
    else:
        cor = "gray"
        label = f"GF: {label_tag} ({const})"

    if not hasattr(plt, "_added_labels"):
        plt._added_labels = set()
    add_label = label not in plt._added_labels

    plt.scatter(df["timestamp"], L_GF_adjusted_final, marker="o", s=20,
                color=cor, label=label if add_label else None, zorder=3)
    if add_label:
        plt._added_labels.add(label)

    # ----------------------------------------------------------
    # 12. Assign leveled data back into DataFrame
    # ----------------------------------------------------------
    col_name = "LGF_combination" if label_tag == "L1-L2" else "LGF_combination15"
    if len(L_GF_adjusted_final) == len(df):
        df[col_name] = L_GF_adjusted_final
    else:
        print(f"Error: {col_name} and df do not have the same length!")

    return {
        "L_adjusted": np.array(L_GF_adjusted_final),
        "P_adjusted": P_GF_adjusted_np,
        "polynomial_fits": polynomial_fits,
        "df": df
    }
# ==============================================================
# Function: RNXlevelling
# ==============================================================
def RNXlevelling(estacao, diretorio_principal, show_plot=True):
    """
    Run the full geometry-free (GF) leveling pipeline over RINEX-derived
    screened tables (.RNX2) for a given station and day.

    This procedure:
        1) Discovers all .RNX2 files in the target directory.
        2) Aggregates core columns into master lists.
        3) Iterates per GNSS constellation class (GPS 'G', optionally 'R').
        4) Filters rows for each satellite PRN.
        5) Builds geometry-free combinations (L1-L2, L1-L5 and code).
        6) Constructs observation arcs using anomaly flags (mini_flag).
        7) Subdivides arcs on large jumps and filters by minimal length.
        8) Calls `process_gf_combination` twice (L1-L2 and L1-L5).
        9) Exports selected columns to `.RNX3` per satellite.
       10) Renders an overview plot (optional).

    Important Notes
    ---------------
    • Flags (OUTLIER, CYCLE-SLIP, GAP) are used to delimit arcs but the samples
      remain in the arrays (Option A) — they get leveled like valid ones.
    • The inter-arc reconnection and Huber corrections are performed inside
      `process_gf_combination`.
    • Export preserves NaN as the sentinel -999999.999 for downstream tools.

    Parameters
    ----------
    estacao : str
        Station code (uppercase on export, e.g., 'ARA1').
    diretorio_principal : str
        Directory path containing the `.RNX2` files for the day/station.
    show_plot : bool, optional
        Whether to show the final overview plot. Default is True.

    Returns
    -------
    None
        All per-satellite `.RNX3` files are written to disk, and an optional
        figure is shown.
    """
    # Silence unrelated library warnings printed to stderr
    sys.stderr = open(os.devnull, 'w')

    # X-axis tick locator (minutes)
    int1 = 120

    # Access the GPS broadcast frequencies used for wavelength computation
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]
    f2 = gps_freqs[2]
    f5 = gps_freqs[5]

    # GLONASS channel map used to derive per-slot frequencies (if needed)
    glonass_channels = {
        'R01':  +1,   'R02':  -4,   'R03':  +5,   'R04':  +6,
        'R05':  +1,   'R06':  -4,   'R07':  +5,   'R08':  +6,
        'R09':  -2,   'R10':  -7,   'R11':   0,   'R12':  -1,
        'R13':  -2,   'R14':  -7,   'R15':   0,   'R16':  -1,
        'R17':  +4,   'R18':  -3,   'R19':  +3,   'R20':  +2,
        'R21':  +4,   'R22':  -3,   'R23':  +3,   'R24':  +2
    }
    df_slots = pd.DataFrame(list(glonass_channels.items()), columns=['Slot', 'Channel'])
    glonass_frequencies = gnss_freqs.FREQUENCY[gnss_freqs.GLO]

    # Build a small table with GLONASS per-slot frequencies (fr1, fr2, fr3)
    data = []
    for _, row in df_slots.iterrows():
        satellite = row['Slot']
        k = row['Channel']
        row_data = [satellite]
        for _, frequency in glonass_frequencies.items():
            freq_value = frequency(k) if callable(frequency) else frequency
            row_data.append(f"{freq_value:.1f}")
        data.append(row_data)
    glo_freqs = pd.DataFrame(data, columns=['Satellite', 'fr1', 'fr2', 'fr3'])

    # ------------------------------------------------------------------
    # Locate and list the .RNX2 files in diretorio_principal
    # ------------------------------------------------------------------
    caminho_ = os.path.join(diretorio_principal)
    if os.path.exists(caminho_):
        conteudo_ = os.listdir(caminho_)
        arquivos = [arquivo for arquivo in conteudo_ if arquivo.endswith(".RNX2")]
        first = arquivos[0]
        doy = first[9:12]
        ano = first[13:17]

        # Sort by satellite number for consistent processing order
        arquivos_ordenados = sorted(arquivos, key=lambda x: int(x.split("_")[1][1:]))

        # Log the discovered files
        for arquivo in arquivos_ordenados:
            print("File:", arquivo)
        print()
        numero_de_arquivos = len(arquivos_ordenados)
        print("Number of RINEX_SCREENED (.RNX2) files in the directory:", numero_de_arquivos)
    else:
        print("The specified directory does not exist.")
    print()

    # ------------------------------------------------------------------
    # Pre-allocate containers for columns aggregated across all files
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Read all .RNX2 tables and append rows into the above lists
    # ------------------------------------------------------------------
    for arquivo in arquivos:
        caminho_arquivo = os.path.join(caminho_, arquivo)

        # Process only GPS in this run (as in your original)
        sat_classes = ['G','R']  # , 'R' can be added if needed

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
                # Append to master lists
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

    # ------------------------------------------------------------------
    # Create a single figure and reuse the axes across satellites
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 6))

    # Loop over satellite classes (GPS only here)
    for sat_class in sat_classes:
        sat = sat_class
        if sat:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        else:
            satellites_to_plot = np.unique(satellites)

        # Per-class filtered containers (reinitialized for each sat_class)
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

        # --------------------------------------------------------------
        # Process each satellite that matches the current sat_class
        # --------------------------------------------------------------
        for satellite in satellites_to_plot:
            print()
            print(f"Processing {satellite} satellite...")
            sat = satellite

            # Resolve frequencies per constellation (GPS or GLONASS)
            if sat.startswith('G'):
                # Keep original f1, f2, f5 (already set)
                f1 = f1
                f2 = f2
                f5 = f5
            elif sat.startswith('R'):
                # If GLONASS is used, look up the per-slot frequencies
                sat_row = glo_freqs.loc[glo_freqs['Satellite'] == sat]
                if not sat_row.empty:
                    f1 = float(sat_row['fr1'].values[0])
                    f2 = float(sat_row['fr2'].values[0])
                    f5 = float(sat_row['fr3'].values[0])
            else:
                f1 = f2 = f5 = None  # Fallback for unsupported constellations

            # Coefficients and wavelengths
            akl = 40.3 * 10 ** 16 * ((1 / f2 ** 2) - (1 / f1 ** 2))
            lambda1 = (speed_of_light / f1)
            lambda2 = (speed_of_light / f2)
            lambda5 = (speed_of_light / f5)

            # Indices in the master arrays corresponding to the current satellite
            indices = np.where(np.array(satellites) == satellite)[0]

            # Reinitialize filtered lists for this satellite
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

            # Copy rows for the current satellite into filtered lists
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

            # ------------------------------------------------------------------
            # Build a working DataFrame for this satellite
            # ------------------------------------------------------------------
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

            # Convert measurement columns to float
            columns_to_convert = ['L1', 'L2', 'L5', 'P1', 'P2', 'P5']
            df[columns_to_convert] = df[columns_to_convert].astype(float)

            # Replace sentinel -999999.999 with NaN
            df.replace(-999999.999, np.nan, inplace=True)

            # Build a timestamp column for plotting and time-aware operations
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time2'])

            # ------------------------------------------------------------------
            # Convert series into arrays (keep NaNs for sentinel only)
            # ------------------------------------------------------------------
            L1_array = np.nan_to_num(np.array(df['L1'].tolist(), dtype=np.float64), nan=-999999.999)
            L2_array = np.nan_to_num(np.array(df['L2'].tolist(), dtype=np.float64), nan=-999999.999)
            L5_array = np.nan_to_num(np.array(df['L5'].tolist(), dtype=np.float64), nan=-999999.999)
            P1_array = np.nan_to_num(np.array(df['P1'].tolist(), dtype=np.float64), nan=-999999.999)
            P2_array = np.nan_to_num(np.array(df['P2'].tolist(), dtype=np.float64), nan=-999999.999)
            P5_array = np.nan_to_num(np.array(df['P5'].tolist(), dtype=np.float64), nan=-999999.999)

            # Replace sentinel with NaN (only for -999999.999)
            L1_array[L1_array == -999999.999] = np.nan
            L2_array[L2_array == -999999.999] = np.nan
            L5_array[L5_array == -999999.999] = np.nan
            P1_array[P1_array == -999999.999] = np.nan
            P2_array[P2_array == -999999.999] = np.nan
            P5_array[P5_array == -999999.999] = np.nan

            # ------------------------------------------------------------------
            # Option A: do NOT zero observables at outlier indices
            # (flags are used only to delimit arcs below)
            # ------------------------------------------------------------------

            # Geometry-free combinations (use all available samples)
            L_GF = levelling_settings.geometry_free_combination_L(speed_of_light / f1, speed_of_light / f2, L1_array, L2_array)
            P_GF = levelling_settings.geometry_free_combination_C(P1_array, P2_array)
            L_GF15 = levelling_settings.geometry_free_combination_L(speed_of_light / f1, speed_of_light / f5, L1_array, L5_array)
            P_GF15 = levelling_settings.geometry_free_combination_C(P1_array, P5_array)

            # Detection thresholds (kept for compatibility; used in arc-length filtering)
            DE = 3
            Thr = DE + 1
            arc_len = 15

            # Use flags to delimit arcs (without removing samples from arrays)
            indices_outliers = df.index[df['outlier_flag'] == 'Y']
            df.loc[indices_outliers, 'cs_flag'] = 'S'

            arcos = []
            arc_atual = []
            df['mini_flag'] = 'N'
            mask = (df['outlier_flag'] == 'S') | (df['outlier_flag'] == 'Y') | (df['cs_flag'] == 'S') | (df['cs_flag'] == 'Y')
            df.loc[mask, 'mini_flag'] = 'Y'

            # Build arcs: sequences where mini_flag == 'N'
            for idx2, value in enumerate(df['mini_flag']):
                if value == 'Y':
                    if arc_atual:
                        arcos.append(arc_atual)
                        arc_atual = []
                else:
                    arc_atual.append(idx2)
            if arc_atual:
                arcos.append(arc_atual)

            print()
            for i, arc in enumerate(arcos):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, "
                      f"Start = {start_index}, End = {end_index}, Obs. = {num_observations}, Status = {status}")

            # Keep only arcs with at least 15 samples
            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1] + 1]) >= 15]

            # Outside arc → NaN (inside arcs keep all samples)
            L_GF2 = np.copy(L_GF)
            marcador_fora_arco = np.ones_like(L_GF, dtype=bool)
            for i_arc, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco[start:end] = False
            L_GF2[marcador_fora_arco] = np.nan
            L_GF = L_GF2

            # P_GF_adjusted with NaN outside arcs; inside arcs keep original values
            P_GF_adjusted = np.array(P_GF, copy=True)
            P_GF_adjusted[:] = np.nan
            for i_arc, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                P_GF_adjusted[start:end] = P_GF[start:end]

            # Subdivide arcs by large jumps in L_GF (threshold = 2)
            L_GF_adjusted = list(L_GF)
            limiar = 2
            subarcos = []
            for arc in arcos:
                L_GF_adjusted_arc = [L_GF_adjusted[i] for i in arc]
                valor = abs(np.diff(L_GF_adjusted_arc, prepend=np.nan))
                outliers = np.where(valor > limiar)[0]
                if len(outliers) > 0:
                    subarco_indices = [arc[0]]
                    for outlier in outliers:
                        subarco_indices.append(arc[outlier])
                        subarco_indices.append(arc[outlier] + 1)
                    subarco_indices.append(arc[-1])
                    for i_idx in range(0, len(subarco_indices), 2):
                        subarcos.append(list(range(subarco_indices[i_idx],
                                                   subarco_indices[i_idx + 1] + 1)))
                else:
                    subarcos.append(arc)

            # Replace arcs with subarcs and log their stats
            arcos = subarcos
            for i, arc in enumerate(arcos):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, "
                      f"Start = {start_index}, End = {end_index}, Obs. = {num_observations}, Status = {status}")

            # Final filtering by minimum arc length
            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1] + 1]) >= 15]

            # ===============================================================
            # Double call: L1-L2 (GPS) / L1-L5 (GPS) — anomalies included
            # ===============================================================
            result_L12 = process_gf_combination(
                L_GF=L_GF,
                P_GF=P_GF,
                arcos=arcos,
                df=df,
                sat_class=sat_class,
                satellite=satellite,
                satellites_to_plot=satellites_to_plot,
                remove_outliers_quartil=remove_outliers_quartil,
                label_tag="L1-L2"
            )

            # result_L15 = process_gf_combination(
            #     L_GF=L_GF15,
            #     P_GF=P_GF15,
            #     arcos=arcos,
            #     df=df,
            #     sat_class=sat_class,
            #     satellite=satellite,
            #     satellites_to_plot=satellites_to_plot,
            #     remove_outliers_quartil=remove_outliers_quartil,
            #     label_tag="L1-L5"
            # )

            # --------------------------------------------------------------
            # Select columns to export to .RNX3
            # --------------------------------------------------------------
            colunas_desejadas = [
                'date', 'time2', 'mjd', 'pos_x', 'pos_y', 'pos_z',
                # 'LGF_combination', 'LGF_combination15', 'mini_flag', 'satellite',
                'LGF_combination', 'mini_flag', 'satellite',
                'sta', 'hght', 'El', 'Lon', 'Lat', 'obs_La', 'obs_Lb',
                'obs_Lc', 'obs_Ca', 'obs_Cb', 'obs_Cc'
            ]
            df_selecionado = df[colunas_desejadas]

            # Output paths and filenames
            output_directory = os.path.join(str(ano), str(doy), estacao.upper())
            full_path = os.path.join(diretorio_principal)
            file_name = f"{estacao}_{satellite}_{doy}_{ano}.RNX3"
            fig_name = f"{estacao}_{doy}_{ano}.png"
            output_file_path = os.path.join(full_path, file_name)
            output_fig_path = os.path.join(full_path, fig_name)

            # Ensure output directory exists and export
            os.makedirs(full_path, exist_ok=True)
            df_selecionado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')

            print(f"Data exported to {output_file_path}.")

        # --------------------------------------------------------------
        # Titles, labels, formatting and legend for the full figure
        # --------------------------------------------------------------
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

    # Show the plot if requested
    if show_plot:
        plt.show()

