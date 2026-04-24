#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============
# Imports
# ===============
import pyOASIS
import georinex as gr
from pyOASIS import settings
from pyOASIS import linear_combinations
from pyOASIS import gnss_freqs
import datetime as dt
import os
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from astropy.time import Time
from scipy.constants import speed_of_light
from scipy.ndimage import uniform_filter1d
import sys
import time
import warnings
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from datetime import datetime
from scipy.optimize import curve_fit
import itertools
from pyOASIS import screening_settings
from collections import OrderedDict


# ==========================================================
# Helper function (from the first block)
# ==========================================================
def rescale_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    final_data = scaled_data * 20 - 10
    return final_data


# ==========================================================
# process_combination (from the first block)
# ==========================================================
def process_combination(df, arcs, column_name, ARCL, LMW, flags, timep, label="L1-L2"):
    """
    df: main DataFrame
    arcs: list of valid arcs (lists of indices)
    column_name: name of the column containing LMW data (e.g., 'LMW2' or 'LMW3')
    ARCL: minimum number of points required for polynomial fitting
    LMW: array or series corresponding to the combination
    flags: modifiable list of flags ('C' and 'S')
    timep: main time series or index (used for printouts)
    label: 'L1-L2' or 'L1-L5'
    """
    idx_total = []

    for arc in arcs:
        arc_data = df.iloc[arc]
        time_idx = df.index[arc]  # local variable; does not change logic

        # Skip short arcs
        if len(arc_data) < ARCL:
            continue

        # Prepare variables
        x = arc_data.index.astype(np.int64) // 10**9  # convert to seconds
        xx = arc_data['time']
        y = arc_data[column_name].values
        y_rescaled = rescale_data(y)
        delta_y = np.diff(y_rescaled, prepend=np.nan)

        # ----------------------------------------------------------
        # Bidirectional anomaly detection (as in your code)
        # ----------------------------------------------------------

        # Polynomial fit forward
        p_fwd = Polynomial.fit(x[1:], delta_y[1:], 3)
        delta_y_fit_fwd = p_fwd(x)
        residuals_fwd = abs(delta_y - delta_y_fit_fwd)

        # Polynomial fit backward
        x_rev = x[::-1]
        delta_y_rev = delta_y[::-1]
        p_bwd = Polynomial.fit(x_rev[1:], delta_y_rev[1:], 3)
        delta_y_fit_bwd = p_bwd(x_rev)
        residuals_bwd = abs(delta_y_rev - delta_y_fit_bwd)[::-1]

        # Combine both directions (maximum deviation)
        residuals = np.maximum(residuals_fwd, residuals_bwd)

        # Quartiles and IQR
        Q1 = np.nanpercentile(residuals, 5)
        Q3 = np.nanpercentile(residuals, 90)
        IQR = Q3 - Q1

        # Thresholds
        outlier_threshold = 5
        high_residual_threshold = 1

        outlier_mask = (residuals < Q1 - outlier_threshold * IQR) | (residuals > Q3 + outlier_threshold * IQR)
        high_residuals_mask = residuals > high_residual_threshold
        other_residuals_mask = ~(outlier_mask | high_residuals_mask)

        indices_outliers = arc[0] + np.where(outlier_mask)[0]
        indices_high_residuals = arc[0] + np.where(high_residuals_mask)[0]

        # Merge indices without duplicates
        combined_indices = np.union1d(indices_outliers, indices_high_residuals)
        idx_total.append(combined_indices)

        print(f"Combined indices ({label}):", combined_indices)

    # Concatenate all combined indices into a single NumPy array
    if idx_total:
        idx_total = np.concatenate(idx_total)

    # Update flags (NaN → 'S')
    if not np.all(np.isnan(LMW)):
        nan_indices = np.where(np.isnan(LMW))[0]
        for idx in nan_indices:
            flags[idx] = 'S'

    # Update flags based on combined indices
    for idx in idx_total:
        if idx < len(flags):
            flags[idx] = 'S'

    # Display flagged indices
    for idx in idx_total:
        print(f"{label} (+)", idx, timep.iloc[idx])

    return flags, idx_total


# ==========================================================
# RNXclean (from the first block) — LOGIC UNCHANGED
# ==========================================================
def RNXclean(station_name, day_of_year, year, input_folder, orbit_folder, output_folder):
    """
    Robust GNSS RINEX Cleaning and SP3 Matching
    -------------------------------------------
    Processes GPS and GLONASS constellations, handling RINEX v2/v3 observables,
    assigning correct frequencies per satellite, and matching SP3 orbits.
    """
    # =====================================================
    # Accessing frequencies of GPS and GLONASS
    # =====================================================
    gps_freqs_local = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs_local[1]
    f2 = gps_freqs_local[2]
    f5 = gps_freqs_local[5]

    # ==========================================
    # GLONASS Channel Table (Defined Inline)
    # ==========================================
    glonass_channels = {
        'R01':  +1,   'R02':  -4,   'R03':  +5,   'R04':  +6,
        'R05':  +1,   'R06':  -4,   'R07':  +5,   'R08':  +6,
        'R09':  -2,   'R10':  -7,   'R11':   0,   'R12':  -1,
        'R13':  -2,   'R14':  -7,   'R15':   0,   'R16':  -1,
        'R17':  +4,   'R18':  -3,   'R19':  +3,   'R20':  +2,
        'R21':  +4,   'R22':  -3,   'R23':  +3,   'R24':  +2
    }

    # Convert dictionary to DataFrame for backward compatibility
    df_slots = pd.DataFrame(list(glonass_channels.items()), columns=['Slot', 'Channel'])

    # Load GLONASS frequency constants (unchanged)
    glonass_frequencies = gnss_freqs.FREQUENCY[gnss_freqs.GLO]

    data = []
    for _, row in df_slots.iterrows():
        satellite = row['Slot']
        k = row['Channel']
        row_data = [satellite]
        for _, frequency in glonass_frequencies.items():
            freq_value = frequency(k) if callable(frequency) else frequency
            row_data.append(f"{freq_value:.1f}")
        data.append(row_data)
    glo_freqs_df = pd.DataFrame(data, columns=['Satellite', 'fr1', 'fr2', 'fr3'])

    # =====================================================
    # Parameters
    # =====================================================
    h1 = 0
    n_hours = 24
    int1 = 120
    ARCL = 15
    constellations = ['G', 'R']  # GPS and GLONASS (so far)

    # =====================================================
    # Process each constellation
    # =====================================================
    for c in constellations:
        sat_class = c
        print(f"\n{'=' * 80}")
        print(f"[INFO] Processing constellation: {sat_class}")
        print(f"{'=' * 80}\n")

        # ------------------------------------------------------
        # RINEX FILE OPENING
        # ------------------------------------------------------
        version_number = '1'
        ext_order = ['o', 'd']  # preference order: .yyo → .yyd
        version_candidates = [version_number, ('0' if version_number == '1' else '1')]
        rinex_file_path = None
        used_ext = None
        used_version = None
        for ext in ext_order:
            for ver in version_candidates:
                year_format = f"{year[-2:]}{ext}"
                candidate = f"{input_folder}/{station_name.lower()}{day_of_year}{ver}.{year_format}"
                if os.path.exists(candidate):
                    rinex_file_path = candidate
                    used_ext = ext
                    used_version = ver
                    break
            if rinex_file_path:
                break

        if not rinex_file_path:
            print(f"[ERROR] No RINEX file found for station {station_name} on DOY {day_of_year} ({year}). Tried the following:")
            for ext in ext_order:
                for ver in version_candidates:
                    yf = f"{year[-2:]}{ext}"
                    print(f"    - {input_folder}/{station_name.lower()}{day_of_year}{ver}.{yf}")
            continue
        else:
            version_number = used_version
            print(f"[INFO] File {rinex_file_path} found (ext=.{used_ext}, version_number={version_number})")

        warnings.filterwarnings(
            "ignore",
            message="Converting non-nanosecond precision datetime values",
            category=UserWarning
        )
        # Robust loading
        try:
            obs_data = gr.load(rinex_file_path)
        except IndexError:
            print(f"[WARNING] IndexError while loading: {rinex_file_path}")
            print("[INFO] Retrying with 'use=None' (all constellations)...")
            obs_data = gr.load(rinex_file_path, use=None)

        all_sats = np.unique(obs_data.sv.values)
        print(f"[INFO] Loaded {len(all_sats)} satellites: {list(all_sats)}")
        # -------------------------------
        # ORBIT FILE OPENING
        # -------------------------------
        interval = getattr(obs_data, "interval", np.nan)
        if np.isnan(interval):
            t = Time(obs_data.time).mjd
            diffs = np.diff(t) * 86400
            freq = int(np.nanmin(diffs)) if len(diffs) > 0 else 30
        else:
            freq = int(interval)

        freq = int(getattr(obs_data, "interval", 30))
        file_path = os.path.join(orbit_folder, f'ORBITS_{year}_{day_of_year}.SP3')
        if not os.path.exists(file_path):
            print(f"[ERROR] SP3 file not found: {file_path}")
            continue
        print(f"[INFO] Processing SP3 file: {file_path}")
        column_names = ["Date", "Time", "Satellite", "X", "Y", "Z"]
        try:
            df2 = pd.read_csv(file_path, sep=r"\s+|\t+", engine="python", header=0, names=column_names)
        except Exception as e:
            print(f"[ERROR] Failed to read SP3 file: {e}")
            continue
        # Normalize satellite names (remove 'P')
        df2["Satellite"] = df2["Satellite"].astype(str).apply(lambda s: s[1:] if s.startswith("P") else s)
        print(f"[INFO] SP3 file loaded with {len(df2)} entries.")
        # -----------------------------------------------------
        # MATCH RINEX x SP3
        # -----------------------------------------------------
        sv_rinex = [str(sv) for sv in np.unique(obs_data.sv.values)]
        sv_sp3 = [str(sv) for sv in np.unique(df2["Satellite"])]
        satellites_to_plot = [sv for sv in sv_rinex if sv.startswith(sat_class)]
        ipp_uniq = [sv for sv in sv_sp3 if sv.startswith(sat_class)]
        common_elements_list = sorted(set(satellites_to_plot).intersection(set(ipp_uniq)), key=lambda x: int(x[-2:]))
        if len(common_elements_list) == 0:
            print(f"[WARNING] No matching satellites for class {sat_class}.")
            continue
        print(f"[INFO] Matched satellites ({sat_class}): {common_elements_list}\n")
        # -----------------------------------------------------
        # SATELLITE LOOP
        # -----------------------------------------------------
        for sat in common_elements_list:
            print(f"\n[PROCESSING] Satellite: {sat}")
            # Assign frequencies
            if sat_class == 'G':
                f1, f2, f5 = gps_freqs_local[1], gps_freqs_local[2], gps_freqs_local[5]
            elif sat_class == 'R':
                sat_row = glo_freqs_df.loc[glo_freqs_df['Satellite'] == sat]
                if not sat_row.empty:
                    f1 = float(sat_row['fr1'].values[0])
                    f2 = float(sat_row['fr2'].values[0])
                    f5 = float(sat_row['fr3'].values[0])
                else:
                    f1 = f2 = f5 = np.nan
            # -------------------------------------------------
            # Carrier Phases
            # -------------------------------------------------
            L1 = np.array(obs_data["L1"].sel(sv=sat)) if "L1" in obs_data else np.full(len(obs_data.time), np.nan)
            L2 = np.array(obs_data["L2"].sel(sv=sat)) if "L2" in obs_data else np.full(len(obs_data.time), np.nan)
            L5 = np.array(obs_data["L5"].sel(sv=sat)) if "L5" in obs_data else np.full(len(obs_data.time), np.nan)
            # -------------------------------------------------
            # Code observables with robust fallback
            # -------------------------------------------------
            # P1/C1
            if "P1" in obs_data:
                P1 = np.array(obs_data["P1"].sel(sv=sat))
                code_obs1 = "P1"
            elif "C1" in obs_data:
                P1 = np.array(obs_data["C1"].sel(sv=sat))
                code_obs1 = "C1"
            else:
                P1 = np.full_like(L1, np.nan)
                code_obs1 = "None"
            if np.all(np.isnan(P1)) and "C1" in obs_data:
                P1 = np.array(obs_data["C1"].sel(sv=sat))
                code_obs1 = "C1"
            # P2/C2
            if "P2" in obs_data:
                P2 = np.array(obs_data["P2"].sel(sv=sat))
                code_obs2 = "P2"
            elif "C2" in obs_data:
                P2 = np.array(obs_data["C2"].sel(sv=sat))
                code_obs2 = "C2"
            else:
                P2 = np.full_like(L2, np.nan)
                code_obs2 = "None"
            if np.all(np.isnan(P2)) and "C2" in obs_data:
                P2 = np.array(obs_data["C2"].sel(sv=sat))
                code_obs2 = "C2"
            # P5/C5
            if "P5" in obs_data:
                P5 = np.array(obs_data["P5"].sel(sv=sat))
                code_obs5 = "P5"
            elif "C5" in obs_data:
                P5 = np.array(obs_data["C5"].sel(sv=sat))
                code_obs5 = "C5"
            else:
                P5 = np.full_like(L5, np.nan)
                code_obs5 = "None"
            if np.all(np.isnan(P5)) and "C5" in obs_data:
                P5 = np.array(obs_data["C5"].sel(sv=sat))
                code_obs5 = "C5"
            if np.all(np.isnan(P5)):
                print(f"[WARNING] {sat}: All L5 ({code_obs5}) are NaN")
            print(f"[INFO] {sat}: Using {code_obs1}/{code_obs2}/{code_obs5}")

            # =====================================================
            # BUILD DATAFRAME
            # =====================================================
            df = pd.DataFrame({'time': obs_data.time})
            df.set_index('time', inplace=True)
            df['index'] = np.arange(len(df))
            df['mjd'] = Time(df.index).mjd
            df['timestamp'] = pd.to_datetime(df.index)
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
            df['L1'], df['L2'], df['L5'] = L1, L2, L5
            df['P1'], df['P2'], df['P5'] = P1, P2, P5
            df['satellite'] = sat
            df['station'] = station_name.upper()
            df['position'] = [obs_data.position] * len(df)
            df['mjd'] = ["{:.12f}".format(v) for v in df['mjd']]

            timep = df['time']
            L1, L2, L5, P1, P2, P5 = df['L1'], df['L2'], df['L5'], df['P1'], df['P2'], df['P5']
            mjd, date_col, time_col = df['mjd'], df['date'], df['time']

            df['LMW12'] = linear_combinations.melbourne_wubbena(f1, f2, L1, L2, P1, P2)
            df['LMW15'] = linear_combinations.melbourne_wubbena(f1, f5, L1, L5, P1, P5)

            station_coords = df['position'].iloc[0]
            # convert coords to floats
            coords_list = [float(coord) for coord in station_coords]
            obs_x, obs_y, obs_z = coords_list[0], coords_list[1], coords_list[2]

            # =====================================================
            # IPP/interpolation (kept)
            # =====================================================
            all_data = []
            all_time_s = []
            all_sat_s = []
            all_longitude_s = []
            all_latitude_s = []
            all_elevation_s = []

            indices = np.where(df2['Satellite'] == sat)[0]
            df_filtered_sp3 = df2.iloc[indices]

            import time as _time
            formatted_time = _time.strftime("%H:%M:%S", _time.localtime(_time.time()))
            from datetime import datetime as _dt, timedelta

            rate = freq

            all_data = []
            all_time_s = []
            all_sat_s = []
            all_longitude_s = []
            all_latitude_s = []
            all_elevation_s = []

            last_data_by_satellite = {}

            for _, row in df_filtered_sp3.iterrows():
                date_s = row["Date"]
                sat_s = row["Satellite"]
                sx = row["X"]
                sy = row["Y"]
                sz = row["Z"]
                time_s = row["Time"]

                lon, lat, alt = settings.convert_coords(obs_x, obs_y, obs_z, to_radians=True)
                ip = settings.IonosphericPiercingPoint(sx, sy, sz, obs_x, obs_y, obs_z)
                elevation = ip.elevation(lat, lon)
                lat_ip, lon_ip = ip.coordinates(lat, lon)

                current_time = _dt.strptime(f"{date_s} {time_s}", "%d-%m-%Y %H:%M:%S")

                if sat_s in last_data_by_satellite:
                    last_known = last_data_by_satellite[sat_s]
                    last_time = last_known['date_time']
                    delta_t = (current_time - last_time).total_seconds()

                    if delta_t > rate:
                        n_steps = int(delta_t // rate)
                        for step in range(1, n_steps):
                            interp_time = last_time + timedelta(seconds=step * rate)
                            frac = step * rate / delta_t
                            lon_interp = last_known['lon'] + frac * (lon_ip - last_known['lon'])
                            lat_interp = last_known['lat'] + frac * (lat_ip - last_known['lat'])
                            elv_interp = last_known['elevation'] + frac * (elevation - last_known['elevation'])

                            all_data.append(interp_time.strftime("%d-%m-%Y"))
                            all_time_s.append(interp_time.strftime("%H:%M:%S"))
                            all_sat_s.append(sat_s)
                            all_longitude_s.append(lon_interp)
                            all_latitude_s.append(lat_interp)
                            all_elevation_s.append(elv_interp)

                last_data_by_satellite[sat_s] = {
                    'date_time': current_time,
                    'lon': lon_ip,
                    'lat': lat_ip,
                    'elevation': elevation
                }

                all_data.append(date_s)
                all_time_s.append(time_s)
                all_sat_s.append(sat_s)
                all_longitude_s.append(lon_ip)
                all_latitude_s.append(lat_ip)
                all_elevation_s.append(elevation)

            final_hour = _dt.strptime(f"{date_s} 23:59:45", "%d-%m-%Y %H:%M:%S")
            for satellite_name, data_last in last_data_by_satellite.items():
                last_record = data_last
                while last_record['date_time'] < final_hour:
                    last_record['date_time'] += timedelta(seconds=rate)
                    all_data.append(last_record['date_time'].strftime("%d-%m-%Y"))
                    all_time_s.append(last_record['date_time'].strftime("%H:%M:%S"))
                    all_sat_s.append(satellite_name)
                    all_longitude_s.append(last_record['lon'])
                    all_latitude_s.append(last_record['lat'])
                    all_elevation_s.append(last_record['elevation'])

            # Convert to Series
            all_date = pd.Series(all_data)
            all_time_series = pd.Series(all_time_s)
            all_sat_series = pd.Series(all_sat_s)
            all_longitude_series = pd.Series(all_longitude_s)
            all_latitude_series = pd.Series(all_latitude_s)
            all_elevation_series = pd.Series(all_elevation_s)

            # Smooth elevation with moving average
            window_size = 100
            all_elevation_smooth = pd.Series(all_elevation_series).rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean()
            all_elevation_series = all_elevation_smooth

            combined_df = pd.DataFrame({
                "Date": all_date,
                "Time": all_time_series,
                "SAT": all_sat_series,
                "Longitude": all_longitude_series,
                "Latitude": all_latitude_series,
                "Elevation": all_elevation_series
            })

            combined_df['Time'] = pd.to_datetime(combined_df['Time'], format='%H:%M:%S')
            df['time'] = df['time'].astype(str)
            combined_df['Time'] = pd.to_datetime(combined_df['Time']).dt.time
            df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time

            common_times = set(combined_df['Time']).intersection(set(df['time']))
            ref_len = min(len(combined_df), len(df))

            combined_df = combined_df[combined_df['Time'].isin(common_times)].iloc[:ref_len]
            df = df[df['time'].isin(common_times)].iloc[:ref_len]

            cols_df1 = ['Elevation', 'Longitude', 'Latitude']
            combined_df_selected = combined_df[cols_df1]

            L1, L2, L5, P1, P2, P5 = df['L1'], df['L2'], df['L5'], df['P1'], df['P2'], df['P5']
            mjd, date_col, time_col = df['mjd'], df['date'], df['time']

            df['pos_x'], df['pos_y'], df['pos_z'] = obs_x, obs_y, obs_z
            df['height'] = np.full(len(L1), 450.0)

            cols_df2 = [
                'date', 'time', 'mjd', 'pos_x', 'pos_y', 'pos_z',
                'L1', 'L2', 'L5', 'P1', 'P2', 'P5', 'satellite', 'station', 'height'
            ]
            df_filtered_selected = df[cols_df2]

            if len(df_filtered_selected) == len(combined_df_selected):
                df_final = pd.concat([df_filtered_selected, combined_df_selected], axis=1)
            else:
                print("Error: df_filtered and combined_df do not have the same length!")
                sys.exit()

            LMW2 = df['LMW12']
            LMW3 = df['LMW15']
            df['LMW2'] = LMW2
            df['LMW3'] = LMW3

            abs_elevation = abs(combined_df_selected['Elevation'])
            combined_df_selected.index = df.index
            indices_low_elevation = combined_df_selected.index[abs_elevation < 10]

            cols_nan = ['LMW2', 'LMW3']
            df.loc[indices_low_elevation, cols_nan] = np.nan

            LMW = np.array(df['LMW2'])
            LMW15 = np.array(df['LMW3'])

            arcs = []
            current_arc = []
            for idx_i, value in enumerate(LMW):
                if np.isnan(value):
                    if current_arc:
                        arcs.append(current_arc)
                        current_arc = []
                else:
                    current_arc.append(idx_i)
            if current_arc:
                arcs.append(current_arc)

            print()
            print('Melbourne-Wubbena combination for L1-L2')
            print()
            for i_arc, arc in enumerate(arcs):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i_arc + 1}: Start index = {start_index}, End index = {end_index}, "
                      f"Number of observations = {num_observations}, Status = {status}")

            arcs15 = []
            current_arc15 = []
            for idx_i, value in enumerate(LMW15):
                if np.isnan(value):
                    if current_arc15:
                        arcs15.append(current_arc15)
                        current_arc15 = []
                else:
                    current_arc15.append(idx_i)
            if current_arc15:
                arcs15.append(current_arc15)

            print()
            print()
            print('Melbourne-Wubbena combination for L1-L5')
            print()
            for i_arc, arc in enumerate(arcs15):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Kept" if num_observations >= 15 else "Discarded"
                print(f"Arc {i_arc + 1}: Start index = {start_index}, End index = {end_index}, "
                      f"Number of observations = {num_observations}, Status = {status}")

            LMW = np.array(LMW)
            LMW15 = np.array(LMW15)

            print()
            print()

            # Flags — call process_combination for L1–L2 (L1–L5 is commented as in your code)
            flags = ['C'] * len(LMW)
            flags, idx_total_L12 = process_combination(df, arcs, 'LMW2', ARCL, LMW, flags, timep, label="L1-L2")
            # flags, idx_total_L15 = process_combination(df, arcs15, 'LMW3', ARCL, LMW15, flags, timep, label="L1-L5")

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
                'mjd': df['mjd'],  # we will format below
                'pos_x': df['pos_x'],
                'pos_y': df['pos_y'],
                'pos_z': df['pos_z'],
                'L1': df['L1'].round(6),
                'L2': df['L2'].round(6),
                'L5': df['L5'].round(6),
                'P1': df['P1'].round(6),
                'P2': df['P2'].round(6),
                'P5': df['P5'].round(6),
                'cs_flags': flags,
                'satellite': satellite_values,
                'sta': station,
                'hght': df['height'].round(2),          # will be zero-padded to 2
                'El': combined_df['Elevation'].round(4),# will be zero-padded to 4
                'Lon': combined_df['Longitude'].round(4),
                'Lat': combined_df['Latitude'].round(4),
            })

            export_df.rename(columns={'P1': code_obs1}, inplace=True)
            export_df.rename(columns={'P2': code_obs2}, inplace=True)
            export_df.rename(columns={'P5': code_obs5}, inplace=True)

            col_order = [
                'date','time','mjd','pos_x','pos_y','pos_z',
                'L1','L2','L5', code_obs1, code_obs2, code_obs5,
                'cs_flags','satellite','sta','hght','El','Lon','Lat'
            ]
            export_df = export_df[col_order]

            export_df['mjd'] = pd.to_numeric(export_df['mjd'], errors='coerce')
            export_df['mjd'] = export_df['mjd'].apply(lambda v: f"{v:.6f}" if pd.notna(v) else np.nan)

            for col, fmt in [('El', '{:.4f}'), ('Lon', '{:.4f}'), ('Lat', '{:.4f}'), ('hght', '{:.2f}')]:
                export_df[col] = pd.to_numeric(export_df[col], errors='coerce')
                export_df[col] = export_df[col].apply(lambda v: fmt.format(v) if pd.notna(v) else np.nan)

            for col in ['pos_x','pos_y','pos_z','L1','L2','L5', code_obs1, code_obs2, code_obs5]:
                export_df[col] = pd.to_numeric(export_df[col], errors='coerce')
                export_df[col] = export_df[col].apply(lambda v: f"{v:.6f}" if pd.notna(v) else np.nan)

            obj_cols = export_df.select_dtypes(include=['object']).columns
            for c in obj_cols:
                export_df[c] = (
                    export_df[c]
                    .astype(str)
                    .str.replace('\t', ' ', regex=False)
                    .str.replace('\r', ' ', regex=False)
                    .str.replace('\n', ' ', regex=False)
                    .str.strip()
                    .replace({'nan': np.nan})
                )

            numeric_cols = ['pos_x','pos_y','pos_z','L1','L2','L5', code_obs1, code_obs2, code_obs5]
            for c in numeric_cols:
                export_df[c] = pd.to_numeric(export_df[c], errors='coerce')

            output_directory = os.path.join(output_folder)
            os.makedirs(output_directory, exist_ok=True)
            file_name = f"{station_name.upper()}_{sat}_{day_of_year}_{year}.RNX1"
            output_file_path = os.path.join(output_directory, file_name)

            export_df.to_csv(
                output_file_path,
                sep='\t',
                index=False,
                na_rep='-999999.999',
                lineterminator='\n'
            )
            print(f"[OK] Saved: {output_file_path}")

            # Store IPPs (kept)
            if 'all_satellites_df' not in locals():
                all_satellites_df = []
            all_satellites_df.append(export_df[['Lon', 'Lat', 'El', 'satellite']])

        # ===========================
        # END satellite loop
        # ===========================

    # ==========================================================
    # >>> AUTOMATIC CONNECTION: call the second code here <<<
    # ==========================================================
    print("\n" + "=" * 80)
    print("[INFO] Triggering RNXScreening in the output directory (RNX1 -> RNX2):")
    print(f"       {output_folder}")
    print("=" * 80 + "\n")
    RNXScreening(output_folder)  # <<< ONLY connection added


# ==========================================================
# detect_and_plot_arcs_before_after (from the second block)
# ==========================================================
def detect_and_plot_arcs_before_after(
    df,
    arcos_validos,
    series,
    label_pair="L1-L2",
    rescale_func=None,
    fit_poly_func=None,
    outlier_flag_column="outlier_flag",
    plot=False,  # disabled by default
    save_dir=None,
    sat_id=None
):
    import numpy as np
    from numpy.polynomial import Polynomial

    thr1 = 2  # IQR threshold factor

    # ensure flag column
    if outlier_flag_column not in df.columns:
        df[outlier_flag_column] = 'N'

    all_removed = []          # global anomaly indices
    all_sign_changes = []     # global sign-change indices

    # Main loop per arc
    for i_arc, arc in enumerate(arcos_validos, start=1):
        start, end = arc[0], arc[-1]
        t = df['timestamp'].iloc[start:end+1].to_numpy()
        y_raw = np.asarray(series[start:end+1], dtype=float)

        if len(t) < 5 or np.all(np.isnan(y_raw)):
            continue

        # 1) rescale + derivative
        y_ref = rescale_func(y_raw) if rescale_func else y_raw.copy()
        dy = np.diff(y_ref, prepend=np.nan)

        # 2) polynomial fit (3rd order)
        x_sec = (df['timestamp'].iloc[start:end+1] - df['timestamp'].iloc[start]).dt.total_seconds().to_numpy()
        if np.isfinite(dy[1:]).sum() >= 4:
            p = Polynomial.fit(x_sec[1:], dy[1:], 3)
            dy_fit = p(x_sec)
        else:
            dy_fit = np.zeros_like(dy)

        resid = dy - dy_fit

        # 3) mini-arcs by sign change
        mini_arcos, mini_atual, prev_sign = [], [], None
        sign_changes_local = []

        for k, val in enumerate(resid):
            s = np.sign(val) if np.isfinite(val) else 0.0
            if prev_sign is None:
                prev_sign = s
            if s != prev_sign:
                sign_changes_local.append(start + k)
                if mini_atual:
                    mini_arcos.append(mini_atual)
                mini_atual = []
            mini_atual.append(k)
            prev_sign = s

        if mini_atual:
            mini_arcos.append(mini_atual)

        mini_kept = [m for m in mini_arcos if len(m) >= 4]
        if len(mini_kept) == 0:
            mini_kept = [list(range(0, len(t)))]

        all_sign_changes.extend(sign_changes_local)

        removed_local = []

        # 4) anomaly detection per mini-arc
        for m in mini_kept:
            m0, m1 = m[0], m[-1]
            idx_loc = np.arange(m0, m1+1)
            res_m = resid[idx_loc].copy()

            if fit_poly_func and np.isfinite(res_m).sum() >= 4:
                try:
                    m_fit   = fit_poly_func(np.arange(res_m.size), res_m, 3)
                    new_res = np.abs(res_m - m_fit)
                except Exception:
                    new_res = np.abs(res_m)
            else:
                new_res = np.abs(res_m)

            if np.all(np.isnan(new_res)):
                mask_out = np.zeros_like(new_res, dtype=bool)
            else:
                q1, q3 = np.nanpercentile(new_res, [15, 85])
                iqr = q3 - q1
                thr_low  = q1 - thr1 * iqr
                thr_high = q3 + thr1 * iqr
                mask_out = (new_res < thr_low) | (new_res > thr_high)

            micro = np.abs(np.diff(new_res, prepend=np.nan))
            if np.all(np.isnan(micro)):
                mask_micro = np.zeros_like(new_res, dtype=bool)
            else:
                q1m, q3m = np.nanpercentile(micro, [15, 85])
                iqrm = q3m - q1m
                thr_micro_low  = q1m - thr1 * iqrm
                thr_micro_high = q3m + thr1 * iqrm
                mask_micro = (micro < thr_micro_low) | (micro > thr_micro_high)

            anom_local_mask = mask_out | mask_micro
            anom_loc_idx  = idx_loc[anom_local_mask]
            anom_glob_idx = start + anom_loc_idx
            removed_local.extend(anom_glob_idx.tolist())

        if removed_local:
            all_removed.extend(removed_local)

    # 5) Update flags in DataFrame
    if all_removed:
        df.loc[all_removed, outlier_flag_column] = 'Y'

    if "sign_change_flag" not in df.columns:
        df["sign_change_flag"] = 'N'
    if all_sign_changes:
        df.loc[all_sign_changes, "sign_change_flag"] = 'S'

    return {
        "removed_indices": sorted(set(all_removed)),
        "sign_change_indices": sorted(set(all_sign_changes)),
        "df": df
    }


# ==========================================================
# RNXScreening (from the second block) — LOGIC UNCHANGED
# ==========================================================
def RNXScreening(destination_directory):
    # List files in the .RNX1 directory
    filess = os.listdir(destination_directory)
    files = [file_ for file_ in filess if file_.endswith("RNX1")]

    for file in files:
        f = os.path.join(destination_directory, file)
        g = os.path.basename(f)
        ano = g[13:17]
        doy = g[9:12]
        estacao = g[0:4]
        sat = g[5:8]

        # Variables and Parameters
        h1 = 0
        n_horas = 24
        int1 = 120

        # GPS freqs
        gps_freqs_local = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
        f1 = gps_freqs_local[1]
        f2 = gps_freqs_local[2]
        f5 = gps_freqs_local[5]

        # GLONASS channels/freqs
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

        data = []
        for _, row in df_slots.iterrows():
            satellite = row['Slot']
            k = row['Channel']
            row_data = [satellite]
            for _, frequency in glonass_frequencies.items():
                freq_value = frequency(k) if callable(frequency) else frequency
                row_data.append(f"{freq_value:.1f}")
            data.append(row_data)
        glo_freqs_df = pd.DataFrame(data, columns=['Satellite', 'fr1', 'fr2', 'fr3'])

        if sat.startswith('G'):
            f1 = f1
            f2 = f2
            f5 = f5
        elif sat.startswith('R'):
            sat_row = glo_freqs_df.loc[glo_freqs_df['Satellite'] == sat]
            if not sat_row.empty:
                f1 = float(sat_row['fr1'].values[0])
                f2 = float(sat_row['fr2'].values[0])
                f5 = float(sat_row['fr3'].values[0])
        else:
            f1 = f2 = f5 = None

        # Load tabulated RNX1
        date = []
        time_s = []
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

        with open(caminho_arquivo, 'r') as fpin:
            header = fpin.readline().strip().split('\t')
            obs_La_header = header[6]
            obs_Lb_header = header[7]
            obs_Lc_header = header[8]
            obs_Ca_header = header[9]
            obs_Cb_header = header[10]
            obs_Cc_header = header[11]

            for linha in fpin:
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

                date.append(registro['date'])
                time_s.append(registro['time'])
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

        # Filter by satellite from file name
        satellite = sat
        print(f"Processing: {satellite}")
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

        for idx in indices:
            date_filtered.append(date[idx])
            time_filtered.append(time_s[idx])
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

        data_df = {
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
            'P5': P2_filtered,  # kept as in your original logic
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
        df = pd.DataFrame(data_df)

        # Conversions and NaNs
        columns_to_convert = ['L1', 'L2', 'L5', 'P1', 'P2', 'P5']
        df[columns_to_convert] = df[columns_to_convert].astype(float)
        df.replace(-999999.999, np.nan, inplace=True)

        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time2'])
        df['time'] = df['timestamp'].dt.time

        L1_array = np.nan_to_num(np.array(df['L1'].tolist(), dtype=np.float64), nan=-999999.999)
        L2_array = np.nan_to_num(np.array(df['L2'].tolist(), dtype=np.float64), nan=-999999.999)
        L5_array = np.nan_to_num(np.array(df['L5'].tolist(), dtype=np.float64), nan=-999999.999)
        P1_array = np.nan_to_num(np.array(df['P1'].tolist(), dtype=np.float64), nan=-999999.999)
        P2_array = np.nan_to_num(np.array(df['P2'].tolist(), dtype=np.float64), nan=-999999.999)
        P5_array = np.nan_to_num(np.array(df['P5'].tolist(), dtype=np.float64), nan=-999999.999)

        L1_array[L1_array == -999999.999] = np.nan
        L2_array[L2_array == -999999.999] = np.nan
        L5_array[L5_array == -999999.999] = np.nan
        P1_array[P1_array == -999999.999] = np.nan
        P2_array[P2_array == -999999.999] = np.nan
        P5_array[P5_array == -999999.999] = np.nan

        MW_combination  = screening_settings.melbourne_wubbena_combination(f1, f2, L1_array, L2_array, P1_array, P2_array)
        MW_combination2 = screening_settings.melbourne_wubbena_combination(f1, f5, L1_array, L5_array, P1_array, P5_array)

        df['MW']  = MW_combination
        df['MW2'] = MW_combination2

        # Arcs from cs_flag
        arcos = []
        arc_atual = []
        for idx, value in enumerate(df['cs_flag']):
            if value == 'S':
                if arc_atual:
                    arcos.append(arc_atual)
                    arc_atual = []
            else:
                arc_atual.append(idx)
        if arc_atual:
            arcos.append(arc_atual)

        print()
        for i, arc in enumerate(arcos):
            start_index = arc[0]
            end_index = arc[-1]
            num_observations = len(arc)
            status = "Kept" if num_observations >= 15 else "Discarded"
            print(f"Arc {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                  f"Obs. = {num_observations}, Status = {status}")

        # Polynomial fits per arc (kept)
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

            x_values = np.arange(len(arc_values))
            polynomial_fit = screening_settings.fit_polynomial(x_values, arc_values, 3)
            arc_data.append(arc_values)
            arc_idx.append(arc_timestamps)
            polynomial_fits.append(polynomial_fit)

            num_observations = len(arc_values)
            num_points_fit = len(polynomial_fit)
            print(f"Arc {i + 1}: Start index = {start}, End index = {end}, "
                  f"Number of observations = {num_observations}, Number of fit points = {num_points_fit}")

        arcos_validos = [arc for arc in arcos if len(MW_combination[arc[0]:arc[-1]+1]) >= 15]
        if len(arcos_validos) == 1:
            arcos_validos.append(arcos_validos[0])

        # Detect anomalies in L1–L2
        res_L12 = detect_and_plot_arcs_before_after(
            df=df,
            arcos_validos=arcos_validos,
            series=MW_combination,
            label_pair="L1-L2",
            rescale_func=screening_settings.rescale_data,
            fit_poly_func=screening_settings.fit_polynomial,
            plot=True,
            save_dir=None,
            sat_id=satellite
        )

        # Detect anomalies in L1–L5
        res_L15 = detect_and_plot_arcs_before_after(
            df=df,
            arcos_validos=arcos_validos,
            series=MW_combination2,
            label_pair="L1-L5",
            rescale_func=screening_settings.rescale_data,
            fit_poly_func=screening_settings.fit_polynomial,
            plot=True,
            save_dir=None,
            sat_id=satellite
        )

        combined_indices = sorted(set(res_L12["removed_indices"] + res_L15["removed_indices"]))
        df.loc[combined_indices, "outlier_flag"] = "Y"
        print(f"\nTotal combined outliers: {len(combined_indices)} samples marked ('Y').\n")

        # EXPORT .RNX2
        output_directory = os.path.join(str(ano), str(doy), estacao.upper())
        full_path = os.path.join(destination_directory)
        os.makedirs(full_path, exist_ok=True)
        file_name = f"{estacao}_{satellite}_{doy}_{ano}.RNX2"
        output_file_path = os.path.join(full_path, file_name)

        colunas_desejadas = [
            'date', 'time', 'mjd',
            'pos_x', 'pos_y', 'pos_z',
            'L1', 'L2', 'L5',
            'P1', 'P2', 'P5',
            'cs_flag', 'outlier_flag', 'satellite', 'sta',
            'hght', 'El', 'Lon', 'Lat',
            'obs_La', 'obs_Lb', 'obs_Lc',
            'obs_Ca', 'obs_Cb', 'obs_Cc'
        ]
        df_selecionado = df[colunas_desejadas].copy()
        df_selecionado = df_selecionado.fillna(-999999.999)

        df_selecionado['mjd'] = pd.to_numeric(df_selecionado['mjd'], errors='coerce')
        df_selecionado['mjd'] = df_selecionado['mjd'].apply(lambda v: f"{v:.6f}" if pd.notna(v) else np.nan)

        for col, fmt in [('El', '{:.4f}'), ('Lon', '{:.4f}'), ('Lat', '{:.4f}'), ('hght', '{:.2f}')]:
            df_selecionado[col] = pd.to_numeric(df_selecionado[col], errors='coerce')
            df_selecionado[col] = df_selecionado[col].apply(lambda v: fmt.format(v) if pd.notna(v) else np.nan)

        for col in ['pos_x','pos_y','pos_z','L1','L2','L5','P1','P2','P5']:
            df_selecionado[col] = pd.to_numeric(df_selecionado[col], errors='coerce')
            df_selecionado[col] = df_selecionado[col].apply(lambda v: f"{v:.6f}" if pd.notna(v) else np.nan)

        obj_cols = df_selecionado.select_dtypes(include=['object']).columns
        for c in obj_cols:
            df_selecionado[c] = (
                df_selecionado[c]
                .astype(str)
                .str.replace('\t', ' ', regex=False)
                .str.replace('\r', ' ', regex=False)
                .str.replace('\n', ' ', regex=False)
                .str.strip()
                .replace({'nan': np.nan})
            )

        for c in ['cs_flag','outlier_flag']:
            # Preserve semantics of pandas' legacy `errors='ignore'` (removed in
            # pandas 3.0): convert the column only if every value is numeric,
            # otherwise leave it untouched.
            try:
                df_selecionado[c] = pd.to_numeric(df_selecionado[c], errors='raise')
            except (ValueError, TypeError):
                pass

        df_selecionado.to_csv(
            output_file_path,
            sep='\t',
            index=False,
            na_rep='-999999.999',
            lineterminator='\n'
        )
        print(f"[OK] Data exported to {output_file_path}.")


# ==========================================================
# (Optional) Programmatic usage example
# ==========================================================
if __name__ == "__main__":
    # Example: set parameters here only if you want to run this file directly.
    # Otherwise, import RNXclean in another script and call it with your parameters.
    #
    # station_name = "BELE"
    # day_of_year  = "266"
    # year         = "2025"
    # input_folder = "/home/debian-giorgio/pyOASIS/pyOASIS/INPUT/RINEX"
    # orbit_folder = "/home/debian-giorgio/pyOASIS/pyOASIS/OUTPUT/2025/266/ORBITS"
    # output_folder= "/home/debian-giorgio/pyOASIS/pyOASIS/OUTPUT/BELE"
    # RNXclean(station_name, day_of_year, year, input_folder, orbit_folder, output_folder)
    pass
