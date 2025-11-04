#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======================================================================
 DTECcalc.py
 Author: Giorgio Picanço
 Date: November 2025
======================================================================
 Description:
     Compute the ΔTEC (DTEC) index from geometry-free leveled GNSS data (.RNX3)
     using the L1-L2 combination only. The function applies a pre-mask using
     the 'mini_flag' column (where mini_flag != 'N' → NaN), then processes
     the clean data to derive DTEC in TECU/hour. The output file includes
     date, time, MJD, coordinates, elevation, and DTEC, all aligned.

 Dependencies:
     - Python ≥ 3.8
     - numpy
     - pandas
     - matplotlib
     - pyOASIS
======================================================================
"""

import os
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
    """

    # ===============================
    # Constants and configuration
    # ===============================
    WINDOW = 60.0 * 2.5     # seconds (2.5 min)
    TDTEC = 60.0            # normalization (TECU/min)
    GAP2 = 3600             # seconds between arcs
    H = 450000              # shell height [m]
    SIGMA = 5               # std multiplier for outlier removal
    DE = 3                  # polynomial degree
    elev_angle = 30         # min elevation [deg]

    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]
    f2 = gps_freqs[2]
    akl = 40.3e16 * ((1 / f2 ** 2) - (1 / f1 ** 2))

    # ===============================
    # File discovery
    # ===============================
    path_ = os.path.join(input_folder)
    if not os.path.exists(path_):
        print("Specified directory does not exist.")
        return

    files = [f for f in os.listdir(path_) if f.startswith(station) and f.endswith(".RNX3")]
    ord_files = sorted(files, key=lambda x: int(x.split("_")[1][1:]))
    print(f"\nNumber of .RNX3 files found: {len(ord_files)}\n")

    if not ord_files:
        print("[ERROR] No input files found.")
        return

    # ===============================
    # Data aggregation
    # ===============================
    all_data = []
    for file_ in ord_files:
        path_file_ = os.path.join(path_, file_)
        try:
            df = pd.read_csv(path_file_, sep="\t", engine="python")
        except Exception as e:
            print(f"[WARN] Could not read {file_}: {e}")
            continue

        # --- Mask invalid mini_flag samples ---
        if "mini_flag" in df.columns:
            mask_invalid = df["mini_flag"].astype(str) != "N"
            n_masked = mask_invalid.sum()
            if n_masked > 0:
                print(f"  [INFO] Masking {n_masked} samples in {file_} where mini_flag != 'N'")
            df.loc[mask_invalid, "LGF_combination"] = np.nan

        # --- Create datetime column ---
        if "date" in df.columns and "time2" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["time2"], errors="coerce")
        else:
            df["datetime"] = pd.NaT

        all_data.append(df)

    if not all_data:
        print("[ERROR] No readable .RNX3 data.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    df_all.replace([-999999.999, 999999.999], np.nan, inplace=True)

    # --- Mask again globally (safety) ---
    if "mini_flag" in df_all.columns:
        df_all.loc[df_all["mini_flag"].astype(str) != "N", "LGF_combination"] = np.nan

    # ===============================
    # Process per satellite
    # ===============================
    sat_classes = ["G", "R"]
    plt.figure(figsize=(12, 6))

    for sat_class in sat_classes:
        satellites = np.unique(df_all["satellite"].astype(str))
        satellites_to_plot = [sv for sv in satellites if sv.startswith(sat_class)]
        if not satellites_to_plot:
            continue

        print(f"\nCalculating DTEC for {station.upper()} | {year} DOY {doy} | {sat_class} system\n")
        sat_data = []

        for idx, sat in enumerate(satellites_to_plot, start=1):
            print(f"  Processing {sat} ({idx}/{len(satellites_to_plot)})...")
            df_sat = df_all[df_all["satellite"] == sat].copy()

            # Ensure numeric types
            num_cols = ["LGF_combination", "mjd", "El", "Lon", "Lat", "hght"]
            for c in num_cols:
                if c in df_sat.columns:
                    df_sat[c] = pd.to_numeric(df_sat[c], errors="coerce")

            df_sat.replace(-999999.999, np.nan, inplace=True)
            df_sat.dropna(subset=["LGF_combination", "mjd"], inplace=True)

            if df_sat.empty:
                continue

            # Convert LGF to STEC (TECU)
            stec = df_sat["LGF_combination"].to_numpy() / akl
            t = df_sat["mjd"].to_numpy()
            lat = df_sat["Lat"].to_numpy()
            lon = df_sat["Lon"].to_numpy()
            elev = df_sat["El"].to_numpy()
            hh = df_sat["hght"].to_numpy()

            # Compute ΔTEC (DTEC)
            d = 86400.0 * np.diff(t)
            d = ma.masked_values(d, 0.0)
            i = np.where(np.append(d.mask, False) == True)[0]
            for j in range(i.size):
                dIFB = stec[i[j] + 1] - stec[i[j]]
                stec[i[j] + 1:] -= dIFB

            if d.mask.any():
                mask = np.append(~d.mask, True)
                t, stec, lat, lon, hh, elev = [arr[mask] for arr in (t, stec, lat, lon, hh, elev)]

            # Define windows
            t0 = t[0]
            i = np.floor(np.round(86400 * (t - t0)) / WINDOW)
            j = np.unique(i)

            alon, alat, ahh, at, elev1, DTEC = [], [], [], [], [], []

            for k in range(j.size):
                l = ma.masked_values(i, j[k])
                if lon[l.mask].size > 1:
                    alon.append(np.nanmean(lon[l.mask]))
                    alat.append(np.nanmean(lat[l.mask]))
                    ahh.append(np.nanmean(hh[l.mask]))
                    at.append(np.nanmean(t[l.mask]))
                    elev1.append(np.nanmean(elev[l.mask]))

                    df_stec = pd.DataFrame({
                        "timestamp": t[l.mask] * 86400.0,
                        "stec": stec[l.mask]
                    }).set_index("timestamp")

                    T15 = 15 if len(df_stec) > 30 else 5
                    T60 = 4 * T15

                    short_avg = df_stec["stec"].rolling(window=T15, center=True, min_periods=1).mean()
                    long_avg = df_stec["stec"].rolling(window=T60, center=True, min_periods=1).mean()

                    dtec_val = short_avg.iloc[-1] - long_avg.iloc[-1]
                    DTEC.append(dtec_val)

            if not DTEC:
                continue

            alon = np.array(alon)
            alat = np.array(alat)
            ahh = np.array(ahh)
            at = np.array(at)
            DTEC = np.array(DTEC)
            elev = np.array(elev1)

            # Outlier filtering via polynomial fit
            d = 86400.0 * np.diff(at)
            d = ma.masked_greater_equal(d, GAP2)
            i1 = np.append(0, np.where(np.append(d.mask, False) == True)[0])
            i2 = np.append(np.where(np.append(d.mask, False) == True)[0] - 1, alon.size - 1)

            y = np.empty(DTEC.size)
            for j in range(i1.size):
                tm = np.mean(at[i1[j]:i2[j] + 1])
                if (at[i2[j]] - at[i1[j]]) != 0.0:
                    x = (at[i1[j]:i2[j] + 1] - tm) / (at[i2[j]] - at[i1[j]])
                    c = np.polyfit(x, DTEC[i1[j]:i2[j] + 1], DE)
                    y[i1[j]:i2[j] + 1] = np.polyval(c, x)
                else:
                    y[i1[j]:i2[j] + 1] = DTEC[i1[j]:i2[j] + 1]
            mask_out = np.abs(DTEC - y) > SIGMA * np.std(DTEC - y)
            alon, alat, ahh, at, DTEC, elev = [a[~mask_out] for a in (alon, alat, ahh, at, DTEC, elev)]
            cutoff = np.where(elev >= elev_angle)
            alon, alat, ahh, at, DTEC, elev = [a[cutoff] for a in (alon, alat, ahh, at, DTEC, elev)]

            # Store results
            sat_data.append(pd.DataFrame({
                "date": [(datetime(1858, 11, 17) + timedelta(days=float(x))).strftime("%Y-%m-%d") for x in at],
                "time": [(datetime(1858, 11, 17) + timedelta(days=float(x))).strftime("%H:%M:%S") for x in at],
                "MJD": at,
                "Longitude": alon,
                "Latitude": alat,
                "Height": ahh,
                "Elevation": elev,
                "DTEC": 10 * DTEC,  # TECU/hour
                "STA": station,
                "SAT": sat
            }))

        if not sat_data:
            continue

        df_concat = pd.concat(sat_data, ignore_index=True)

        # ===============================
        # Save output
        # ===============================
        os.makedirs(destination_directory, exist_ok=True)
        file_name = f"{station}_{doy}_{year}_{sat_class}_DTEC.txt"
        output_path = os.path.join(destination_directory, file_name)

        df_concat.to_csv(
            output_path,
            sep="\t",
            index=False,
            na_rep="-999999.999",
            float_format="%11.5f",
            columns=["date", "time", "MJD", "Longitude", "Latitude",
                     "Height", "Elevation", "DTEC", "STA", "SAT"]
        )
        print(f"[OUT] Saved {output_path}")

        # ===============================
        # Plot results
        # ===============================
        base_date = datetime(1858, 11, 17)
        df_concat["datetime"] = df_concat["MJD"].astype(float).apply(lambda x: base_date + timedelta(days=x))
        color = "navy" if sat_class == "G" else "red"
        label = f"DTEC: L1-L2 ({'GPS' if sat_class == 'G' else 'GLONASS'})"
        plt.scatter(df_concat["datetime"], df_concat["DTEC"], color=color, s=30, label=label)

    # ===============================
    # Final plot styling
    # ===============================
    hours_fmt = mdates.DateFormatter("%H")
    hour_locator = mdates.HourLocator(interval=2)
    plt.gca().xaxis.set_major_formatter(hours_fmt)
    plt.gca().xaxis.set_major_locator(hour_locator)
    plt.xlabel("Time (UT)", fontsize=16)
    plt.ylabel("DTEC (TECU/hour)", fontsize=16)
    plt.title(f"Station: {station.upper()} | Year: {year} | DOY: {doy}", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", linewidth=1, color="gray")
    plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left")
    plt.tight_layout()

    file_name_png = f"{station}_{doy}_{year}_DTEC.png"
    plt.savefig(os.path.join(destination_directory, file_name_png), dpi=300)
    if show_plot:
        plt.show()
