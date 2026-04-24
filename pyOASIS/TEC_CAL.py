"""
STEC Calibration Algorithm (Integrated In-Memory Version with Plotting)
------------------------------------------------------------------------
This version combines:
  (1) SAT-like data generation directly from .RNX3 files
  (2) Calibration of sTEC and vTEC using Amalia Meza’s segmented least-squares model
  (3) Visualization of sTEC and vTEC time series

Outputs (for each frequency pair):
  - {output_dir}/{STATION}_{DOY}_{YEAR}_L1L2.DCB : Differential Code Biases per arc
  - {output_dir}/{STATION}_{DOY}_{YEAR}_L1L2.TEC : Calibrated sTEC and vTEC
  - {output_dir}/{STATION}_{DOY}_{YEAR}_L1L2.png : Figure with STEC, VTEC and IPP
  (and corresponding _L1L5.* files when available)

Author: Giorgio Picanço
Years: 2024–2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sqrt, atan2, degrees
from datetime import datetime, timedelta
from pathlib import Path
from astropy.time import Time
from .settings import *


# --- Smooth interpolation backends (optional SciPy) ---
try:
    from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
except Exception:
    PchipInterpolator = None
    Akima1DInterpolator = None



def TECcalc(station: str,
            doy: str,
            year: str,
            input_dir: str,
            output_dir: str,
            show_plot: bool = True) -> None:
    """
    Run the full TEC calibration pipeline and plotting, reading .RNX3 files
    from `input_dir` and writing outputs to `output_dir`.

    Typical call:
        pyOASIS.TECcalc(sta, doy, year, sta_output, sta_output, show_plot=True)

    Parameters
    ----------
    station : str
        Station code (e.g., "P331").
    doy : str
        Day Of Year (e.g., "266"). Accepts "266" or "0266" (will be zero-padded to 3 digits).
    year : str
        4-digit year (e.g., "2025").
    input_dir : str
        Directory containing .RNX3 files (folder with RNX3 TSVs).
    output_dir : str
        Directory where .DCB, .TEC and .png will be written.
    show_plot : bool
        If True shows the figure on screen; if False only saves the .png.
    """

    # ===============================
    # CONFIGURATION (algorithmic constants)
    # ===============================
    RE = 6371.0
    H_IONO = 450.0
    N_OBS = 190000
    N_UNK = 900
    K_GPS = 9.5362

    # ===============================
    # GEODETIC CONSTANTS (ellipsoid)
    # ===============================
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b ** 2 / a ** 2)

    # ===============================
    # Internal helpers (kept 1:1 with original logic)
    # ===============================
    def ecef_to_geodetic(x, y, z):
        """Convert ECEF (x,y,z) to geodetic latitude, longitude, height (WGS-84)."""
        r = sqrt(x ** 2 + y ** 2)
        lon = atan2(y, x)
        lat = atan2(z, r * (1 - e2))
        for _ in range(5):
            N = a / sqrt(1 - e2 * np.sin(lat) ** 2)
            h = r / np.cos(lat) - N
            lat = atan2(z, r * (1 - e2 * N / (N + h)))
        return degrees(lat), degrees(lon), h

    def convert_satellite_id(sat_code):
        """Map RINEX satellite labels to integer PRN domain used by the solver."""
        sat_code = str(sat_code).strip().upper()
        if len(sat_code) < 2:
            return "000"
        system = sat_code[0]
        try:
            prn = int(sat_code[1:])
        except Exception:
            return "000"
        if system == "G":
            return f"{prn:03d}"
        elif system == "R":
            return f"{100 + prn:03d}"
        return "000"

    # ===============================
    # STEP 1: GENERATE SAT DATA (L1-L2)
    # ===============================
    def generate_sat_data(rnx3_dir: Path):
        """
        Build SAT-like rows from RNX3 files using LGF_combination (L1-L2).
        Each row: [prn, mjd, lgf, hght_m, lon_rad, lat_diff_deg, elevation_deg, azimuth_rad]
        """
        sat_data = []
        lat_sta = lon_sta = None

        # Get station coordinates (first RNX3 containing pos_x/pos_y/pos_z)
        for filename in os.listdir(rnx3_dir):
            if filename.endswith(".RNX3"):
                df = pd.read_csv(rnx3_dir / filename, sep="\t", engine="python")
                df.replace([999999.999, -999999.999], np.nan, inplace=True)
                if {'pos_x', 'pos_y', 'pos_z'}.issubset(df.columns):
                    df = df.dropna(subset=['pos_x', 'pos_y', 'pos_z'])
                    if not df.empty:
                        x, y, z = df.iloc[0][['pos_x', 'pos_y', 'pos_z']]
                        lat_sta, lon_sta, _ = ecef_to_geodetic(x, y, z)
                        break

        if lat_sta is None or lon_sta is None:
            raise ValueError("Failed to obtain station coordinates.")

        # Process RNX3 files
        for filename in os.listdir(rnx3_dir):
            if not filename.endswith(".RNX3"):
                continue
            df = pd.read_csv(rnx3_dir / filename, sep="\t", engine="python")
            df.replace([999999.999, -999999.999], np.nan, inplace=True)

            required_cols = [
                'mjd', 'Lon', 'Lat', 'hght', 'El', 'satellite', 'LGF_combination'
            ]
            if not all(col in df.columns for col in required_cols):
                continue

            df = df.dropna(subset=required_cols)

            for _, row in df.iterrows():
                try:
                    prn_str = convert_satellite_id(row['satellite'])
                    if prn_str == "000":
                        continue
                    prn = int(prn_str)

                    lgf = float(row.get('LGF_combination', -999999.999))
                    if np.isnan(lgf) or lgf <= -999999.0:
                        continue

                    mjd = float(row['mjd'])
                    lat_ipp = float(row['Lat'])
                    lon_ipp = float(row['Lon'])
                    lon_norm = (lon_ipp + 180) % 360 - 180
                    lon_rad = radians(lon_norm)
                    hght = float(row['hght']) * 1000
                    el = float(row['El'])
                    lat_diff = lat_ipp - lat_sta

                    # Azimuth
                    delta_lon = radians(lon_ipp - lon_sta)
                    lat1 = radians(lat_sta)
                    lat2 = radians(lat_ipp)
                    X = np.sin(delta_lon) * np.cos(lat2)
                    Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
                    azim = atan2(X, Y)
                    if azim < 0:
                        azim += 2 * np.pi

                    sat_data.append([prn, mjd, lgf, hght, lon_rad, lat_diff, el, azim])

                except Exception:
                    continue

        # ==============================================================
        # Smooth per-satellite interpolation BEFORE TEC calibration
        #   - Reconstruct a regular time axis per PRN
        #   - Do NOT bridge large gaps (> max_gap_hours)
        #   - Keep each satellite completely independent
        #   - Use smooth cubic interpolation (PCHIP or Akima) if SciPy available
        # ==============================================================

        # Settings
        max_gap_hours = 1.0            # do not interpolate gaps longer than this
        interp_resolution_sec = 15      # cadence of the regularized series (e.g., 30 s)
        interp_method = 'pchip'         # 'pchip' | 'akima' | 'linear'
        smooth_after_interp = True      # optional Savitzky-Golay smoothing on LGF only

        df_all = pd.DataFrame(
            sat_data,
            columns=['prn', 'mjd', 'lgf', 'hght', 'lon_rad', 'lat_diff', 'el', 'azim']
        )

        df_interp_all = []

        def _make_interpolator(x, y):
            """Return an interpolator according to interp_method and availability."""
            if interp_method.lower() == 'akima' and Akima1DInterpolator is not None and len(x) >= 5:
                return Akima1DInterpolator(x, y)
            if interp_method.lower() == 'pchip' and PchipInterpolator is not None and len(x) >= 3:
                return PchipInterpolator(x, y, extrapolate=False)
            # fallback (linear)
            return None  # will use np.interp

        for prn, df_sat in df_all.groupby('prn'):
            df_sat = df_sat.sort_values('mjd').reset_index(drop=True)
            if len(df_sat) < 3:
                df_interp_all.append(df_sat)
                continue

            # Detect large gaps (in hours)
            df_sat['delta_t'] = df_sat['mjd'].diff() * 24.0
            large_gap_mask = df_sat['delta_t'] > max_gap_hours
            gap_breaks = df_sat.index[large_gap_mask].tolist()

            # Split into segments that do NOT cross large gaps
            segments = []
            start_idx = 0
            for g in gap_breaks + [len(df_sat)]:
                segments.append(df_sat.iloc[start_idx:g])
                start_idx = g

            for seg in segments:
                seg = seg.sort_values('mjd')
                if len(seg) < 2:
                    # nothing to interpolate
                    df_interp_all.append(seg)
                    continue

                # Regular time grid per segment
                t0 = seg['mjd'].iloc[0]
                t1 = seg['mjd'].iloc[-1]
                if t1 <= t0:
                    df_interp_all.append(seg)
                    continue

                t_regular = np.arange(t0, t1 + interp_resolution_sec / 86400.0, interp_resolution_sec / 86400.0)

                seg_interp = pd.DataFrame({'mjd': t_regular})
                # Interpolate scalar series with chosen method or fallback to linear
                for col in ['lgf', 'hght', 'lon_rad', 'lat_diff', 'el']:
                    x = seg['mjd'].to_numpy()
                    y = seg[col].to_numpy()
                    f = _make_interpolator(x, y)
                    if f is None:
                        seg_interp[col] = np.interp(t_regular, x, y)
                    else:
                        yi = f(t_regular)
                        # fill NaNs at edges (no extrapolation): fall back to nearest valid via numpy.interp
                        if np.isnan(yi).any():
                            yi = np.where(np.isnan(yi), np.interp(t_regular, x, y), yi)
                        seg_interp[col] = yi

                # Handle azimuth smartly (unwrap to avoid 2π jumps)
                x = seg['mjd'].to_numpy()
                az = seg['azim'].to_numpy()
                az_unw = np.unwrap(az)  # radians
                f_az = _make_interpolator(x, az_unw)
                if f_az is None:
                    az_interp = np.interp(t_regular, x, az_unw)
                else:
                    az_interp = f_az(t_regular)
                    if np.isnan(az_interp).any():
                        az_interp = np.where(np.isnan(az_interp), np.interp(t_regular, x, az_unw), az_interp)
                # re-wrap to [0, 2π)
                az_interp = np.mod(az_interp, 2 * np.pi)
                seg_interp['azim'] = az_interp

                # Optional gentle smoothing on LGF only (keeps features, reduces jaggies)
                if smooth_after_interp:
                    try:
                        from scipy.signal import savgol_filter
                        # window ≈ 5 min worth of samples, at least 7 and odd
                        win = max(7, int(round((300.0 / interp_resolution_sec))))
                        if win % 2 == 0:
                            win += 1
                        if win > 3 and len(seg_interp['lgf']) >= win:
                            seg_interp['lgf'] = savgol_filter(seg_interp['lgf'].to_numpy(), window_length=win, polyorder=2, mode='interp')
                    except Exception:
                        # no SciPy or too-short segment: silently skip
                        pass

                seg_interp['prn'] = prn
                df_interp_all.append(seg_interp)

        df_interp_all = pd.concat(df_interp_all, ignore_index=True).sort_values(['prn', 'mjd']).reset_index(drop=True)
        sat_data_sorted = df_interp_all[['prn', 'mjd', 'lgf', 'hght', 'lon_rad', 'lat_diff', 'el', 'azim']].values
        print(f"Smoothly interpolated per-satellite LGF series before TEC ({len(df_interp_all)} points total) using '{interp_method}'.")
        return np.array(sat_data_sorted), lat_sta, lon_sta


    # ===============================
    # STEP 1: GENERATE SAT DATA (L1-L5)
    # ===============================
    def generate_sat_data_l15(rnx3_dir: Path):
        """
        Same as generate_sat_data, but uses LGF_combination15 (L1-L5).
        Compatible with the new RNX3 (mini_flag may exist, not used here).
        """
        sat_data = []
        lat_sta = lon_sta = None

        # Station coordinates
        for filename in os.listdir(rnx3_dir):
            if filename.endswith(".RNX3"):
                df = pd.read_csv(rnx3_dir / filename, sep="\t", engine="python")
                df.replace([999999.999, -999999.999], np.nan, inplace=True)
                if {'pos_x', 'pos_y', 'pos_z'}.issubset(df.columns):
                    df = df.dropna(subset=['pos_x', 'pos_y', 'pos_z'])
                    if not df.empty:
                        x, y, z = df.iloc[0][['pos_x', 'pos_y', 'pos_z']]
                        lat_sta, lon_sta, _ = ecef_to_geodetic(x, y, z)
                        break

        if lat_sta is None or lon_sta is None:
            raise ValueError("Failed to obtain station coordinates.")

        # Process RNX3 files
        for filename in os.listdir(rnx3_dir):
            if not filename.endswith(".RNX3"):
                continue
            df = pd.read_csv(rnx3_dir / filename, sep="\t", engine="python")
            df.replace([999999.999, -999999.999], np.nan, inplace=True)
            required_cols = ['mjd', 'Lon', 'Lat', 'hght', 'El', 'satellite', 'LGF_combination15']
            if not all(col in df.columns for col in required_cols):
                continue
            df = df.dropna(subset=required_cols)

            for _, row in df.iterrows():
                try:
                    prn_str = convert_satellite_id(row['satellite'])
                    if prn_str == "000":
                        continue
                    prn = int(prn_str)

                    lgf = float(row.get('LGF_combination15', -999999.999))
                    if np.isnan(lgf) or lgf <= -999999.0:
                        continue

                    mjd = float(row['mjd'])
                    lat_ipp = float(row['Lat'])
                    lon_ipp = float(row['Lon'])
                    lon_norm = (lon_ipp + 180) % 360 - 180
                    lon_rad = radians(lon_norm)
                    hght = float(row['hght']) * 1000
                    el = float(row['El'])
                    lat_diff = lat_ipp - lat_sta

                    delta_lon = radians(lon_ipp - lon_sta)
                    lat1 = radians(lat_sta)
                    lat2 = radians(lat_ipp)
                    X = np.sin(delta_lon) * np.cos(lat2)
                    Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
                    azim = atan2(X, Y)
                    if azim < 0:
                        azim += 2 * np.pi

                    sat_data.append([prn, mjd, lgf, hght, lon_rad, lat_diff, el, azim])

                except Exception:
                    continue

        sat_data_sorted = sorted(sat_data, key=lambda x: (x[0], x[1]))
        return np.array(sat_data_sorted), lat_sta, lon_sta

    # ===============================
    # STEP 2: CALIBRATION
    # ===============================
    def glonass_frequency(channel):
        """Map GLONASS channel to frequency factor used in m2TECU scaling."""
        f1 = (1602.0 + 0.5625 * channel) ** 2
        f2 = (1246.0 + 0.4375 * channel) ** 2
        return -(f1 * f2 / (f2 - f1)) / 40.3 / 10000

    def process_station_memory(data, station, lat_sta, lon_sta, doy_int, out_suffix="", base_out_dir: Path = Path(".")):
        """
        Build normal equations, solve for ionospheric parameters and DCBs,
        write .DCB, .TEC, and plot .png.
        """
        name = station
        crd = [lon_sta, lat_sta, 0]

        aTa = np.zeros((N_UNK, N_UNK))
        aTy = np.zeros(N_UNK)
        ic = np.zeros(200, dtype=int)
        G_data = np.zeros((N_OBS, 7, 200))
        unk2prn = np.zeros(N_UNK, dtype=int)
        ele_mask = 20

        kk = np.zeros(200)
        glonass_table = {101: 1, 102: -4, 103: 5, 104: 6, 105: 1, 106: -4, 107: 5, 108: 6,
                         109: -2, 110: -7, 111: 0, 112: -1, 113: -2, 114: -7, 115: 0, 116: -1,
                         117: 4, 118: -3, 119: 3, 120: 2, 121: 4, 122: -3, 123: 3, 124: 2}
        for prn, val in glonass_table.items():
            kk[prn] = val

        # Fill G_data matrix
        for row in data:
            sv, t, smt, h, lon, lat, ele, azi = row
            sv = int(sv)
            m2tecu = K_GPS if sv <= 99 else glonass_frequency(kk[sv])
            lon_deg = np.degrees(lon)
            lat_deg = lat + crd[1]
            ele_deg = ele
            smt = m2tecu * smt

            if ele_deg < ele_mask:
                continue
            idx = ic[sv]
            G_data[idx, 0, sv] = t
            G_data[idx, 1, sv] = smt
            G_data[idx, 2, sv] = lon_deg
            G_data[idx, 3, sv] = lat_deg
            G_data[idx, 4, sv] = ele_deg
            G_data[idx, 5, sv] = azi
            ic[sv] += 1

        # Arc segmentation (map by arc ID, no +144 shift here)
        ukw = 0
        IncInd = np.zeros(N_UNK, dtype=int)
        for prn in range(200):
            if ic[prn] == 0:
                continue
            ukw += 1
            G_data[0, 6, prn] = ukw
            unk2prn[ukw] = prn
            for i in range(1, ic[prn]):
                dt = (G_data[i, 0, prn] - G_data[i - 1, 0, prn]) * 24
                ds = abs(G_data[i, 1, prn] - G_data[i - 1, 1, prn])
                mf = mapfun(G_data[i, 4, prn])
                if dt > 5.0 / 60.0 or ds > 5.0 / mf:
                    ukw += 1
                    G_data[i, 6, prn] = ukw
                    unk2prn[ukw] = prn
                else:
                    G_data[i, 6, prn] = G_data[i - 1, 6, prn]

        # STEC normalization inside each arc (zero at arc start)
        for prn in range(200):
            if ic[prn] == 0:
                continue
            arc_ids = np.unique(G_data[:ic[prn], 6, prn])
            for arc in arc_ids:
                mask = (G_data[:ic[prn], 6, prn] == arc)
                if np.sum(mask) < 2:
                    continue
                g_slice = G_data[:ic[prn], 1, prn]
                mask = (G_data[:ic[prn], 6, prn] == arc)
                first_val = g_slice[mask][0]
                g_slice[mask] -= first_val
                G_data[:ic[prn], 1, prn] = g_slice

        # Normal equations
        for prn in range(200):
            for i in range(ic[prn]):
                Alfa = int(G_data[i, 6, prn])
                ut = (G_data[i, 0, prn] % 1) * 24
                lon1 = G_data[i, 2, prn]
                if lon1 > 180:
                    lon1 -= 360
                elif lon1 < -180:
                    lon1 += 360
                lat1 = G_data[i, 3, prn]
                mf = mapfun(G_data[i, 4, prn])
                smt1 = G_data[i, 1, prn]

                a_loc = np.zeros(6)
                a_loc[0] = 1
                a_loc[1] = ut - int(ut)
                a_loc[2] = (lon1 - crd[0]) * np.pi / 180
                a_loc[3] = a_loc[1] * a_loc[2]
                a_loc[4] = (lat1 - crd[1]) * np.pi / 180
                a_loc[5] = a_loc[1] * a_loc[4]

                iPer = int(ut) * 6
                for ii in range(6):
                    for jj in range(ii, 6):
                        aTa[iPer + ii, iPer + jj] += a_loc[ii] * a_loc[jj]
                        if ii != jj:
                            aTa[iPer + jj, iPer + ii] += a_loc[ii] * a_loc[jj]
                    aTa[iPer + ii, 144 + Alfa] += a_loc[ii] * mf
                    aTa[144 + Alfa, iPer + ii] += a_loc[ii] * mf
                    aTy[iPer + ii] += a_loc[ii] * smt1 * mf
                aTa[144 + Alfa, 144 + Alfa] += mf * mf
                aTy[144 + Alfa] += smt1 * mf * mf

        nonzero_cols = np.where(aTy != 0)[0]
        inc = len(nonzero_cols)
        IncInd[nonzero_cols] = np.arange(1, inc + 1)

        aa = []
        for j in range(inc):
            for i in range(j + 1):
                aa.append(aTa[nonzero_cols[i], nonzero_cols[j]])
        yy = aTy[nonzero_cols]

        if len(aa) == 0 or len(yy) == 0:
            print(f"No valid data for {station}.")
            return

        x_sol = cholesky_solve(aa, yy.tolist())

        # Epoch of the first arc (id=1)
        mjd0 = G_data[0, 0, int(unk2prn[1])]
        t_ast = Time(mjd0, format='mjd')
        yx = t_ast.datetime.year
        mx = t_ast.datetime.month
        domx = t_ast.datetime.day
        doyx = t_ast.datetime.timetuple().tm_yday  # kept for compatibility

        # ===============================
        # Output names/paths
        # ===============================
        base_name = f"{name}_{doy_int:03d}_{year}{out_suffix}"
        base_path = base_out_dir / base_name

        dcb_path = base_path.with_suffix(".DCB")
        tec_path = base_path.with_suffix(".TEC")
        fig_path = base_path.with_suffix(".png")

        # Save DCB
        with open(dcb_path, 'w') as fx:
            for i in range(144 + ukw):
                val = x_sol[IncInd[i] - 1] if IncInd[i] != 0 else 0.0
                prn_out = 0 if i < 144 else int(unk2prn[i - 144])
                fx.write(f"{i + 1:03d}  {val:12.5f} {prn_out:4d}\n")

        # Save TEC
        with open(tec_path, 'w') as fcal:
            for prn in range(200):
                for j in range(ic[prn]):
                    Alfa = int(G_data[j, 6, prn])
                    if IncInd[144 + Alfa] == 0:
                        continue
                    dcb = x_sol[IncInd[144 + Alfa] - 1]
                    smt = G_data[j, 1, prn] - dcb
                    mf = mapfun(G_data[j, 4, prn])
                    smtv = smt * mf
                    lon = G_data[j, 2, prn]
                    lat = G_data[j, 3, prn]
                    ele = G_data[j, 4, prn]
                    az = G_data[j, 5, prn]
                    mjd = G_data[j, 0, prn]
                    fcal.write(f"{prn:4d}  {mjd:13.6f}  {smt:11.5f}  {lon:9.3f} {lat:9.3f} {ele:9.3f} {az:9.3f} {smtv:11.5f}\n")

        print(f"Calibration complete: {tec_path} and {dcb_path} generated.")

        # ===============================
        # PLOT (kept as in the original, with additions)
        # ===============================
        def mjd_to_datetime(mjd_val):
            return datetime(1858, 11, 17) + timedelta(days=mjd_val)

        def plot_tec_results(file_path, station_plot, doy_plot, lat_sta_plot=None, lon_sta_plot=None,
                             filter_by_elevation=True, elev_threshold=30.0, save_to: Path = None):
            """
            Read .TEC, optionally filter by elevation, correct large gaps (>1h) by
            vertically reconnecting STEC and VTEC arcs, and plot STEC/VTEC/IPP.
            """
            col_names = ['sat', 'mjd', 'stec', 'lon', 'lat', 'elevation', 'azimuth', 'vtec']
            df_plot = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)

            # Convert MJD to datetime and UT hour
            df_plot['datetime'] = df_plot['mjd'].apply(mjd_to_datetime)
            df_plot['hour'] = (
                df_plot['datetime'].dt.hour +
                df_plot['datetime'].dt.minute / 60.0 +
                df_plot['datetime'].dt.second / 3600.0
            )

            # Elevation filtering
            if filter_by_elevation:
                df_plot = df_plot[df_plot['elevation'].abs() >= elev_threshold].reset_index(drop=True)
                print(f"Filtered by elevation ±{elev_threshold}° → {len(df_plot)} records")

            mask_plot = 30
            df_plot = df_plot[df_plot["elevation"] >= mask_plot].reset_index(drop=True)
            print(f"Applied filter: elevation ≥ {mask_plot}° → {len(df_plot)} remaining observations")

            # ==============================================================
            # Global gap correction (progressive reconnection across the full series)
            #   - Treats the entire data as a single continuous series (no sat separation)
            #   - Detects gaps > 0.5 hour based on MJD
            #   - Computes mean values around each gap (before/after)
            #   - Fits a linear trend between those means
            #   - Calculates midpoint value in time and adjusts both sides
            #     by half the vertical difference along the trend
            # ==============================================================

            # Sort chronologically
            df_plot = df_plot.sort_values('mjd').reset_index(drop=True)

            # Compute Δt between consecutive points (in hours)
            df_plot['delta_t'] = df_plot['mjd'].diff() * 24.0

            # Threshold for detecting a gap
            gap_threshold = 0.25  # hours
            gap_indices = df_plot.index[df_plot['delta_t'] > gap_threshold].tolist()

            if gap_indices:
                print(f"Detected {len(gap_indices)} gaps larger than {gap_threshold} h")
                df_plot = df_plot.copy()

                # Number of samples before/after gap to compute local means
                window_pts = 5

                for gap_idx in gap_indices:
                    # Skip invalid edges
                    if gap_idx <= window_pts or gap_idx >= len(df_plot) - window_pts:
                        continue

                    # Select local windows before and after the gap
                    before_idx = range(gap_idx - window_pts, gap_idx)
                    after_idx = range(gap_idx, gap_idx + window_pts)

                    mjd_before = df_plot.loc[list(before_idx), 'mjd']
                    mjd_after = df_plot.loc[list(after_idx), 'mjd']

                    # Mean times
                    mean_t_before = mjd_before.mean()
                    mean_t_after = mjd_after.mean()

                    # Mean STEC/VTEC before and after gap
                    mean_stec_before = df_plot.loc[list(before_idx), 'stec'].mean()
                    mean_stec_after = df_plot.loc[list(after_idx), 'stec'].mean()

                    mean_vtec_before = df_plot.loc[list(before_idx), 'vtec'].mean()
                    mean_vtec_after = df_plot.loc[list(after_idx), 'vtec'].mean()

                    # Fit simple linear trend connecting both means
                    dt = (mean_t_after - mean_t_before) * 24.0  # hours
                    if dt <= 0:
                        continue

                    slope_stec = (mean_stec_after - mean_stec_before) / dt
                    slope_vtec = (mean_vtec_after - mean_vtec_before) / dt

                    # Compute midpoint in time (halfway between means)
                    t_mid = (mean_t_before + mean_t_after) / 2.0

                    # Corresponding midpoint values on the fitted line
                    mid_stec_ref = mean_stec_before + slope_stec * ((t_mid - mean_t_before) * 24.0)
                    mid_vtec_ref = mean_vtec_before + slope_vtec * ((t_mid - mean_t_before) * 24.0)

                    # Actual mean values before/after gap
                    mid_stec_before = mean_stec_before
                    mid_stec_after = mean_stec_after
                    mid_vtec_before = mean_vtec_before
                    mid_vtec_after = mean_vtec_after

                    # Vertical offsets relative to the ideal trend
                    diff_stec_before = mid_stec_ref - mid_stec_before
                    diff_stec_after = mid_stec_after - mid_stec_ref
                    diff_vtec_before = mid_vtec_ref - mid_vtec_before
                    diff_vtec_after = mid_vtec_after - mid_vtec_ref

                    # Correction amplification factor (controls how strongly arcs are pulled together)
                    k = 2  # 1.0 = original (half difference), 2.0 = full correction

                    # Apply symmetric vertical sliding with amplification
                    df_plot.loc[df_plot['mjd'] <= mean_t_before, 'stec'] += k * diff_stec_before / 2.0
                    df_plot.loc[df_plot['mjd'] >= mean_t_after, 'stec'] -= k * diff_stec_after / 2.0
                    df_plot.loc[df_plot['mjd'] <= mean_t_before, 'vtec'] += k * diff_vtec_before / 2.0
                    df_plot.loc[df_plot['mjd'] >= mean_t_after, 'vtec'] -= k * diff_vtec_after / 2.0


                print("Progressive inter-arc reconnection applied (global series level).")
            else:
                print("No significant gaps (>0.5h) found for correction.")


            # ===============================
            # Figure layout: 2x2 (bottom row merged)
            # ===============================
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
            ax1 = fig.add_subplot(gs[0, 0])  # STEC
            ax2 = fig.add_subplot(gs[0, 1])  # VTEC
            ax3 = fig.add_subplot(gs[1, :])  # IPP Map

            fig.suptitle(
                f"STEC, VTEC, and IPP Distribution - Station {station_plot} - DOY {doy_plot}",
                fontsize=14, fontweight='bold'
            )


            # # ==============================================================
            # # Média robusta + suavização gaussiana (estilo IRTAM)
            # # ==============================================================
            # from scipy.ndimage import gaussian_filter1d
            #
            # def _iter_sigma_clip_mean(x, sigma=2.0, max_iter=5):
            #     vals = pd.Series(x).dropna().to_numpy()
            #     if vals.size == 0:
            #         return np.nan
            #     mask = np.ones(vals.size, dtype=bool)
            #     for _ in range(max_iter):
            #         v = vals[mask]
            #         if v.size == 0:
            #             return np.nan
            #         mu = np.mean(v)
            #         sd = np.std(v, ddof=1) if v.size > 1 else 0.0
            #         if sd == 0:
            #             break
            #         newmask = np.abs(vals - mu) <= sigma * sd
            #         if newmask.sum() == mask.sum():
            #             break
            #         mask = newmask
            #     v = vals[mask]
            #     return np.mean(v) if v.size else np.nan
            #
            # def robust_rolling_mean(series, window_minutes=60, sigma=2.0, max_iter=5, min_samples=5):
            #     s = series.dropna().sort_index()
            #     return s.rolling(f"{window_minutes}min", min_periods=min_samples, center=True)\
            #             .apply(lambda x: _iter_sigma_clip_mean(x, sigma=sigma, max_iter=max_iter), raw=True)
            #
            #
            # # Calcula médias robustas e suavizadas
            # window_minutes = 60
            # sigma_clip = 2.0
            # gaussian_sigma = 4.0
            #
            # df_plot_sorted = df_plot.sort_values('datetime').set_index('datetime')
            #
            # df_plot_sorted['stec_mean'] = robust_rolling_mean(df_plot_sorted['stec'], window_minutes, sigma_clip)
            # df_plot_sorted['vtec_mean'] = robust_rolling_mean(df_plot_sorted['vtec'], window_minutes, sigma_clip)
            #
            # df_plot_sorted['stec_mean_gauss'] = pd.Series(
            #     gaussian_filter1d(df_plot_sorted['stec_mean'].interpolate(limit_direction='both'), sigma=gaussian_sigma),
            #     index=df_plot_sorted.index
            # )
            # df_plot_sorted['vtec_mean_gauss'] = pd.Series(
            #     gaussian_filter1d(df_plot_sorted['vtec_mean'].interpolate(limit_direction='both'), sigma=gaussian_sigma),
            #     index=df_plot_sorted.index
            # )
            #
            # df_plot_sorted['hour'] = (
            #     df_plot_sorted.index.hour +
            #     df_plot_sorted.index.minute / 60.0 +
            #     df_plot_sorted.index.second / 3600.0
            # )
            #
            # print("Gaussian-smoothed 2σ rolling means computed for STEC and VTEC.")
            #
            # # ==============================================================
            # # Visualization toggle: colormap or fixed color (for STEC & VTEC)
            # # ==============================================================
            # use_cmap = True           # True = use colormap; False = fixed color
            # cmap_name = 'jet'          # Used only when use_cmap=True
            # color_stec = 'blue'        # Fixed color for STEC when use_cmap=False
            # color_vtec = 'red'         # Fixed color for VTEC when use_cmap=False
            #
            # # -------------------------------
            # # Panel 1: STEC
            # # -------------------------------
            # if use_cmap:
            #     sc1 = ax1.scatter(
            #         df_plot_sorted['hour'], df_plot_sorted['stec'],
            #         c=df_plot_sorted['stec'],
            #         cmap=cmap_name,
            #         s=1, alpha=0.8,
            #         vmin=0, vmax=150,
            #         label='STEC'
            #     )
            # else:
            #     sc1 = ax1.scatter(
            #         df_plot_sorted['hour'], df_plot_sorted['stec'],
            #         color=color_stec,
            #         s=1, alpha=0.8,
            #         label='STEC'
            #     )
            #
            # # curva média-suavizada (amarela)
            # ax1.plot(
            #     df_plot_sorted['hour'], df_plot_sorted['stec_mean_gauss'],
            #     color='#ffb000', linewidth=2.0, label='STEC mean (2σ+Gauss)'
            # )
            #
            # # -------------------------------
            # # Panel 2: VTEC
            # # -------------------------------
            # if use_cmap:
            #     sc2 = ax2.scatter(
            #         df_plot_sorted['hour'], df_plot_sorted['vtec'],
            #         c=df_plot_sorted['vtec'],
            #         cmap=cmap_name,
            #         s=1, alpha=0.8,
            #         vmin=0, vmax=60,
            #         label='VTEC'
            #     )
            # else:
            #     sc2 = ax2.scatter(
            #         df_plot_sorted['hour'], df_plot_sorted['vtec'],
            #         color=color_vtec,
            #         s=1, alpha=0.8,
            #         label='VTEC'
            #     )
            #
            # # curva média-suavizada (amarela)
            # ax2.plot(
            #     df_plot_sorted['hour'], df_plot_sorted['vtec_mean_gauss'],
            #     color='#ffb000', linewidth=2.0, label='VTEC mean (2σ+Gauss)'
            # )






            # ==============================================================
            # Média robusta + suavização gaussiana (estilo IRTAM) — OPCIONAL
            # ==============================================================
            from scipy.ndimage import gaussian_filter1d

            compute_means = False  # <<<<< defina False se quiser desativar tudo

            if compute_means:

                def _iter_sigma_clip_mean(x, sigma=2.0, max_iter=5):
                    vals = pd.Series(x).dropna().to_numpy()
                    if vals.size == 0:
                        return np.nan
                    mask = np.ones(vals.size, dtype=bool)
                    for _ in range(max_iter):
                        v = vals[mask]
                        if v.size == 0:
                            return np.nan
                        mu = np.mean(v)
                        sd = np.std(v, ddof=1) if v.size > 1 else 0.0
                        if sd == 0:
                            break
                        newmask = np.abs(vals - mu) <= sigma * sd
                        if newmask.sum() == mask.sum():
                            break
                        mask = newmask
                    return np.mean(vals[mask]) if mask.any() else np.nan

                def robust_rolling_mean(series, window_minutes=60, sigma=2.0, max_iter=5, min_samples=5):
                    s = series.dropna().sort_index()
                    return s.rolling(f"{window_minutes}min", min_periods=min_samples, center=True)\
                            .apply(lambda x: _iter_sigma_clip_mean(x, sigma=sigma, max_iter=max_iter), raw=True)

                # Calcula médias robustas e suavizadas
                window_minutes = 60
                sigma_clip = 2.0
                gaussian_sigma = 4.0

                df_plot_sorted = df_plot.sort_values('datetime').set_index('datetime')
                df_plot_sorted['stec_mean'] = robust_rolling_mean(df_plot_sorted['stec'], window_minutes, sigma_clip)
                df_plot_sorted['vtec_mean'] = robust_rolling_mean(df_plot_sorted['vtec'], window_minutes, sigma_clip)

                df_plot_sorted['stec_mean_gauss'] = pd.Series(
                    gaussian_filter1d(df_plot_sorted['stec_mean'].interpolate(limit_direction='both'), sigma=gaussian_sigma),
                    index=df_plot_sorted.index
                )
                df_plot_sorted['vtec_mean_gauss'] = pd.Series(
                    gaussian_filter1d(df_plot_sorted['vtec_mean'].interpolate(limit_direction='both'), sigma=gaussian_sigma),
                    index=df_plot_sorted.index
                )

                print("Gaussian-smoothed 2σ rolling means computed for STEC and VTEC.")

            else:
                df_plot_sorted = df_plot.sort_values('datetime').set_index('datetime')
                df_plot_sorted['stec_mean_gauss'] = np.nan
                df_plot_sorted['vtec_mean_gauss'] = np.nan

            # ==============================================================
            # Visualization toggle: colormap or fixed color (for STEC & VTEC)
            # ==============================================================

            use_cmap = True
            cmap_name = 'jet'
            color_stec = 'blue'
            color_vtec = 'red'

            # -------------------------------
            # Panel 1: STEC
            # -------------------------------
            if use_cmap:
                sc1 = ax1.scatter(
                    df_plot_sorted['hour'], df_plot_sorted['stec'],
                    c=df_plot_sorted['stec'], cmap=cmap_name,
                    s=1, alpha=0.8, vmin=0, vmax=150, label='STEC'
                )
            else:
                sc1 = ax1.scatter(
                    df_plot_sorted['hour'], df_plot_sorted['stec'],
                    color=color_stec, s=1, alpha=0.8, label='STEC'
                )

            if compute_means:
                ax1.plot(df_plot_sorted['hour'], df_plot_sorted['stec_mean_gauss'],
                        color='#ffb000', linewidth=2.0, label='STEC mean (2σ+Gauss)')

            # -------------------------------
            # Panel 2: VTEC
            # -------------------------------
            if use_cmap:
                sc2 = ax2.scatter(
                    df_plot_sorted['hour'], df_plot_sorted['vtec'],
                    c=df_plot_sorted['vtec'], cmap=cmap_name,
                    s=1, alpha=0.8, vmin=0, vmax=60, label='VTEC'
                )
            else:
                sc2 = ax2.scatter(
                    df_plot_sorted['hour'], df_plot_sorted['vtec'],
                    color=color_vtec, s=1, alpha=0.8, label='VTEC'
                )

            if compute_means:
                ax2.plot(df_plot_sorted['hour'], df_plot_sorted['vtec_mean_gauss'],
                        color='#ffb000', linewidth=2.0, label='VTEC mean (2σ+Gauss)')



            # Axes formatting
            ax1.set_xlim(0, 24)
            ax1.set_xticks(np.arange(0, 25, 3))
            ax1.set_title('Slant TEC (STEC) vs UT')
            ax1.set_xlabel('Universal Time (UT)')
            ax1.set_ylabel('STEC (TECU)', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.grid(True, ls='--', alpha=0.4)
            # Colorbar for STEC (safe when using fixed color too, but meaningful when use_cmap=True)
            cbar1 = fig.colorbar(sc1, ax=ax1, orientation='horizontal', pad=0.12, fraction=0.05)
            cbar1.set_label('STEC (TECU)', fontsize=10)

            # # Optional smoothing (kept for compatibility, not plotted directly)
            # df_sorted = df_plot.sort_values('hour')
            # window_size = 60
            # df_sorted['vtec_smooth'] = df_sorted['vtec'].rolling(window=window_size, center=True).mean()

            ax2.set_xlim(0, 24)
            ax2.set_xticks(np.arange(0, 25, 3))
            ax2.set_title('Vertical TEC (VTEC) vs UT')
            ax2.set_xlabel('Universal Time (UT)')
            ax2.set_ylabel('VTEC (TECU)', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.grid(True, ls='--', alpha=0.4)
            cbar2 = fig.colorbar(sc2, ax=ax2, orientation='horizontal', pad=0.12, fraction=0.05)
            cbar2.set_label('VTEC (TECU)', fontsize=10)

            # -------------------------------
            # Panel 3: IPP (colored by VTEC, 'jet')
            # -------------------------------
            sc3 = ax3.scatter(
                df_plot['lon'], df_plot['lat'],
                c=df_plot['vtec'],
                cmap='jet',
                s=5, alpha=0.8,
                vmin=0, vmax=60
            )
            ax3.set_title('IPP Geographic Distribution (colored by VTEC)')
            ax3.set_xlabel('Longitude (°)')
            ax3.set_ylabel('Latitude (°)')
            ax3.grid(True, ls='--', alpha=0.4)
            ax3.set_xlim(df_plot['lon'].min() - 2, df_plot['lon'].max() + 2)
            ax3.set_ylim(df_plot['lat'].min() - 2, df_plot['lat'].max() + 2)
            if lat_sta_plot is not None and lon_sta_plot is not None:
                ax3.scatter(lon_sta_plot, lat_sta_plot, color='black', marker='*', s=120, label='Station')
            cbar3 = fig.colorbar(sc3, ax=ax3, orientation='horizontal', pad=0.1, fraction=0.05)
            cbar3.set_label('VTEC (TECU)', fontsize=10)
            ax3.legend(loc='upper right', fontsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save figure
            if save_to is not None:
                try:
                    fig.savefig(save_to, dpi=200)
                    print(f"Figure saved to: {save_to}")
                except Exception as e:
                    print(f"[WARN] Could not save figure: {e}")

            # Show (if requested)
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        # Call plotting (preserving the original look & outputs)
        plot_tec_results(
            file_path=tec_path,
            station_plot=station,
            doy_plot=f"{doy_int:03d}",
            lat_sta_plot=crd[1],
            lon_sta_plot=crd[0],
            filter_by_elevation=True,
            elev_threshold=30.0,
            save_to=fig_path
        )

    # ===============================
    # Pipeline execution
    # ===============================
    # Normalize paths
    rnx3_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize DOY (3 digits)
    try:
        doy_int = int(doy)
    except Exception:
        raise ValueError(f"Invalid DOY: {doy}")
    doy_int = int(doy_int)
    if doy_int < 0 or doy_int > 366:
        raise ValueError(f"DOY out of range [0, 366]: {doy_int}")

    # --- Pass 1: LGF_combination (L1-L2) ---
    print(f"[INFO] Processing L1-L2 for station={station}, DOY={doy_int:03d}, YEAR={year}")
    sat_data_L12, lat_sta_L12, lon_sta_L12 = generate_sat_data(rnx3_dir)
    process_station_memory(
        sat_data_L12, station, lat_sta_L12, lon_sta_L12, doy_int,
        out_suffix="_L1L2", base_out_dir=out_dir
    )

    # --- Pass 2: LGF_combination15 (L1-L5) ---
    try:
        print(f"[INFO] Processing L1-L5 for station={station}, DOY={doy_int:03d}, YEAR={year}")
        sat_data_L15, lat_sta_L15, lon_sta_L15 = generate_sat_data_l15(rnx3_dir)
        process_station_memory(
            sat_data_L15, station, lat_sta_L15, lon_sta_L15, doy_int,
            out_suffix="_L1L5", base_out_dir=out_dir
        )
    except Exception as e:
        print(f"[WARN] LGF_combination15 unavailable or no valid data: {e}")

    print("[DONE] TECcalc pipeline finished.")
