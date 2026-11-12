"""RINEX2/3 -> SP3 geometry -> MEDUSA-equation GF and arc leveling.

Integration stage 2. Uses pyOASIS.gnss.rinex_io. No GeoRinex/RTKLIB.
NOT a port of MEDUSA iterative C++ quality control: classification is explicitly
PRELIMINARY; results must not be treated as science-QC-certified outputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import csv
import math
import numpy as np
from .rinex_io import read_observations

C = 299792458.0
OMEGA_E = 7.2921151467e-5
EARTH_R = 6378136.3
SHELL_R = 6378136.3 + 350000.0


def utc64(t):
    return np.datetime64(t, 'ns')


def seconds(t):
    return float(np.datetime64(t, 'ns').astype('int64')) / 1e9


class Sp3Orbit:
    """Native SP3 P records; 11-point Neville interpolation as in MEDUSA Sp3Orbit.cpp.

    Uses SP3 epoch labels as GNSS-scale epochs. RINEX must use the same time
    scale (GPS expected). No automatic UTC/GPS conversion is attempted.
    """
    def __init__(self, path):
        self.records = defaultdict(list)
        epoch = None
        with open(path, encoding='ascii', errors='replace') as f:
            for line in f:
                if line.startswith('*'):
                    z = line[1:].split()
                    if len(z) < 6:
                        continue
                    y, mo, d, h, mi = map(int, z[:5]); sec = float(z[5])
                    epoch = np.datetime64(datetime(y, mo, d, h, mi), 'ns') + np.timedelta64(round(sec * 1e9), 'ns')
                elif epoch is not None and line.startswith('P') and len(line) >= 46:
                    sv = line[1:4].strip()
                    try:
                        p = np.array([float(line[a:b]) for a, b in ((4, 18), (18, 32), (32, 46))])
                    except ValueError:
                        continue
                    if np.isfinite(p).all() and (np.abs(p) < 999999).all():
                        self.records[sv].append((seconds(epoch), p * 1000))
        if not self.records:
            raise ValueError('SP3 has no valid satellite positions')
        for sv in list(self.records):
            seq = self.records[sv]
            seq.sort(key=lambda z: z[0]); self.records[sv] = (np.array([e[0] for e in seq]), np.array([e[1] for e in seq]))

    def position(self, satellite, time):
        if satellite not in self.records:
            return None
        x, y = self.records[satellite]
        t = seconds(time)
        if not x[0] <= t <= x[-1]:
            return None
        j = int(np.searchsorted(x, t))
        if j < len(x) and abs(x[j] - t) < 1e-7:
            return y[j].copy()
        if len(x) < 11:
            return None  # consistent with Sp3Orbit.cpp: no lower-degree fallback
        first = min(max(j - 5, 0), len(x) - 11)
        xx = x[first:first+11] - t
        q = y[first:first+11].copy()
        for k in range(1, 11):
            for i in range(11-k):
                den = xx[i] - xx[i+k]
                if den == 0: return None
                q[i] = ((-xx[i+k]) * q[i] + xx[i] * q[i+1]) / den
        return q[0]


def receiver_geodetic(xyz):
    x, y, z = [float(v) for v in xyz]
    if np.linalg.norm([x, y, z]) < 1e6: raise ValueError('Invalid RINEX receiver ECEF')
    a = 6378137.0; e2 = 6.69437999014e-3
    lon = math.atan2(y, x); p = math.hypot(x, y); lat = math.atan2(z, p * (1-e2))
    for _ in range(12):
        n = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        lat = math.atan2(z + e2 * n * math.sin(lat), p)
    return lat, lon


def link_geometry(receiver, satellite):
    """ECEF geometric range with Sagnac, ENU az/el, thin-shell spherical IPP.

    This is a Python counterpart, NOT byte-for-byte RTKLIB ionppp; validate
    numerical differences before claiming geometrical equivalence to MEDUSA.
    """
    rr = np.asarray(receiver, float); rs = np.asarray(satellite, float)
    d = rs - rr; dist = float(np.linalg.norm(d))
    if dist < 1: raise ValueError('Invalid geometry')
    los = d / dist
    rho = dist + OMEGA_E * (rs[0]*rr[1] - rs[1]*rr[0]) / C
    lat, lon = receiver_geodetic(rr)
    east = -math.sin(lon)*los[0] + math.cos(lon)*los[1]
    north = (-math.sin(lat)*math.cos(lon)*los[0] - math.sin(lat)*math.sin(lon)*los[1] + math.cos(lat)*los[2])
    up = math.cos(lat)*math.cos(lon)*los[0] + math.cos(lat)*math.sin(lon)*los[1] + math.sin(lat)*los[2]
    el = math.asin(max(-1, min(1, up))); az = math.atan2(east, north) % (2*math.pi)
    res = dict(sat_x_m=rs[0], sat_y_m=rs[1], sat_z_m=rs[2], range_m=rho, azimuth_deg=math.degrees(az), elevation_deg=math.degrees(el), ipp_lat_deg=math.nan, ipp_lon_deg=math.nan, mapping_factor=math.nan)
    if el <= 0: return res
    # Sphere intersection along receiver->satellite line (receiver norm used)
    b = float(np.dot(rr, los)); discr = b*b + SHELL_R**2 - float(np.dot(rr, rr))
    if discr <= 0: return res
    path = -b + math.sqrt(discr)
    if path < 0: return res
    ipp = rr + path*los
    res['ipp_lat_deg'] = math.degrees(math.atan2(ipp[2], math.hypot(ipp[0], ipp[1])))
    res['ipp_lon_deg'] = (math.degrees(math.atan2(ipp[1], ipp[0])) + 180) % 360 - 180
    # Standard thin-shell obliquity for spherical earth (approximation)
    res['mapping_factor'] = 1 / math.sqrt(1 - (EARTH_R/SHELL_R * math.cos(el))**2)
    return res


def signal_codes(observations, satellite, first, second):
    """Signal pairs provided explicitly as tracking suffix (3.x) or obs codes (2.x).

    e.g. first='1C' second='2W' for RINEX3 -> C1C/L1C, C2W/L2W;
    RINEX2 -> first='L1,C1' second='L2,P2'.
    """
    if observations.version >= 3:
        return 'L'+first, 'C'+first, 'L'+second, 'C'+second
    try:
        l1, p1 = first.split(','); l2, p2 = second.split(',')
    except ValueError as e:
        raise ValueError('RINEX2 pair must be phase,code e.g. L1,C1 L2,P2') from e
    return l1, p1, l2, p2


def resolve_frequency(observations, satellite, phasecode):
    s = observations.get(satellite, phasecode)
    if s is not None and s.frequency_hz and s.wavelength_m:
        return float(s.frequency_hz), float(s.wavelength_m)
    if observations.version < 3 and satellite.startswith('G'):
        band = phasecode[1:2]
        freq = {'1': 1575.42e6, '2': 1227.60e6, '5': 1176.45e6}.get(band)
        if freq: return freq, C/freq
    raise ValueError(f'Frequency unavailable: {satellite}/{phasecode}; no guessing GLONASS channels')


def is_loss_of_lock(lli):
    """RINEX LLI bit 0 means loss of lock only for a PRESENT LLI value.

    Our RINEX reader uses -1 for an absent LLI; Python's (-1 & 1) == 1,
    so missing indicators must not be interpreted as cycle slips.
    Other LLI bits do not, by themselves, signify loss of lock.
    """
    n = int(lli)
    return n >= 0 and bool(n & 1)


def process(rinex_path, sp3_path, *, systems='G', pairs=None, min_elevation_deg=10.0, gap_factor=1.5):
    """Return leveled rows with provisional QC. Not validated MEDUSA C++ QC parity.

    pairs example {'G': [('1C','2W')]} RINEX3; {'G':[('L1,C1','L2,P2')]} RINEX2.
    """
    obs = read_observations(rinex_path, systems=systems)
    if obs.time_system not in (None, '', 'GPS', 'GPST'):
        raise ValueError(f'Time system {obs.time_system} requires explicit conversion before SP3 interpolation')
    if not obs.receiver_xyz_m: raise ValueError('RINEX missing APPROX POSITION XYZ')
    orbit = Sp3Orbit(sp3_path)
    if pairs is None:
        pairs = {'G': [('1C','2W')]} if obs.version >= 3 else {'G':[('L1,C1','L2,P2')]}
    if len(obs.times) < 2: raise ValueError('Need at least 2 epochs for cadence/QC')
    dt = np.diff(obs.times.astype('datetime64[ns]').astype('int64'))/1e9
    nominal = float(np.min(dt[dt>0]))  # conservative shortest observed cadence; caller should override for irregular datasets
    rows = []
    for sv in obs.satellites:
        if sv[0] not in systems: continue
        for pair1, pair2 in pairs.get(sv[0], []):
            l1, p1, l2, p2 = signal_codes(obs, sv, pair1, pair2)
            f1, w1 = resolve_frequency(obs, sv, l1); f2, w2 = resolve_frequency(obs, sv, l2)
            if abs(f1-f2) < 1: raise ValueError('Signals must have distinct frequencies')
            v1 = obs.dense(sv, l1); v2 = obs.dense(sv, l2)
            c1 = obs.dense(sv, p1); c2 = obs.dense(sv, p2)
            lli1 = obs.get(sv,l1).lli if obs.get(sv,l1) else np.zeros(len(obs.times),np.int8)
            lli2 = obs.get(sv,l2).lli if obs.get(sv,l2) else np.zeros(len(obs.times),np.int8)
            prev_time = None
            for i, t in enumerate(obs.times):
                if not (np.isfinite(v1[i]) and np.isfinite(v2[i]) and np.isfinite(c1[i]) and np.isfinite(c2[i])):
                    prev_time = None
                    continue
                xyz = orbit.position(sv, t)
                if xyz is None:
                    prev_time = None
                    continue
                geom = link_geometry(obs.receiver_xyz_m, xyz)
                # Select only usable elevations. No hidden cut applied to output QC flags.
                if geom['elevation_deg'] <= min_elevation_deg:
                    prev_time = None
                    continue
                phase1 = float(v1[i]*w1); phase2 = float(v2[i]*w2)
                lgf = phase1-phase2; cgf = float(c2[i]-c1[i])
                wide_phase = (f1*phase1-f2*phase2)/(f1-f2)
                narrow_code = (f1*c1[i]+f2*c2[i])/(f1+f2)
                mw_m = float(wide_phase-narrow_code)
                gap = prev_time is None or (seconds(t)-prev_time > nominal*gap_factor)
                lli_slip = is_loss_of_lock(lli1[i]) or is_loss_of_lock(lli2[i])
                prev_time = seconds(t)
                rows.append(dict(epoch_index=i,time=str(t),satellite=sv,signal_1=pair1,signal_2=pair2,phase_1_code=l1,phase_2_code=l2,code_1_code=p1,code_2_code=p2,phase_1_cycles=float(v1[i]),phase_2_cycles=float(v2[i]),code_1_m=float(c1[i]),code_2_m=float(c2[i]),frequency_1_hz=f1,frequency_2_hz=f2,lgf_m=lgf,cgf_m=cgf,mw_m=mw_m,gap_before=gap,lli_slip_before=lli_slip,qc_status='PROVISIONAL_LLI_GAP_ONLY',**geom))
    # MEDUSA diagnostic leveling convention; outlier/slip classification is not C++-equivalent yet.
    streams = defaultdict(list)
    for r in rows: streams[(r['satellite'],r['signal_1'],r['signal_2'])].append(r)
    arcs = []; arc_id = 0
    for key in sorted(streams):
        current = []
        for r in sorted(streams[key], key=lambda a:a['epoch_index']):
            if current and (r['gap_before'] or r['lli_slip_before']):
                arcs.append((arc_id,current));arc_id+=1;current=[]
            current.append(r)
        if current: arcs.append((arc_id,current));arc_id+=1
    result = []
    for aid, arc in arcs:
        # Same level bias formula as src/gnss/tools/gf_arc_leveling.py
        bias = float(np.mean([r['lgf_m']-r['cgf_m'] for r in arc]))
        for r in arc:
            x = dict(r,arc_id=aid,arc_size=len(arc),leveling_points=len(arc),level_bias_m=bias,legf_m=r['lgf_m']-bias)
            result.append(x)
    result.sort(key=lambda r:(r['epoch_index'],r['satellite'],r['signal_1'],r['signal_2']))
    return result


def export_csv(rows, path):
    if not rows: raise ValueError('No usable output; check signal mapping, SP3 coverage and elevation')
    p = Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return p
