"""Station link statistics on fixed UTC bins; no interpolation across gaps."""
from collections import defaultdict
import numpy as np
from scipy.signal import savgol_filter

PRODUCTS = ('DELTATEC', 'VTEC', 'DTEC')
WINDOW_MS = {'DELTATEC': 150000, 'ROTI': 150000, 'SIDX': 60000}


def station_statistics(data, product, *, grid_ms, cadence_ms, minimum=20., min_satellites=4, method='mean', smooth_minutes=0.):
    """Equal satellite weights; missing bins and insufficient support stay NaN.

    grid_ms is the original RINEX UTC epoch grid. DeltaTEC uses fixed UTC
    150-second windows, labelled by their centers. No resampling of links.
    """
    if not np.isfinite(minimum) or not 0 <= minimum <= 90:
        raise ValueError('Invalid elevation cutoff')
    if min_satellites < 1 or method not in ('mean', 'median') or not np.isfinite(smooth_minutes) or smooth_minutes < 0:
        raise ValueError('Invalid station statistics configuration')
    grid = np.asarray(grid_ms, dtype=np.int64)
    step = WINDOW_MS.get(product, int(cadence_ms))
    if step <= 0 or not len(grid):
        raise ValueError('Positive cadence and nonempty UTC grid required')
    if product in WINDOW_MS:
        grid = np.arange(grid[0]//step*step, grid[-1]//step*step+step, step, dtype=np.int64)
    else:
        # Retain the original sampling phase, including the GPST-to-UTC offset.
        grid = np.arange(grid[0], grid[-1]+1, step, dtype=np.int64)
    bins = defaultdict(dict)
    key = 'window_start_utc_ms' if product in WINDOW_MS else 'utc_ms'
    for sv,d in data.items():
        v = np.asarray(d['value'], float); e = np.asarray(d['elevation_deg'], float)
        good = np.isfinite(v) & np.isfinite(e) & (e >= minimum)
        if 'geometry_available' in d: good &= np.asarray(d['geometry_available'], bool)
        # Unphysical negative absolute TEC is excluded from the station estimate.
        # Signed detrended TEC and DeltaTEC are valid.
        if product == 'VTEC': good &= v >= 0
        for t,x in zip(np.asarray(d[key], np.int64)[good], v[good]):
            bins[int(t)].setdefault(sv, []).append(float(x))
    raw = np.full(len(grid), np.nan); median = raw.copy(); count = np.zeros(len(grid), np.int32)
    for i,t in enumerate(grid):
        links = bins.get(int(t), {})
        if not links: continue
        values = [np.mean(v) for v in links.values()]
        count[i] = len(values)
        raw[i] = np.mean(values); median[i] = np.median(values)
    selected = (raw if method == 'mean' else median).copy()
    selected[count < min_satellites] = np.nan
    smooth = np.full(len(grid), np.nan)
    # Largest odd window whose endpoint span does not exceed requested duration.
    half = int(smooth_minutes*60000//(2*step)); width = 2*half+1
    if smooth_minutes > 0 and width >= 3:
        valid = np.flatnonzero(np.isfinite(selected))
        for seg in np.split(valid, np.flatnonzero(np.diff(valid)>1)+1):
            if len(seg) >= width:
                z = savgol_filter(selected[seg], width, 1)
                smooth[seg[half:len(seg)-half]] = z[half:len(seg)-half]
    return dict(utc_ms=grid+(step//2 if product in WINDOW_MS else 0), mean=raw, median=median,
                count=count, selected=selected, smoothed=smooth)


def epoch_grid(geo):
    from .ionosphere_products import gpst_array_to_utc_ms
    sat = next(iter(geo.satellites.values()))
    t = gpst_array_to_utc_ms(sat.times)
    return t, int(np.median(np.diff(t))) if len(t)>1 else 0


def roti_station_data(rows):
    """Adapt fixed 150 s ROTI records without changing their values."""
    data = defaultdict(lambda: defaultdict(list))
    for r in rows or []:
        if int(r['window_seconds']) != 150:
            raise ValueError('Expected fixed 150-second ROTI')
        d = data[str(r['satellite'])]
        d['window_start_utc_ms'].append(int(r.get('window_start_utc_ms', int(r['utc_ms']) - 75000)))
        d['value'].append(float(r['roti_tecu_min']))
        d['elevation_deg'].append(float(r['elevation_deg']))
    return {sv: dict(d) for sv, d in data.items()}


def lgf_station_data(geo):
    """Accepted levelled phase, on the original epoch grid, in metres."""
    from .ionosphere_products import gpst_array_to_utc_ms
    return {sv: dict(utc_ms=gpst_array_to_utc_ms(s.times),
                     value=np.where(np.asarray(s.arc_ids)>=0, s.l4_leveled_m, np.nan),
                     elevation_deg=s.geometry['elevation_deg'],
                     geometry_available=s.geometry_valid)
            for sv, s in geo.satellites.items()}
