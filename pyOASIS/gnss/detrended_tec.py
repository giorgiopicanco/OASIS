"""VTEC detrending with a centered linear Savitzky–Golay filter.

Zhang-style method, adapted to a 900 s window and native cadence.
No interpolation, no extrapolation, no cross-arc smoothing.
"""
from collections import Counter
import numpy as np
from scipy.signal import savgol_filter
from .rnx_qc import satellite_key
from .ionosphere_products import gpst_array_to_utc_ms

WINDOW_SECONDS = 900
POLYORDER = 1
MIN_ELEVATION_DEG = 30.0


def _final_slip_mask(q, size):
    """Project original final UPDARC slip indices onto the full epoch grid."""
    mask = np.zeros(size, dtype=bool)
    source = np.asarray(getattr(q, 'source_indices', []), dtype=np.int64)
    cursor = 0
    for result in getattr(q, 'original_b111_result', ()):
        n = len(result.source_indices)
        original = source[cursor:cursor+n]
        cursor += n
        if len(original) != n:
            raise ValueError('Incomplete RNX_QC source mapping')
        up = getattr(result, 'final_updarc', None)
        if up is None:
            continue
        for arc in up.arcs:
            slip = np.asarray(arc.slip_indices if arc.slip_indices is not None else [], dtype=np.int64)
            if np.any(slip < 0) or np.any(slip >= n):
                raise ValueError('Invalid final UPDARC slip indices')
            mask[original[slip]] = True
    if cursor and cursor != len(source):
        raise ValueError('Incomplete RNX_QC source mapping')
    return mask


def compute_detrended_tec(geo, vtec):
    """Return accepted center samples and provenance for each QC segment."""
    output = {}
    stats = Counter()
    for sv, data in sorted(vtec.items(), key=lambda item: satellite_key(item[0])):
        sat = geo.satellites[sv]
        grid = gpst_array_to_utc_ms(sat.times)
        t = np.asarray(data['utc_ms'], dtype=np.int64)
        y = np.asarray(data['value'], dtype=float)
        arcs = np.asarray(data['arc_id'], dtype=np.int64)
        if len(t) == 0:
            continue
        if np.any(np.diff(grid) <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError(f'{sv}: non-increasing epoch grid')
        if len(grid) < 2:
            continue
        cadence = int(np.median(np.diff(grid)))
        if cadence <= 0 or cadence > WINDOW_SECONDS * 1000 // 2:
            stats['unsupported_cadence_satellites'] += 1
            continue
        n = int(round(WINDOW_SECONDS * 1000 / cadence))
        if n % 2 == 0:
            n += 1
        n = max(n, 3)
        half = n // 2
        ix = np.searchsorted(grid, t)
        if np.any(ix >= len(grid)) or not np.array_equal(grid[ix], t):
            raise ValueError(f'{sv}: VTEC epochs do not match RNX_QC grid')
        b = sat.b111
        flags = np.asarray(b.flags_final, dtype=np.int64)[ix]
        final_slips = _final_slip_mask(b, len(grid))[ix]
        if not np.array_equal(arcs, np.asarray(b.arc_ids)[ix]):
            raise ValueError(f'{sv}: VTEC arc provenance mismatch')
        good = (np.isfinite(y) & (y >= 0) & (arcs >= 0) & (flags >= 0)
                & ((flags & 1) == 0) & (np.asarray(data['elevation_deg']) >= MIN_ELEVATION_DEG)
                & np.asarray(data['physical_nonnegative'], dtype=bool))
        stats['input_vtec_samples'] += len(t)
        stats['excluded_invalid_or_low_elevation'] += int((~good).sum())
        j = np.flatnonzero(good)
        if not len(j):
            continue
        # Break even if a corrected slip is inside the same ambiguity arc.
        cuts = np.flatnonzero((np.diff(j) != 1) | (np.diff(ix[j]) != 1)
                             | (np.diff(t[j]) != cadence) | (np.diff(arcs[j]) != 0)
                             | ((flags[j[1:]] & 6) != 0) | final_slips[j[1:]]) + 1
        chunks = []
        for segid, seg in enumerate(np.split(j, cuts)):
            if len(seg) < n:
                stats['short_segment_samples'] += len(seg)
                continue
            bg = savgol_filter(y[seg], n, POLYORDER, mode='interp')
            center = seg[half:len(seg)-half]
            smooth = bg[half:len(seg)-half]
            stats['edge_samples_omitted'] += 2 * half
            chunks.append((center, smooth, np.full(len(center), segid, dtype=np.int64)))
        if not chunks:
            continue
        j = np.concatenate([c[0] for c in chunks]); bg = np.concatenate([c[1] for c in chunks])
        d = {k: np.asarray(data[k])[j] for k in ('utc_ms','arc_id','elevation_deg','ipp_lat_deg','ipp_lon_deg','ipp_height_km')}
        d.update(value=y[j]-bg, vtec_tecu=y[j], background_vtec_tecu=bg,
                 segment_id=np.concatenate([c[2] for c in chunks]),
                 source_epoch_index=ix[j], window_samples=np.full(len(j),n,dtype=np.int64),
                 actual_window_span_ms=np.full(len(j),(n-1)*cadence,dtype=np.int64),
                 cadence_ms=np.full(len(j),cadence,dtype=np.int64))
        output[sv] = d
        stats['accepted_samples'] += len(j)
    return output, dict(stats, window_seconds=WINDOW_SECONDS, polynomial_order=POLYORDER,
                        elevation_cutoff_deg=MIN_ELEVATION_DEG,
                        method='VTEC_MINUS_CENTERED_SAVITZKY_GOLAY',
                        calibration_dependency='EXPERIMENTAL_MODEL_CALIBRATION',
                        edge_policy='FULL_CENTERED_WINDOW_ONLY')
