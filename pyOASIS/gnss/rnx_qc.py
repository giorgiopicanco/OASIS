"""Public RNX_QC interface; original MEDUSA historical QC is unchanged."""
from __future__ import annotations
import numpy as np

SYSTEM_ORDER = 'GREC'


def satellite_key(satellite):
    """Order GPS, GLONASS, Galileo, BeiDou, then ascending PRN."""
    return (SYSTEM_ORDER.index(str(satellite)[0]), str(satellite))


def diagnostic_arrays(q):
    """Project original compact-stage diagnostic arrays onto the RINEX grid.

    -1 means a stage did not process that epoch; NaN means no reported slip size.
    A candidate is not automatically a confirmed corrected cycle slip.
    """
    n = len(q.flags_final)
    details = {}
    source = np.asarray(getattr(q, 'source_indices', []), dtype=int)
    cursor = 0
    for r in getattr(q, 'original_b111_result', ()):
        count = len(r.source_indices)
        ix = source[cursor:cursor + count]
        cursor += count
        if len(ix) != count or np.any(ix < 0) or np.any(ix >= n):
            raise ValueError('RNX_QC diagnostic source alignment mismatch')
        for name, stage in [('initial', r.initial_updarc),
                            ('mw', r.chkwub_result), ('phase', r.chkl4p_result)]:
            flags = np.asarray(stage.flags, dtype=np.int64)
            if len(flags) != count:
                raise ValueError('RNX_QC diagnostic flag alignment mismatch')
            details.setdefault(name + '_flags', np.full(n, -1, dtype=np.int64))[ix] = flags
        for name, stage, field in [('mw_slip_cycles', r.chkwub_result, 'slip_cycles'),
                                   ('phase1_slip_cycles', r.chkl4p_result, 'cycl1'),
                                   ('phase2_slip_cycles', r.chkl4p_result, 'cycl2')]:
            indices = np.asarray(stage.slip_indices, dtype=int)
            values = np.asarray(getattr(stage, field), dtype=float)
            if len(indices) != len(values) or np.any(indices < 0) or np.any(indices >= count):
                raise ValueError(f'RNX_QC {name} diagnostic alignment mismatch')
            details.setdefault(name, np.full(n, np.nan))[ix[indices]] = values
    if cursor and cursor != len(source):
        raise ValueError('RNX_QC incomplete diagnostic source mapping')
    return details


def diagnostic_masks(q, details=None):
    """Boolean masks from actual original flags, never inferred from jumps."""
    flags = np.asarray(q.flags_final, dtype=np.int64)
    known = flags >= 0
    raw_phase = np.asarray(q.l4_raw_m, dtype=float)
    raw_code = np.asarray(q.code_gf_m, dtype=float)
    masks = {
        'missing_phase': ~np.isfinite(raw_phase),
        'missing_code': ~np.isfinite(raw_code),
        'excluded_before_final_qc': ~known & np.isfinite(raw_phase) & np.isfinite(raw_code),
        'outlier_or_bad': known & ((flags & 1) != 0),
        'new_ambiguity': known & ((flags & 2) != 0),
        'corrected_cycle_slip': known & ((flags & 4) != 0),
        'phase_slip_candidate': np.asarray(q.slip, dtype=bool),
        'accepted_level': np.isfinite(q.l4_leveled_m) & (np.asarray(q.arc_ids) >= 0) & known & ((flags & 1) == 0),
    }
    details = diagnostic_arrays(q) if details is None else details
    for stage in ('initial', 'mw', 'phase'):
        key = stage + '_flags'
        if key in details:
            f = details[key]
            masks[stage + '_bad'] = (f >= 0) & ((f & 1) != 0)
    if 'mw_slip_cycles' in details:
        masks['mw_slip_candidate'] = np.isfinite(details['mw_slip_cycles'])
    if 'phase1_slip_cycles' in details:
        masks['phase_slip_reported'] = np.isfinite(details['phase1_slip_cycles'])
    return masks


def run_rnx_qc(*args, **kwargs):
    """Run the preserved original engine through its existing RINEX adapter."""
    from .medusa_b111 import run_b111
    return run_b111(*args, **kwargs)
