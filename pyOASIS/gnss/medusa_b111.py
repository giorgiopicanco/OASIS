"""Run the original MEDUSA B111 QC core on pyOASIS RINEX 2/3 observations.

No GeoRinex and no intermediate files. The original MEDUSA B111 functions
are imported unchanged; this module only adapts input/output representations.
The archived historical B111 code itself documents unresolved Bernese pieces;
this adapter does not claim Bernese equivalence or override these limitations.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import ast
from importlib import import_module
from ..runtime import bundled_medusa_root
import re
from types import SimpleNamespace
import sys
import numpy as np

from .rinex_io import read_observations, RinexObservations


@dataclass(frozen=True)
class B111Satellite:
    satellite: str
    signals: tuple[str, str, str, str]
    times: np.ndarray
    l4_raw_m: np.ndarray
    l4_corrected_m: np.ndarray
    l4_leveled_m: np.ndarray
    code_gf_m: np.ndarray
    mw_raw_cycles: np.ndarray
    mw_corrected_cycles: np.ndarray
    flags_final: np.ndarray
    arc_ids: np.ndarray
    arc_bias_m: np.ndarray
    residual_m: np.ndarray
    slip: np.ndarray
    source_indices: np.ndarray
    original_b111_result: object
    f1_hz: float | None = None
    f2_hz: float | None = None
    source_observables: tuple[str, str, str, str] | None = None


    @property
    def lgf_raw_m(self):
        return self.l4_raw_m

    @property
    def lgf_corrected_m(self):
        return self.l4_corrected_m

    @property
    def lgf_leveled_m(self):
        return self.l4_leveled_m

    @property
    def pgf_raw_m(self):
        return self.code_gf_m


@dataclass(frozen=True)
class B111Dataset:
    source: str
    satellites: dict[str, B111Satellite]
    status: str = 'MEDUSA_B111_HISTORICAL_CORE'
    skipped: dict[str, str] | None = None
    glonass_channel_sources: dict[str, str] | None = None
    glonass_channel_conflicts: dict[str, tuple[int, int]] | None = None


def _medusa_original_chooser(root: Path):
    """Load ONLY MEDUSA's original choose_combination function from its source.

    The MEDUSA diagnostic runner disables Matplotlib legends globally at import
    time (Axes.legend = lambda ...: None). Importing it directly would silently
    corrupt all pyOASIS figures. Extracting its unmodified function AST avoids
    executing those unrelated runner-level side effects. Scientific selection
    remains EXACTLY the function shipped in the supplied MEDUSA source tree.
    """
    src = root / 'tools' / 'historical' / 'b111' / 'run_real_all_satellites.py'
    parsed = ast.parse(src.read_text(encoding='utf-8'), filename=str(src))
    matches = [node for node in parsed.body
               if isinstance(node, ast.FunctionDef) and node.name == 'choose_combination']
    if len(matches) != 1:
        raise RuntimeError('Original MEDUSA choose_combination() unavailable/ambiguous')
    module = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    ns = {'np': np}
    exec(compile(module, str(src), 'exec'), ns)
    return ns['choose_combination']


def _medusa_original_config(root: Path, offset):
    """Read MEDUSA live B111_CONFIG unchanged without importing the UI runner."""
    src = root / 'tools' / 'realtime' / 'b111_live.py'
    parsed = ast.parse(src.read_text(encoding='utf-8'), filename=str(src))
    matches = [node for node in parsed.body
               if isinstance(node, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'B111_CONFIG' for t in node.targets)]
    if len(matches) != 1:
        raise RuntimeError('MEDUSA B111_CONFIG missing/ambiguous')
    module = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    ns = {'bernese_offset': offset}
    exec(compile(module, str(src), 'exec'), ns)
    return dict(ns['B111_CONFIG'])


def _core(medusa_root):
    """Import the original B111 QC and physical signal routines unchanged."""
    root = Path(medusa_root).resolve()
    if not (root / 'tools' / 'historical' / 'b111' / 'rinex_driver.py').is_file():
        raise FileNotFoundError(f'MEDUSA B111 not found under {root}')
    if root == bundled_medusa_root().resolve():
        prefix = 'pyOASIS._vendor.medusa.tools.historical.b111'
    else:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        prefix = 'tools.historical.b111'
    driver = import_module(prefix + '.rinex_driver')
    screening = import_module(prefix + '.screening')
    build_rinex_combinations_from_stream = driver.build_rinex_combinations_from_stream
    run_b111_rinex_qc = driver.run_b111_rinex_qc
    bernese_offset = screening.bernese_offset
    B111_CONFIG = _medusa_original_config(root, bernese_offset)
    choose_combination = _medusa_original_chooser(root)
    return (build_rinex_combinations_from_stream, run_b111_rinex_qc,
            choose_combination, bernese_offset, B111_CONFIG)


def _stream_from_rinex(obs: RinexObservations):
    """Adapter matching MEDUSA B111's stream/header shape, no new RINEX decoder."""
    n = len(obs.times)
    streams = {}
    for sat, signal_map in obs.series.items():
        streams[sat] = {k: np.asarray(v.values, dtype=np.float64).copy() for k, v in signal_map.items()}
        # Existing B111 expects each observed signal aligned to the global epoch grid.
        for code, arr in streams[sat].items():
            if len(arr) != n:
                raise ValueError(f'{sat}/{code}: invalid observation length')
    header = SimpleNamespace(version=obs.version,
        observation_types={k: list(v) for k, v in obs.declared_signals.items()},
        glonass_channels=dict(obs.glonass_channels))
    stream = SimpleNamespace(times=np.asarray(obs.times, dtype='datetime64[ns]'), satellites=streams)
    return stream, header


def resolve_glonass_channels(header_channels, external_channels=None):
    """Prefer per-RINEX GLONASS SLOT / FRQ #; use explicit fallback only if absent.

    Returns (effective_channels, source_by_sv, conflicts). Conflicts are
    recorded as (RINEX_header_k, external_k) and NEVER override the header.
    An external table is not assumed valid for the RINEX date; provenance is
    retained so its use is auditable.
    """
    header_channels = dict(header_channels or {})
    external_channels = dict(external_channels or {})
    def check(sv, k):
        if not isinstance(sv, str) or len(sv) != 3 or sv[0] != 'R' or not sv[1:].isdigit():
            raise ValueError(f'Invalid GLONASS satellite identifier: {sv!r}')
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not -7 <= int(k) <= 6:
            raise ValueError(f'{sv}: invalid GLONASS FDMA channel: {k!r}')
        return int(k)
    a = {sv: check(sv, k) for sv, k in header_channels.items()}
    b = {sv: check(sv, k) for sv, k in external_channels.items()}
    effective = dict(b)
    effective.update(a)  # authoritative metadata, even when conflicting
    sources = {sv: ('RINEX_HEADER' if sv in a else 'EXTERNAL_FALLBACK') for sv in effective}
    conflicts = {sv: (a[sv], b[sv]) for sv in a.keys() & b.keys() if a[sv] != b[sv]}
    return effective, sources, conflicts


def _legacy_mixed_stream_from_rinex(obs: RinexObservations, *, glonass_channels=None):
    """Translate legacy RINEX 2 *names* for MEDUSA's original G/R/E/C core.

    RINEX 2 lacks a tracking-mode suffix: the suffixes below are compatibility
    labels only, NOT inferred physical modulation codes. Values, masks, and
    epoch positions are preserved unchanged. R channels must be supplied from
    verified epoch-appropriate metadata; no hidden station-independent table.
    """
    if not 2.0 <= float(obs.version) < 3.0:
        raise ValueError('RINEX 2 adapter called on a non-RINEX 2 dataset')
    if any(sv.startswith('C') for sv in obs.satellites):
        raise ValueError('RINEX2 BeiDou tracking aliases are not established; use RINEX3 BeiDou signals')
    stream, header = _stream_from_rinex(obs)
    if glonass_channels is not None:
        for sv, channel in dict(glonass_channels).items():
            if not (isinstance(sv, str) and len(sv) == 3 and sv[0] == 'R' and sv[1:].isdigit()):
                raise ValueError(f'Invalid GLONASS SV in channel metadata: {sv!r}')
            if not isinstance(channel, (int, np.integer)) or not -7 <= int(channel) <= 6:
                raise ValueError(f'{sv}: invalid GLONASS FDMA channel {channel!r}')
            if sv in header.glonass_channels and header.glonass_channels[sv] != int(channel):
                raise ValueError(f'{sv}: conflicting RINEX/header and supplied FDMA channels')
            header.glonass_channels[sv] = int(channel)

    # Historical RINEX 2 cannot identify the exact tracking-mode suffix.
    # Map only code/phase pairs on the same *physical band*; never copy signals
    # across frequency bands. These names are solely for MEDUSA's exact
    # Cxx/Lxx matching and its original physical_pairs() frequency resolver.
    band_candidates = {
        'G': (('1', (('P1', 'P'), ('C1', 'C'))),
              ('2', (('P2', 'P'), ('C2', 'C')))),
        'R': (('1', (('P1', 'P'), ('C1', 'C'))),
              ('2', (('P2', 'P'), ('C2', 'C')))),
        'E': (('1', (('C1', 'C'), ('P1', 'P'))),
              ('7', (('C7', 'Q'),)),
              ('5', (('C5', 'Q'),)),
              ('8', (('C8', 'Q'),))),
    }
    aliases_by_sat = {}
    declared_by_system = {}
    for sat in obs.satellites:
        system = sat[0]
        raw = stream.satellites[sat]
        converted = {}
        aliases = {}
        for digit, candidates in band_candidates.get(system, ()):
            phase = 'L' + digit
            if phase not in raw:
                continue
            phase_values = np.asarray(raw[phase], dtype=float)
            for code, suffix in candidates:
                if code not in raw:
                    continue
                code_values = np.asarray(raw[code], dtype=float)
                if not np.any(np.isfinite(phase_values) & np.isfinite(code_values)):
                    continue
                # The historical RINEX 2 code names are preserved in aliases.
                phase_alias, code_alias = 'L'+digit+suffix, 'C'+digit+suffix
                converted[phase_alias] = phase_values.copy()
                converted[code_alias] = code_values.copy()
                aliases[phase_alias] = phase
                aliases[code_alias] = code
        stream.satellites[sat] = converted
        aliases_by_sat[sat] = aliases
        declared_by_system.setdefault(system, set()).update(converted)
    header.observation_types = {sys: sorted(types) for sys, types in declared_by_system.items()}
    return stream, header, aliases_by_sat


# Backward-compatible name for existing GPS RINEX2 bridge tests.
def _legacy_gps_stream_from_rinex(obs):
    """Back-compat GPS-only adapter: retain original P-before-C policy/tests.

    Production mixed-RINEX2 processing uses _legacy_mixed_stream_from_rinex,
    which passes all valid band/code alternatives to MEDUSA's native chooser.
    """
    if any(not sv.startswith('G') for sv in obs.satellites):
        raise ValueError('Legacy GPS only helper received non-GPS data')
    stream, header, aliases = _legacy_mixed_stream_from_rinex(obs)
    chosen_aliases = {}
    declared = set()
    for sat in obs.satellites:
        a = aliases[sat]
        best = {}
        for band in ('1', '2'):
            suffix = next((c for c in ('P', 'C')
                           if 'L'+band+c in a and 'C'+band+c in a), None)
            if suffix is None:
                best = {}
                break
            best['L'+band+suffix] = a['L'+band+suffix]
            best['C'+band+suffix] = a['C'+band+suffix]
        chosen_aliases[sat] = best
        stream.satellites[sat] = {name: stream.satellites[sat][name]
                                  for name in best}
        declared.update(best)
    header.observation_types = {'G': sorted(declared)}
    return stream, header, chosen_aliases


def _expand(arr, source_indices, n, *, dtype=float, fill=np.nan):
    out = np.full(n, fill, dtype=dtype)
    vals = np.asarray(arr)
    if vals.size != len(source_indices):
        raise ValueError('MEDUSA B111 returned inconsistent compact result length')
    out[source_indices] = vals
    return out


def _rejected_updarc_arcs(pre, *, minobs):
    """Return arcs unsafe for CHKWUB OFFSET, using MEDUSA's UPDARC flags."""
    rejected = []
    for arc in pre.updarc_result.arcs:
        flags = pre.updarc_result.flags[int(arc.start):int(arc.end) + 1]
        usable = int(np.count_nonzero((flags & 5) == 0))
        if usable < 2:
            rejected.append((int(arc.start), int(arc.end), usable))
    return rejected


def _valid_arc_segments(pre):
    """Map eligible UPDARC compact arcs back to full RINEX epoch indices."""
    compact = np.asarray(pre.source_indices, dtype=np.int64)
    flags = np.asarray(pre.updarc_result.flags, dtype=np.int64)
    result = []
    rejected = []
    for arc in pre.updarc_result.arcs:
        start, end = int(arc.start), int(arc.end) + 1
        if not (0 <= start < end <= len(compact)):
            raise ValueError('Invalid UPDARC arc bounds')
        usable = int(np.count_nonzero((flags[start:end] & 5) == 0))
        if usable >= 2:
            result.append(compact[start:end].copy())
        else:
            rejected.append((start, end - 1, usable))
    return result, rejected


def _slice_combination(chosen, original_indices):
    """Keep original B111 observable units, taking one valid UPDARC arc."""
    from dataclasses import replace
    fields = ('times', 'phase1_cycles', 'phase2_cycles', 'code1_m',
              'code2_m', 'l4_m', 'code_gf_m', 'l5_cycles', 'valid')
    return replace(chosen, **{
        name: np.asarray(getattr(chosen, name))[original_indices].copy()
        for name in fields
    })


def _process_arc_segments(chosen, pre, qc, cfg, n):
    """Run MEDUSA's unchanged B111 QC on eligible arcs, restoring full-grid indices."""
    indices_by_arc, rejected = _valid_arc_segments(pre)
    names = ('l4_corrected_m', 'l4_leveled_m', 'mw_corrected_cycles',
             'arc_bias_m', 'residual_m')
    data = {name: np.full(n, np.nan, dtype=float) for name in names}
    flags = np.full(n, -1, dtype=np.int64)
    arc_ids = np.full(n, -1, dtype=np.int64)
    slips = np.zeros(n, dtype=np.int8)
    processed_idx = []
    qc_results = []
    next_arc_id = 0
    for full_indices in indices_by_arc:
        segment = _slice_combination(chosen, full_indices)
        try:
            result = qc(segment, **cfg)
        except ValueError as exc:
            # Screening may remove the remaining usable samples after UPDARC.
            # Never fabricate an OFFSET or retain a partially processed arc.
            match = re.fullmatch(
                r"Bernese OFFSET requires at least two usable observations \(NTOT=([01])\)",
                str(exc),
            )
            if match is None:
                raise
            rejected.append((int(full_indices[0]), int(full_indices[-1]),
                             int(match.group(1)), 'OFFSET_INSUFFICIENT_USABLE'))
            continue
        local = np.asarray(result.source_indices, dtype=np.int64)
        if np.any(local < 0) or np.any(local >= len(full_indices)):
            raise ValueError('B111 QC returned invalid segment indices')
        original = full_indices[local]
        if len(original) != len(set(original.tolist())):
            raise ValueError('B111 QC returned duplicate indices')
        level, upd, mw, l4 = (result.leveling_result, result.final_updarc,
                              result.chkwub_result, result.chkl4p_result)
        values = {
            'l4_corrected_m': upd.observations,
            'l4_leveled_m': level.leveled_gf_m,
            'mw_corrected_cycles': mw.observations,
            'arc_bias_m': level.arc_bias_m,
            'residual_m': level.residual_m,
        }
        for name, source in values.items():
            source = np.asarray(source, dtype=float)
            if len(source) != len(original):
                raise ValueError(f'B111 {name}: size mismatch')
            data[name][original] = source
        flags[original] = np.asarray(upd.flags, dtype=np.int64)
        local_arc = np.asarray(level.arc_id, dtype=np.int64)
        valid_ids = sorted(set(int(x) for x in local_arc if x >= 0))
        translated = {v: next_arc_id + i for i, v in enumerate(valid_ids)}
        arc_ids[original] = [translated.get(int(x), -1) for x in local_arc]
        next_arc_id += len(valid_ids)
        for i in l4.slip_indices:
            i = int(i)
            if not 0 <= i < len(original):
                raise ValueError('B111 invalid slip index')
            slips[original[i]] = 1
        processed_idx.extend(original.tolist())
        qc_results.append(result)
    return data, flags, arc_ids, slips, np.asarray(processed_idx, dtype=np.int64), qc_results, rejected


def run_b111(rinex_path, medusa_root, *, systems='GREC', config=None, glonass_channels=None, observations=None) -> B111Dataset:
    """Original MEDUSA B111 on pyOASIS RINEX 2/3 GREC with valid-arc isolation.

    No intermediate files. Rejected short arcs are excluded *individually*;
    eligible arcs of the same satellite remain processable. Not yet production
    qualified: compare per-arc output against an independent MEDUSA baseline.
    """
    builder, qc, chooser, offset, default_config = _core(medusa_root)
    obs = observations if observations is not None else read_observations(rinex_path, systems=systems)
    effective_channels, channel_sources, channel_conflicts = resolve_glonass_channels(
        obs.glonass_channels, glonass_channels)
    if obs.version < 3.0:
        stream, header, _legacy_aliases = _legacy_mixed_stream_from_rinex(
            obs, glonass_channels=effective_channels)
        missing_r_channels = [sv for sv in obs.satellites if sv.startswith('R')
                              and stream.satellites.get(sv) and sv not in header.glonass_channels]
        if missing_r_channels:
            raise ValueError('RINEX 2 GLONASS requires verified FDMA channels for '+
                             ', '.join(missing_r_channels)+
                             '; pass --glonass-channels <JSON> with channels for the RINEX epoch')
    else:
        stream, header = _stream_from_rinex(obs)
        # The MEDUSA original frequency resolver consumes header.glonass_channels.
        # Never replace an in-file GLONASS FDMA channel with the legacy table.
        header.glonass_channels = effective_channels
    cfg = dict(default_config)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f'Unknown B111 settings: {sorted(unknown)}')
        cfg.update(config)
    cfg['offset_func'] = offset
    build_rinex_updarc = import_module(builder.__module__).build_rinex_updarc
    out, skipped = {}, {}
    n = len(obs.times)
    for sat in obs.satellites:
        if sat[0] not in systems:
            continue
        if obs.version < 3.0 and not stream.satellites.get(sat):
            skipped[sat] = 'NO_COMPATIBLE_DUAL_FREQUENCY_PAIR'
            continue
        try:
            combinations = builder(rinex_path, sat, stream=stream, header=header)
        except ValueError as exc:
            # MEDUSA's GLONASS FDMA resolver requires real satellite channel
            # metadata. Skip ONLY this satellite if the channel is missing.
            if sat.startswith('R') and 'channel' in str(exc).lower():
                skipped[sat] = f'MISSING_GLONASS_FDMA_CHANNEL: {exc}'
                continue
            raise
        chosen = chooser(combinations, sat[0])
        if chosen is None:
            skipped[sat] = 'NO_COMPATIBLE_DUAL_FREQUENCY_PAIR'
            continue
        pre = build_rinex_updarc(chosen, gaparc=cfg['gaparc'], minobs=cfg['minobs'])
        data, flags, arcids, slips, processed, results, rejected = _process_arc_segments(
            chosen, pre, qc, cfg, n
        )
        offset_rejected = sum(len(item) == 4 for item in rejected)
        if not results:
            skipped[sat] = (f'NO_ELIGIBLE_UPDARC_ARCS; rejected={len(rejected)}; '
                            f'OFFSET_INSUFFICIENT_USABLE={offset_rejected}')
            continue
        out[sat] = B111Satellite(
            satellite=sat,
            signals=(chosen.phase1, chosen.phase2, chosen.code1, chosen.code2),
            times=np.asarray(obs.times).copy(),
            l4_raw_m=np.asarray(chosen.l4_m).copy(),
            l4_corrected_m=data['l4_corrected_m'],
            l4_leveled_m=data['l4_leveled_m'],
            code_gf_m=np.asarray(chosen.code_gf_m).copy(),
            mw_raw_cycles=np.asarray(chosen.l5_cycles).copy(),
            mw_corrected_cycles=data['mw_corrected_cycles'],
            flags_final=flags, arc_ids=arcids,
            arc_bias_m=data['arc_bias_m'], residual_m=data['residual_m'],
            slip=slips, source_indices=processed,
            original_b111_result=tuple(results),
            f1_hz=float(chosen.f1_hz), f2_hz=float(chosen.f2_hz),
            source_observables=(tuple(_legacy_aliases[sat][key] for key in
                                      (chosen.phase1, chosen.phase2, chosen.code1, chosen.code2))
                                if obs.version < 3.0 else
                                (chosen.phase1, chosen.phase2, chosen.code1, chosen.code2)),
        )
        if rejected:
            skipped[sat] = (f'{len(rejected)-offset_rejected} SHORT_ARC(S) SKIPPED; '
                            f'OFFSET_INSUFFICIENT_USABLE={offset_rejected}; eligible arcs processed')
    status = ('MEDUSA_B111_HISTORICAL_CORE' if out and not skipped else
              'MEDUSA_B111_PARTIAL_WITH_SKIPPED_ARCS' if out else
              'MEDUSA_B111_NO_PROCESSABLE_SATELLITES')
    return B111Dataset(str(rinex_path), out, status, skipped,
                       {sv: channel_sources[sv] for sv in channel_sources if sv in obs.satellites},
                       {sv: channel_conflicts[sv] for sv in channel_conflicts if sv in obs.satellites})
