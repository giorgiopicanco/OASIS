"""pyOASIS command-line pipeline and reusable processing entry point.

Stage 24: RNX_QC, phase LGF, orbit geometry, MEDUSA V6 ROTI and optional
experimental OASIS deltaTEC, SIDX, calibrated TEC and detrended VTEC.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from pyOASIS.gnss.rnx_qc import satellite_key
from pyOASIS.runtime import bundled_medusa_root


FIGURE_PRODUCTS = frozenset({'LGF','ROTI','DELTATEC','SIDX','STEC','VTEC','DTEC','IPP'})


def parse_figure_products(value):
    selected = {x.strip().replace('Δ','DELTA').upper() for x in value.split(',')}
    if selected == {'ALL'}:
        return None
    if not selected or selected - FIGURE_PRODUCTS:
        raise argparse.ArgumentTypeError('Unknown figure product; use LGF,ROTI,DELTATEC,SIDX,STEC,VTEC,dTEC,IPP or ALL')
    return selected


def daily_output_dir(root, station, year, doy):
    # Avoid path traversal and invalid Windows filename characters in marker names.
    import re
    station = str(station).strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9_-]*', station):
        raise ValueError('Invalid station identifier; supply --station with a safe station code')
    return Path(root) / station / str(year) / f'{int(doy):03d}'


def build_parser():
    p = argparse.ArgumentParser(description='OASIS: Open-Access System for Ionospheric Studies')
    p.add_argument('--rinex', required=True, type=Path, help='RINEX 2/3 observation file')
    p.add_argument('--sp3', type=Path, default=None,
                   help='Explicit matching SP3 path. If omitted, resolve by RINEX date in --orbit-dir.')
    p.add_argument('--orbit-dir', type=Path, default=Path('INPUT/ORBITS'),
                   help='SP3 cache directory (default INPUT/ORBITS)')
    orbit_actions = p.add_mutually_exclusive_group()
    orbit_actions.add_argument('--no-download', action='store_true',
                              help='Offline: require existing valid SP3; never access NASA')
    orbit_actions.add_argument('--force-download', action='store_true',
                              help='Download a fresh orbit even if SP3 exists')
    p.add_argument('--earthdata-user', default=None,
                   help='NASA Earthdata login (only needed if the orbit must be downloaded)')
    p.add_argument('--orbit-prefix', choices=['GFZ0MGXRAP', 'JAX0MGXFIN'],
                   default=None, help='Override automatic orbit product prefix')
    p.add_argument('--medusa-root', type=Path,
                   default=bundled_medusa_root(),
                   help='Optional external MEDUSA override; defaults to the bundled runtime')
    p.add_argument('--systems', default='GREC', help='Request G/R/E/C jointly (default GREC). Missing input constellations are reported, not invented.')
    p.add_argument('--require-grec', action='store_true', help='Fail if the RINEX/RNX_QC lacks ANY of G,R,E,C; never claim GPS-only as GREC')
    p.add_argument('--glonass-channels', type=Path, default=None, help='Optional GLONASS FDMA fallback (RINEX 2/3). RINEX 3 header always wins.')
    p.add_argument('--station', default=None)
    p.add_argument('--supplemental-sp3', type=Path, action='append', default=[],
                   help='Additional existing SP3 file; used only if the primary SP3 cannot supply a satellite position. Repeatable.')
    p.add_argument('--out', type=Path, default=Path('OUTPUT'), help='Output root; station/year/DOY is appended automatically')
    figure_choice = p.add_mutually_exclusive_group()
    figure_choice.add_argument('--figures', dest='figures', action='store_true',
                               help='Generate selected OASIS overview and RNX_QC figures')
    figure_choice.add_argument('--no-figures', dest='figures', action='store_false',
                               help='Do not create figures (default)')
    p.set_defaults(figures=False)
    p.add_argument('--qc-figures', action=argparse.BooleanOptionalAction, default=False, help='Enable per-satellite RNX_QC diagnostics (default: disabled)')
    p.add_argument('--overview-figures', action=argparse.BooleanOptionalAction, default=True, help='Enable or disable summary product and IPP figures')
    p.add_argument('--plot-min-elevation', type=float, default=20.0, help='Minimum elevation in degrees for overview figures only (default: 20)')
    p.add_argument('--mean-line', action=argparse.BooleanOptionalAction, default=True, help='Single station mean for VTEC, DELTATEC and DTEC only')
    p.add_argument('--mean-min-satellites', type=int, default=4)
    p.add_argument('--mean-method', choices=['mean','median'], default='mean')
    p.add_argument('--mean-smooth-minutes', type=float, default=0., help=argparse.SUPPRESS)
    p.add_argument('--product-figure-color-mode', '--figure-color-mode', choices=['constellation','blue_red'], default='constellation')
    p.add_argument('--figure-products', type=parse_figure_products, default=None, help='Comma-separated LGF,ROTI,DELTATEC,SIDX,STEC,VTEC,dTEC,IPP or ALL; default all available')
    p.add_argument('--qc-elevation', action=argparse.BooleanOptionalAction, default=False, help='Add satellite elevation panel to RNX_QC figures')
    p.add_argument('--qc-zoom', action=argparse.BooleanOptionalAction, default=False, help='Zoom RNX_QC figures to the observed pass')
    p.add_argument('--figure-dir', type=Path, default=None,
                   help='Override overview directory; default: station/year/DOY folder')
    p.add_argument('--max-qc-figures', type=int, default=None,
                   help='Limit RNX_QC per-satellite QC figures (default: all; 0 means none)')
    p.add_argument('--figure-dpi', type=int, default=300,
                   help='PNG resolution (default: 300 DPI)')
    p.add_argument('--verbose-satellites', action='store_true',
                   help=argparse.SUPPRESS)
    p.add_argument('--export', action='store_true', help='Legacy scientific exports remain disabled; use --save-output hdf5')
    p.add_argument('--products', '--product', default='LGF', type=lambda x: x.replace("Δ", "DELTA").upper(),
                   help='Comma-separated products: LGF,ROTI,DELTATEC,SIDX,STEC,VTEC,dTEC, or ALL (TEC includes both STEC and VTEC)')
    p.add_argument('--save-output', choices=['none','hdf5'], default='hdf5',
                   help='Write one combined HDF5 with all processed satellites (or none)')
    p.add_argument('--output-file', type=Path, default=None,
                   help='Output HDF5 filename; default: STATION_YEAR_DOY_ION.h5')
    p.add_argument('--allow-missing-geometry', action='store_true',
                   help='Backward-compatible no-op: missing SP3 geometry is now preserved as NaN by default.')
    p.add_argument('--require-complete-geometry', action='store_true',
                   help='Strict mode: fail if any satellite with accepted LGF has zero valid IPP points.')
    p.add_argument('--quiet-progress', action='store_true',
                   help='Suppress progress messages on stderr; a compact result remains on stdout.')
    p.add_argument('--overwrite-output', action='store_true',
                   help='Explicitly replace existing HDF5 output file')
    return p


def validate_inputs(args, p):
    from pyOASIS.gnss.ionosphere_products import PRODUCTS
    requested = {x.strip() for x in args.products.upper().replace('Δ','DELTA').split(',')}
    if requested == {'ALL'}: requested=set(PRODUCTS)
    if 'TEC' in requested:
        requested.remove('TEC'); requested.update(('STEC','VTEC'))
    if not requested or requested-set(PRODUCTS): p.error('Unknown product selection')
    args.products = ','.join(x for x in PRODUCTS if x in requested)
    args.systems = ''.join(x for x in 'GREC' if x in args.systems) if set(args.systems)<=set('GREC') and len(set(args.systems))==len(args.systems) else args.systems

    if not args.rinex.is_file():
        p.error(f'RINEX not found: {args.rinex}')
    if args.sp3 is not None and not args.sp3.is_file():
        p.error(f'Explicit SP3 not found: {args.sp3}')
    if args.sp3 is not None and args.force_download:
        p.error('--force-download conflicts with --sp3; omit --sp3 for automatic retrieval')
    if not args.medusa_root.joinpath('tools/historical/b111/rinex_driver.py').is_file():
        p.error(f'MEDUSA B111 original not found: {args.medusa_root}')
    if not args.systems or len(set(args.systems)) != len(args.systems) or set(args.systems) - set('GREC'):
        p.error('--systems accepts unique letters drawn from GREC')
    if args.export:
        p.error('Legacy text export is disabled. Use --save-output hdf5.')
    if args.require_complete_geometry and args.allow_missing_geometry:
        p.error('Use only one geometry policy: --require-complete-geometry or --allow-missing-geometry.')
    for source in args.supplemental_sp3:
        if not source.is_file():
            p.error(f'Supplemental SP3 does not exist: {source}')
    if args.max_qc_figures is not None and args.max_qc_figures < 0:
        p.error('--max-qc-figures must be >= 0')
    if args.mean_min_satellites < 1 or not np.isfinite(args.mean_smooth_minutes) or args.mean_smooth_minutes < 0:
        p.error('Invalid station mean settings')
    if not 0 <= args.plot_min_elevation <= 90:
        p.error('--plot-min-elevation must be between 0 and 90')
    if not 72 <= args.figure_dpi <= 600:
        p.error('--figure-dpi must be between 72 and 600')
    if args.output_file is not None and args.save_output != 'hdf5':
        p.error('--output-file requires --save-output hdf5')



def resolve_orbit(args):
    """Resolve an orbit for the given RINEX, using a validated local cache first.

    No network access, import of requests or password prompt happens when
    an existing valid SP3 is selected. Explicit --sp3 is never replaced.
    """
    from pyOASIS.gnss.mgex_sp3 import (
        date_from_rinex_filename, inspect_sp3, main as download_main, product_for
    )

    inspected = None
    if args.sp3 is not None:
        selected = args.sp3
        source = 'EXPLICIT_SP3'
    else:
        obs_date = date_from_rinex_filename(args.rinex)
        basename, url = product_for(obs_date, args.orbit_prefix)
        selected = args.orbit_dir / basename
        source = 'LOCAL_CACHE' if selected.is_file() else 'CDDIS_DOWNLOAD'
        local_ok = False
        if selected.is_file() and not args.force_download:
            try:
                epochs, satellites = inspect_sp3(selected)
                local_ok = epochs >= 2 and any(satellites.values())
                if local_ok:
                    inspected = (epochs, satellites)
            except (OSError, ValueError):
                local_ok = False
        if not local_ok:
            if args.no_download:
                raise FileNotFoundError(
                    f'Offline mode: missing or invalid SP3: {selected}. '
                    'Download automatically by omitting --no-download.'
                )
            if args.force_download:
                source = 'CDDIS_FORCED_REFRESH'
            elif selected.exists():
                source = 'CDDIS_REPLACED_INVALID_CACHE'
            cmd = ['--date', obs_date.isoformat(), '--out', str(args.orbit_dir)]
            if args.orbit_prefix:
                cmd += ['--prefix', args.orbit_prefix]
            if args.earthdata_user:
                cmd += ['--username', args.earthdata_user]
            if args.force_download or selected.is_file():
                cmd += ['--force']
            if download_main(cmd) != 0:
                raise RuntimeError(f'Cannot retrieve and validate matching SP3: {url}')
    epochs, satellites = inspected if inspected is not None else inspect_sp3(selected)
    if epochs < 2 or not any(satellites.values()):
        raise ValueError(f'Invalid SP3 orbit {selected}: epochs={epochs}, GREC={satellites}')
    requested = set(args.systems)
    present = {c for c, n in satellites.items() if n > 0}
    if not requested.intersection(present):
        raise ValueError(f'SP3 {selected} has no requested GNSS systems: {args.systems}')
    args.sp3 = selected
    return {'status': source, 'file': str(selected), 'epochs': epochs,
            'satellites_by_system': satellites}


def _label(rinex, systems, station, *, observations=None):
    from pyOASIS.gnss.rinex_io import read_observations
    obs = observations if observations is not None else read_observations(rinex, systems=systems)
    if len(obs.times) == 0:
        raise ValueError('RINEX observation file has no epochs')
    date = obs.times[0].astype('datetime64[D]').astype(object)
    return (station or obs.marker or rinex.name[:4]).strip().upper(), f'{date.timetuple().tm_yday:03d}', str(date.year)


def _verify_figures(paths, folder):
    paths = [Path(p) for p in paths]
    if not paths:
        raise RuntimeError('No figures were produced: output count must not report success')
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size < 100]
    if missing:
        raise RuntimeError(f'PNG generation failed or files are empty: {missing}')
    if len(set(p.resolve() for p in paths)) != len(paths):
        raise RuntimeError('Figure renderer returned duplicate output paths')
    return {'count': len(paths), 'folder': str(folder), 'files_verified': True,
            'names': [p.name for p in paths]}


def _progress(message, *, enabled=True):
    if enabled:
        print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', file=sys.stderr, flush=True)


def run(args):
    verbose = not args.quiet_progress
    start = time.perf_counter()
    _progress('[STEP 1/8] Resolving SP3 orbit files (using local cache when available)', enabled=verbose)
    orbit_report = resolve_orbit(args)
    _progress(f'[STEP 1/8] Primary SP3: {orbit_report["file"]} | source={orbit_report["status"]} | epochs={orbit_report["epochs"]}', enabled=verbose)
    for supplemental in args.supplemental_sp3:
        _progress(f'[STEP 1/8] Supplemental SP3: {supplemental}', enabled=verbose)
    # Uses the original B111 functions via existing pyOASIS MEDUSA bridge.
    from pyOASIS.gnss.rnx_qc import run_rnx_qc
    from pyOASIS.gnss.medusa_b111_sp3 import attach_sp3_geometry

    channels = None
    if args.glonass_channels is not None:
        if not args.glonass_channels.is_file():
            raise FileNotFoundError(f'GLONASS channel map not found: {args.glonass_channels}')
        channels = json.loads(args.glonass_channels.read_text(encoding='utf-8'))
        if not isinstance(channels, dict):
            raise ValueError('GLONASS channel JSON must contain an Rxx -> integer object')
    from pyOASIS.gnss.rinex_io import read_observations
    _progress(f'[STEP 2/8] Reading RINEX: {args.rinex}', enabled=verbose)
    observations = read_observations(args.rinex, systems=args.systems)
    _progress(f'[STEP 2/8] Epochs={len(observations.times)} | observed satellites={len(observations.satellites)}', enabled=verbose)
    station, doy, year = _label(args.rinex, args.systems, args.station, observations=observations)
    output_dir = daily_output_dir(args.out, station, year, doy)
    dest = args.output_file or (output_dir / f'{station}_{year}_{doy}_ION.h5')
    _progress(f'[OUTPUT] Daily directory: {output_dir.resolve()}', enabled=verbose)
    if args.save_output == 'hdf5' and dest.exists() and not args.overwrite_output:
        raise FileExistsError(f'HDF5 exists: {dest}. Use --overwrite-output or a different output directory.')
    _progress('[STEP 3/8] Running RNX_QC: signal selection, QC, arcs and LGF leveling', enabled=verbose)
    t_b111 = time.perf_counter()
    qc = run_rnx_qc(args.rinex, args.medusa_root, systems=args.systems,
                  glonass_channels=channels, observations=observations)
    _progress(f'[STEP 3/8] RNX_QC completed in {time.perf_counter()-t_b111:.1f}s | processed satellites={len(qc.satellites)}', enabled=verbose)
    counts = Counter(sv[0] for sv in qc.satellites)
    if not counts:
        raise RuntimeError('RNX_QC produced no accepted satellite: cannot generate scientific figures')
    if args.require_grec:
        absent = [sys for sys in 'GREC' if counts.get(sys, 0) == 0]
        if absent:
            raise RuntimeError(
                f'REAL GREC REQUESTED but RNX_QC found no accepted data for {absent}. '
                f'Actual systems: {dict(counts)}. Check input coverage, valid pairs and FDMA channels. '
                'not a MEDUSA QC conversion. Use a multiconstellation RINEX such as CHPI. '
                'No GPS-only figures/products were mislabeled as GREC.'
            )
    # Compact in-memory result; no general run report or satellite audit file.
    report = {
        'products': {},
        'output_directory': str(output_dir.resolve()),
        'systems_processed_by_rnx_qc': dict(counts),
        'glonass_fdma_external_validation': (
            'REQUIRED_FOR_FALLBACK_CHANNELS' if 'EXTERNAL_FALLBACK' in
            (qc.glonass_channel_sources or {}).values() else 'NOT_APPLICABLE'),
    }
    from pyOASIS.gnss.orbit_sources import CompositeSp3Orbit
    _progress('[STEP 4/8] Attaching SP3 geometry at signal transmission time', enabled=verbose)
    t_geo = time.perf_counter()
    orbit_catalog = CompositeSp3Orbit([args.sp3, *args.supplemental_sp3])
    geo = attach_sp3_geometry(qc, args.rinex, args.sp3, systems=args.systems,
                              observations=observations, orbit=orbit_catalog)
    _progress(f'[STEP 4/8] Geometry completed in {time.perf_counter()-t_geo:.1f}s', enabled=verbose)
    summary = geo.summary()
    _progress(f'[STEP 4/8] LGF samples={summary["leveled_samples"]} | valid IPP samples={summary["geometry_valid_samples"]}', enabled=verbose)
    coverage_warnings = []
    offset_rejections = {sv: reason for sv, reason in qc.skipped.items()
                         if re.search(r'OFFSET_INSUFFICIENT_USABLE=[1-9][0-9]*', reason)}
    if offset_rejections:
        coverage_warnings.append('RNX_QC rejected arcs after OFFSET screening: ' +
                                 '; '.join(f'{sv}: {reason}' for sv, reason in offset_rejections.items()))
    if summary['orbit_samples'] == 0 or summary['geometry_valid_samples'] == 0:
        warning = ('No valid orbital geometry was found. LGF can still be exported, '
                   'ROTI remains available when RNX_QC passes; IPP is flagged unavailable. '
                   'Check date, satellite IDs, SP3 coverage, and time scale.')
        coverage_warnings.append(warning)
        _progress('[WARNING] ' + warning, enabled=True)
    # Require at least a geometric association for *each* GNSS system with B111 samples.
    # Avoid presenting a GREC run as complete when a constellation is missing in SP3.
    geometry_by_system = {sys: int(sum(np.count_nonzero(s.geometry_valid)
                       for sv, s in geo.satellites.items() if sv[0] == sys))
                       for sys in 'GREC' if sys in counts}
    missed = [sys for sys, n in geometry_by_system.items() if n == 0]
    if missed:
        warning = ('No SP3 geometry for RNX_QC constellations: ' + ', '.join(missed)
                   + '. Preserving LGF and recording missing geometry.')
        coverage_warnings.append(warning)
        _progress('[WARNING] ' + warning, enabled=True)
    report['products'] = {'LGF': summary['leveled_samples']}
    roti_rows=None
    if 'ROTI' not in args.products:
        _progress('[STEP 5/8] ROTI not requested', enabled=verbose)
    if 'ROTI' in args.products:
        from pyOASIS.gnss.medusa_roti_adapter import compute_medusa_roti
        _progress('[STEP 5/8] Computing ROTI (150 s) from leveled phase LGF', enabled=verbose)
        t_roti=time.perf_counter()
        roti_rows,roti_stats=compute_medusa_roti(geo)
        report['products']['ROTI'] = len(roti_rows)
        _progress(f'[STEP 5/8] ROTI records={len(roti_rows)} | duration={time.perf_counter()-t_roti:.1f}s | window=150 s (UTC fixed slots)',enabled=verbose)

    from pyOASIS.gnss.ionosphere_products import compute_indices, compute_calibrated_tec, PRODUCTS
    requested=set(args.products.split(','))
    _progress('[STEP 6/8] Computing requested OASIS indices and model calibration', enabled=verbose)
    extra_products=compute_indices(geo,requested)
    if requested & {'STEC','VTEC','DTEC'}:
        from pyOASIS.gnss.medusa_geometry import ecef2pos
        xyz=np.asarray(observations.receiver_xyz_m,dtype=float)
        pos=ecef2pos(xyz)
        tec,diag=compute_calibrated_tec(geo,float(np.degrees(pos[0])),float(np.degrees(pos[1])))
        report['tec_diagnostics']=diag
        if diag.get('status')!='EXPERIMENTAL_MODEL_CALIBRATION':
            coverage_warnings.append('TEC calibration unavailable: '+str(diag.get('status')))
        elif diag.get('negative_stec_count',0):
            coverage_warnings.append(f"TEC model produced {diag['negative_stec_count']} negative STEC estimates. Preserved and marked nonphysical; absolute TEC validation pending.")
        extra_products.update({p:v for p,v in tec.items() if p in requested})
        if 'DTEC' in requested:
            from pyOASIS.gnss.detrended_tec import compute_detrended_tec
            extra_products['DTEC'], report['dtec_diagnostics'] = compute_detrended_tec(geo, tec['VTEC'])
            if not extra_products['DTEC']:
                coverage_warnings.append('dTEC unavailable: no complete valid VTEC detrending windows.')
    for product, data in extra_products.items():
        report['products'][product]=sum(len(v['value']) for v in data.values())
        _progress(f'[PRODUCTS] { {"DELTATEC":"ΔTEC","DTEC":"dTEC"}.get(product,product)}: {report["products"][product]} records', enabled=verbose)
    scientific_limitations = [
        'dTEC uses experimental model-calibrated VTEC; centered 15-minute linear Savitzky-Golay adaptation, not externally validated',
        'ROTI uses MEDUSA ION V6; DELTATEC/SIDX and model-calibrated STEC/VTEC are experimental OASIS adaptations, not regression validated' ,
        'Geometry uses MEDUSA transmission-time SP3 path and RTKLIB-formula Python port; independent C++ numerical parity pending',
        'IPP uses 350 km RTKLIB ionppp formula; verify against compiled RTKLIB before scientific certification',
    ]
    _progress('[STEP 7/8] Checking essential geometry and calibration warnings', enabled=verbose)
    missing = [sv for sv, sat in geo.satellites.items()
               if not np.any(sat.geometry_valid) and np.isfinite(sat.l4_leveled_m).any()]
    report['geometry_missing_entirely'] = missing
    report['warnings'] = list(coverage_warnings)
    for warning in coverage_warnings:
        if warning.startswith(('TEC','dTEC')): _progress('[WARNING] '+warning,enabled=True)
    if missing:
        message = ('Missing geometry for satellites: ' + ', '.join(sorted(missing,key=satellite_key))
                   + '. Their LGF remains valid for export; ROTI/DELTATEC/SIDX may remain available without IPP; model-calibrated TEC and dTEC require geometry.')
        report['warnings'].append(message)
        _progress('[WARNING] ' + message, enabled=True)
        if args.require_complete_geometry:
            raise RuntimeError(message + ' Strict geometry validation requested; no new HDF5 was written.')
    if args.save_output == 'hdf5':
        _progress('[STEP 8/8] Writing one HDF5 file containing all RNX_QC satellites', enabled=verbose)
        t_save = time.perf_counter()
        from pyOASIS.io.hdf5 import save_lgf_hdf5
        # Reuse the preflight-checked HDF5 destination.
        # Keep only scientific metadata inside HDF5, not a duplicate run report.
        provenance = {
            'tec_diagnostics': report.get('tec_diagnostics', {}),
            'scientific_limitations': scientific_limitations,
            'warnings': report['warnings'],
            'glonass_fdma_external_validation': report['glonass_fdma_external_validation'],
        }
        path = save_lgf_hdf5(dest, geo, observations, report=provenance,
                             overwrite=args.overwrite_output, roti_rows=roti_rows, extra_products=extra_products, station_settings=dict(minimum=args.plot_min_elevation, min_satellites=args.mean_min_satellites, method=args.mean_method, smooth_minutes=0.))
        report['hdf5_file'] = str(path.resolve())
        _progress(f'[STEP 8/8] HDF5 saved: {path.resolve()} | duration={time.perf_counter()-t_save:.1f}s', enabled=verbose)
    else:
        _progress('[STEP 8/8] HDF5 export explicitly disabled (--save-output none)', enabled=verbose)

    report['figure_files'] = []
    if args.figures:
        from pyOASIS.visualization.oasis_figures import render_oasis_figures
        directory = args.figure_dir or output_dir
        _progress(f'[FIGURES] Generating original OASIS-style figures in {directory}', enabled=verbose)
        report['figure_files'] = render_oasis_figures(
            geo, roti_rows, directory, station=station, doy=doy, year=year,
            max_qc_satellites=args.max_qc_figures if args.qc_figures else 0, extra_products=extra_products, overview_figures=args.overview_figures,
            qc_elevation=args.qc_elevation, qc_zoom=args.qc_zoom, figure_products=args.figure_products, qc_directory=directory / "QC", dpi=args.figure_dpi, min_elevation=args.plot_min_elevation, mean_line=args.mean_line, mean_min_satellites=args.mean_min_satellites, mean_method=args.mean_method, mean_smooth_minutes=0., color_mode=args.product_figure_color_mode, notify=lambda line: _progress(line, enabled=verbose))
        _progress(f'[FIGURES] Saved {len(report["figure_files"])} PNG files', enabled=verbose)

    report['stage'] = 'RNX_QC_LGF_IONOSPHERE_PRODUCTS'
    report['run_status']='COMPLETE_WITH_WARNINGS' if report['warnings'] else 'COMPLETE'
    report['requested_products']=args.products.split(',')
    report['elapsed_seconds'] = round(time.perf_counter() - start, 3)
    # No auxiliary CSV/JSON files. Legacy path keys remain null for callers.
    report['run_report_file'] = None
    report['geometry_audit_csv'] = None
    report.pop('tec_diagnostics', None)
    report.pop('dtec_diagnostics', None)
    _progress(f'[COMPLETE] Total processing time: {report["elapsed_seconds"]} s', enabled=verbose)
    return report



def _plot_hdf5_main(argv):
    parser = argparse.ArgumentParser(prog='pyOASIS plot-hdf5',
                                     description='Create OASIS-style figures from an existing pyOASIS HDF5 file')
    parser.add_argument('--input', type=Path, required=True, help='Input pyOASIS HDF5 file')
    parser.add_argument('--out', type=Path, default=None, help='PNG output folder (default: HDF5 parent)')
    parser.add_argument('--plot-min-elevation', type=float, default=20.0)
    parser.add_argument('--mean-line', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mean-min-satellites', type=int, default=4)
    parser.add_argument('--mean-method', choices=['mean','median'], default='mean')
    parser.add_argument('--mean-smooth-minutes', type=float, default=0., help=argparse.SUPPRESS)
    parser.add_argument('--product-figure-color-mode', '--figure-color-mode', choices=['constellation','blue_red'], default='constellation')
    parser.add_argument('--figure-products', type=parse_figure_products, default=None)
    parser.add_argument('--max-qc-figures', type=int, default=None,
                        help='Maximum number of RNX_QC panels (default: all)')
    parser.add_argument('--figure-dpi', type=int, default=300)
    parser.add_argument('--qc-figures',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--overview-figures',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--qc-elevation',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--qc-zoom',action=argparse.BooleanOptionalAction,default=False)
    args = parser.parse_args(argv)
    if args.mean_min_satellites < 1 or not np.isfinite(args.mean_smooth_minutes) or args.mean_smooth_minutes < 0:
        parser.error('Invalid station mean settings')
    from pyOASIS.visualization.oasis_figures import render_hdf5_figures
    directory = args.out or args.input.parent
    paths = render_hdf5_figures(args.input, directory,
                                max_qc_satellites=args.max_qc_figures if args.qc_figures else 0, overview_figures=args.overview_figures,
                                qc_elevation=args.qc_elevation, qc_zoom=args.qc_zoom, figure_products=args.figure_products, qc_directory=directory / "QC", dpi=args.figure_dpi, min_elevation=args.plot_min_elevation, mean_line=args.mean_line, mean_min_satellites=args.mean_min_satellites, mean_method=args.mean_method, mean_smooth_minutes=0., color_mode=args.product_figure_color_mode,
                                notify=lambda message: _progress(message))
    print(json.dumps({'source_hdf5': str(args.input.resolve()),
                      'figures': len(paths), 'files': paths}, indent=2))


def _inspect_hdf5_main(argv):
    parser = argparse.ArgumentParser(prog='pyOASIS inspect-hdf5',
                                     description='Inspect a pyOASIS LGF/ROTI HDF5 without loading arrays')
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(argv)
    from pyOASIS.io.hdf5 import load_lgf_hdf5_summary
    print(json.dumps(load_lgf_hdf5_summary(args.input), indent=2, default=str))


def main(argv=None):
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] == 'plot-hdf5':
        return _plot_hdf5_main(argv[1:])
    if argv and argv[0] == 'inspect-hdf5':
        return _inspect_hdf5_main(argv[1:])
    if argv and argv[0] == 'process':
        argv = argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_inputs(args, parser)
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == '__main__':
    main()
