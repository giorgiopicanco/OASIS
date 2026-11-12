"""Single-file, aligned RNX_QC/geometry/ionospheric product HDF5 export.

Scientific values are provided by the processing pipeline; only unit conversions and station SIDX aggregation are performed here.
Every sample is aligned to a common GNSS-time epoch grid.  IPP is the
MEDUSA-method 350 km ionospheric shell output of the SP3 Python adapter.
Compiled MEDUSA/RTKLIB numerical parity is still pending.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import defaultdict
import json
import os
import uuid

import h5py
import numpy as np

FORMAT_NAME = 'pyOASIS-LGF'
FORMAT_VERSION = '2.1'
IPP_HEIGHT_M = 350_000.0


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _string_dataset(parent, name, data):
    typ = h5py.string_dtype(encoding='utf-8')
    if isinstance(data, str):
        return parent.create_dataset(name, data=data, dtype=typ)
    return parent.create_dataset(name, data=np.asarray(list(data), dtype=object), dtype=typ)


def _numeric(parent, name, values, n=None, *, unit=None, description=None):
    a = np.asarray(values)
    if a.dtype.kind not in 'fiub':
        raise TypeError(f'{name}: numeric or boolean dtype required, got {a.dtype}')
    if n is not None and a.shape != (n,):
        raise ValueError(f'{name}: expected {n} epochs; got {a.shape}')
    ds = parent.create_dataset(name, data=a, compression='gzip' if a.ndim else None,
                               compression_opts=4 if a.ndim else None, shuffle=bool(a.ndim))
    if unit is not None:
        ds.attrs['units'] = unit
    if description is not None:
        ds.attrs['description'] = description
    return ds


def save_lgf_hdf5(path, geo, observations, *, report=None, overwrite=False, roti_rows=None, extra_products=None, station_settings=None):
    """Write one HDF5 for ALL B111-processed satellites, with no lossy filtering.

    `geo` is B111GeoDataset from attach_sp3_geometry, `observations` is the
    RinexObservations object read from the SAME input RINEX.  Missing data
    stay NaN, rejected arcs stay -1.  Writes atomically to avoid partial files.
    Returns Path; raises rather than silently truncating or replacing files.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'HDF5 exists: {path}. Use --overwrite-output to replace it.')
    if not geo.satellites:
        raise ValueError('No B111 satellites to save')
    epochs = np.asarray(observations.times, dtype='datetime64[ns]')
    if epochs.ndim != 1 or len(epochs) == 0 or np.isnat(epochs).any():
        raise ValueError('Invalid RINEX epoch grid')
    if getattr(observations, 'time_system', None) not in ('GPS','GPST'):
        raise ValueError('This HDF5 interface requires explicit GPS/GPST epochs')
    if np.any(np.diff(epochs.astype('int64')) <= 0):
        raise ValueError('Epoch grid must be strictly increasing')
    n = len(epochs)
    for sv, sat in geo.satellites.items():
        if not np.array_equal(np.asarray(sat.times, dtype='datetime64[ns]'), epochs):
            raise ValueError(f'{sv}: epoch grid mismatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with h5py.File(tmp, 'w') as f:
            f.attrs['format'] = FORMAT_NAME
            f.attrs['format_version'] = FORMAT_VERSION
            products = ['LGF'] + (['ROTI'] if roti_rows is not None else []) + list(extra_products or {})
            f.attrs['product'] = '+'.join(products)
            f.attrs['requested_products_json'] = _json(products)
            available=['LGF'] if any(np.isfinite(s.l4_leveled_m).any() for s in geo.satellites.values()) else []
            if roti_rows: available.append('ROTI')
            available.extend(p for p,data in (extra_products or {}).items() if any(len(d['value']) for d in data.values()))
            f.attrs['available_products_json'] = _json(available)
            f.attrs['qc_algorithm'] = 'RNX_QC (original MEDUSA B111)'
            f.attrs['time_system'] = 'GPST'
            f.attrs['epoch_encoding'] = 'int64 nanoseconds since 1970-01-01 00:00:00 on nominal GPST calendar; NOT POSIX UTC'
            f.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
            f.attrs['station'] = str(observations.marker or '')
            f.attrs['rinex_source'] = str(observations.source)
            f.attrs['rinex_version'] = float(observations.version)
            f.attrs['sp3_source'] = str(geo.sp3_path)
            f.attrs['rnx_qc_source_status'] = str(geo.b111.status)
            f.attrs['geometry_status'] = str(geo.status)
            f.attrs['ipp_shell_height_m'] = IPP_HEIGHT_M
            f.attrs['ipp_shell_model'] = 'MEDUSA_RTKLIB_IONPPP_FORMULA_350KM_PYTHON_TRANSLATION'
            f.attrs['geometry_qualification'] = 'MEDUSA_METHOD_PYTHON_PORT_NUMERICAL_PARITY_PENDING'
            f.attrs['tec_calibration'] = str((report or {}).get('tec_diagnostics', {}).get('status', 'NOT_COMPUTED'))
            meta = f.create_group('metadata')
            _string_dataset(meta, 'declared_signals_json', _json(observations.declared_signals))
            _string_dataset(meta, 'glonass_channels_json', _json(observations.glonass_channels))
            _string_dataset(meta, 'glonass_channel_sources_json', _json(geo.b111.glonass_channel_sources or {}))
            _string_dataset(meta, 'glonass_channel_conflicts_json', _json(geo.b111.glonass_channel_conflicts or {}))
            _string_dataset(meta, 'rnx_qc_skipped_json', _json(geo.b111.skipped or {}))
            # Preserve compact scientific provenance, without a duplicate run report.
            for key in ('tec_diagnostics', 'scientific_limitations', 'warnings', 'glonass_fdma_external_validation'):
                if key in (report or {}):
                    _string_dataset(meta, key + '_json', _json(report[key]))
            _string_dataset(meta, 'orbit_sources_json', _json(list(getattr(geo, 'sp3_sources', ()) or [geo.sp3_path])))
            _string_dataset(meta, 'all_rinex_satellites', sorted(observations.satellites))
            xyz = getattr(observations, 'receiver_xyz_m', None)
            _numeric(meta, 'receiver_ecef_xyz_m', np.asarray(xyz if xyz is not None else [np.nan]*3,dtype=float),unit='m')
            egroup = f.create_group('epochs')
            _numeric(egroup, 'time_gpst_ns', epochs.astype('int64'), n, unit='ns',
                     description='Nominal GPS time calendar, not Unix UTC timestamps')
            _string_dataset(egroup, 'datetime_gpst_iso', np.datetime_as_string(epochs, unit='ns'))
            from pyOASIS.gnss.ionosphere_products import gpst_array_to_utc_ms
            utc_ms = gpst_array_to_utc_ms(epochs)
            _numeric(egroup, 'time_utc_ms', utc_ms, n, unit='ms', description='Unix UTC milliseconds')
            _string_dataset(egroup, 'datetime_utc_iso', np.datetime_as_string(utc_ms.astype('datetime64[ms]'), unit='ms'))
            _numeric(egroup, 'mjd_utc', 40587.+utc_ms/86400000., n, unit='day')
            _numeric(egroup, 'gpst_minus_utc_s', (epochs.astype('datetime64[ms]').astype('int64')-utc_ms)//1000,n,unit='s')
            _numeric(egroup, 'rinex_epoch_flag', observations.epoch_flags, n)
            satellites = f.create_group('satellites', track_order=True)
            for sv, sat in sorted(geo.satellites.items(), key=lambda item: ('GREC'.index(item[0][0]), item[0])):
                b = sat.b111
                g = satellites.create_group(sv)
                g.attrs['satellite'] = sv
                g.attrs['constellation'] = sv[0]
                g.attrs['f1_hz'] = np.nan if b.f1_hz is None else float(b.f1_hz)
                g.attrs['f2_hz'] = np.nan if b.f2_hz is None else float(b.f2_hz)
                g.attrs['phase1_code'] = str(b.signals[0]); g.attrs['phase2_code'] = str(b.signals[1])
                g.attrs['code1_code'] = str(b.signals[2]); g.attrs['code2_code'] = str(b.signals[3])
                g.attrs['source_observables_json'] = _json(list(b.source_observables or b.signals))
                g.attrs['glonass_channel_source'] = (geo.b111.glonass_channel_sources or {}).get(sv,'NOT_APPLICABLE')
                if sv in observations.glonass_channels:
                    g.attrs['glonass_fdma_channel'] = int(observations.glonass_channels[sv])
                g.attrs['epoch_axis'] = '/epochs/time_gpst_ns'
                g.attrs['ipp_shell_height_m'] = IPP_HEIGHT_M
                q = g.create_group('rnx_qc')
                field_map = (
                    ('lgf_raw_m','l4_raw_m','m'), ('lgf_corrected_m','l4_corrected_m','m'),
                    ('lgf_leveled_m','l4_leveled_m','m'), ('pgf_raw_m','code_gf_m','m'),
                    ('mw_raw_cycles','mw_raw_cycles','cycle'), ('mw_corrected_cycles','mw_corrected_cycles','cycle'),
                    ('flags_final','flags_final',None), ('arc_ids','arc_ids',None),
                    ('arc_bias_m','arc_bias_m','m'), ('residual_m','residual_m','m'),
                    ('phase_slip_candidate','slip',None))
                for public, original, unit in field_map:
                    _numeric(q, public, getattr(b,original), n, unit=unit)
                q.attrs['phase_combination'] = 'lambda1*phase1_cycles - lambda2*phase2_cycles'
                q.attrs['code_combination'] = 'pseudorange2_m - pseudorange1_m'
                q.attrs['levelled_observable'] = 'carrier phase LGF levelled to code PGF; PGF is not levelled'
                q.attrs['flag_bits'] = 'bit 0 bad/outlier; bit 1 new ambiguity; bit 2 corrected cycle slip; -1 not processed'
                from pyOASIS.gnss.rnx_qc import diagnostic_masks, diagnostic_arrays
                details = diagnostic_arrays(b)
                if details:
                    stages=q.create_group('stage_diagnostics')
                    for key, values in details.items():
                        _numeric(stages,key,values,n,unit='cycle' if key.endswith('_cycles') else None,description='Original stage output; -1 flags or NaN slip size indicate unavailable')
                masks = q.create_group('masks',track_order=True)
                for name, values in diagnostic_masks(b,details).items():
                    _numeric(masks,name,values,n,description='True where this original QC event or eligibility condition applies')
                _numeric(q, 'source_indices', b.source_indices, unit='epoch index (zero-based)')
                _numeric(q, 'leveled_valid', np.isfinite(b.l4_leveled_m)&(np.asarray(b.arc_ids)>=0), n)
                if set(extra_products or {}) & {'STEC','VTEC','DTEC'}:
                    factor=40.3e16*(1/float(b.f2_hz)**2-1/float(b.f1_hz)**2)
                    _numeric(q,'lgf_equivalent_stec_tecu',np.asarray(b.l4_leveled_m)/factor,n,unit='TECU',description='Direct LGF conversion; arc and instrumental offsets are not calibrated')
                gg = g.create_group('geometry')
                gg.attrs['orbit_method'] = 'MEDUSA_SP3_TXTIME_EARTH_ROTATION_RTKLIB_FORMULAS_PYTHON_PORT'
                gg.attrs['broadcast_fallback'] = 'NOT_AVAILABLE_WITH_OBSERVATION_ONLY_RINEX'
                _string_dataset(gg, 'orbit_source', (sat.orbit_source if getattr(sat, 'orbit_source', None) is not None else
                                                    ['SP3' if bool(v) else 'UNAVAILABLE' for v in sat.orbit_available]))
                for name, values in sat.geometry.items():
                    unit = ('m' if name.endswith('_m') else 's' if name.endswith('_s') else 'degree' if name.endswith('_deg') else 'dimensionless')
                    _numeric(gg,name,values,n,unit=unit)
                for name in ('orbit_available','above_horizon','geometry_valid'):
                    _numeric(gg,name,getattr(sat,name),n)
                ipph = np.where(sat.geometry_valid,IPP_HEIGHT_M,np.nan)
                _numeric(gg,'ipp_shell_height_m',ipph,n,unit='m',
                         description='Fixed 350 km shell assumed for calculated valid IPP')
                raw = g.create_group('selected_rinex_observables')
                # Actual source codes, not aliases used for RINEX2 MEDUSA.
                selected = list(dict.fromkeys(b.source_observables or b.signals))
                for code in selected:
                    series = observations.get(sv,code)
                    if series is None:
                        # Declared but missing: preserve distinction from zero measurements.
                        continue
                    cg = raw.create_group(code)
                    _numeric(cg,'values',series.values,n,unit='cycle' if code.startswith('L') else 'm' if code.startswith(('C','P')) else None)
                    _numeric(cg,'lli',series.lli,n)
                    _numeric(cg,'ssi',series.ssi,n)
                    cg.attrs['frequency_hz'] = float(series.frequency_hz) if series.frequency_hz is not None else np.nan
                    cg.attrs['wavelength_m'] = float(series.wavelength_m) if series.wavelength_m is not None else np.nan
                    cg.attrs['source_code'] = code
            if roti_rows is not None:
                rgroup=f.require_group('products').create_group('ROTI')
                rgroup.attrs['algorithm'] = 'MEDUSA_ION_V6_FIXED_150S'
                rgroup.attrs['source'] = 'MEDUSA/tools/realtime/medusa_ion_v6/b111_adapter_v6.py'
                rgroup.attrs['time_scale'] = 'UTC'
                rgroup.attrs['units'] = 'TECU/min'
                rgroup.attrs['window_seconds'] = 150
                rgroup.attrs['record_count'] = len(roti_rows)
                rgroup.attrs['timestamp_semantics'] = 'UTC window midpoint'
                rgroup.attrs['ion_flags'] = '1=no IPP; 2=partial live block; 4=30 s cadence; 8=edge context'
                if any(int(r['window_seconds']) != 150 for r in roti_rows):
                    raise ValueError('Only V6 fixed 150-second ROTI is supported')
                grouped = defaultdict(list)
                for row in roti_rows:
                    grouped[str(row['satellite'])].append(row)
                for window in (150,):
                    wg=rgroup.create_group(f'{window}s',track_order=True)
                    for sv in sorted(grouped,key=lambda sv: ('GREC'.index(sv[0]),sv)):
                        rows=grouped[sv]
                        g=wg.create_group(sv)
                        for key in ('utc_ms','roti_tecu_min','rot_count','latitude_ipp_deg','longitude_ipp_deg',
                                    'elevation_deg','cadence_seconds','window_start_utc_ms','window_end_utc_ms','mjd_utc','ipp_height_km','ion_flags'):
                            vals=[r.get(key) for r in rows]
                            dtype=np.int64 if key in ('utc_ms','rot_count','window_start_utc_ms','window_end_utc_ms','ion_flags') else np.float64
                            _numeric(g,key,np.asarray([np.nan if x is None else x for x in vals],dtype=dtype))
                        _string_dataset(g,'arc',[str(r['arc']) for r in rows])
                        _string_dataset(g,'signal_pair',[str(r['signal_pair']) for r in rows])
            for product, satellites_data in (extra_products or {}).items():
                pg = f.require_group('products').create_group(product, track_order=True)
                pg.attrs['units'] = {'DELTATEC':'TECU (legacy scaled amplitude)','SIDX':'TECU/min','STEC':'TECU','VTEC':'TECU','DTEC':'TECU'}[product]
                pg.attrs['status'] = 'EXPERIMENTAL_OASIS_ADAPTATION'
                pg.attrs['time_scale'] = 'UTC'
                pg.attrs['timestamp_semantics'] = 'Mean observation UTC epoch within window' if product in ('DELTATEC','SIDX') else 'Original observation UTC epoch'
                pg.attrs['record_count']=sum(len(d['value']) for d in satellites_data.values())
                if product in ('STEC','VTEC','DTEC'): pg.attrs['calibration_status']=f.attrs['tec_calibration']
                pg.attrs['algorithm'] = {'DELTATEC':'OASIS_DTEC_ROLLING_MEAN_DIFFERENCE_150S','SIDX':'OASIS_SIDX_MEAN_ABS_ROT_60S','STEC':'OASIS_TEC_HOURLY_SIX_COEFFICIENT_ARC_OFFSETS','VTEC':'OASIS_TEC_HOURLY_SIX_COEFFICIENT_ARC_OFFSETS','DTEC':'VTEC_MINUS_SAVITZKY_GOLAY_900S_LINEAR'}[product]
                pg.attrs['display_name'] = {'DELTATEC':'ΔTEC','DTEC':'dTEC'}.get(product,product)
                if product == 'DTEC':
                    pg.attrs['window_seconds'] = 900
                    pg.attrs['polynomial_order'] = 1
                    pg.attrs['elevation_cutoff_deg'] = 30.0
                    pg.attrs['edge_policy'] = 'FULL_CENTERED_WINDOW_ONLY'
                    pg.attrs['source_product'] = 'VTEC (experimental model calibration)'
                if product in ('DELTATEC','SIDX'):
                    pg.attrs['window_seconds'] = 150 if product=='DELTATEC' else 60
                    pg.attrs['original_elevation_cutoff_deg']=30 if product=='DELTATEC' else 20
                    pg.attrs['elevation_policy']='All valid numerical windows preserved; original cutoff stored as original_elevation_ok'
                for sv, data in satellites_data.items():
                    sg=pg.create_group(sv)
                    b=geo.satellites[sv].b111
                    sg.attrs['phase_signal_pair']=':'.join(b.signals[:2])
                    sg.attrs['f1_hz']=float(b.f1_hz); sg.attrs['f2_hz']=float(b.f2_hz)
                    for key, values in data.items():
                        _numeric(sg,key,np.asarray(values,dtype=np.int64 if key in ('utc_ms','arc_id','window_start_utc_ms','window_end_utc_ms','sample_count','segment_id','source_epoch_index','window_samples','cadence_ms','actual_window_span_ms') else bool if key in ('geometry_available','original_elevation_ok','physical_nonnegative') else float))
                    sg[product.lower()+'_value']=sg['value']
                    sg['value'].attrs['units']=pg.attrs['units']
                    if product == 'DTEC':
                        sg['vtec_tecu'].attrs['units'] = 'TECU'
                        sg['background_vtec_tecu'].attrs['units'] = 'TECU'
                    _numeric(sg,'mjd_utc',40587.+np.asarray(data['utc_ms'],dtype=float)/86400000.,unit='day')
                    _string_dataset(sg,'datetime_utc_iso',np.datetime_as_string(np.asarray(data['utc_ms'],dtype='int64').astype('datetime64[ms]'),unit='ms'))
                    if product=='SIDX': _numeric(sg,'sidx_mtecu_s',np.asarray(data['value'])*1000/60,unit='mTECU/s')
            if 'SIDX' in (extra_products or {}):
                from pyOASIS.gnss.ionosphere_products import aggregate_sidx
                station_group=f.create_group('station_products').create_group('SIDX',track_order=True)
                station_group.attrs['elevation_cutoff_deg']=20
                station_group.attrs['aggregation']='Equal-weight mean of one value per satellite in each fixed UTC minute'
                for system,data in aggregate_sidx(extra_products['SIDX']).items():
                    sg=station_group.create_group(system)
                    for key,values in data.items():
                        _numeric(sg,key,values,unit='mTECU/s' if key=='value_mtecu_s' else 'TECU/min' if key=='value_tecu_min' else None)
            from pyOASIS.gnss.station_statistics import PRODUCTS, station_statistics, epoch_grid
            settings = station_settings or dict(minimum=20., min_satellites=4, method='mean', smooth_minutes=0.)
            grid, cadence = epoch_grid(geo)
            station_inputs = dict(extra_products or {})
            for product in PRODUCTS:
                if product not in station_inputs or cadence <= 0: continue
                stats = station_statistics(station_inputs[product], product, grid_ms=grid, cadence_ms=cadence, **settings)
                sg = f.require_group('station_products').require_group(product)
                for key, value in settings.items(): sg.attrs[key] = value
                sg.attrs['aggregation'] = 'Equal weight per satellite per UTC epoch or native fixed window; no interpolation'
                sg.attrs['negative_vtec_policy'] = 'Excluded from station statistics only'
                sg.attrs['smoothing'] = 'Optional linear centered Savitzky-Golay; full windows only; no gap bridging'
                for key, values in stats.items():
                    _numeric(sg, key, values, unit='ms' if key=='utc_ms' else 'satellites' if key=='count' else 'm' if product=='LGF' else f['products/'+product].attrs['units'])
            _string_dataset(meta,'rnx_qc_satellites',sorted(geo.satellites,key=lambda sv: ('GREC'.index(sv[0]),sv)))
            f.flush()
        if path.exists() and not overwrite:
            raise FileExistsError(f'HDF5 exists: {path}')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_lgf_hdf5_summary(path):
    """Lightweight public reader; does not load large observation arrays."""
    with h5py.File(path,'r') as f:
        if f.attrs.get('format') != FORMAT_NAME:
            raise ValueError('Not a pyOASIS LGF HDF5')
        return {'station': str(f.attrs['station']), 'product': str(f.attrs['product']),
                'format_version':str(f.attrs['format_version']),
                'available_products':json.loads(f.attrs.get('available_products_json','[]')),
                'epoch_count': len(f['epochs/time_gpst_ns']),
                'satellites': sorted(f['satellites'].keys(),key=lambda sv: ('GREC'.index(sv[0]),sv)),
                'ipp_shell_height_m': float(f.attrs['ipp_shell_height_m']),
                'geometry_status': str(f.attrs['geometry_status'])}
