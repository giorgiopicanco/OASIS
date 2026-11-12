"""Original OASIS multi-GNSS figure style, using validated output arrays only.

The constellation palette, markers, axes, titles and QC panel layout follow
previous OASIS figures, with the requested GREC palette. Scientific products are supplied by the processing pipeline.
ROTI values are taken unchanged from MEDUSA ION V6 output.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

SYSTEMS = {
    'G': ('GPS', 'blue', 'o'),
    'R': ('GLONASS', 'red', 'o'),
    'E': ('Galileo', 'lime', 'o'),
    'C': ('BeiDou', 'violet', 'o'),
}


def _plot_imports():
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as exc:
        raise RuntimeError('Figure generation requires matplotlib. Install with: python -m pip install matplotlib') from exc
    return plt, mdates


def _date_array(times):
    from pyOASIS.gnss.ionosphere_products import gpst_array_to_utc_ms
    return gpst_array_to_utc_ms(times).astype('datetime64[ms]').astype(object)


def _finish(fig, file_path, dpi):
    plt, _ = _plot_imports()
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=dpi)
    plt.close(fig)
    if not file_path.is_file() or file_path.stat().st_size < 100:
        raise RuntimeError(f'Figure file is missing or empty: {file_path}')
    return file_path


def _time_axis(ax, ylabel, mdates, *, hours=2, scale='UTC'):
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel('Time (UTC)' if scale == 'UTC' else 'Time (GPST)', fontsize=16)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,hours)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.grid(True, linestyle='--', linewidth=1, color='gray')
    ax.tick_params(axis='both', which='major', labelsize=14)


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')


def elevation_mask(elevation, minimum=20.0):
    """Display-only finite elevation threshold, inclusive."""
    minimum = float(minimum)
    if not np.isfinite(minimum) or not 0 <= minimum <= 90:
        raise ValueError('Plot minimum elevation must be between 0 and 90 degrees')
    e = np.asarray(elevation, dtype=float)
    return np.isfinite(e) & (e >= minimum)


def station_mean(data, product, minimum=20.0):
    """Equal weight per satellite per epoch (or fixed 150 s ΔTEC window).

    No interpolation, smoothing or mixing different time bins. Duplicate
    contributions from a satellite in a bin are averaged before pooling.
    """
    from collections import defaultdict
    bins = defaultdict(dict)
    for sv, d in data.items():
        v = np.asarray(d['value'], dtype=float)
        good = np.isfinite(v) & elevation_mask(d['elevation_deg'], minimum)
        key = 'window_start_utc_ms' if product == 'DELTATEC' else 'utc_ms'
        for t, value in zip(np.asarray(d[key], dtype=np.int64)[good], v[good]):
            bins[int(t)].setdefault(sv, []).append(float(value))
    t = np.array(sorted(bins), dtype=np.int64)
    y = np.array([np.mean([np.mean(v) for v in bins[k].values()]) for k in t])
    n = np.array([len(bins[k]) for k in t], dtype=int)
    if product == 'DELTATEC': t += 75000
    return t, y, n


def product_style(system, color_mode):
    if color_mode == 'blue_red':
        return 'Satellite values', 'blue', 'o'
    if color_mode != 'constellation':
        raise ValueError('Unknown product figure color mode')
    return SYSTEMS[system]


def _draw_station_mean(ax, data, product, minimum, grid_ms, cadence_ms,
                       min_satellites=4, method='mean', smooth_minutes=0., color_mode='constellation'):
    from pyOASIS.gnss.station_statistics import PRODUCTS, station_statistics
    if product not in PRODUCTS:
        return
    stats = station_statistics(data, product, grid_ms=grid_ms, cadence_ms=cadence_ms,
                               minimum=minimum, min_satellites=min_satellites,
                               method=method, smooth_minutes=0.)
    dates = stats['utc_ms'].astype('datetime64[ms]').astype(object)
    values = stats['selected'] * (1000/60 if product == 'SIDX' else 1)
    if np.isfinite(values).any():
        ax.plot(dates, values, color='red' if color_mode == 'blue_red' else 'black',
                lw=1.2, zorder=10, label='Station mean' if method=='mean' else 'Station median')


def plot_lgf_overview(geo, folder, *, station, doy, year, dpi=300, min_elevation=20.0, color_mode="constellation", mean_line=True, mean_min_satellites=4, mean_method="mean"):
    plt, mdates = _plot_imports()
    fig, ax = plt.subplots(figsize=(12, 6))
    seen = set()
    sample_count = 0
    for sv, sat in sorted(geo.satellites.items(),key=lambda item: ('GREC'.index(item[0][0]),item[0])):
        name, color, marker = product_style(sv[0], color_mode)
        lev = np.asarray(sat.l4_leveled_m, dtype=float)
        mask = np.isfinite(lev) & (np.asarray(sat.arc_ids) >= 0) & elevation_mask(sat.geometry['elevation_deg'], min_elevation)
        if not np.any(mask):
            continue
        sample_count += int(mask.sum())
        signals = getattr(sat.b111, 'signals', ())
        pair = '-'.join(str(x) for x in signals[:2]) if signals else name
        key = 'all' if color_mode == 'blue_red' else (sv[0], pair)
        label = (name if color_mode == 'blue_red' else f'LGF: {pair} ({name})') if key not in seen else '_nolegend_'
        seen.add(key)
        ax.scatter(_date_array(sat.times)[mask], lev[mask], s=3,
                   color=color, marker=marker, label=label, rasterized=True, linewidths=0)
    if sample_count == 0:
        plt.close(fig)
        return None
    _time_axis(ax, 'Levelled phase geometry-free (LGF, m)', mdates)
    _legend(ax)
    ax.set_title(f'Station: {station.upper()}  |  Year: {year}  |  DOY: {doy}', fontsize=16)
    fig.tight_layout()
    return _finish(fig, Path(folder) / f'{station}_{doy}_{year}_LGF.png', dpi)


def plot_rnx_qc(geo, folder, *, station, doy, year, dpi=300,
                 max_qc_satellites=None, notify=None, qc_elevation=False, qc_zoom=False, min_elevation=20.0):
    """Five compact panels; retain every original QC field in HDF5."""
    plt, mdates = _plot_imports()
    from pyOASIS.gnss.rnx_qc import diagnostic_masks, diagnostic_arrays
    files = []
    items = sorted(geo.satellites.items(), key=lambda item: ('GREC'.index(item[0][0]), item[0]))
    if max_qc_satellites is not None:
        items = items[:max_qc_satellites]
    for index, (sv, sat) in enumerate(items, 1):
        b = sat.b111; t = _date_array(sat.times)
        name, color, _ = SYSTEMS[sv[0]]
        details = getattr(b, 'diagnostic_arrays', None)
        if details is None: details = diagnostic_arrays(b)
        masks = getattr(b, 'diagnostic_masks', None) or diagnostic_masks(b, details)
        empty = np.zeros(len(t), dtype=bool)
        def mask(k): return np.asarray(masks.get(k, empty), dtype=bool)
        accepted = mask('accepted_level'); arcs = np.asarray(b.arc_ids)
        raw = np.asarray(b.l4_raw_m); pgf = np.asarray(b.code_gf_m)
        observed = np.isfinite(raw) & np.isfinite(pgf)
        events = [('Outliers', mask('outlier_or_bad'), 'black', 'x'),
                  ('MW candidates', mask('mw_slip_candidate'), 'darkorange', 'v'),
                  ('LGF candidates', mask('phase_slip_candidate') | mask('phase_slip_reported'), 'purple', '^'),
                  ('Corrected slips', mask('corrected_cycle_slip'), 'red', 'o'),
                  ('New ambiguity', mask('new_ambiguity'), 'teal', 's')]
        fig, axes = plt.subplots(6 if qc_elevation else 5, 1, figsize=(13, 12 if qc_elevation else 10), sharex=True)
        panels = [(0, raw, 'Raw phase LGF', 'gray', 3),
                  (0, b.l4_corrected_m, 'Corrected phase LGF', color, 1.2),
                  (1, b.mw_raw_cycles, 'Raw MW', 'gray', 2),
                  (1, b.mw_corrected_cycles, 'Screened MW', color, 1.2),
                  (2, pgf, 'Raw pseudorange PGF', 'gray', 2),
                  (2, b.l4_leveled_m, 'Levelled carrier-phase LGF', color, 1.2),
                  (3, b.residual_m, 'Levelling residual', color, 1.2)]
        for i, values, label, c, size in panels:
            axes[i].scatter(t, values, s=size, color=c, label=label, linewidths=0, rasterized=True)
        axes[3].axhline(0, color='gray', linewidth=.5, linestyle=':')
        # Plot accepted arc extents as spans, not a fictitious continuous series.
        bias = np.asarray(getattr(b, 'arc_bias_m', np.full(len(t), np.nan)))
        for k, arc in enumerate(np.unique(arcs[accepted])):
            j = np.flatnonzero(accepted & (arcs == arc))
            if not len(j): continue
            for ax in axes:
                ax.axvspan(t[j[0]], t[j[-1]], color=color, alpha=.035 if k%2 == 0 else .075, zorder=0)
                for edge in (j[0],j[-1]): ax.axvline(t[edge], color='gray', lw=.45, ls=':')
            finite = bias[j][np.isfinite(bias[j])]
            label = f'Arc {arc}' + (f' | bias {np.median(finite):+.3f} m' if len(finite) else '')
            axes[2].text(t[j[len(j)//2]], .95-(k%3)*.12, label, transform=axes[2].get_xaxis_transform(), ha='center', va='top', fontsize=7, clip_on=True)
        for row,(label,m,c,marker) in enumerate(events):
            axes[4].scatter(t[m],np.full(m.sum(),row),s=14,marker=marker,color=c,linewidths=.6)
            if row in (0,1,2,3):
                idx = 1 if row == 1 else 0
                v = np.asarray(b.mw_raw_cycles if idx == 1 else (b.l4_corrected_m if row==3 else raw))
                valid=m & np.isfinite(v)
                if valid.any(): axes[idx].scatter(t[valid],v[valid],s=16,marker=marker,color=c,linewidths=.6,label=label)
        # Magnitudes are original stage-reported candidates, not inferred corrections.
        for key,label,panel in [('mw_slip_cycles','MW',1),('phase1_slip_cycles','L1',0),('phase2_slip_cycles','L2',0)]:
            if key not in details: continue
            v=np.asarray(details[key]); rawv=np.asarray(b.mw_raw_cycles if panel==1 else raw)
            for j in np.flatnonzero(np.isfinite(v) & np.isfinite(rawv)):
                axes[panel].annotate(f'{label}: {v[j]:+.2f} cyc', (t[j],rawv[j]), xytext=(3,7 if label!='L2' else -11), textcoords='offset points', fontsize=6, color='darkorange')
        mw_available = 'mw_slip_candidate' in masks
        axes[4].set_yticks(range(len(events)), [f'{label} ({int(m.sum()) if row!=1 or mw_available else "N/A"})' for row,(label,m,_,_) in enumerate(events)], fontsize=8)
        axes[4].set_ylim(-.6,4.6)
        labels=['Phase LGF (m)','MW (cycles)','GF (m)','Residual (m)','QC events']
        titles=['Raw and corrected carrier-phase LGF','Melbourne–Wübbena combination','Phase-to-pseudorange levelling','Levelling residuals','Quality-control events']
        if qc_elevation:
            el=np.asarray(sat.geometry.get('elevation_deg',np.full(len(t),np.nan)))
            axes[5].scatter(t,el,s=1.5,color=color,linewidths=0)
            axes[5].axhline(min_elevation,color='gray',ls=':',lw=.6); axes[5].set_ylim(0,90)
            labels.append('Elevation (deg)'); titles.append('Satellite elevation')
        for i,(ax,yl,title) in enumerate(zip(axes,labels,titles)):
            ax.set_ylabel(yl,fontsize=10); ax.set_title(f'({chr(97+i)}) {title}',loc='left',fontsize=10)
            ax.grid(ls='--',linewidth=.45,color='gray',alpha=.7); ax.tick_params(labelsize=9)
            if i<4: _legend(ax)
        axes[-1].set_xlabel('Time (UTC)',fontsize=12)
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5,maxticks=10) if qc_zoom else mdates.HourLocator(byhour=range(0,24,2)))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M' if qc_zoom else '%H'))
        if qc_zoom and observed.any():
            from datetime import timedelta
            j=np.flatnonzero(observed); pad=timedelta(minutes=5)
            axes[-1].set_xlim(t[j[0]]-pad,t[j[-1]]+pad)
        elif len(t):
            axes[-1].set_xlim(t[0],t[-1])
        signals=' / '.join(str(x) for x in getattr(b,'signals',())[:2])
        fig.suptitle(f'{station.upper()} | {name} {sv} | {year} DOY {doy} | RNX_QC\n'
                     f'{signals} | Accepted: {accepted.sum()}/{observed.sum()} | Outliers: {events[0][1].sum()} | '
                     f'MW/LGF candidates: {events[1][1].sum() if mw_available else "N/A"}/{events[2][1].sum()} | Corrected: {events[3][1].sum()} | Arcs: {len(np.unique(arcs[accepted]))}',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94))
        files.append(_finish(fig,Path(folder)/f'{station}_{sv}_{doy}_{year}_RNX_QC.png',dpi))
        if notify is not None and (index % 10 == 0 or index == len(items)):
            notify(f'[FIGURES] RNX_QC plots: {index}/{len(items)} satellites completed')
    return files


def plot_ipp_overview(geo, folder, *, station, doy, year, dpi=300, min_elevation=20.0):
    """Display available IPPs only; does not compute TEC or fill orbital gaps."""
    plt, _ = _plot_imports()
    fig, ax = plt.subplots(figsize=(12, 6))
    seen = set()
    count = 0
    for sv, sat in sorted(geo.satellites.items(),key=lambda item: ('GREC'.index(item[0][0]),item[0])):
        name, color, marker = SYSTEMS.get(sv[0], ('Unknown', '#737373', 'o'))
        lat = np.asarray(sat.geometry['ipp_lat_deg'], dtype=float)
        lon = np.asarray(sat.geometry['ipp_lon_deg'], dtype=float)
        valid = np.asarray(sat.geometry_valid, dtype=bool) & np.isfinite(lat) & np.isfinite(lon) & elevation_mask(sat.geometry['elevation_deg'], min_elevation)
        if not valid.any():
            continue
        count += int(valid.sum())
        label = f'{name} ({sv[0]})' if sv[0] not in seen else '_nolegend_'
        seen.add(sv[0])
        ax.scatter(lon[valid], lat[valid], s=2, alpha=0.25,
                   color=color, marker=marker, label=label, rasterized=True, linewidths=0)
    if count == 0:
        plt.close(fig)
        return None
    ax.set_xlabel('IPP longitude (degrees)')
    ax.set_ylabel('IPP latitude (degrees)')
    ax.grid(ls='--', alpha=0.3)
    _legend(ax)
    ax.set_title(f'{station.upper()} | {year} DOY {doy} | IPP at 350 km\n'
                 'MEDUSA-method SP3 geometry; numerical C++ parity pending')
    fig.tight_layout()
    return _finish(fig, Path(folder) / f'{station}_{doy}_{year}_IPP_350KM.png', dpi)


def plot_roti_windows(roti_rows, folder, *, station, doy, year, dpi=300, min_elevation=20.0, color_mode="constellation", mean_line=True, mean_min_satellites=4, mean_method="mean", geo=None):
    """OASIS ROTI_CALC.py style; elevation and axis limits are display only."""
    plt, mdates = _plot_imports()
    rows = list(roti_rows or [])
    if any(int(r['window_seconds']) != 150 for r in rows):
        raise ValueError('Legacy ROTI data: recompute with MEDUSA ION V6 before plotting')
    if not rows: return []
    fig, ax = plt.subplots(figsize=(12, 6))
    seen = set()
    for system in SYSTEMS:
        name, color, marker = product_style(system, color_mode)
        selected = [r for r in rows if str(r['satellite']).startswith(system)
                    and float(r['elevation_deg']) >= min_elevation and np.isfinite(r['roti_tecu_min'])]
        if not selected: continue

        pairs = sorted({str(r['signal_pair']) for r in selected})
        for pair in pairs:
            subset = [r for r in selected if str(r['signal_pair']) == pair]
            t = np.asarray([r['utc_ms'] for r in subset], dtype='int64').astype('datetime64[ms]').astype(object)
            ax.scatter(t, [r['roti_tecu_min'] for r in subset], marker='o', s=5, color=color, linewidths=0,
                       label=(name if color_mode == 'blue_red' else f'ROTI: {pair} ({name})') if (color_mode != 'blue_red' or not seen) else '_nolegend_')
            seen.add(name)
    if not ax.has_data():
        plt.close(fig)
        return []
    _time_axis(ax, 'ROTI (TECU/min)', mdates, scale='UTC')
    ax.set_ylim(0, 5)
    ax.set_title(f'Station: {station.upper()}  |  Year: {year}  |  DOY: {doy}', fontsize=16)
    _legend(ax)
    fig.tight_layout()
    return [_finish(fig, Path(folder) / f'{station}_{doy}_{year}_ROTI.png', dpi)]


def render_oasis_figures(geo, roti_rows, directory, *, station, doy, year,
                         max_qc_satellites=None, dpi=300, notify=None, extra_products=None, overview_figures=True, qc_elevation=False, qc_zoom=False, figure_products=None, qc_directory=None, min_elevation=20.0, mean_line=True, mean_min_satellites=4, mean_method="mean", mean_smooth_minutes=0., color_mode="constellation"):
    """Return only verified figure files. Empty optional products are not fabricated."""
    if max_qc_satellites is not None and max_qc_satellites < 0:
        raise ValueError('max_qc_satellites must be >= 0')
    if dpi < 72 or dpi > 600:
        raise ValueError('figure DPI must be between 72 and 600')
    elevation_mask([], min_elevation)
    product_style("G", color_mode)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    def add(p):
        if p is not None:
            paths.append(str(p.resolve()))
            if notify:
                notify(f'[FIGURES] Saved {p}')
    def selected(product):
        return overview_figures and (figure_products is None or product in figure_products)
    if selected('LGF'):
        add(plot_lgf_overview(geo, directory, station=station, doy=doy, year=year, dpi=dpi, min_elevation=min_elevation, color_mode=color_mode, mean_line=mean_line, mean_min_satellites=mean_min_satellites, mean_method=mean_method))
    if selected('IPP'):
        add(plot_ipp_overview(geo, directory, station=station, doy=doy, year=year, dpi=dpi, min_elevation=min_elevation))
    if selected('ROTI') and roti_rows is not None:
        for path in plot_roti_windows(roti_rows, directory, station=station, doy=doy, year=year, dpi=dpi, min_elevation=min_elevation, color_mode=color_mode, mean_line=mean_line, mean_min_satellites=mean_min_satellites, mean_method=mean_method, geo=geo):
            add(path)
    for path in plot_rnx_qc(geo, qc_directory or directory, station=station, doy=doy, year=year, dpi=dpi,
                            max_qc_satellites=max_qc_satellites, notify=notify, qc_elevation=qc_elevation, qc_zoom=qc_zoom, min_elevation=min_elevation):
        paths.append(str(path.resolve()))
    for product, data in (extra_products or {}).items() if overview_figures else ():
        if not selected(product): continue
        plt,mdates=_plot_imports(); fig,ax=plt.subplots(figsize=(12,6)); seen=set()
        visible_max = 120.0
        if product=='SIDX':
            from pyOASIS.gnss.ionosphere_products import aggregate_sidx
            for system,d in aggregate_sidx(data, min_elevation=min_elevation).items():
                name,color,_=SYSTEMS[system]
                times=d['window_start_utc_ms']; y=d['value_mtecu_s']; ts=d['utc_ms'].astype('datetime64[ms]').astype(object)
                for seg in np.split(np.arange(len(times)),np.flatnonzero(np.diff(times)>60000)+1):
                    ax.plot(ts[seg],y[seg],color=color,linewidth=.7,marker='o',markersize=1.5,label=name if name not in seen else '_nolegend_'); seen.add(name)
        else:
            for sv,d in sorted(data.items(),key=lambda item: ('GREC'.index(item[0][0]),item[0])):
                name,color,_=product_style(sv[0], color_mode)
                v=np.asarray(d['value'], float); e=np.asarray(d['elevation_deg'])
                mask=np.isfinite(v)&elevation_mask(e,min_elevation)
                if 'geometry_available' in d: mask &= np.asarray(d['geometry_available'], bool)
                if not mask.any(): continue
                if product in ('STEC', 'VTEC'):
                    visible_max = max(visible_max, float(np.max(v[mask])))
                t=np.asarray(d['utc_ms'],dtype='int64').astype('datetime64[ms]').astype(object)
                ax.scatter(t[mask],v[mask],s=3,color=color,label=name if name not in seen else '_nolegend_',linewidths=0); seen.add(name)
        if not seen: plt.close(fig); continue
        if mean_line and product in ('VTEC', 'DELTATEC', 'DTEC'):
            from pyOASIS.gnss.station_statistics import epoch_grid
            grid_ms, cadence_ms = epoch_grid(geo)
            if cadence_ms > 0:
                _draw_station_mean(ax, data, product, min_elevation, grid_ms, cadence_ms,
                                   mean_min_satellites, mean_method, color_mode=color_mode)
        unit={'DELTATEC':'scaled TECU amplitude','DTEC':'TECU','SIDX':'mTECU/s','STEC':'TECU','VTEC':'TECU'}[product]
        display={'DELTATEC':'ΔTEC','DTEC':'dTEC'}.get(product,product)
        suffix={'DELTATEC':'deltaTEC','DTEC':'dTEC'}.get(product,product)
        _time_axis(ax,f'{display} ({unit})',mdates); _legend(ax)
        method = ' | 15-min Savitzky–Golay' if product=='DTEC' else ''
        ax.set_title(f'Station: {station.upper()} | Year: {year} | DOY: {doy}{method} | Experimental',fontsize=16)
        if product in ('STEC','VTEC','DTEC','DELTATEC'): ax.axhline(0,color='gray',linewidth=.6,linestyle=':')
        if product in ('STEC', 'VTEC'):
            ax.set_ylim(0.0, visible_max)
        fig.tight_layout()
        add(_finish(fig,directory/f'{station}_{doy}_{year}_{suffix}.png',dpi))
    return paths


def _from_h5(path):
    import h5py
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'HDF5 file not found: {path}')
    with h5py.File(path, 'r') as f:
        if str(f.attrs.get('format', '')) != 'pyOASIS-LGF':
            raise ValueError(f'Not a pyOASIS LGF file: {path}')
        epochs = np.asarray(f['epochs/time_gpst_ns'], dtype='int64').astype('datetime64[ns]')
        sats = {}
        for sv, group in f['satellites'].items():
            bgroup=group['rnx_qc'] if 'rnx_qc' in group else group['medusa_b111']
            fields={'l4_raw_m':'lgf_raw_m','l4_corrected_m':'lgf_corrected_m','l4_leveled_m':'lgf_leveled_m','code_gf_m':'pgf_raw_m','slip':'phase_slip_candidate'}
            keys=('l4_raw_m','l4_corrected_m','l4_leveled_m','mw_raw_cycles','mw_corrected_cycles','arc_ids','code_gf_m','residual_m','flags_final','slip','arc_bias_m')
            b=SimpleNamespace(**{k:np.asarray(bgroup[fields.get(k,k) if fields.get(k,k) in bgroup else k]) for k in keys})
            if 'masks' in bgroup: b.diagnostic_masks={k:v[:] for k,v in bgroup['masks'].items()}
            if 'stage_diagnostics' in bgroup: b.diagnostic_arrays={k:v[:] for k,v in bgroup['stage_diagnostics'].items()}
            b.signals = (group.attrs.get('phase1_code', ''), group.attrs.get('phase2_code', ''))
            gg = group['geometry']
            geo = {key: np.asarray(gg[key]) for key in ('ipp_lat_deg', 'ipp_lon_deg','elevation_deg')}
            sats[sv] = SimpleNamespace(times=epochs, l4_leveled_m=b.l4_leveled_m,
                                       arc_ids=b.arc_ids, b111=b, geometry=geo,
                                       geometry_valid=np.asarray(gg['geometry_valid'], dtype=bool))
        geo = SimpleNamespace(satellites=sats)
        roti_rows = None
        if 'products' in f and 'ROTI' in f['products']:
            roti_rows = []
            rg = f['products/ROTI']
            if rg.attrs.get('algorithm') != 'MEDUSA_ION_V6_FIXED_150S':
                raise ValueError('Legacy ROTI HDF5: recompute using Stage 22')
            for sv, sg in rg['150s'].items():
                for t, v, e, pair in zip(sg['utc_ms'][:], sg['roti_tecu_min'][:],
                                         sg['elevation_deg'][:], sg['signal_pair'].asstr()[:]):
                    roti_rows.append(dict(window_seconds=150, satellite=sv, utc_ms=int(t),
                                          roti_tecu_min=float(v), elevation_deg=float(e), signal_pair=pair))
        station = str(f.attrs.get('station') or 'SITE').strip() or 'SITE'
        day = epochs[0].astype('datetime64[D]').astype(object)
        return geo, roti_rows, station, f'{day.timetuple().tm_yday:03d}', str(day.year)


def render_hdf5_figures(path, directory, *, max_qc_satellites=None, dpi=300, notify=None, overview_figures=True, qc_elevation=False, qc_zoom=False, figure_products=None, qc_directory=None, min_elevation=20.0, mean_line=True, mean_min_satellites=4, mean_method="mean", mean_smooth_minutes=0., color_mode="constellation"):
    """Generate legacy-style figures directly from a previously saved HDF5."""
    geo, roti_rows, station, doy, year = _from_h5(path)
    import h5py
    extras={}
    with h5py.File(path,'r') as f:
        for p in ('DELTATEC','DTEC','SIDX','STEC','VTEC'):
            if 'products/'+p not in f: continue
            group=f['products/'+p]
            # Older HDF5 DTEC means the legacy rolling-mean index, NOT detrended VTEC.
            if p=='DTEC':
                algorithm=group.attrs.get('algorithm','')
                if algorithm=='OASIS_DTEC_ROLLING_MEAN_DIFFERENCE_150S': key='DELTATEC'
                elif algorithm=='VTEC_MINUS_SAVITZKY_GOLAY_900S_LINEAR': key='DTEC'
                else: raise ValueError('Unrecognized DTEC algorithm; cannot safely label this file')
            else: key=p
            if key in extras: raise ValueError(f'Ambiguous product: {key}')
            extras[key]={sv:{k:v[:] for k,v in g.items()} for sv,g in group.items()}
    return render_oasis_figures(geo, roti_rows, directory, station=station, doy=doy, year=year, extra_products=extras, overview_figures=overview_figures,
                                max_qc_satellites=max_qc_satellites, dpi=dpi, notify=notify, qc_elevation=qc_elevation, qc_zoom=qc_zoom, figure_products=figure_products, qc_directory=qc_directory, min_elevation=min_elevation, mean_line=mean_line, mean_min_satellites=mean_min_satellites, mean_method=mean_method, mean_smooth_minutes=0., color_mode=color_mode)

# Backward-compatible function alias for earlier Python callers.
plot_b111_qc = plot_rnx_qc
