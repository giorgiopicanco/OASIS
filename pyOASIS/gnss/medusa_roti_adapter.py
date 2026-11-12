"""Historical adapter for the MEDUSA ION V6 fixed-window ROTI kernel."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import numpy as np
from .medusa_v6_window import window_records, SLOT_MS, HALF_MS
from .rnx_qc import satellite_key

_OFFSETS = [
 ("1980-01-06T00:00:00",0),("1981-07-01T00:00:01",1),
 ("1982-07-01T00:00:02",2),("1983-07-01T00:00:03",3),
 ("1985-07-01T00:00:04",4),("1988-01-01T00:00:05",5),
 ("1990-01-01T00:00:06",6),("1991-01-01T00:00:07",7),
 ("1992-07-01T00:00:08",8),("1993-07-01T00:00:09",9),
 ("1994-07-01T00:00:10",10),("1996-01-01T00:00:11",11),
 ("1997-07-01T00:00:12",12),("1999-01-01T00:00:13",13),
 ("2006-01-01T00:00:14",14),("2009-01-01T00:00:15",15),
 ("2012-07-01T00:00:16",16),("2015-07-01T00:00:17",17),
 ("2017-01-01T00:00:18",18)]


def gpst_to_utc_ms(t):
    t=np.datetime64(t,'ms')
    if t < np.datetime64(_OFFSETS[0][0]):raise ValueError('GPST before GPS epoch')
    offset=0
    for epoch, n in _OFFSETS:
        if t < np.datetime64(epoch): break
        offset=n
    return int(t.astype('int64'))-offset*1000


def compute_medusa_roti(geo):
    """Preserve the V6 window kernel; adapt the existing B111 result arrays."""
    output=[]
    counts=Counter()
    for sv, sat in sorted(geo.satellites.items(),key=lambda item:satellite_key(item[0])):
        b=sat.b111
        f1,f2=float(b.f1_hz),float(b.f2_hz)
        if not all(math.isfinite(f) and f>1e8 for f in (f1,f2)):
            raise ValueError(f'{sv}: invalid B111 frequencies')
        k=40.3e16*(1/f2**2-1/f1**2)
        if abs(k)<1e-10:
            raise ValueError(f'{sv}: invalid B111 frequency pair')
        pair=(str(b.signals[0])+':'+str(b.signals[1])).replace('L','')
        segments=[]
        arrays=[b.l4_leveled_m,b.arc_ids,b.flags_final,b.slip,sat.geometry_valid]
        if any(len(x)!=len(sat.times) for x in arrays):
            raise ValueError(f'{sv}: B111 time alignment mismatch')
        # Use final UPDARC boundaries and slip indices, not preliminary CHKL4P slips.
        source=np.asarray(b.source_indices,dtype=int)
        cursor=0
        for result in b.original_b111_result:
            up=result.final_updarc
            n=len(result.source_indices)
            original=source[cursor:cursor+n]; cursor+=n
            if len(original)!=n or len(up.flags)!=n:
                raise ValueError(f'{sv}: final UPDARC source alignment mismatch')
            for arc in up.arcs:
                start,end=int(arc.start),int(arc.end)+1
                if not 0<=start<end<=n:
                    raise ValueError(f'{sv}: invalid final UPDARC arc')
                slips=set(int(x) for x in (arc.slip_indices or []))
                segment=[]
                for j in range(start,end):
                    i=int(original[j]); epoch=sat.times[i]
                    t=gpst_to_utc_ms(epoch)
                    slot=(t+SLOT_MS//2)//SLOT_MS*SLOT_MS
                    arc_id=int(b.arc_ids[i]); value=float(b.l4_leveled_m[i])
                    good=(int(up.flags[j])&1)==0 and math.isfinite(value)
                    if good and arc_id<0:
                        raise ValueError(f'{sv}: valid level without arc ID')
                    if segment and (not good or j in slips or arc_id!=segment[-1][6] or
                                    slot-segment[-1][1] not in (15000,30000) or
                                    not 0<t-segment[-1][0]<=31500):
                        segments.append(segment); segment=[]
                    if not good: continue
                    lat=lon=elev=float('nan')
                    if sat.geometry_valid[i]:
                        lat,lon,elev=(float(sat.geometry[x][i]) for x in ('ipp_lat_deg','ipp_lon_deg','elevation_deg'))
                    segment.append((t,slot,value/k,lat,lon,elev,arc_id))
                if segment: segments.append(segment)
        if cursor!=len(source):
            raise ValueError(f'{sv}: unconsumed B111 source indices')
        # Index segments by 300-second block, preserving original segment order.
        blocks=defaultdict(list)
        for seg in segments:
            for start in sorted({x[1]//300000*300000 for x in seg}):
                blocks[start].append([x for x in seg if start<=x[1]<start+300000])
        for start, segs in sorted(blocks.items()):
            records, stats=window_records(segs,start,'',sv,pair)
            counts.update(stats)
            for a in records:
                begin,end,mjd,_,_,_,_,value,nrot,cadence,lat,lon,height,elev,flags,arc=a
                output.append(dict(window_seconds=150, satellite=sv, utc_ms=begin+75000,
                    window_start_utc_ms=begin,window_end_utc_ms=end,mjd_utc=mjd,
                    roti_tecu_min=value,rot_count=nrot,cadence_seconds=cadence,
                    latitude_ipp_deg=lat,longitude_ipp_deg=lon,ipp_height_km=height,
                    elevation_deg=elev,mean_elevation_deg=elev,ion_flags=flags,
                    arc=arc,signal_pair=pair))
    return output,dict(counts,source_algorithm='MEDUSA ION V6 b111_adapter_v6.py window loop',
                       window_seconds=150,geometry_requirement='None; elevation cutoff is display only',
                       qc_reference='Historical B111 arrays; live block-context B111 parity pending')
