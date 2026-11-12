"""Window loop extracted unchanged from MEDUSA ION V6 b111_adapter_v6.py.
The wrapper supplies the historical B111 segments instead of rerunning live QC.
Source SHA256: df3e21543736b1f552f8b4124a456e9e24c328872b913d6ed482b79ef3ed5086
"""
import math
import numpy as np
from collections import Counter
SLOT_MS=15000
HALF_MS=150000
FLAG_NO_IPP=1
FLAG_PARTIAL=2
FLAG_CADENCE_30S=4
FLAG_EDGE_CONTEXT=8

def window_records(segments, start, station, sat, pair, partial=False):
    roti_out=[]
    c=Counter()
    for window in (start,start+HALF_MS):
        eligible=[]
        for seg in segments:
            rows=[x for x in seg if window<=x[1]<window+HALF_MS]
            if len(rows)<5:continue
            steps=np.diff([x[1] for x in rows]);
            if len(steps)<1:continue
            cadence=int(np.median(steps))
            if cadence not in (15000,30000):continue
            expected=HALF_MS//cadence
            if len(rows)!=expected or np.any(steps!=cadence):continue
            # Allow distinct 30-s phases (0/30/... or 15/45/...).
            indices=[(x[1]-window)//SLOT_MS for x in rows]
            if len(set(indices))!=len(indices):continue
            if cadence==15000 and indices!=list(range(10)):continue
            if cadence==30000 and indices not in (list(range(0,10,2)),list(range(1,10,2))):continue
            eligible.append((cadence,rows))
        if not eligible:
            c['b111_no_complete_150s_arc']+=1;continue
        # One result max per station-satellite-window.
        cadence,rows=max(eligible,key=lambda z:len(z[1]))
        rot=[(b[2]-a[2])*60000.0/(b[0]-a[0]) for a,b in zip(rows,rows[1:])]
        if len(rot)<2 or any(not math.isfinite(x) for x in rot):
            c['invalid_rot']+=1;continue
        # User-configured post-B111 ROT screening; never publish extreme
        # single-epoch phase gradients as a valid ROTI estimate.
        if any(abs(x)>5.0 for x in rot):
            c['post_b111_rot_gt_5_tecu_per_min']+=1;continue
        value=float(np.std(np.asarray(rot,dtype=float),ddof=0))
        # No display clipping at QC stage. B111 live 10 TECU/min cutoff as a quality screen.
        if value<0 or value>10:
            c['b111_roti_gt_10']+=1;continue
        loc=[x for x in rows if math.isfinite(x[3]) and math.isfinite(x[4])]
        if loc:
            near=min(loc,key=lambda x:abs(x[1]-(window+HALF_MS//2)))
            lat,lon,elev=near[3:6]
        else:lat=lon=elev=float('nan')
        fl=(0 if loc else FLAG_NO_IPP)|(FLAG_PARTIAL if partial else 0)|\
           (FLAG_CADENCE_30S if cadence==30000 else 0)
        roti_out.append((window,window+HALF_MS,40587.+(window+HALF_MS//2)/86400000.,
               station.encode()[:64],sat.encode()[:8],sat[0].encode(),pair.encode()[:32],
               value,len(rot),cadence/1000.,lat,lon,350. if loc else float('nan'),
               elev,fl,rows[-1][6]))
    return roti_out, c
