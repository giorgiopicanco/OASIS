"""In-memory adaptations of OASIS DTEC_CALC (deltaTEC), SIDX_CALC and TEC_CAL.
No synthetic observations or inter-arc bridging. Experimental calibration.
"""
from collections import defaultdict
import numpy as np
from .medusa_roti_adapter import _OFFSETS
from .rnx_qc import satellite_key

def gpst_array_to_utc_ms(times):
    times=np.asarray(times,dtype='datetime64[ms]')
    if np.isnat(times).any(): raise ValueError('Invalid GPST epoch')
    t=times.astype(np.int64)
    threshold=np.array([np.datetime64(epoch,"ms").astype(np.int64) for epoch,_ in _OFFSETS])
    ix=np.searchsorted(threshold,t,side="right")-1
    if np.any(ix<0): raise ValueError("GPST before GPS epoch")
    return t-np.array([n for _,n in _OFFSETS],dtype=np.int64)[ix]*1000

PRODUCTS = ('LGF', 'ROTI', 'DELTATEC', 'SIDX', 'STEC', 'VTEC', 'DTEC')

def satellite_order(item):
    return satellite_key(item[0])

def epoch_data(sat):
    b = sat.b111
    t = gpst_array_to_utc_ms(sat.times)
    k = 40.3e16*(1/float(b.f2_hz)**2-1/float(b.f1_hz)**2)
    if not np.isfinite(k) or abs(k)<1e-10:
        raise ValueError('Invalid geometry-free frequency factor')
    x = np.asarray(b.l4_leveled_m)/k
    a = np.asarray(b.arc_ids)
    flags = np.asarray(b.flags_final)
    good = np.isfinite(x)&(a>=0)&(flags>=0)&((flags&1)==0)
    return t,x,a,good

def compute_indices(geo, requested):
    result={p:{} for p in ('DELTATEC','SIDX') if p in requested}
    if not result: return result
    for sv,s in sorted(geo.satellites.items(),key=satellite_order):
        t,x,a,good=epoch_data(s)
        flags=np.asarray(s.b111.flags_final,dtype=int)
        cadence=int(np.median(np.diff(t))) if len(t)>1 else 0
        for p,output in result.items():
            width=150000 if p=='DELTATEC' else 60000
            rows=[]
            # Reject ambiguous windows with more than one accepted QC arc.
            arc_count=defaultdict(set)
            for epoch,arc in zip(t[good],a[good]): arc_count[int(epoch//width)].add(int(arc))
            for arc in np.unique(a[good]):
                ix=np.flatnonzero(good&(a==arc))
                bins=t[ix]//width
                for j in np.split(ix,np.flatnonzero(np.diff(bins)!=0)+1):
                    win=t[j[0]]//width
                    if len(j)<2 or cadence<=0 or np.any(np.diff(t[j])!=cadence): continue
                    if len(arc_count[int(win)])!=1 or np.any(flags[j[1:]]&6): continue
                    if p=='DELTATEC':
                        # Original OASIS centered rolling short/long means, last sample.
                        z=x[j]; short=15 if len(z)>30 else 5
                        value=10*(np.mean(z[-(short//2+1):])-np.mean(z[-(2*short+1):]))
                    else:
                        value=np.mean(np.abs(np.diff(x[j])*60000/np.diff(t[j])))
                    if not np.isfinite(value): continue
                    g=s.geometry; valid=np.asarray(s.geometry_valid)[j]
                    e=float(np.mean(g['elevation_deg'][j][valid])) if valid.any() else np.nan
                    lat=float(np.mean(g['ipp_lat_deg'][j][valid])) if valid.any() else np.nan
                    lon=float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(g['ipp_lon_deg'][j][valid])))))) if valid.any() else np.nan
                    rows.append((int(np.mean(t[j])),value,int(arc),e,lat,lon,int(win*width),int((win+1)*width),len(j)))
            # Original polynomial residual screen, separately per existing QC arc.
            arr=np.array(rows,dtype=float).reshape(-1,9)
            keep=np.ones(len(arr),bool)
            for arc in np.unique(arr[:,2]):
                j=np.flatnonzero(arr[:,2]==arc)
                if len(j)<5: continue
                z=arr[j,0]; z=(z-z.mean())/np.ptp(z)
                res=arr[j,1]-np.polyval(np.polyfit(z,arr[j,1],3),z)
                keep[j]=np.abs(res)<=max(5*np.std(res),1e-12*max(1.,np.max(np.abs(arr[j,1]))))
            arr=arr[keep]
            arr=arr[np.argsort(arr[:,0],kind="stable")]
            output[sv]=dict(zip(('utc_ms','value','arc_id','elevation_deg','ipp_lat_deg','ipp_lon_deg','window_start_utc_ms','window_end_utc_ms','sample_count'),arr.T))
            output[sv]['ipp_height_km']=np.where(np.isfinite(arr[:,4])&np.isfinite(arr[:,5]),350.,np.nan)
            output[sv]['geometry_available']=np.isfinite(arr[:,4])&np.isfinite(arr[:,5])&np.isfinite(arr[:,3])
            output[sv]['original_elevation_ok']=output[sv]['geometry_available']&(arr[:,3]>=(30 if p=='DELTATEC' else 20))
    return result

def compute_calibrated_tec(geo, receiver_lat_deg, receiver_lon_deg):
    """Original six coefficients per UTC hour + one offset per QC arc.
    Solve weighted observation equations directly, without legacy interpolation.
    Fail closed when rank deficient. Offsets are model estimates, not external DCBs.
    """
    from scipy.sparse import coo_matrix
    from scipy.linalg import eigh, solve
    if not np.isfinite(receiver_lat_deg) or not np.isfinite(receiver_lon_deg):
        raise ValueError('Invalid receiver geodetic coordinates')
    samples=[]; arcs={}; hours={}
    for sv,s in sorted(geo.satellites.items(),key=satellite_order):
        t,x,a,good=epoch_data(s); g=s.geometry
        good &= np.asarray(s.geometry_valid)&(g['elevation_deg']>=20)&np.isfinite(g['mapping_factor'])&(g['mapping_factor']>=1)&np.isfinite(g['ipp_lat_deg'])&np.isfinite(g['ipp_lon_deg'])
        for arc in np.unique(a[good]):
            ix=np.flatnonzero(good&(a==arc))
            if len(ix)<2: continue
            key=(sv,int(arc)); arcs[key]=len(arcs)
            for i in ix:
                hour=int(t[i]//3600000); hours.setdefault(hour,len(hours))
                u=(t[i]%3600000)/3600000
                lon=np.radians((g['ipp_lon_deg'][i]-receiver_lon_deg+180)%360-180)
                lat=np.radians(g['ipp_lat_deg'][i]-receiver_lat_deg)
                m=1/g['mapping_factor'][i]
                samples.append((key,hour,i,t[i],x[i]-x[ix[0]],m,[1,u,lon,u*lon,lat,u*lat]))
    out={'STEC':{},'VTEC':{}}
    if not samples: return out,{'status':'NO_VALID_GEOMETRY'}
    # A six-parameter hour with fewer than six independent geometric rows
    # cannot be estimated (e.g., the preceding UTC day has only 18 seconds).
    by_hour=defaultdict(list)
    for row in samples: by_hour[row[1]].append(row[6])
    valid_hours={h for h, design in by_hour.items() if np.linalg.matrix_rank(np.asarray(design))==6}
    excluded_hours=sorted(set(hours)-valid_hours)
    samples=[row for row in samples if row[1] in valid_hours]
    if not samples: return out,{'status':'NO_SOLVABLE_UTC_HOURS','excluded_utc_hours':excluded_hours}
    # Drop arcs that have fewer than two samples after the hour selection.
    count=defaultdict(int)
    for row in samples: count[row[0]]+=1
    samples=[row for row in samples if count[row[0]]>=2]
    if not samples: return out,{'status':'NO_SOLVABLE_ARCS','excluded_utc_hours':excluded_hours}
    hours={h:i for i,h in enumerate(sorted({row[1] for row in samples}))}
    arcs={a:i for i,a in enumerate(sorted({row[0] for row in samples}))}
    n=6*len(hours)+len(arcs); ri=[]; ci=[]; vals=[]; y=[]
    if n>5000: return out,{'status':'MODEL_TOO_LARGE_FOR_DENSE_RANK_CHECK','unknowns':n,'observations':len(samples)}
    for j,(key,hour,i,t,x,m,coeff) in enumerate(samples):
        ri.extend([j]*7); ci.extend([6*hours[hour]+k for k in range(6)]+[6*len(hours)+arcs[key]])
        vals.extend(coeff+[m]); y.append(x*m)
    A=coo_matrix((vals,(ri,ci)),shape=(len(y),n)).tocsr()
    normal=(A.T@A).toarray(); rhs=A.T@np.asarray(y)
    scale=np.sqrt(np.diag(normal)); scale[scale==0]=1
    normal/=scale[:,None]*scale[None,:]; rhs/=scale
    eig=eigh(normal,eigvals_only=True); rank=int(np.sum(eig>max(eig[-1]*1e-10,1e-14)))
    if rank<n: return out,{'status':'RANK_DEFICIENT','rank':rank,'unknowns':n,'observations':len(y),'excluded_utc_hours':excluded_hours}
    sol=solve(normal,rhs,assume_a='pos')/scale
    for key,hour,i,t,x,m,coeff in samples:
        sv,arc=key; offset=sol[6*len(hours)+arcs[key]]; stec=x-offset
        g=geo.satellites[sv].geometry
        for p,v in [('STEC',stec),('VTEC',stec*m)]:
            d=out[p].setdefault(sv,defaultdict(list))
            for k,val in dict(utc_ms=t,value=v,arc_id=arc,elevation_deg=g['elevation_deg'][i],ipp_lat_deg=g['ipp_lat_deg'][i],ipp_lon_deg=g['ipp_lon_deg'][i],model_arc_offset_tecu=offset,ipp_height_km=350.,mapping_factor=1/m,physical_nonnegative=bool(stec>=0),lgf_equivalent_stec_tecu=float(geo.satellites[sv].b111.l4_leveled_m[i]/(40.3e16*(1/float(geo.satellites[sv].b111.f2_hz)**2-1/float(geo.satellites[sv].b111.f1_hz)**2)))).items(): d[k].append(val)
    return out,{'status':'EXPERIMENTAL_MODEL_CALIBRATION','rank':rank,'unknowns':n,'observations':len(y),'residual_rms_tecu':float(np.sqrt(np.mean((A@sol-y)**2))), 'excluded_utc_hours':excluded_hours, 'normalized_normal_condition':float(eig[-1]/eig[0]), 'hour_coefficients':{str(h):sol[6*i:6*i+6].tolist() for h,i in hours.items()}, 'arc_offsets_tecu':{f'{sv}:{arc}':float(sol[6*len(hours)+i]) for (sv,arc),i in arcs.items()}, 'negative_stec_count':sum(int(np.count_nonzero(np.asarray(d['value'])<0)) for d in out['STEC'].values())}


def aggregate_sidx(satellite_data, *, min_elevation=20.0):
    """Station-level SIDX, one equal-weight contribution per satellite/window.

    Uses the original OASIS SIDX elevation cutoff (20 degrees) and fixed UTC
    window midpoints. The native units are TECU/min; conversion to mTECU/s
    is exactly 1000/60, not the legacy arbitrary plotting multiplier of ten.
    """
    result={}
    for system in 'GREC':
        bins=defaultdict(dict)
        for sv,data in satellite_data.items():
            if not sv.startswith(system): continue
            for t,v,e in zip(data['window_start_utc_ms'],data['value'],data['elevation_deg']):
                if np.isfinite(v) and np.isfinite(e) and e>=min_elevation:
                    if sv in bins[int(t)]: raise ValueError(f'Duplicate SIDX window for {sv}')
                    bins[int(t)][sv]=float(v)
        if not bins: continue
        times=np.array(sorted(bins),dtype=np.int64)
        values=np.array([np.mean(list(bins[t].values())) for t in times],dtype=float)
        result[system]={'utc_ms':times+30000,'window_start_utc_ms':times,
                        'value_tecu_min':values,'value_mtecu_s':values*1000/60,
                        'satellite_count':np.array([len(bins[t]) for t in times],dtype=np.int64)}
    return result
