"""Python translation of MEDUSA precise SP3 branch and RTKLIB rtkcmn.c geometry.

Sources: SatelliteGeometryProcessor.cpp (transmit time + Earth rotation),
LinkGeometry.cpp (geodist/satazel/ionppp), RTKLIB-2.4.3 rtkcmn.c.
This is NOT the original compiled C++ and lacks RTCM broadcast fallback.
Numerical cross-language validation is required before claiming exact parity.
"""
from __future__ import annotations
import math
import numpy as np

CLIGHT = 299792458.0
OMGE = 7.2921151467e-5
RE_WGS84 = 6378137.0
FE_WGS84 = 1.0/298.257223563


def ecef2pos(r):
    """rtkcmn.c ecef2pos, same convergence condition (1e-4 m)."""
    x,y,z0=(float(v) for v in r)
    e2=FE_WGS84*(2.0-FE_WGS84)
    r2=x*x+y*y
    z=z0; zk=0.; v=RE_WGS84
    for _ in range(100):
        if abs(z-zk)<1.0e-4: break
        zk=z
        sinp=z/math.sqrt(r2+z*z)
        v=RE_WGS84/math.sqrt(1.0-e2*sinp*sinp)
        z=z0+v*e2*sinp
    else:
        raise ArithmeticError('RTKLIB ecef2pos failed to converge')
    lat=math.atan(z/math.sqrt(r2)) if r2>1e-12 else (math.pi/2 if z0>0 else -math.pi/2)
    lon=math.atan2(y,x) if r2>1e-12 else 0.0
    h=math.sqrt(r2+z*z)-v
    return lat,lon,h


def rtklib_link_geometry(receiver, sat, *, shell_km=350.0):
    """Direct translation of MEDUSA LinkGeometry.cpp precise-position overload.

    `mapping_factor` is RTKLIB ionppp slant factor (STEC/VTEC).
    """
    rr=np.asarray(receiver,dtype=float);rs=np.asarray(sat,dtype=float)
    if rr.shape!=(3,) or rs.shape!=(3,) or not np.all(np.isfinite(rr)) or not np.all(np.isfinite(rs)):
        return None
    if np.linalg.norm(rr)<1e6 or np.linalg.norm(rs)<1e7:return None
    d=rs-rr; length=float(np.linalg.norm(d))
    if length<=0:return None
    los=d/length
    rho=length+OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/CLIGHT
    lat,lon,h=ecef2pos(rr)
    east=-math.sin(lon)*los[0]+math.cos(lon)*los[1]
    north=-math.sin(lat)*math.cos(lon)*los[0]-math.sin(lat)*math.sin(lon)*los[1]+math.cos(lat)*los[2]
    up=math.cos(lat)*math.cos(lon)*los[0]+math.cos(lat)*math.sin(lon)*los[1]+math.sin(lat)*los[2]
    az=math.atan2(east,north) if east*east+north*north>=1e-12 else 0.0
    if az<0:az+=2*math.pi
    el=math.asin(max(-1.0,min(1.0,up)))
    out=dict(sat_x_m=float(rs[0]),sat_y_m=float(rs[1]),sat_z_m=float(rs[2]),range_m=float(rho),
             azimuth_deg=math.degrees(az),elevation_deg=math.degrees(el),
             ipp_lat_deg=math.nan,ipp_lon_deg=math.nan,mapping_factor=math.nan)
    if el<=0:return out
    # Exact RTKLIB 2.4.3 ionppp formula, including polar longitude branch.
    rp=6378.1363/(6378.1363+shell_km)*math.cos(el)
    ap=math.pi/2-el-math.asin(rp)
    sinap=math.sin(ap);tanap=math.tan(ap);cosaz=math.cos(az)
    ipp_lat=math.asin(max(-1.0,min(1.0,math.sin(lat)*math.cos(ap)+math.cos(lat)*sinap*cosaz)))
    divisor=math.cos(ipp_lat)
    if abs(divisor)<1e-14:return out
    argument=max(-1.0,min(1.0,sinap*math.sin(az)/divisor))
    polar=(lat>math.radians(70) and tanap*cosaz>math.tan(math.pi/2-lat)) or (lat<-math.radians(70) and -tanap*cosaz>math.tan(math.pi/2+lat))
    ipp_lon=lon+(math.pi-math.asin(argument) if polar else math.asin(argument))
    lon_deg=math.degrees(ipp_lon)
    while lon_deg>=180.:lon_deg-=360.
    while lon_deg< -180.:lon_deg+=360.
    out.update(ipp_lat_deg=math.degrees(ipp_lat),ipp_lon_deg=lon_deg,mapping_factor=1.0/math.sqrt(1.0-rp*rp))
    return out


def geometry_from_sp3(orbit, satellite, gpst, pseudorange_m, receiver_xyz, *, shell_km=350.0):
    """MEDUSA SP3 path: ttx=t_rx-P/c; Sp3Orbit.position(ttx); Earth rotation.

    Returns None when SP3 unavailable; no invented broadcast ephemeris.
    """
    p=float(pseudorange_m)
    if not math.isfinite(p) or p<=0:return None
    dt=p/CLIGHT
    tx=np.datetime64(gpst,'ns')-np.timedelta64(round(dt*1e9),'ns')
    sat=orbit.position(satellite,tx)
    if sat is None:return None
    x,y,z=(float(v) for v in sat)
    angle=OMGE*dt
    c=math.cos(angle);s=math.sin(angle)
    rotated=[c*x+s*y,-s*x+c*y,z]
    return rtklib_link_geometry(receiver_xyz,rotated,shell_km=shell_km)
