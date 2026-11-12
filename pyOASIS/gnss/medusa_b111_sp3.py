"""Join actual MEDUSA B111 leveled GNSS observations to SP3 geometry, in memory.

The MEDUSA B111 QC is executed by medusa_b111.run_b111 unchanged.
The geometry matches the original MEDUSA SP3 algorithm sequence with a Python
translation of RTKLIB formulas; numerical parity is NOT yet certified.
No intermediate RINEX or CSV is written by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from .medusa_b111 import B111Dataset, B111Satellite, run_b111
from .medusa_processing import Sp3Orbit
from .medusa_geometry import geometry_from_sp3
from .rinex_io import read_observations

GEOMETRY_FIELDS = (
    'sat_x_m', 'sat_y_m', 'sat_z_m', 'range_m', 'azimuth_deg',
    'elevation_deg', 'ipp_lat_deg', 'ipp_lon_deg', 'mapping_factor',
    'pseudorange_m', 'signal_travel_time_s',
)


@dataclass(frozen=True)
class B111GeoSatellite:
    """Same global epoch grid as B111Satellite, geometry only for valid B111 levels."""
    b111: B111Satellite
    geometry: dict[str, np.ndarray]
    orbit_available: np.ndarray
    above_horizon: np.ndarray
    geometry_valid: np.ndarray
    orbit_source: np.ndarray | None = None

    @property
    def satellite(self) -> str:
        return self.b111.satellite

    @property
    def times(self) -> np.ndarray:
        return self.b111.times

    @property
    def l4_leveled_m(self) -> np.ndarray:
        return self.b111.l4_leveled_m

    @property
    def lgf_leveled_m(self):
        return self.b111.l4_leveled_m

    @property
    def rnx_qc(self):
        return self.b111

    @property
    def arc_ids(self) -> np.ndarray:
        return self.b111.arc_ids

    @property
    def flags_final(self) -> np.ndarray:
        return self.b111.flags_final

    def __getitem__(self, field: str) -> np.ndarray:
        return self.geometry[field]


@dataclass(frozen=True)
class B111GeoDataset:
    rinex_path: str
    sp3_path: str
    b111: B111Dataset
    satellites: dict[str, B111GeoSatellite]
    status: str = 'MEDUSA_SP3_TRANSMIT_TIME_RTKLIB_FORMULAS_PYTHON_PORT_UNVERIFIED_PARITY'
    sp3_sources: tuple[str, ...] = ()

    @property
    def rnx_qc(self):
        return self.b111

    @property
    def skipped(self):
        return self.b111.skipped

    def summary(self) -> dict[str, int | str]:
        return {
            'status': self.status,
            'satellites': len(self.satellites),
            'leveled_samples': sum(int(np.isfinite(s.l4_leveled_m).sum()) for s in self.satellites.values()),
            'orbit_samples': sum(int(s.orbit_available.sum()) for s in self.satellites.values()),
            'geometry_valid_samples': sum(int(s.geometry_valid.sum()) for s in self.satellites.values()),
            'b111_status': self.b111.status,
        }


def _readonly(values, dtype=None):
    a = np.array(values, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


def attach_sp3_geometry(
    b111: B111Dataset,
    rinex_path: str | Path,
    sp3_path: str | Path,
    *,
    systems: str = 'GREC',
    observations: Any = None,
    orbit: Any = None,
) -> B111GeoDataset:
    """Attach SP3 geometry to B111 successes only; preserve full-grid epoch indices.

    All geometry fields are NaN on a QC-rejected observation or if SP3 is missing.
    Below-horizon links preserve range/az/el but IPP and mapping remain NaN.
    Requires known GNSS time scale matching the SP3 orbit time scale (GPS/GPST).
    Uses MEDUSA transmission-time + Earth-rotation SP3 method, without
    original C++/RTKLIB parity certification or RTCM broadcast fallback.
    """
    obs = observations if observations is not None else read_observations(rinex_path, systems=systems)
    if getattr(obs, 'time_system', None) not in ('GPS', 'GPST'):
        raise ValueError(
            'SP3 association requires explicit RINEX GPS/GPST time system; '
            f'got {getattr(obs, "time_system", None)!r}. No implicit time conversion.'
        )
    rr = getattr(obs, 'receiver_xyz_m', None)
    if rr is None or np.asarray(rr).shape != (3,) or not np.isfinite(rr).all():
        raise ValueError('RINEX lacks valid receiver APPROX POSITION XYZ')
    rr = np.asarray(rr, dtype=float)
    if np.linalg.norm(rr) < 1e6:
        raise ValueError('RINEX receiver ECEF XYZ is invalid')
    sp3 = orbit if orbit is not None else Sp3Orbit(sp3_path)
    aligned = {}
    epochs = np.asarray(obs.times, dtype='datetime64[ns]')
    for sv, sat in b111.satellites.items():
        times = np.asarray(sat.times, dtype='datetime64[ns]')
        n = len(times)
        if len(epochs) != n or not np.array_equal(epochs, times):
            raise ValueError(f'{sv}: B111 epoch grid differs from RINEX; refusing index-based join')
        if len(sat.l4_leveled_m) != n or len(sat.arc_ids) != n or len(sat.flags_final) != n:
            raise ValueError(f'{sv}: B111 arrays are not aligned to the RINEX epoch grid')
        geo = {k: np.full(n, np.nan, dtype=float) for k in GEOMETRY_FIELDS}
        available = np.zeros(n, dtype=bool)
        above = np.zeros(n, dtype=bool)
        valid = np.zeros(n, dtype=bool)
        sources = np.full(n, 'UNAVAILABLE', dtype=object)
        # Do not attach geometry to rejected/not-leveled B111 samples.
        accepted = np.flatnonzero(np.isfinite(sat.l4_leveled_m) & (np.asarray(sat.arc_ids) >= 0))
        for i in accepted:
            # Exact MEDUSA SP3 *method*: require pseudorange, derive transmit time,
            # interpolate SP3 there, rotate Earth, run RTKLIB geometry formulas.
            # Does NOT silently invent a broadcast ephemeris if SP3 is absent.
            source_codes = tuple(sat.source_observables or sat.signals)
            pvalue = None
            for code in source_codes[2:]:
                if not code.startswith(('C','P')): continue
                series = obs.get(sv, code)
                if series is None: continue
                candidate = float(series.values[i])
                if np.isfinite(candidate) and candidate > 0:
                    pvalue = candidate
                    break
            if pvalue is None:
                continue
            # The original MEDUSA hasSatellite/position() checks.  The Python
            # interpolator returns None on missing satellite/epoch instead.
            values = geometry_from_sp3(sp3, sv, times[i], pvalue, rr)
            if values is None:
                continue
            available[i] = True
            sources[i] = str(getattr(sp3, 'last_source', None) or str(sp3_path))
            values['pseudorange_m'] = pvalue
            values['signal_travel_time_s'] = pvalue / 299792458.0
            for field in GEOMETRY_FIELDS:
                geo[field][i] = float(values[field])
            above[i] = bool(np.isfinite(geo['elevation_deg'][i]) and geo['elevation_deg'][i] > 0)
            valid[i] = bool(above[i] and np.isfinite(geo['ipp_lat_deg'][i]) and
                            np.isfinite(geo['ipp_lon_deg'][i]) and
                            np.isfinite(geo['mapping_factor'][i]))
        aligned[sv] = B111GeoSatellite(
            b111=sat,
            geometry={k: _readonly(a) for k, a in geo.items()},
            orbit_available=_readonly(available),
            above_horizon=_readonly(above),
            geometry_valid=_readonly(valid),
            orbit_source=_readonly(sources),
        )
    return B111GeoDataset(
        rinex_path=str(rinex_path), sp3_path=str(sp3_path),
        b111=b111, satellites=aligned,
        sp3_sources=tuple(str(p) for p in getattr(sp3, 'paths', (sp3_path,))),
    )


def run_b111_with_sp3(rinex_path, sp3_path, medusa_root, *, systems='GREC', config=None) -> B111GeoDataset:
    """Single-call B111 official historical QC + preliminary SP3 geometry.

    Does not generate scientific TEC/ROTI/DTEC/SIDX, and does not write files.
    This API currently accepts RINEX3 through run_b111; RINEX2 B111 signal
    mapping is still unvalidated and raises NotImplementedError upstream.
    """
    b111 = run_b111(rinex_path, medusa_root, systems=systems, config=config)
    return attach_sp3_geometry(b111, rinex_path, sp3_path, systems=systems)
