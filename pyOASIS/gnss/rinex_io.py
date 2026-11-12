"""Canonical, lossless (for numeric OBS fields) RINEX 2/3 input for pyOASIS.

Uses the already validated pyOASIS.rinex reader.  No GeoRinex, xarray,
implicit signal substitutions, orbit calculation, or quality control.
RINEX 4 intentionally excluded from this release.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
from pyOASIS.rinex import read_rinex, carrier_frequency


@dataclass(frozen=True)
class SignalSeries:
    satellite: str
    code: str
    values: np.ndarray
    lli: np.ndarray
    ssi: np.ndarray
    frequency_hz: float | None
    wavelength_m: float | None

    def __post_init__(self):
        for name in ('values', 'lli', 'ssi'):
            arr = getattr(self, name)
            arr.setflags(write=False)


@dataclass(frozen=True)
class RinexObservations:
    source: str
    version: float
    marker: str
    times: np.ndarray
    epoch_flags: np.ndarray
    receiver_xyz_m: tuple[float, float, float] | None
    time_system: str | None
    declared_signals: dict[str, tuple[str, ...]]
    glonass_channels: dict[str, int]
    series: dict[str, dict[str, SignalSeries]]

    def __post_init__(self):
        self.times.setflags(write=False)
        self.epoch_flags.setflags(write=False)

    @property
    def satellites(self) -> tuple[str, ...]:
        return tuple(sorted(self.series))

    def get(self, satellite: str, code: str) -> SignalSeries | None:
        """Return None when no finite field for this SV/code (never synthesize zero)."""
        return self.series.get(satellite, {}).get(code)

    def dense(self, satellite: str, code: str) -> np.ndarray:
        """Aligned observation samples; absent signal => all NaN."""
        s = self.get(satellite, code)
        return s.values if s is not None else np.full(len(self.times), np.nan)

    def iter_series(self) -> Iterator[SignalSeries]:
        for sv in self.satellites:
            for code in sorted(self.series[sv]):
                yield self.series[sv][code]


def read_observations(path: str | Path, *, systems: str = 'GREC') -> RinexObservations:
    """Read native RINEX 2/3 OBS without selecting tracking signals.

    Note: .crx/.crx.gz (Hatanaka) must first be converted to .rnx.
    GLONASS FDMA frequency is left None if its channel is unavailable.
    RINEX 2 frequencies are not inferred without an explicit signal policy.
    """
    p = Path(path)
    if p.name.lower().endswith(('.crx', '.crx.gz')):
        raise ValueError('Convert Hatanaka CRX to RINEX OBS before reading')
    # Header is inspected up front so that the 4.x experimental reader never
    # accidentally enables 4.x in this release.
    from pyOASIS.rinex import read_header
    header = read_header(p)
    if not 2.0 <= header.version < 4.0:
        raise ValueError(f'Only RINEX 2/3 OBS is supported here: {header.version}')
    d = read_rinex(p, systems=systems)
    out: dict[str, dict[str, SignalSeries]] = {}
    for sv, codes in d.satellites.items():
        for code, values in codes.items():
            freq = None
            wavelength = None
            if d.header.version >= 3 and code[0] in 'CLDS':
                try:
                    c = carrier_frequency(sv, code, header=d.header)
                    freq, wavelength = c.frequency_hz, c.wavelength_m
                except ValueError:
                    # Do not guess unmapped bands or GLONASS channels.
                    pass
            out.setdefault(sv, {})[code] = SignalSeries(
                satellite=sv, code=code,
                values=np.array(values, dtype=np.float64, copy=True),
                lli=np.array(d.lli[sv][code], dtype=np.int8, copy=True),
                ssi=np.array(d.ssi[sv][code], dtype=np.int8, copy=True),
                frequency_hz=freq, wavelength_m=wavelength,
            )
    return RinexObservations(
        source=str(p), version=d.header.version, marker=d.header.marker_name,
        times=np.array(d.times, dtype='datetime64[ns]', copy=True),
        epoch_flags=np.array(d.epoch_flags, dtype=np.int8, copy=True),
        receiver_xyz_m=tuple(d.header.position_xyz) if d.header.position_xyz is not None else None,
        time_system=d.header.time_system,
        declared_signals=dict(d.header.observation_types),
        glonass_channels=dict(d.header.glonass_channels),
        series=out,
    )
