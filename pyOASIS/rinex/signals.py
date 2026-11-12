"""Physical carrier frequencies from RINEX 3/4 observation codes.

Does not pick a preferred tracking code or silently guess GLONASS FDMA channels.
Frequencies in Hz. RINEX 2 two-character codes need explicit translation by
constellation and receiver metadata, not a fabricated RINEX 3 tracking suffix.
"""
from __future__ import annotations
from dataclasses import dataclass
from .reader import RinexHeader

C = 299_792_458.0

# Physical frequency digit in RINEX 3/4, dependent on GNSS system.
FREQUENCIES = {
    'G': {'1': 1575.42e6, '2': 1227.60e6, '5': 1176.45e6},
    'E': {'1': 1575.42e6, '5': 1176.45e6, '7': 1207.14e6,
          '8': 1191.795e6, '6': 1278.75e6},
    'C': {'1': 1575.42e6, '2': 1561.098e6, '5': 1176.45e6,
          '6': 1268.52e6, '7': 1207.14e6, '8': 1191.795e6},
    'I': {'1': 1575.42e6, '5': 1176.45e6, '9': 2492.028e6},
    'J': {'1': 1575.42e6, '2': 1227.60e6, '5': 1176.45e6, '6': 1278.75e6},
    'S': {'1': 1575.42e6, '5': 1176.45e6},
    'R': {'3': 1202.025e6, '4': 1600.995e6, '6': 1248.06e6},
}

@dataclass(frozen=True)
class Carrier:
    satellite: str
    observable: str
    frequency_hz: float

    @property
    def wavelength_m(self) -> float:
        return C / self.frequency_hz


def carrier_frequency(satellite: str, observable: str, *, header: RinexHeader) -> Carrier:
    if len(satellite) != 3 or len(observable) < 2:
        raise ValueError(f'Invalid satellite/signal: {satellite!r}, {observable!r}')
    sys, digit = satellite[0], observable[1]
    if header.version < 3:
        raise ValueError('RINEX 2 signal-to-band interpretation requires the dedicated legacy adapter')
    if sys == 'R' and digit in ('1', '2'):
        if satellite not in header.glonass_channels:
            raise ValueError(f'GLONASS FDMA channel missing for {satellite}; do not assume k=0')
        k = header.glonass_channels[satellite]
        f = (1602e6 + 562500 * k) if digit == '1' else (1246e6 + 437500 * k)
    else:
        try:
            f = FREQUENCIES[sys][digit]
        except KeyError as ex:
            raise ValueError(f'Unknown physical carrier: {satellite} {observable}') from ex
    return Carrier(satellite, observable, f)
