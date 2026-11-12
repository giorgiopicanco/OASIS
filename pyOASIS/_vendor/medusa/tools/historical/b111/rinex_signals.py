"""
B111 — RINEX 3/4 GNSS signal interpretation
============================================

Maps RINEX observation codes to physical GNSS carrier bands.

This module does NOT:
    - choose a preferred dual-frequency pair;
    - apply D111 logic;
    - perform leveling;
    - invent missing GLONASS channels.

The RINEX observation code is preserved.  The frequency band is derived
from the constellation and RINEX frequency digit.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path

from .combinations import (
    GnssSignal,

    GPS_L1,
    GPS_L2,
    GPS_L5,

    GALILEO_E1,
    GALILEO_E5A,
    GALILEO_E5B,
    GALILEO_E5,
    GALILEO_E6,

    BEIDOU_B1I,
    BEIDOU_B1C,
    BEIDOU_B2I,
    BEIDOU_B2A,
    BEIDOU_B2B,
    BEIDOU_B3I,

    GLONASS_G3,
    glonass_signal,
)


SYSTEM_NAMES = {
    "G": "GPS",
    "R": "GLONASS",
    "E": "GALILEO",
    "C": "BEIDOU",
    "J": "QZSS",
    "S": "SBAS",
    "I": "IRNSS",
}


@dataclass(frozen=True)
class RinexSignal:
    system: str
    observable: str
    measurement: str
    frequency_digit: str
    tracking_code: str
    physical_band: str | None


@dataclass
class RinexHeaderSignals:
    version: float | None
    observation_types: dict[str, list[str]]
    glonass_channels: dict[str, int]


def _open_text(path):
    path = Path(path)

    if path.suffix == ".gz":
        return gzip.open(
            path,
            "rt",
            encoding="ascii",
            errors="replace",
        )

    return path.open(
        "rt",
        encoding="ascii",
        errors="replace",
    )


def read_rinex_header(path) -> RinexHeaderSignals:
    """
    Read the RINEX header without loading the observation body.
    """

    observation_types: dict[str, list[str]] = {}
    glonass_channels: dict[str, int] = {}

    version = None
    current_system = None

    with _open_text(path) as f:

        for line in f:

            label = (
                line[60:80].strip()
                if len(line) >= 60
                else ""
            )

            if label == "RINEX VERSION / TYPE":
                try:
                    version = float(line[:9])
                except ValueError:
                    version = None

            elif label == "SYS / # / OBS TYPES":

                system = line[0].strip()

                if system:
                    current_system = system
                    observation_types.setdefault(
                        system,
                        [],
                    )

                if current_system is None:
                    raise ValueError(
                        "Malformed SYS / # / OBS TYPES continuation."
                    )

                # RINEX 3 observation fields occupy columns 8-60.
                fields = line[7:60].split()

                observation_types[
                    current_system
                ].extend(fields)

            elif label == "GLONASS SLOT / FRQ #":

                # Entries are satellite/channel pairs:
                #
                # R01  1 R03  5 ...
                #
                # The first integer in the first record is the number
                # of entries and is intentionally ignored here.

                tokens = line[:60].split()

                i = 0

                if (
                    tokens
                    and tokens[0].lstrip("+-").isdigit()
                ):
                    i = 1

                while i + 1 < len(tokens):

                    sv = tokens[i]

                    if not (
                        len(sv) == 3
                        and sv[0] == "R"
                        and sv[1:].isdigit()
                    ):
                        i += 1
                        continue

                    try:
                        channel = int(tokens[i + 1])
                    except ValueError:
                        i += 1
                        continue

                    glonass_channels[sv] = channel
                    i += 2

            elif label == "END OF HEADER":
                break

    return RinexHeaderSignals(
        version=version,
        observation_types=observation_types,
        glonass_channels=glonass_channels,
    )


def classify_observable(
    system: str,
    observable: str,
) -> RinexSignal:
    """
    Interpret the frequency digit of a RINEX 3/4 observation code.

    This classification preserves the full RINEX code because the same
    frequency digit may represent different physical signals depending
    on the constellation and tracking code.
    """

    if len(observable) < 2:
        raise ValueError(
            f"Invalid RINEX observable: {observable!r}"
        )

    measurement = observable[0]
    digit = observable[1]
    tracking = (
        observable[2:]
        if len(observable) > 2
        else ""
    )

    band = None

    if system == "G":
        band = {
            "1": "L1",
            "2": "L2",
            "5": "L5",
        }.get(digit)

    elif system == "R":
        band = {
            "1": "G1",
            "2": "G2",
            "3": "G3",
        }.get(digit)

    elif system == "E":
        band = {
            "1": "E1",
            "5": "E5a",
            "7": "E5b",
            "8": "E5",
            "6": "E6",
        }.get(digit)

    elif system == "C":
        #
        # BeiDou requires more care because the RINEX frequency digit
        # identifies the carrier frequency, while names such as B1I,
        # B1C, B2I, B2a and B2b also depend on signal generation/code.
        #
        # We preserve the tracking code and resolve the physical signal
        # below.
        #
        band = {
            "1": "B1C",
            "2": "B1I",
            "5": "B2a",
            "6": "B3I",
            "7": "B2I/B2b",
        }.get(digit)

    return RinexSignal(
        system=system,
        observable=observable,
        measurement=measurement,
        frequency_digit=digit,
        tracking_code=tracking,
        physical_band=band,
    )


def carrier_signal(
    system: str,
    observable: str,
    *,
    sv: str | None = None,
    glonass_channels: dict[str, int] | None = None,
) -> GnssSignal:
    """
    Convert a RINEX carrier observable to the physical carrier frequency.

    The input should normally be a carrier-phase observable beginning
    with L, but C/D/S observations on the same signal may also be used
    to identify the carrier frequency.
    """

    info = classify_observable(
        system,
        observable,
    )

    digit = info.frequency_digit

    if system == "G":

        table = {
            "1": GPS_L1,
            "2": GPS_L2,
            "5": GPS_L5,
        }

        if digit in table:
            return table[digit]

    elif system == "E":

        table = {
            "1": GALILEO_E1,
            "5": GALILEO_E5A,
            "7": GALILEO_E5B,
            "8": GALILEO_E5,
            "6": GALILEO_E6,
        }

        if digit in table:
            return table[digit]

    elif system == "R":

        if digit == "3":
            return GLONASS_G3

        if digit in {"1", "2"}:

            if sv is None:
                raise ValueError(
                    "GLONASS G1/G2 requires the satellite ID."
                )

            if glonass_channels is None:
                raise ValueError(
                    "GLONASS channel table is required."
                )

            if sv not in glonass_channels:
                raise ValueError(
                    f"No GLONASS frequency channel available for {sv}."
                )

            return glonass_signal(
                "G1" if digit == "1" else "G2",
                glonass_channels[sv],
            )

    elif system == "C":

        tracking = info.tracking_code.upper()

        # RINEX BDS frequency number 1: B1C
        if digit == "1":
            return BEIDOU_B1C

        # Frequency number 2: B1I
        if digit == "2":
            return BEIDOU_B1I

        # Frequency number 5: B2a
        if digit == "5":
            return BEIDOU_B2A

        # Frequency number 6: B3I
        if digit == "6":
            return BEIDOU_B3I

        # Frequency number 7 has the same carrier frequency
        # (1207.14 MHz) for B2I/B2b.  Preserve the distinction in
        # classification, but the physical carrier frequency is equal.
        if digit == "7":

            if tracking == "I":
                return BEIDOU_B2I

            return BEIDOU_B2B

    raise ValueError(
        "Unsupported B111 carrier mapping: "
        f"system={system!r}, observable={observable!r}"
    )
