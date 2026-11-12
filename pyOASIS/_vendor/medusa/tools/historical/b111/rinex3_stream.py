"""
Lightweight streaming RINEX 3 observation reader for MEDUSA/B111.

Purpose
-------
Read RINEX 3 observation files sequentially without GeoRinex/xarray.

* Supports plain .rnx and gzip-compressed .rnx.gz.
* Reads the file exactly once.
* Supports GPS (G), GLONASS (R), Galileo (E), BeiDou (C).
* Uses SYS / # / OBS TYPES from the RINEX header.
* Preserves the original epoch grid.
* Missing observations are represented by NaN.

This module performs RINEX decoding only.
It does NOT perform B111 QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import gzip

import numpy as np


@dataclass
class Rinex3StreamData:
    times: np.ndarray
    systems: dict[str, tuple[str, ...]]
    satellites: dict[str, dict[str, np.ndarray]]


def _open_text(path):
    path = Path(path)

    if path.suffix.lower() == ".gz":
        return gzip.open(
            path,
            "rt",
            encoding="ascii",
            errors="replace",
            newline="",
        )

    return path.open(
        "rt",
        encoding="ascii",
        errors="replace",
        newline="",
    )


def _parse_header(f):
    obs_types = {}

    while True:
        line = f.readline()

        if not line:
            raise EOFError(
                "Unexpected EOF before END OF HEADER"
            )

        line = line.rstrip("\r\n")
        label = line[60:80].strip() if len(line) >= 60 else ""

        if label == "END OF HEADER":
            break

        if label != "SYS / # / OBS TYPES":
            continue

        system = line[0]
        nobs = int(line[3:6])
        names = line[7:60].split()

        while len(names) < nobs:
            continuation = f.readline()

            if not continuation:
                raise EOFError(
                    "Unexpected EOF in SYS / # / OBS TYPES"
                )

            continuation = continuation.rstrip("\r\n")

            names.extend(
                continuation[7:60].split()
            )

        obs_types[system] = tuple(names[:nobs])

    return obs_types


def _parse_epoch(line):
    fields = line[1:].split()

    if len(fields) < 8:
        raise ValueError(
            f"Malformed RINEX epoch line: {line!r}"
        )

    year = int(fields[0])
    month = int(fields[1])
    day = int(fields[2])
    hour = int(fields[3])
    minute = int(fields[4])

    sec_float = float(fields[5])
    second = int(sec_float)
    microsecond = int(
        round(
            (sec_float - second)
            * 1_000_000
        )
    )

    if microsecond >= 1_000_000:
        second += 1
        microsecond -= 1_000_000

    flag = int(fields[6])
    nsat = int(fields[7])

    timestamp = np.datetime64(
        datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
        ),
        "ns",
    )

    return timestamp, flag, nsat


def _parse_observation_fields(text, nobs):
    """
    Parse fixed-width RINEX observation fields.

    Each observation occupies 16 characters:
        columns 1-14 : numeric value
        column 15    : LLI
        column 16    : signal strength

    Only the numeric value is required by B111 here.
    """

    values = np.full(
        nobs,
        np.nan,
        dtype=np.float64,
    )

    for i in range(nobs):
        field = text[
            i * 16:
            (i + 1) * 16
        ]

        if not field:
            break

        value_text = field[:14].strip()

        if not value_text:
            continue

        try:
            values[i] = float(value_text)
        except ValueError:
            pass

    return values


def read_rinex3_stream(
    path,
    *,
    systems="GREC",
):
    """
    Decode a RINEX 3 observation file sequentially.

    Returns data on the original RINEX epoch grid.
    """

    wanted = set(systems)

    times = []

    # Temporary sparse storage:
    #
    # satellite ->
    #     observable ->
    #         {epoch_index: value}
    sparse = {}

    with _open_text(path) as f:

        obs_types = _parse_header(f)

        selected_obs_types = {
            system: names
            for system, names in obs_types.items()
            if system in wanted
        }

        epoch_index = -1

        while True:

            line = f.readline()

            if not line:
                break

            line = line.rstrip("\r\n")

            if not line:
                continue

            if not line.startswith(">"):
                continue

            timestamp, flag, nsat = _parse_epoch(
                line
            )

            epoch_index += 1
            times.append(timestamp)

            # RINEX epoch flags > 1 may contain event/special
            # records rather than normal satellite observations.
            #
            # We still consume exactly nsat following records.
            normal_epoch = flag in (0, 1)

            for _ in range(nsat):

                first = f.readline()

                if not first:
                    raise EOFError(
                        "Unexpected EOF inside epoch"
                    )

                first = first.rstrip("\r\n")

                if len(first) < 3:
                    continue

                satellite = first[:3]
                system = satellite[0]

                names = obs_types.get(system)

                if names is None:
                    # Cannot determine continuation length safely
                    # without the system observation definition.
                    raise ValueError(
                        f"No observation types for {system}"
                    )

                nobs = len(names)

                # First physical line has PRN in columns 1-3,
                # observations begin at column 4.
                payload = first[3:]

                # This RINEX 3 file stores each satellite
                # observation record on ONE physical line.
                #
                # The line may terminate before all observation
                # types declared in SYS / # / OBS TYPES. Missing
                # trailing fields are therefore left as NaN by
                # _parse_observation_fields().
                #
                # IMPORTANT:
                # Do NOT consume the following physical line here.
                # It is either the next satellite record or the
                # next epoch record beginning with ">".
                if (
                    not normal_epoch
                    or system not in wanted
                ):
                    continue

                values = _parse_observation_fields(
                    payload,
                    nobs,
                )

                sat_store = sparse.setdefault(
                    satellite,
                    {
                        name: {}
                        for name in names
                    },
                )

                for name, value in zip(
                    names,
                    values,
                ):
                    if np.isfinite(value):
                        sat_store[name][
                            epoch_index
                        ] = value

    times_array = np.asarray(
        times,
        dtype="datetime64[ns]",
    )

    nepoch = len(times_array)

    satellites = {}

    for satellite, obs_store in sparse.items():

        satellite_data = {}

        for name, samples in obs_store.items():

            array = np.full(
                nepoch,
                np.nan,
                dtype=np.float64,
            )

            if samples:
                indices = np.fromiter(
                    samples.keys(),
                    dtype=np.int64,
                )

                values = np.fromiter(
                    samples.values(),
                    dtype=np.float64,
                )

                array[indices] = values

            satellite_data[name] = array

        satellites[satellite] = satellite_data

    return Rinex3StreamData(
        times=times_array,
        systems={
            system: tuple(names)
            for system, names
            in selected_obs_types.items()
        },
        satellites=satellites,
    )
