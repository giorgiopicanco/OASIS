"""Independent RINEX 2/3/4 OBS reader. No georinex or xarray.

Strict by default: unexpected line structure raises RinexFormatError rather than
silently shifting values into the wrong measurement columns.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TextIO
from contextlib import contextmanager
import gzip
import re
import numpy as np


class RinexFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    value: float
    lli: int | None = None
    ssi: int | None = None


@dataclass
class RinexHeader:
    version: float = 0.0
    file_type: str = ""
    system: str = ""
    marker_name: str = ""
    position_xyz: tuple[float, float, float] | None = None
    interval: float | None = None
    time_system: str | None = None
    observation_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    glonass_channels: dict[str, int] = field(default_factory=dict)
    raw_header: list[str] = field(default_factory=list)


@dataclass
class RinexEpoch:
    time: np.datetime64
    flag: int
    satellites: dict[str, dict[str, Observation]]
    receiver_clock_offset: float | None = None
    # Original RINEX-4 fractional seconds are not rounded away in metadata.
    seconds_text: str = ""


@dataclass
class RinexData:
    header: RinexHeader
    times: np.ndarray
    satellites: dict[str, dict[str, np.ndarray]]
    lli: dict[str, dict[str, np.ndarray]]
    ssi: dict[str, dict[str, np.ndarray]]
    epoch_flags: np.ndarray
    epoch_seconds_text: list[str]


@contextmanager
def open_rinex(path):
    p = Path(path)
    if p.suffix.lower() == ".gz":
        handle = gzip.open(p, "rt", encoding="ascii", errors="replace", newline="")
    else:
        handle = p.open("rt", encoding="ascii", errors="replace", newline="")
    try:
        yield handle
    finally:
        handle.close()


def _readline(f: TextIO) -> str:
    return f.readline().rstrip("\r\n")


def _label(line: str) -> str:
    return line[60:80].strip()


def _parse_header(f: TextIO) -> RinexHeader:
    h = RinexHeader()
    obs2: list[str] = []
    obs2_count: int | None = None
    obs3_count: dict[str, int] = {}
    current_system: str | None = None
    while True:
        raw = f.readline()
        if not raw:
            raise RinexFormatError("END OF HEADER not found")
        line = raw.rstrip("\r\n")
        h.raw_header.append(line)
        label = _label(line)
        if label == "RINEX VERSION / TYPE":
            h.version = float(line[:9])
            h.file_type = line[20:21].strip().upper()
            h.system = line[40:41].strip().upper()
            if h.file_type != "O":
                raise RinexFormatError(f"Expected observation RINEX (O), got {h.file_type!r}")
            if not (2 <= h.version < 5):
                raise RinexFormatError(f"Unsupported RINEX version {h.version}")
        elif label == "MARKER NAME":
            h.marker_name = line[:60].strip()
        elif label == "APPROX POSITION XYZ":
            try:
                h.position_xyz = tuple(map(float, line[:60].split()[:3]))
            except ValueError:
                pass
        elif label == "INTERVAL":
            try:
                h.interval = float(line[:10])
            except ValueError:
                pass
        elif label == "TIME OF FIRST OBS":
            h.time_system = line[48:51].strip() or None
        elif label == "# / TYPES OF OBSERV":
            # RINEX 2: six-character slots, max 9 types per physical line.
            if line[:6].strip():
                obs2_count = int(line[:6])
                obs2 = []
            for i in range(9):
                code = line[6 + 6 * i: 12 + 6 * i].strip()
                if code:
                    obs2.append(code)
        elif label == "SYS / # / OBS TYPES":
            sys = line[0:1].strip()
            if sys:
                current_system = sys
                obs3_count[sys] = int(line[3:6])
                h.observation_types[sys] = ()
            if current_system is None:
                raise RinexFormatError("Orphan SYS / # / OBS TYPES continuation")
            fields = tuple(line[7:60].split())
            h.observation_types[current_system] += fields
        elif label == "GLONASS SLOT / FRQ #":
            tokens = line[:60].split()
            i = 1 if tokens and tokens[0].lstrip("+-").isdigit() else 0
            while i + 1 < len(tokens):
                sv, k = tokens[i], tokens[i + 1]
                if re.fullmatch(r"R\d{2}", sv):
                    try:
                        val = int(k)
                    except ValueError:
                        raise RinexFormatError(f"Invalid GLONASS channel: {sv} {k}")
                    if not -7 <= val <= 6:
                        raise RinexFormatError(f"GLONASS FDMA channel out of range: {sv} {val}")
                    if sv in h.glonass_channels and h.glonass_channels[sv] != val:
                        raise RinexFormatError(f"Conflicting GLONASS FDMA header values: {sv}")
                    h.glonass_channels[sv] = val
                    i += 2
                else:
                    i += 1
        elif label == "END OF HEADER":
            break
    if h.version < 3:
        if obs2_count is None or len(obs2) < obs2_count:
            raise RinexFormatError("Incomplete RINEX 2 observation type list")
        # RINEX 2 uses one global observation list for all constellations.
        for sys in "GRECISJ":
            h.observation_types[sys] = tuple(obs2[:obs2_count])
    else:
        for sys, count in obs3_count.items():
            got = h.observation_types[sys]
            if len(got) != count:
                raise RinexFormatError(f"{sys}: expected {count} OBS types, got {len(got)}")
    return h


def read_header(path) -> RinexHeader:
    with open_rinex(path) as f:
        return _parse_header(f)


def _obs_field(text: str) -> Observation:
    value = text[:14].strip()
    def flag(c):
        return int(c) if c and c.isdigit() else None
    try:
        v = float(value.replace("D", "E")) if value else float("nan")
    except ValueError as exc:
        raise RinexFormatError(f"Invalid RINEX observation value: {value!r}") from exc
    return Observation(v, flag(text[14:15]), flag(text[15:16]))


def _timestamp(y, m, d, hh, mm, second_text):
    # Datetime64 supports ns. Preserve the original textual seconds separately.
    from datetime import datetime, timedelta
    from decimal import Decimal, ROUND_HALF_UP
    s = Decimal(second_text)
    whole = int(s)
    nanos = int(((s - whole) * 1_000_000_000).to_integral_value(rounding=ROUND_HALF_UP))
    dt = datetime(y, m, d, hh, mm) + timedelta(seconds=whole)
    return np.datetime64(dt, "ns") + np.timedelta64(nanos, "ns")


def _epoch3(line):
    tok = line[1:].split()
    if len(tok) < 8:
        raise RinexFormatError(f"Malformed RINEX 3/4 epoch: {line!r}")
    y, m, d, hh, mm = map(int, tok[:5])
    sec = tok[5]
    flag, n = int(tok[6]), int(tok[7])
    clock = None
    # Optional clock offset uses fixed columns 42-56, not split tokens after nsat.
    try:
        clock = float(line[41:56]) if line[41:56].strip() else None
    except ValueError:
        pass
    return _timestamp(y, m, d, hh, mm, sec), sec, flag, n, clock


def _epoch2(line, f):
    try:
        yy = int(line[1:3]); m = int(line[4:6]); d = int(line[7:9])
        hh = int(line[10:12]); mm = int(line[13:15]); sec = line[15:26].strip()
        flag = int(line[28:29]); n = int(line[29:32])
    except ValueError as exc:
        raise RinexFormatError(f"Malformed RINEX 2 epoch: {line!r}") from exc
    year = 2000 + yy if yy < 80 else 1900 + yy
    sv_text = line[32:68]
    while len(sv_text) < n * 3:
        extra = _readline(f)
        if not extra:
            raise EOFError("Truncated RINEX 2 satellite list")
        sv_text += extra[32:68]
    sats = [sv_text[i*3:(i+1)*3].strip() for i in range(n)]
    try:
        clock = float(line[68:80]) if line[68:80].strip() else None
    except ValueError:
        clock = None
    return _timestamp(year, m, d, hh, mm, sec), sec, flag, n, sats, clock


def _satellite3(first, f, nobs):
    sv = first[:3].strip()
    if not re.fullmatch(r"[A-Z]\d{2}", sv):
        raise RinexFormatError(f"Malformed satellite identifier: {sv!r}")
    payload = first[3:]
    # The RINEX 3/4 record can occupy one long line, or continuation lines
    # with three leading blanks. An absent trailing field is padded with NaN.
    while len(payload) < nobs * 16:
        offset = f.tell()
        following = f.readline()
        if not following:
            break
        cont = following.rstrip("\r\n")
        if cont.startswith("   ") and cont[3:].strip():
            payload += cont[3:]
        else:
            f.seek(offset)
            break
    values = {}
    return sv, payload


def iter_epochs(path, *, systems="GRECISJ") -> Iterator[RinexEpoch]:
    """Yield normal observation epochs; consume events without treating them as observations.

    Event flag 4 (new header records) is rejected: dynamic signal-list changes
    require a dedicated header-state update implementation.
    """
    wanted = set(systems)
    with open_rinex(path) as f:
        h = _parse_header(f)
        if h.version >= 3:
            while True:
                line = _readline(f)
                if not line:
                    break
                if not line.startswith(">"):
                    raise RinexFormatError(f"Expected epoch '>', got {line[:48]!r}")
                timestamp, sec, flag, n, clk = _epoch3(line)
                if flag not in (0, 1):
                    if flag == 4:
                        raise RinexFormatError("RINEX header-update event (flag 4) requires dynamic header handling")
                    for _ in range(n):
                        if not f.readline():
                            raise EOFError("Truncated special event")
                    continue
                sats = {}
                for _ in range(n):
                    first = _readline(f)
                    if not first:
                        raise EOFError("Truncated satellite record")
                    sv = first[:3].strip()
                    if not sv or sv[0] not in h.observation_types:
                        raise RinexFormatError(f"No OBS types for satellite in line: {first[:30]!r}")
                    names = h.observation_types[sv[0]]
                    name, payload = _satellite3(first, f, len(names))
                    if name[0] in wanted:
                        sats[name] = {code: _obs_field(payload[16*i:16*(i+1)])
                                      for i, code in enumerate(names)}
                yield RinexEpoch(timestamp, flag, sats, clk, sec)
        else:
            while True:
                line = _readline(f)
                if not line:
                    break
                if not line.strip():
                    continue
                timestamp, sec, flag, n, satnames, clk = _epoch2(line, f)
                if flag not in (0, 1):
                    if flag == 4:
                        raise RinexFormatError("RINEX header-update event (flag 4) requires dynamic header handling")
                    for _ in range(n):
                        if not f.readline():
                            raise EOFError("Truncated RINEX 2 event")
                    continue
                sats = {}
                names = h.observation_types["G"]
                lines_per_sv = (len(names) + 4) // 5
                for sv in satnames:
                    if not sv:
                        raise RinexFormatError("Empty satellite in RINEX 2 epoch")
                    if sv[0].isdigit():
                        sv = (h.system if h.system not in ("", "M") else "G") + sv.zfill(2)
                    payload = ""
                    for _ in range(lines_per_sv):
                        raw_line = f.readline()
                        if raw_line == "":
                            raise EOFError("Truncated RINEX 2 observation lines")
                        # A blank observation line is legal: it encodes absent
                        # values (NaNs), not end-of-file. Do not confuse it
                        # with EOF, which is the zero-length raw read above.
                        part = raw_line.rstrip("\r\n")
                        payload += part.ljust(80)
                    if sv[0] in wanted:
                        sats[sv] = {code: _obs_field(payload[16*i:16*(i+1)])
                                    for i, code in enumerate(names)}
                yield RinexEpoch(timestamp, flag, sats, clk, sec)


def read_rinex(path, *, systems="GRECISJ") -> RinexData:
    """Convenience dense time-aligned arrays; prefer iter_epochs for long files."""
    h = read_header(path)
    times = []
    flags = []
    second_texts = []
    sparse: dict[str, dict[str, dict[int, Observation]]] = {}
    for idx, ep in enumerate(iter_epochs(path, systems=systems)):
        times.append(ep.time)
        flags.append(ep.flag)
        second_texts.append(ep.seconds_text)
        for sv, signals in ep.satellites.items():
            store = sparse.setdefault(sv, {})
            for code, obs in signals.items():
                if np.isfinite(obs.value) or obs.lli is not None or obs.ssi is not None:
                    store.setdefault(code, {})[idx] = obs
    n = len(times)
    values, lli, ssi = {}, {}, {}
    for sv, codes in sparse.items():
        values[sv] = {}; lli[sv] = {}; ssi[sv] = {}
        for code, samples in codes.items():
            v = np.full(n, np.nan)
            l = np.full(n, -1, dtype=np.int8)
            s = np.full(n, -1, dtype=np.int8)
            for idx, obs in samples.items():
                v[idx] = obs.value
                if obs.lli is not None: l[idx] = obs.lli
                if obs.ssi is not None: s[idx] = obs.ssi
            values[sv][code] = v
            lli[sv][code] = l
            ssi[sv][code] = s
    return RinexData(h, np.asarray(times, dtype="datetime64[ns]"), values,
                     lli, ssi, np.asarray(flags, dtype=np.int8), second_texts)
