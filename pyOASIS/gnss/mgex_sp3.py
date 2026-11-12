#!/usr/bin/env python3
"""Download a daily MGEX precise SP3 orbit file from NASA CDDIS.

Example from OASIS root:
    python download_mgex_sp3.py --date 2025-04-30
    python download_mgex_sp3.py --rinex INPUT/RINEX_TESTS/EARTHSCOPE/CHPI00BRA_R_20251200000_01D_30S_MO_short.rnx

Requires: python -m pip install requests
CDDIS authentication: ~/.netrc for urs.earthdata.nasa.gov, or --username
(password prompted; never saved by this script).
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import getpass
import gzip
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    sys.exit("Missing dependency 'requests'. Install with: python -m pip install requests")

CDDIS = 'https://cddis.nasa.gov/archive/gnss/products'
EARTHDATA = 'urs.earthdata.nasa.gov'
GPS_EPOCH = date(1980, 1, 6)


class EarthdataSession(requests.Session):
    """Preserve authentication when redirecting between CDDIS and NASA URS.

    Based on NASA Earthdata recommended requests recipe. The credentials are
    only forwarded across hosts when one host is urs.earthdata.nasa.gov.
    """
    def rebuild_auth(self, prepared_request, response):
        old_host = urlparse(response.request.url).hostname
        new_host = urlparse(prepared_request.url).hostname
        if old_host != new_host:
            if EARTHDATA not in (old_host, new_host):
                prepared_request.headers.pop('Authorization', None)
        if new_host == EARTHDATA and self.trust_env:
            credentials = requests.utils.get_netrc_auth(prepared_request.url)
            if credentials:
                prepared_request.prepare_auth(credentials)
        # Do not call superclass: it drops auth when CDDIS redirects to URS.


def gps_week(d: date) -> int:
    if d < GPS_EPOCH:
        raise ValueError('GPS week is undefined before 1980-01-06')
    return (d - GPS_EPOCH).days // 7


def date_from_rinex_filename(path: str) -> date:
    name = Path(path).name
    # RINEX 3/4 long name: STATION_R_YYYYDDDHHMM_...
    m = re.search(r'_[RSU]_(\d{4})(\d{3})(\d{4})_', name, re.I)
    if m:
        return date(int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2))-1)
    # RINEX 2 short: ssssDDDF.YYo, also ssssDDDF.YYd (compressed Hatanaka)
    m = re.match(r'^.{4}(\d{3})[0-9a-z]\.([0-9]{2})[od]$', name, re.I)
    if m:
        year = int(m.group(2))
        year += 2000 if year <= 79 else 1900
        return date(year, 1, 1) + timedelta(days=int(m.group(1))-1)
    raise ValueError(f'Cannot infer date from RINEX filename: {name}. Use --date YYYY-MM-DD.')


def product_for(d: date, prefix: str | None = None):
    doy = d.timetuple().tm_yday
    prefix = prefix or ('GFZ0MGXRAP' if d.year >= 2018 else 'JAX0MGXFIN')
    basename = f'{prefix}_{d.year}{doy:03d}0000_01D_05M_ORB.SP3'
    return basename, f'{CDDIS}/{gps_week(d):04d}/{basename}.gz'


def inspect_sp3(path: Path):
    constellation_counts = {}
    seen = set()
    epochs = 0
    with path.open('rt', encoding='ascii', errors='replace') as f:
        first = f.readline()
        if not first.startswith('#'):
            raise ValueError('Downloaded content does not look like an SP3 file')
        for line in f:
            if line.startswith('*'):
                epochs += 1
            if len(line) >= 4 and line.startswith('P') and line[1] in 'GREC' and line[2:4].isdigit():
                seen.add(line[1:4])
    for c in 'GREC':
        constellation_counts[c] = len([sv for sv in seen if sv.startswith(c)])
    return epochs, constellation_counts


def main(argv=None):
    ap = argparse.ArgumentParser(description='Download and unpack NASA CDDIS MGEX .SP3.gz for pyOASIS')
    selector = ap.add_mutually_exclusive_group(required=True)
    selector.add_argument('--date', help='Observation date UTC: YYYY-MM-DD')
    selector.add_argument('--rinex', help='RINEX file; date extracted from standard filename')
    ap.add_argument('--out', default='INPUT/ORBITS', help='Output directory (default: INPUT/ORBITS)')
    ap.add_argument('--username', help='Earthdata Login username; password securely prompted')
    ap.add_argument('--prefix', choices=['GFZ0MGXRAP','JAX0MGXFIN'], help='Override product prefix')
    ap.add_argument('--force', action='store_true', help='Overwrite existing SP3')
    ap.add_argument('--keep-gz', action='store_true', help='Keep downloaded .gz after extraction')
    ap.add_argument('--dry-run', action='store_true', help='Only print the resolved filename and URL')
    args = ap.parse_args(argv)

    try:
        d = date.fromisoformat(args.date) if args.date else date_from_rinex_filename(args.rinex)
        # Validate that a day of year from a filename is not beyond end of year
        if args.rinex and re.search(r'_[RSU]_(\d{4})(\d{3})(\d{4})_', Path(args.rinex).name, re.I):
            m = re.search(r'_[RSU]_(\d{4})(\d{3})(\d{4})_', Path(args.rinex).name, re.I)
            if d.year != int(m.group(1)):
                raise ValueError('RINEX filename has an invalid day of year')
        basename, url = product_for(d, args.prefix)
    except ValueError as exc:
        ap.error(str(exc))

    out = Path(args.out)
    target = out / basename
    print(f'Date UTC: {d.isoformat()} | DOY: {d.timetuple().tm_yday:03d} | GPS week: {gps_week(d)}')
    print(f'Product: {basename}')
    print(f'URL: {url}')
    if args.dry_run:
        return 0
    if target.exists() and not args.force:
        try:
            epochs, counts = inspect_sp3(target)
            if epochs < 2 or not any(counts.values()):
                raise ValueError(f'Invalid SP3: epochs={epochs}, satellites={counts}')
        except (ValueError, OSError) as exc:
            print(f'Existing SP3 invalid; downloading replacement: {exc}')
        else:
            print(f'Already exists: {target.resolve()} | epochs={epochs} | satellites={counts}')
            return 0

    out.mkdir(parents=True, exist_ok=True)
    gz_file = out / (basename + '.gz')
    tmp_gz = out / (basename + '.gz.partial')
    tmp_sp3 = out / (basename + '.partial')

    session = EarthdataSession()
    session.headers['User-Agent'] = 'pyOASIS-MGEX-SP3-downloader/1.0'
    if args.username:
        session.auth = (args.username, getpass.getpass('NASA Earthdata password: '))
    else:
        print('Authentication: using Earthdata credentials from your ~/.netrc if required.')

    try:
        with session.get(url, timeout=(20, 120), stream=True, allow_redirects=True) as response:
            response.raise_for_status()
            final = response.url.lower()
            if EARTHDATA in (urlparse(final).hostname or "") or 'login' in final:
                raise RuntimeError('Earthdata redirected to login. Configure ~/.netrc or use --username.')
            it = response.iter_content(chunk_size=256 * 1024)
            with tmp_gz.open('wb') as f:
                for chunk in it:
                    if chunk:
                        f.write(chunk)
        with tmp_gz.open('rb') as f:
            if f.read(2) != b'\x1f\x8b':
                raise RuntimeError('Server returned something other than gzip (possibly an HTML login page).')
        with gzip.open(tmp_gz, 'rb') as src, tmp_sp3.open('wb') as dst:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                dst.write(buf)
        epochs, counts = inspect_sp3(tmp_sp3)
        if epochs < 2 or sum(counts.values()) == 0:
            raise RuntimeError(f'Invalid SP3: epochs={epochs}, GNSS satellites={counts}')
        os.replace(tmp_sp3, target)
        if args.keep_gz:
            os.replace(tmp_gz, gz_file)
        else:
            tmp_gz.unlink(missing_ok=True)
        print(f'SAVED: {target.resolve()}')
        print(f'SP3 verified: {epochs} epochs, satellites by system: {counts}')
        if not all(counts[c] > 0 for c in 'GREC'):
            print('WARNING: This SP3 does not contain all 4 GREC constellations.')
        return 0
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else '?'
        print(f'HTTP {code}: {exc}', file=sys.stderr)
        print('Check Earthdata login and the product existence in this GPS week.', file=sys.stderr)
        return 1
    except (requests.RequestException, OSError, EOFError, RuntimeError, ValueError) as exc:
        print(f'Download/validation failed: {exc}', file=sys.stderr)
        return 1
    finally:
        tmp_gz.unlink(missing_ok=True)
        tmp_sp3.unlink(missing_ok=True)
        session.close()


if __name__ == '__main__':
    sys.exit(main())
