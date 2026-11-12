import gzip
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'pyOASIS'))
from rinex import read_rinex, iter_epochs, read_header, RinexFormatError


def hdr(content, label):
    return f'{content:<60}{label:<20}\n'


def obs(v, lli=None, ssi=None):
    return f'{v:14.3f}{" " if lli is None else lli}{" " if ssi is None else ssi}'


def make_rnx3(version='3.04', cont=False, sec='0.0000000'):
    s = hdr(f'{version:>9}           OBSERVATION DATA    M', 'RINEX VERSION / TYPE')
    codes = ['C1C','L1C','C2W','L2W','S1C','D1C']
    s += hdr('G  ' + f'{len(codes):3d}' + ' ' + ' '.join(codes), 'SYS / # / OBS TYPES')
    s += hdr(' 30.000', 'INTERVAL') + hdr('', 'END OF HEADER')
    s += f'> 2023 02 18 00 00 {sec:>11}  0  1\n'
    fields = [obs(20e6), obs(105e6,1,7), obs(20e6+2), obs(82e6), obs(45), obs(-1200)]
    if cont:
        s += 'G01' + ''.join(fields[:3]) + '\n'
        s += '   ' + ''.join(fields[3:]) + '\n'
    else:
        s += 'G01' + ''.join(fields) + '\n'
    s += '> 2023 02 18 00 00 30.0000000  0  1\n'
    s += 'G01' + obs(20e6+30) + obs(105e6+150) + '\n'
    return s


def make_rnx2():
    s = hdr('     2.11           OBSERVATION DATA    G', 'RINEX VERSION / TYPE')
    # Standard RINEX 2 six-character fields (9 per line).
    codes = ['C1','L1','P2','L2','S1','D1']
    s += hdr(f'{len(codes):6d}' + ''.join(f'{x:>6}' for x in codes), '# / TYPES OF OBSERV')
    s += hdr('', 'END OF HEADER')
    prefix = f' {23:02d} {2:02d} {18:02d} {0:02d} {0:02d}{0.:11.7f}  {0:1d}{1:3d}'
    s += prefix + 'G01' + '\n'
    data = [obs(20e6), obs(105e6,1,5), obs(20e6+2), obs(82e6), obs(40), obs(-200)]
    s += ''.join(data[:5]) + '\n'
    s += ''.join(data[5:]) + '\n'
    return s


class TestRinexReader(unittest.TestCase):
    def _read(self, text, suffix='.rnx'):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ('test' + suffix)
            if suffix.endswith('.gz'):
                with gzip.open(path, 'wt', encoding='ascii') as f: f.write(text)
            else:
                path.write_text(text, encoding='ascii')
            return read_rinex(path)

    def test_rnx3_single_line(self):
        d = self._read(make_rnx3())
        self.assertEqual(len(d.times), 2)
        self.assertEqual(d.header.version, 3.04)
        self.assertEqual(d.satellites['G01']['L1C'][0], 105e6)
        self.assertEqual(d.lli['G01']['L1C'][0], 1)
        self.assertEqual(d.ssi['G01']['L1C'][0], 7)
        self.assertTrue(np.isnan(d.satellites['G01']['L2W'][1]))
        self.assertEqual(int((d.times[1]-d.times[0]) / np.timedelta64(1,'s')),30)

    def test_rnx3_continuation(self):
        d = self._read(make_rnx3(cont=True))
        self.assertEqual(d.satellites['G01']['D1C'][0], -1200)

    def test_rnx4_obs(self):
        d = self._read(make_rnx3(version='4.02', sec='0.0000001'))
        self.assertEqual(d.header.version, 4.02)
        self.assertEqual(d.times[0].astype('datetime64[ns]').astype('int64') % 1_000_000_000, 100)

    def test_rnx2_multiline(self):
        d = self._read(make_rnx2(), '.23o')
        self.assertEqual(d.header.version, 2.11)
        self.assertEqual(d.satellites['G01']['L1'][0], 105e6)
        self.assertEqual(d.lli['G01']['L1'][0], 1)
        self.assertEqual(d.ssi['G01']['L1'][0], 5)
        self.assertEqual(d.satellites['G01']['D1'][0], -200)

    def test_requested_cadences(self):
        for cadence in (1, 5, 15, 30):
            raw = make_rnx3().replace('> 2023 02 18 00 00 30.0000000  0  1',
                                      f'> 2023 02 18 00 00 {cadence:02d}.0000000  0  1')
            d = self._read(raw)
            measured = (d.times[1] - d.times[0]) / np.timedelta64(1, 's')
            self.assertEqual(measured, cadence)

    def test_gzip(self):
        d = self._read(make_rnx3(), '.rnx.gz')
        self.assertEqual(d.header.interval, 30.0)


if __name__ == '__main__': unittest.main()
