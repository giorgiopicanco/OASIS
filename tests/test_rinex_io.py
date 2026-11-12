import tempfile
import unittest
from pathlib import Path
import numpy as np
from pyOASIS.gnss.rinex_io import read_observations


def hdr(content, label):
    return f'{content:<60}{label:<20}\n'


def sample(version='3.04'):
    lines = [hdr(f'{version:>9}           OBSERVATION DATA    M','RINEX VERSION / TYPE'),
             hdr('G    3 C1C L1C L5Q','SYS / # / OBS TYPES'),
             hdr('  1000.0  2000.0  3000.0','APPROX POSITION XYZ'),
             hdr('', 'END OF HEADER'),
             '> 2025 04 30 00 00  0.0000000  0  1\n',
             'G05' + f'{20000000.:14.3f}  ' + f'{105000000.:14.3f}17' + ' '*16 + '\n',
             '> 2025 04 30 00 00 30.0000000  0  1\n',
             'G05' + f'{20000001.:14.3f}  ' + f'{105000001.:14.3f}  ' + ' '*16 + '\n']
    return ''.join(lines)


class TestRinexIO(unittest.TestCase):
    def test_series_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'sample.rnx'; p.write_text(sample(), encoding='ascii')
            d = read_observations(p, systems='G')
            self.assertEqual(len(d.times), 2)
            self.assertEqual(d.satellites, ('G05',))
            self.assertTrue(np.isnan(d.dense('G05','L5Q')).all())
            self.assertEqual(int(d.get('G05','L1C').lli[0]), 1)
            self.assertEqual(int(d.get('G05','L1C').ssi[0]), 7)
            self.assertAlmostEqual(d.get('G05','L1C').frequency_hz, 1575.42e6)
            with self.assertRaises(ValueError):
                d.get('G05','L1C').values[0] = 0

    def test_rinex4_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'sample.rnx'; p.write_text(sample('4.02'), encoding='ascii')
            with self.assertRaisesRegex(ValueError, 'Only RINEX 2/3'):
                read_observations(p)


if __name__ == '__main__':
    unittest.main()
