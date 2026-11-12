import tempfile
import unittest
from pathlib import Path
from pyOASIS.gnss.medusa_b111 import resolve_glonass_channels
from pyOASIS.rinex import read_header
from pyOASIS.rinex.reader import RinexFormatError


def line(value, label):
    return f'{value:<60}{label:<20}\n'

class ChannelPriority(unittest.TestCase):
    def test_header_overrides_conflicting_legacy_table(self):
        channels, source, conflicts = resolve_glonass_channels(
            {'R11': 0, 'R12': -1}, {'R11': 5, 'R13': -4})
        self.assertEqual(channels, {'R11': 0, 'R12': -1, 'R13': -4})
        self.assertEqual(source['R11'], 'RINEX_HEADER')
        self.assertEqual(source['R13'], 'EXTERNAL_FALLBACK')
        self.assertEqual(conflicts, {'R11': (0, 5)})

    def test_no_fabricated_channels(self):
        channels, source, conflicts = resolve_glonass_channels({}, None)
        self.assertEqual((channels, source, conflicts), ({}, {}, {}))

    def test_invalid_channels_rejected(self):
        for v in (-8, 7, '1', None, True):
            with self.subTest(v=v), self.assertRaises(ValueError):
                resolve_glonass_channels({}, {'R11': v})

    def test_real_rinex3_header_parse(self):
        header = (line('     3.04           OBSERVATION DATA    M (MIXED)', 'RINEX VERSION / TYPE') +
                  line('  2 R11  0 R12 -1', 'GLONASS SLOT / FRQ #') +
                  line('', 'END OF HEADER'))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'mixed.rnx'
            p.write_text(header)
            h = read_header(p)
            self.assertEqual(h.glonass_channels, {'R11': 0, 'R12': -1})

    def test_conflicting_channels_in_rinex_header_rejected(self):
        header = (line('     3.04           OBSERVATION DATA    M (MIXED)', 'RINEX VERSION / TYPE') +
                  line('  1 R11  0', 'GLONASS SLOT / FRQ #') +
                  line('  1 R11  1', 'GLONASS SLOT / FRQ #') +
                  line('', 'END OF HEADER'))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'mixed.rnx'
            p.write_text(header)
            with self.assertRaises(RinexFormatError):
                read_header(p)

if __name__ == '__main__': unittest.main()
