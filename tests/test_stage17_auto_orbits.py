import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]

def module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

# Import in isolation: avoid importing legacy pyOASIS and native MEDUSA modules.
import types
for name in ('pyOASIS', 'pyOASIS.gnss'):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg
mgex = module('pyOASIS.gnss.mgex_sp3', ROOT/'pyOASIS/gnss/mgex_sp3.py')
runner = module('run_oasis_stage17_isolated', ROOT/'run_oasis_grec.py')


def sp3(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#cP2025 04 30 00 00 00.00000000      96 ORBIT\n'
                    '*  2025 04 30 00 00 00.00000000\n'
                    'PG01  12000.000000 14000.000000 18000.000000 0.000000\n'
                    'PR01  12000.000000 14000.000000 18000.000000 0.000000\n'
                    'PE01  12000.000000 14000.000000 18000.000000 0.000000\n'
                    'PC01  12000.000000 14000.000000 18000.000000 0.000000\n'
                    '*  2025 04 30 00 05 00.00000000\n', encoding='ascii')

class AutoOrbitTests(unittest.TestCase):
    def args(self, td):
        return SimpleNamespace(
            rinex=Path(td)/'CHPI00BRA_R_20251200000_01D_30S_MO_short.rnx',
            sp3=None, orbit_dir=Path(td)/'INPUT/ORBITS',
            orbit_prefix=None, no_download=False, force_download=False,
            earthdata_user=None, systems='GREC')

    def test_gps_week_and_2025_product(self):
        from datetime import date
        d = date(2025, 4, 30)
        self.assertEqual(mgex.gps_week(d), 2364)
        self.assertEqual(mgex.product_for(d)[0], 'GFZ0MGXRAP_20251200000_01D_05M_ORB.SP3')

    def test_reuse_cached_sp3_without_network_or_password(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.args(td)
            path = a.orbit_dir / mgex.product_for(mgex.date_from_rinex_filename(a.rinex))[0]
            sp3(path)
            with patch.object(mgex, 'main', side_effect=AssertionError('download attempted')):
                report = runner.resolve_orbit(a)
            self.assertEqual(report['status'], 'LOCAL_CACHE')
            self.assertEqual(report['satellites_by_system'], dict.fromkeys('GREC', 1))
            self.assertEqual(a.sp3, path)

    def test_missing_offline_must_not_download(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.args(td)
            a.no_download = True
            with patch.object(mgex, 'main', side_effect=AssertionError('network attempted')):
                with self.assertRaises(FileNotFoundError):
                    runner.resolve_orbit(a)

    def test_explicit_orbit_does_not_download(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.args(td)
            a.sp3 = Path(td)/'manual.sp3'
            sp3(a.sp3)
            with patch.object(mgex, 'main', side_effect=AssertionError('network attempted')):
                report = runner.resolve_orbit(a)
            self.assertEqual(report['status'], 'EXPLICIT_SP3')

    def test_missing_calls_downloader_once(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.args(td)
            a.earthdata_user = 'example-user'
            def fake_download(argv):
                self.assertIn('--username', argv)
                sp3(a.orbit_dir / mgex.product_for(mgex.date_from_rinex_filename(a.rinex))[0])
                return 0
            with patch.object(mgex, 'main', side_effect=fake_download) as mock:
                report = runner.resolve_orbit(a)
            mock.assert_called_once()
            self.assertEqual(report['status'], 'CDDIS_DOWNLOAD')

    def test_force_download_even_if_cached(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.args(td)
            a.force_download = True
            sp3(a.orbit_dir / mgex.product_for(mgex.date_from_rinex_filename(a.rinex))[0])
            with patch.object(mgex, 'main', return_value=0) as mock:
                report = runner.resolve_orbit(a)
            self.assertIn('--force', mock.call_args.args[0])
            self.assertEqual(report['status'], 'CDDIS_FORCED_REFRESH')

    def test_bad_cached_file_redownloaded(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.args(td)
            target = a.orbit_dir / mgex.product_for(mgex.date_from_rinex_filename(a.rinex))[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('<html>login required</html>')
            def fake_download(argv):
                self.assertIn('--force', argv)
                sp3(target)
                return 0
            with patch.object(mgex, 'main', side_effect=fake_download):
                report = runner.resolve_orbit(a)
            self.assertEqual(report['status'], 'CDDIS_REPLACED_INVALID_CACHE')

if __name__ == '__main__':
    unittest.main()
