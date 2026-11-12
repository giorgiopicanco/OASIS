"""Regression tests for display-only limits and compact outputs."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from pyOASIS.visualization import oasis_figures as f


class FastOutputTests(unittest.TestCase):
    def test_tec_limits_follow_only_visible_values(self):
        plt, _ = f._plot_imports()
        t = np.arange(4, dtype=np.int64) * 30000
        for product in ('VTEC', 'STEC'):
            for max_visible in (95., 173.):
                data = {'G01': dict(utc_ms=t,
                    value=np.array([max_visible, 999., -3., 10.]),
                    elevation_deg=np.array([20., 19.9, 45., 50.]),
                    geometry_available=np.array([True, True, True, True]))}
                with TemporaryDirectory() as folder:
                    def check(fig, path, dpi):
                        ax = fig.axes[0]
                        self.assertEqual(ax.get_ylim(), (0., max(120., max_visible)))
                        offsets = np.asarray(ax.collections[0].get_offsets())
                        np.testing.assert_equal(offsets[:, 1], [max_visible, -3., 10.])
                        plt.close(fig)
                        return Path(path)
                    with patch.object(f, '_finish', side_effect=check):
                        f.render_oasis_figures(NS(satellites={}), None, folder,
                            station='TEST', doy='001', year='2025', max_qc_satellites=0,
                            extra_products={product: data}, figure_products={product},
                            min_elevation=20., mean_line=False)

    def test_roti_display_limit_and_elevation(self):
        plt, _ = f._plot_imports()
        rows = [dict(satellite='G01', utc_ms=150000*i, window_seconds=150,
                     signal_pair='L1-L2', elevation_deg=e, roti_tecu_min=v)
                for i,(e,v) in enumerate([(19.9, 2.), (20., 3.), (45., 7.)])]
        with TemporaryDirectory() as folder:
            def check(fig, path, dpi):
                ax = fig.axes[0]
                self.assertEqual(ax.get_ylim(), (0., 5.))
                np.testing.assert_equal(np.asarray(ax.collections[0].get_offsets())[:,1], [3., 7.])
                plt.close(fig)
                return Path(path)
            with patch.object(f, '_finish', side_effect=check):
                f.plot_roti_windows(rows, folder, station='TEST', doy='001', year='2025', min_elevation=20.)

if __name__ == '__main__':
    unittest.main()
