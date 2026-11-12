import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pyOASIS.visualization import oasis_figures as figures

class SidxRestoreTests(unittest.TestCase):
    def test_original_constellation_lines_in_both_modes(self):
        times = np.array([0, 60000, 180000], dtype=np.int64)
        data = {}
        for system in 'GREC':
            for i, value in enumerate([6., 18., 900.], 1):
                data[f'{system}{i:02}'] = dict(window_start_utc_ms=times,
                    utc_ms=times+30000, value=np.full(3,value),
                    elevation_deg=np.full(3, 10. if i == 3 else 45.))
        plt, _ = figures._plot_imports()
        for mode in ('constellation','blue_red'):
            with tempfile.TemporaryDirectory() as folder:
                def check(fig, path, dpi):
                    ax = fig.axes[0]
                    self.assertEqual(len(ax.collections),0)
                    self.assertEqual(len(ax.lines),8)  # two segments per system
                    for j, (name,color,_) in enumerate(figures.SYSTEMS.values()):
                        self.assertEqual(ax.lines[2*j].get_label(),name)
                        for line in ax.lines[2*j:2*j+2]:
                            self.assertEqual(line.get_color(),color)
                            np.testing.assert_allclose(line.get_ydata(),200.)
                    plt.close(fig)
                    return Path(path)
                with patch.object(figures,'_finish',side_effect=check):
                    figures.render_oasis_figures(SimpleNamespace(satellites={}),None,folder,
                        station='TEST',doy='001',year='2025',max_qc_satellites=0,
                        extra_products={'SIDX':data},figure_products=('SIDX',),
                        min_elevation=20.,color_mode=mode,mean_line=True)
