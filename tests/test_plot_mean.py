import unittest
import numpy as np
from pyOASIS.visualization.oasis_figures import station_mean, elevation_mask
from pyOASIS.cli import build_parser

class PlotMeanTests(unittest.TestCase):
    def test_equal_satellite_weight_and_mask(self):
        data={'G01':{'utc_ms':[0,0,30000], 'value':[2,4,8], 'elevation_deg':[20,30,19]},
              'R01':{'utc_ms':[0,30000], 'value':[9,10], 'elevation_deg':[40,np.nan]}}
        t,y,n=station_mean(data,'VTEC',20)
        np.testing.assert_array_equal(t,[0]); np.testing.assert_allclose(y,[6]); np.testing.assert_array_equal(n,[2])
        np.testing.assert_allclose(station_mean(data,'VTEC',35)[1],[9])
    def test_delta_window(self):
        data={'G01':{'utc_ms':[60000], 'window_start_utc_ms':[0], 'value':[2], 'elevation_deg':[30]},
              'R01':{'utc_ms':[90000], 'window_start_utc_ms':[0], 'value':[4], 'elevation_deg':[30]}}
        t,y,n=station_mean(data,'DELTATEC'); np.testing.assert_array_equal(t,[75000]); np.testing.assert_allclose(y,[3]); np.testing.assert_array_equal(n,[2])
    def test_options(self):
        a=build_parser().parse_args(['--rinex','x']); self.assertEqual(a.plot_min_elevation,20); self.assertTrue(a.mean_line)
        a=build_parser().parse_args(['--rinex','x','--plot-min-elevation','35','--no-mean-line']); self.assertEqual(a.plot_min_elevation,35); self.assertFalse(a.mean_line)
        with self.assertRaises(ValueError): elevation_mask([20],float('nan'))
