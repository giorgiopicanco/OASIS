import unittest
import numpy as np
from pyOASIS.gnss.station_statistics import station_statistics

class StationStatisticsTests(unittest.TestCase):
    def test_support_gaps_and_negative(self):
        t=np.arange(80)*30000+12000
        data={f'G{i:02}':dict(utc_ms=t,value=np.full(len(t),float(i)),elevation_deg=np.full(len(t),40.)) for i in range(1,5)}
        data['G01']['value'][40]=-1
        s=station_statistics(data,'VTEC',grid_ms=t,cadence_ms=30000,smooth_minutes=15)
        self.assertEqual(s['count'][40],3); self.assertTrue(np.isnan(s['selected'][40])); self.assertEqual(s['mean'][40],3)
        self.assertTrue(np.isnan(s['smoothed'][25:56]).all()); self.assertAlmostEqual(s['smoothed'][20],2.5)
        s=station_statistics(data,'DTEC',grid_ms=t,cadence_ms=30000)
        self.assertEqual(s['count'][40],4)
    def test_delta_grid_and_median(self):
        data={f'G{i:02}':dict(window_start_utc_ms=[0],value=[v],elevation_deg=[30]) for i,v in enumerate([1,2,3,100])}
        s=station_statistics(data,'DELTATEC',grid_ms=[12000,312000],cadence_ms=30000,method='median')
        np.testing.assert_array_equal(s['utc_ms'],[75000,225000,375000]); self.assertEqual(s['selected'][0],2.5); self.assertTrue(np.isnan(s['selected'][1:]).all())
