import unittest
from types import SimpleNamespace as NS
import numpy as np
from pyOASIS.gnss.medusa_roti_adapter import compute_medusa_roti
from pyOASIS.gnss.medusa_v6_window import window_records

class V6Tests(unittest.TestCase):
    def fixture(self, cadence=30, slip=None, missing=False, extreme=False):
        n=300//cadence
        times=np.datetime64('2025-04-30T00:00:18')+np.arange(n)*np.timedelta64(cadence,'s')
        f1,f2=1575.42e6,1227.60e6
        k=40.3e16*(1/f2**2-1/f1**2)
        stec=np.arange(n,dtype=float)*.1
        if extreme: stec[2]+=100
        flags=np.zeros(n,int)
        if missing: flags[2]=1
        arc=NS(start=0,end=n-1,slip_indices=[] if slip is None else [slip])
        up=NS(flags=flags,arcs=[arc]); result=NS(source_indices=np.arange(n),final_updarc=up)
        b=NS(f1_hz=f1,f2_hz=f2,signals=('L1C','L2W'),l4_leveled_m=stec*k,arc_ids=np.zeros(n,int),flags_final=flags,slip=np.zeros(n),source_indices=np.arange(n),original_b111_result=(result,))
        sat=NS(b111=b,times=times,geometry_valid=np.zeros(n,bool),geometry={})
        return NS(satellites={'R27':sat})
    def test_fixed_windows_missing_geometry(self):
        for cadence in (15,30):
            rows,_=compute_medusa_roti(self.fixture(cadence))
            self.assertEqual(len(rows),2)
            for r in rows:
                self.assertEqual(r['window_start_utc_ms']%150000,0)
                self.assertEqual(r['window_seconds'],150)
                self.assertAlmostEqual(r['roti_tecu_min'],0,places=12)
                self.assertEqual(r['rot_count'],150//cadence-1)
                self.assertTrue(r['ion_flags']&1)
                self.assertTrue(np.isnan(r['elevation_deg']))
    def test_rejected_window(self):
        for kwargs in ({'slip':2},{'missing':True},{'extreme':True}):
            rows,_=compute_medusa_roti(self.fixture(**kwargs))
            self.assertEqual(len(rows),1)
    def test_reference_window(self):
        t=1745971200000
        seg=[(t+30000*i,t+30000*i,float(i*i)*.1,0.,0.,10.,0) for i in range(5)]
        rows,_=window_records([seg],t,'X','G01','1:2')
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0][7],np.std([.2,.6,1.,1.4]))
        self.assertEqual(rows[0][13],10.)
if __name__=='__main__': unittest.main()
