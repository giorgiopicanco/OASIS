"""Regression for isolated OFFSET failures; no changes to MEDUSA core."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from pyOASIS.gnss import medusa_b111 as b

class OffsetRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.indices = [np.array([0,1]), np.array([2,3]), np.array([4,5])]
        self.pre = NS(source_indices=np.arange(6), updarc_result=NS(
            flags=np.zeros(6,dtype=int), arcs=[NS(start=i,end=i+1) for i in (0,2,4)]))
        self.valid = NS(source_indices=np.arange(2), leveling_result=NS(
            leveled_gf_m=np.array([4.,5.]), arc_bias_m=np.zeros(2), residual_m=np.zeros(2), arc_id=np.zeros(2,dtype=int)),
            final_updarc=NS(observations=np.array([4.,5.]),flags=np.zeros(2,dtype=int)),
            chkwub_result=NS(observations=np.zeros(2)), chkl4p_result=NS(slip_indices=[]))

    def test_reject_only_failed_arc(self):
        for ntot in (0,1):
            with self.subTest(ntot=ntot), patch.object(b,'_slice_combination',side_effect=lambda c,i:i):
                def qc(segment, **cfg):
                    if segment[0] == 2:
                        raise ValueError(f'Bernese OFFSET requires at least two usable observations (NTOT={ntot})')
                    return self.valid
                data,flags,ids,slips,processed,results,rejected=b._process_arc_segments(None,self.pre,qc,{},6)
                np.testing.assert_equal(processed,[0,1,4,5])
                np.testing.assert_equal(ids,[0,0,-1,-1,1,1])
                np.testing.assert_equal(flags,[0,0,-1,-1,0,0])
                self.assertTrue(np.isnan(data['l4_leveled_m'][2:4]).all())
                self.assertEqual(len(results),2)
                self.assertEqual(rejected,[(2,3,ntot,'OFFSET_INSUFFICIENT_USABLE')])

    def test_other_errors_propagate(self):
        with patch.object(b,'_slice_combination',return_value=None):
            def qc(segment, **cfg): raise ValueError('Unrelated failure')
            with self.assertRaisesRegex(ValueError,'Unrelated failure'):
                b._process_arc_segments(None,self.pre,qc,{},6)

    def test_all_arcs_rejected(self):
        with patch.object(b,'_slice_combination',return_value=None):
            def qc(segment, **cfg): raise ValueError('Bernese OFFSET requires at least two usable observations (NTOT=0)')
            data,flags,ids,slips,processed,results,rejected=b._process_arc_segments(None,self.pre,qc,{},6)
            self.assertFalse(results)
            self.assertEqual(len(rejected),3)
            self.assertTrue(np.isnan(data['l4_leveled_m']).all())
            self.assertEqual(len(processed),0)

class BatchResumeTests(unittest.TestCase):
    def test_existing_hdf5_skips_processing(self):
        import main_rbmc_rinex2 as batch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rinex = root/'rinex'; rinex.mkdir()
            (rinex/'amte3161.25o').touch()
            medusa = root/'medusa'; medusa.mkdir()
            orbits = root/'orbits'; orbits.mkdir()
            channels = orbits/'channels.json'; channels.write_text('{}')
            out = root/'out'
            dest = out/'AMTE'/'2025'/'316'/'AMTE_2025_316_ION.h5'
            dest.parent.mkdir(parents=True); dest.touch()
            with patch.multiple(batch,RINEX_DIR=rinex,MEDUSA_ROOT=medusa,ORBIT_DIR=orbits,
                                GLONASS_CHANNELS=channels,OUTPUT_DIR=out,OVERWRITE=False,SAVE_HDF5=True), patch.object(batch,'process_station') as process:
                self.assertEqual(batch.main(),0)
                process.assert_not_called()
