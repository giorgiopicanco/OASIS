"""Stage 23 unit and schema tests (no network or real MEDUSA data required)."""
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import unittest
import numpy as np
import h5py
from pyOASIS.gnss.ionosphere_products import (
    gpst_array_to_utc_ms, compute_indices, compute_calibrated_tec, aggregate_sidx)
from pyOASIS.gnss.rnx_qc import diagnostic_masks, diagnostic_arrays
from pyOASIS.io.hdf5 import save_lgf_hdf5, load_lgf_hdf5_summary
from pyOASIS.visualization.oasis_figures import SYSTEMS, render_hdf5_figures
from pyOASIS.cli import build_parser, validate_inputs

FREQ={'G':(1575.42e6,1227.6e6),'R':(1599.1875e6,1243.8125e6),'E':(1575.42e6,1176.45e6),'C':(1561.098e6,1207.14e6)}


def satellite(sv='G01', stec=None, *, missing=False, count=10):
    f1,f2=FREQ[sv[0]]; k=40.3e16*(1/f2**2-1/f1**2)
    if stec is None: stec=np.arange(count,dtype=float)
    n=len(stec); times=np.datetime64('2025-04-30T00:00:18')+np.arange(n)*np.timedelta64(30,'s')
    x=np.asarray(stec)*k; flags=np.zeros(n,dtype=int); arc=np.zeros(n,dtype=int)
    stage=NS(flags=flags.copy(),slip_indices=[2],slip_cycles=[4.],cycl1=[3.],cycl2=[-1.])
    src=NS(source_indices=np.arange(n),initial_updarc=NS(flags=flags.copy()),chkwub_result=stage,chkl4p_result=stage)
    b=NS(satellite=sv,signals=('L1C','L2W','C1C','C2W'),source_observables=('L1C','L2W','C1C','C2W'),
         f1_hz=f1,f2_hz=f2,times=times,l4_raw_m=x+2.,l4_corrected_m=x+2.,l4_leveled_m=x,
         code_gf_m=x,mw_raw_cycles=x,mw_corrected_cycles=x,flags_final=flags,arc_ids=arc,
         arc_bias_m=np.full(n,2.),residual_m=np.zeros(n),slip=(np.arange(n)==2),
         source_indices=np.arange(n),original_b111_result=(src,))
    g={key:np.full(n,np.nan if missing else val) for key,val in [('ipp_lat_deg',-22.),('ipp_lon_deg',-45.),('elevation_deg',45.),('mapping_factor',1.3)]}
    valid=np.full(n,not missing,dtype=bool)
    return NS(b111=b,times=times,l4_leveled_m=x,arc_ids=arc,geometry=g,geometry_valid=valid,
              orbit_available=valid,above_horizon=valid,orbit_source=np.array(['fixture']*n))


class TestStage23(unittest.TestCase):
    def test_time_scale(self):
        t=np.array(['2025-04-30T00:00:00','2025-04-30T00:00:30'],dtype='datetime64[ms]')
        np.testing.assert_array_equal(gpst_array_to_utc_ms(t).astype('datetime64[ms]'),
                                      np.array(['2025-04-29T23:59:42','2025-04-30T00:00:12'],dtype='datetime64[ms]'))
    def test_frequency_scaling_and_windows(self):
        geo=NS(satellites={sv:satellite(sv) for sv in ('C01','E01','R01','G01')})
        products=compute_indices(geo,{'DELTATEC','SIDX'})
        self.assertEqual(list(products['DELTATEC']),['G01','R01','E01','C01'])
        for sv in geo.satellites:
            np.testing.assert_allclose(products['DELTATEC'][sv]['value'],[10.,10.])
            np.testing.assert_allclose(products['SIDX'][sv]['value'],[2.]*5)
        station=aggregate_sidx(products['SIDX'])
        np.testing.assert_allclose(station['G']['value_mtecu_s'],[2000/60]*5)
        self.assertEqual(list(station),list('GREC'))
    def test_reject_gap_and_slip_without_requiring_ipp(self):
        sat=satellite('R27',missing=True)
        sat.b111.flags_final[2]=4
        p=compute_indices(NS(satellites={'R27':sat}),{'DELTATEC','SIDX'})
        self.assertEqual(len(p['DELTATEC']['R27']['value']),1)
        self.assertFalse(p['DELTATEC']['R27']['geometry_available'].any())
        self.assertFalse(p['SIDX']['R27']['original_elevation_ok'].any())
        sat.b111.flags_final[2]=1
        p=compute_indices(NS(satellites={'R27':sat}),{'DELTATEC'})
        self.assertEqual(len(p['DELTATEC']['R27']['value']),1)
    def test_masks_are_original_flags(self):
        sat=satellite(); b=sat.b111
        b.flags_final[0]=-1; b.flags_final[1]=1; b.flags_final[2]=4; b.flags_final[3]=2
        masks=diagnostic_masks(b)
        self.assertTrue(masks['excluded_before_final_qc'][0]); self.assertTrue(masks['outlier_or_bad'][1]); self.assertTrue(masks['corrected_cycle_slip'][2]); self.assertTrue(masks['new_ambiguity'][3]); self.assertFalse(masks['outlier_or_bad'][0])
        arrays=diagnostic_arrays(b)
        self.assertEqual(arrays['phase1_slip_cycles'][2],3.)
        self.assertEqual(arrays['phase2_slip_cycles'][2],-1.)
        self.assertTrue(np.isnan(arrays['phase1_slip_cycles'][1]))
    def test_rank_deficient_model(self):
        p,r=compute_calibrated_tec(NS(satellites={'G01':satellite()}),-22,-45)
        self.assertFalse(p['STEC']); self.assertNotEqual(r['status'],'EXPERIMENTAL_MODEL_CALIBRATION')
    def test_model_recovers_known_solution(self):
        rng=np.random.default_rng(17); sats={}; expected={}
        for sv in ('G01','R01','E01','C01'):
            n=100; u=np.arange(n)*30/3600
            lat=-22+rng.uniform(-7,7,n); lon=-45+rng.uniform(-7,7,n)
            mapping=rng.uniform(1.1,2.5,n)
            a=np.radians(lon+45); b=np.radians(lat+22)
            vtec=30+5*u+10*a+2*u*a-4*b+3*u*b
            stec=vtec*mapping
            sat=satellite(sv,stec=stec+100)
            sat.geometry.update(ipp_lat_deg=lat,ipp_lon_deg=lon,mapping_factor=mapping)
            sats[sv]=sat; expected[sv]=(stec,vtec)
        p,r=compute_calibrated_tec(NS(satellites=sats),-22,-45)
        self.assertEqual(r['status'],'EXPERIMENTAL_MODEL_CALIBRATION')
        for sv,(stec,vtec) in expected.items():
            np.testing.assert_allclose(p['STEC'][sv]['value'],stec,atol=1e-7)
            np.testing.assert_allclose(p['VTEC'][sv]['value'],vtec,atol=1e-7)
        self.assertEqual(r['negative_stec_count'],0)
    def test_canonical_hdf5_and_figure_switches(self):
        sats={sv:satellite(sv) for sv in ('C01','E01','R01','G01')}
        t=sats['G01'].times
        qc=NS(status='original',skipped={},glonass_channel_sources={},glonass_channel_conflicts={})
        geo=NS(satellites=sats,sp3_path='test.sp3',status='fixture',b111=qc)
        obs=NS(times=t,time_system='GPS',marker='TEST',source='test.rnx',version=3.04,declared_signals={},glonass_channels={},satellites=list(sats),epoch_flags=np.zeros(len(t),int),get=lambda *args:None)
        extra=compute_indices(geo,{'DELTATEC','SIDX'})
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'test.h5'; save_lgf_hdf5(f,geo,obs,extra_products=extra)
            with h5py.File(f) as h:
                self.assertEqual(h.attrs['format_version'],'2.1')
                self.assertEqual(list(h['satellites']),['G01','R01','E01','C01'])
                self.assertNotIn('medusa_b111',h['satellites/G01'])
                self.assertIn('lgf_leveled_m',h['satellites/G01/rnx_qc'])
                self.assertIn('pgf_raw_m',h['satellites/G01/rnx_qc'])
                self.assertEqual(h['satellites/G01/rnx_qc/stage_diagnostics/phase1_slip_cycles'][2],3.)
                self.assertIn('station_products/SIDX/G',h)
                self.assertEqual(h['epochs/gpst_minus_utc_s'][0],18)
            self.assertEqual(load_lgf_hdf5_summary(f)['available_products'],['LGF','DELTATEC','SIDX'])
            self.assertEqual(render_hdf5_figures(f,Path(tmp)/'none',max_qc_satellites=0,overview_figures=False,dpi=72),[])
            paths=render_hdf5_figures(f,Path(tmp)/'qc',max_qc_satellites=1,overview_figures=False,dpi=72)
            self.assertEqual(len(paths),1); self.assertTrue(paths[0].endswith('G01_120_2025_RNX_QC.png'))
            self.assertEqual([v[1] for v in SYSTEMS.values()],['blue','red','lime','violet'])
    def test_parser_and_product_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rinex=root/'test.rnx'; rinex.touch(); m=root/'tools/historical/b111'; m.mkdir(parents=True); (m/'rinex_driver.py').touch()
            parser=build_parser(); args=parser.parse_args(['--rinex',str(rinex),'--medusa-root',str(root),'--products','ALL','--figures','--no-qc-figures','--no-overview-figures','--systems','CERG'])
            validate_inputs(args,parser)
            self.assertEqual(args.products,'LGF,ROTI,DELTATEC,SIDX,STEC,VTEC,DTEC'); self.assertEqual(args.systems,'GREC'); self.assertFalse(args.qc_figures); self.assertFalse(args.overview_figures)

if __name__=='__main__': unittest.main()
