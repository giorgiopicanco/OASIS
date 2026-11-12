"""Stage 24 regression, arc-safety and output tests."""
from types import SimpleNamespace as NS
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import h5py
from scipy.signal import savgol_filter
from test_stage23_products import satellite
from pyOASIS.gnss.ionosphere_products import gpst_array_to_utc_ms, compute_indices
from pyOASIS.gnss.detrended_tec import compute_detrended_tec
from pyOASIS.io.hdf5 import save_lgf_hdf5, load_lgf_hdf5_summary
from pyOASIS.visualization.oasis_figures import render_hdf5_figures
from pyOASIS.cli import build_parser, validate_inputs


def fixture(cadence=30, count=180):
    s = satellite(count=count)
    s.times = np.datetime64('2025-04-30T00:00:18') + np.arange(count)*np.timedelta64(cadence,'s')
    s.b111.times = s.times
    t = gpst_array_to_utc_ms(s.times)
    v = 25 + np.arange(count)*.02 + .5*np.sin(np.arange(count)*2*np.pi/10)
    d = dict(utc_ms=t, value=v, arc_id=s.b111.arc_ids.copy(),
             elevation_deg=np.full(count, 45.), ipp_lat_deg=np.full(count,-22.),
             ipp_lon_deg=np.full(count,-45.), ipp_height_km=np.full(count,350.),
             physical_nonnegative=np.ones(count,bool))
    return NS(satellites={'G01':s}), {'G01':d}


class Stage24Tests(unittest.TestCase):
    def test_native_cadence_and_reference_filter(self):
        for cadence, width in ((1,901),(5,181),(15,61),(30,31),(60,15)):
            n=width*3
            geo, v = fixture(cadence,n)
            result,diag=compute_detrended_tec(geo,v)
            d=result['G01']; half=width//2
            bg=savgol_filter(v['G01']['value'],width,1)[half:-half]
            np.testing.assert_allclose(d['background_vtec_tecu'], bg, atol=1e-10)
            np.testing.assert_allclose(d['value'],v['G01']['value'][half:-half]-bg,atol=1e-10)
            self.assertEqual(len(d['value']),n-width+1)
            self.assertEqual(d['window_samples'][0],width)
            self.assertEqual(diag['edge_samples_omitted'],width-1)
            # Zero residual for a pure linear trend.
            v['G01']['value']=20+np.arange(n)*.1
            d=compute_detrended_tec(geo,v)[0]['G01']
            np.testing.assert_allclose(d['value'],0,atol=1e-10)
    def test_no_cross_arc_or_bad_epoch_or_final_slip(self):
        geo,v=fixture(count=240); d=v['G01']; s=geo.satellites['G01']
        s.b111.arc_ids[120:]=1; d['arc_id'][120:]=1
        d['value'][:120]=20; d['value'][120:]=100
        s.b111.flags_final[60]=1
        s.b111.flags_final[180]=4
        s.b111.original_b111_result[0].final_updarc=NS(arcs=[NS(slip_indices=[90])])
        out,_=compute_detrended_tec(geo,v); r=out['G01']; j=r['source_epoch_index']
        np.testing.assert_allclose(r['value'],0,atol=1e-10)
        for event in (60,90,120,180): self.assertFalse(np.any((j-15<event)&(j+15>=event)))
        self.assertFalse(np.any(np.abs(j-60)<=15))
        np.testing.assert_array_equal(r['arc_id'],s.b111.arc_ids[j])
    def test_missing_negative_low_elevation_and_short_segments(self):
        geo,v=fixture(count=100); d=v['G01'];
        d['value'][25]=-1; d['physical_nonnegative'][25]=False; d['elevation_deg'][50]=20
        # A missing VTEC row cannot be bridged by the filter.
        for k in list(d): d[k]=np.delete(d[k],75)
        out,_=compute_detrended_tec(geo,v)
        self.assertFalse(out)
    def test_empty_and_provenance_errors(self):
        geo,v=fixture(); self.assertEqual(compute_detrended_tec(geo,{})[0],{})
        v['G01']['arc_id'][5]=9
        with self.assertRaisesRegex(ValueError,'arc provenance'): compute_detrended_tec(geo,v)
    def test_hdf5_two_distinct_products_and_optional_figures(self):
        geo,v=fixture(count=180); s=geo.satellites['G01']; t=s.times
        geo.sp3_path='test.sp3'; geo.status='fixture'; geo.b111=NS(status='original',skipped={},glonass_channel_sources={},glonass_channel_conflicts={})
        obs=NS(times=t,time_system='GPS',marker='TEST',source='test.rnx',version=3.04,declared_signals={},glonass_channels={},satellites=['G01'],epoch_flags=np.zeros(len(t),int),get=lambda *args:None)
        old=compute_indices(geo,{'DELTATEC'}); new,diag=compute_detrended_tec(geo,v)
        with TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'TEST_2025_120_ION.h5'
            save_lgf_hdf5(path,geo,obs,extra_products={**old,'DTEC':new},report={'tec_diagnostics':{'status':'EXPERIMENTAL_MODEL_CALIBRATION'},'dtec_diagnostics':diag})
            with h5py.File(path) as h:
                self.assertEqual(h.attrs['format_version'],'2.1')
                self.assertEqual(h['products/DELTATEC'].attrs['display_name'],'ΔTEC')
                self.assertEqual(h['products/DTEC'].attrs['display_name'],'dTEC')
                r=h['products/DTEC/G01']; self.assertEqual(r['source_epoch_index'].dtype.kind,'i')
                np.testing.assert_allclose(r['value'][:],r['vtec_tecu'][:]-r['background_vtec_tecu'][:])
            self.assertEqual(load_lgf_hdf5_summary(path)['available_products'],['LGF','DELTATEC','DTEC'])
            no=render_hdf5_figures(path,root/'none',overview_figures=False,max_qc_satellites=0,dpi=72)
            self.assertEqual(no,[])
            paths=render_hdf5_figures(path,root/'yes',max_qc_satellites=1,qc_elevation=True,qc_zoom=True,dpi=72)
            self.assertTrue(any(p.endswith('_deltaTEC.png') for p in paths)); self.assertTrue(any(p.endswith('_dTEC.png') for p in paths)); self.assertTrue(any(p.endswith('_RNX_QC.png') for p in paths))
            # Old Stage 23 HDF5 must not be interpreted as new dTEC.
            with h5py.File(path,'r+') as h:
                del h['products/DTEC']; h['products'].move('DELTATEC','DTEC'); h.attrs['format_version']='2.0'
            paths=render_hdf5_figures(path,root/'legacy',max_qc_satellites=0,dpi=72)
            self.assertTrue(any(p.endswith('_deltaTEC.png') for p in paths)); self.assertFalse(any(p.endswith('_dTEC.png') for p in paths))
    def test_selection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); rinex=root/'test.rnx'; rinex.touch(); m=root/'tools/historical/b111'; m.mkdir(parents=True); (m/'rinex_driver.py').touch()
            parser=build_parser(); args=parser.parse_args(['--rinex',str(rinex),'--medusa-root',str(root),'--products','ΔTEC,dTEC','--qc-elevation','--qc-zoom'])
            validate_inputs(args,parser); self.assertEqual(args.products,'DELTATEC,DTEC'); self.assertTrue(args.qc_elevation); self.assertTrue(args.qc_zoom)

if __name__=='__main__': unittest.main()
