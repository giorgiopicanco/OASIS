"""Regression tests for station/day organization and opt-in figures."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
from pyOASIS.cli import build_parser, daily_output_dir, parse_figure_products
from pyOASIS.api import OASISConfig
from pyOASIS.visualization.oasis_figures import render_oasis_figures

class OutputLayoutTests(unittest.TestCase):
    def test_daily_path(self):
        self.assertEqual(daily_output_dir('OUTPUT','chpi',2025,120),Path('OUTPUT/CHPI/2025/120'))
        self.assertEqual(daily_output_dir('OUTPUT','boav',2023,49),Path('OUTPUT/BOAV/2023/049'))
        with self.assertRaises(ValueError): daily_output_dir('OUTPUT','../x',2025,120)
    def test_cli_api_defaults(self):
        args=build_parser().parse_args(['--rinex','test.rnx','--figures'])
        self.assertFalse(args.qc_figures)
        self.assertEqual(args.out,Path('OUTPUT'))
        config=OASISConfig(Path('x'),Path('m'),Path('o'))
        self.assertFalse(config.qc_figures)
        self.assertEqual(config.output_dir,Path('OUTPUT'))
        self.assertEqual(parse_figure_products('LGF,ΔTEC,dTEC'),{'LGF','DELTATEC','DTEC'})
    def test_select_overview_and_qc_directory(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            geo=SimpleNamespace(satellites={})
            with patch('pyOASIS.visualization.oasis_figures.plot_lgf_overview',return_value=None) as lgf, patch('pyOASIS.visualization.oasis_figures.plot_ipp_overview',return_value=None) as ipp, patch('pyOASIS.visualization.oasis_figures.plot_rnx_qc',return_value=[]) as qc:
                render_oasis_figures(geo,None,root,station='TEST',year='2025',doy='120',figure_products={'LGF'},max_qc_satellites=0,qc_directory=root/'QC')
                lgf.assert_called_once(); ipp.assert_not_called()
                self.assertEqual(qc.call_args.args[1],root/'QC')
                self.assertEqual(qc.call_args.kwargs['max_qc_satellites'],0)
                self.assertFalse((root/'QC').exists())
