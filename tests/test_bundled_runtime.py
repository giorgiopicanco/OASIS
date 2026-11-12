import unittest
from pyOASIS import OASISConfig
from pyOASIS.cli import build_parser
from pyOASIS.runtime import bundled_medusa_root
from pyOASIS.gnss.medusa_b111 import _core
from pathlib import Path


class BundledRuntimeTests(unittest.TestCase):
    def test_default_paths(self):
        self.assertIsNone(OASISConfig(rinex=Path('sample.rnx')).medusa_root)
        self.assertEqual(build_parser().parse_args(['--rinex', 'sample.rnx']).medusa_root,
                         bundled_medusa_root())

    def test_bundled_qc_import(self):
        builder, qc, chooser, offset, config = _core(bundled_medusa_root())
        self.assertTrue(all(callable(f) for f in (builder, qc, chooser, offset)))
        self.assertIs(config['offset_func'], offset)
        from importlib import import_module
        self.assertTrue(callable(import_module(builder.__module__).build_rinex_updarc))
