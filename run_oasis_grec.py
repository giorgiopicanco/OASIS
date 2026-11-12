"""Backward-compatible command entry point for the pyOASIS GNSS library.

All processing logic is maintained in pyOASIS.cli; this script is retained to
support existing stage-18/19/20 PowerShell commands and user workflows.
"""
from pyOASIS.cli import build_parser, validate_inputs, resolve_orbit, run, main

__all__ = ['build_parser', 'validate_inputs', 'resolve_orbit', 'run', 'main']

if __name__ == '__main__':
    main()
