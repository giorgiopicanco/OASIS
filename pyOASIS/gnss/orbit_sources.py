"""Read-only multiple-SP3 orbit resolver with explicit satellite provenance.

The first source is the primary SP3. Supplemental SP3 files are used only if
that source cannot supply an actual interpolated coordinate for a satellite at
its signal transmission time. No satellite positions are manufactured.
"""
from __future__ import annotations

from pathlib import Path

from .medusa_processing import Sp3Orbit


class CompositeSp3Orbit:
    """Provide Sp3Orbit.position() while retaining which SP3 supplied it."""

    def __init__(self, paths, *, orbit_factory=Sp3Orbit):
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError('At least one SP3 orbit source is required')
        normalized = [str(path.resolve()) for path in self.paths]
        if len(normalized) != len(set(normalized)):
            raise ValueError('Duplicate SP3 orbit source paths are not allowed')
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f'SP3 orbit source not found: {path}')
        self.sources = [(path, orbit_factory(path)) for path in self.paths]
        self.records = {}
        for path, orbit in self.sources:
            for satellite in orbit.records:
                self.records.setdefault(satellite, []).append(str(path))
        self.last_source = None

    def position(self, satellite, time):
        self.last_source = None
        for path, orbit in self.sources:
            result = orbit.position(satellite, time)
            if result is not None:
                self.last_source = str(path)
                return result
        return None

    def candidates(self, satellite):
        """List actual files containing orbit records for a satellite."""
        return tuple(self.records.get(satellite, ()))

    def inventory(self):
        return {
            'files': [str(p) for p in self.paths],
            'satellites_by_source': {
                str(path): sorted(orbit.records)
                for path, orbit in self.sources
            },
            'union_satellites': sorted(self.records),
        }
