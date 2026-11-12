"""Small supported Python API; future scientific products can use the same pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from .cli import build_parser, run, validate_inputs

SUPPORTED_PRODUCTS = frozenset({'LGF','ROTI','DELTATEC','SIDX','STEC','VTEC','DTEC'})



@dataclass(frozen=True)
class OASISConfig:
    rinex: Path
    medusa_root: Path | None = None
    orbit_dir: Path = Path('INPUT/ORBITS')
    output_dir: Path = Path('OUTPUT')
    systems: str = 'GREC'
    products: tuple[str, ...] = ('LGF',)
    figures: bool = False
    qc_figures: bool = False
    figure_products: tuple[str, ...] | None = None
    overview_figures: bool = True
    qc_elevation: bool = False
    qc_zoom: bool = False
    save_hdf5: bool = True
    primary_sp3: Path | None = None
    supplemental_sp3: tuple[Path, ...] = ()
    figure_dpi: int = 300
    plot_min_elevation: float = 20.0
    mean_line: bool = True
    mean_min_satellites: int = 4
    mean_method: str = "mean"
    mean_smooth_minutes: float = 0.  # Legacy option; no extra curve is drawn.
    product_figure_color_mode: str = "constellation"
    max_qc_figures: int | None = None
    overwrite_output: bool = False
    require_complete_geometry: bool = False
    offline: bool = False
    glonass_channels: Path | None = None


def process(config: OASISConfig) -> dict:
    """Process GNSS observations using the same validated CLI argument contract."""
    products = tuple(p.replace("Δ", "DELTA").upper() for p in config.products)
    if not products or not set(products).issubset(SUPPORTED_PRODUCTS | {'ALL','TEC'}):
        raise ValueError(f'Available products: {sorted(SUPPORTED_PRODUCTS)}; ALL and TEC aliases are supported.')
    args = ['--rinex', str(config.rinex),
            '--orbit-dir', str(config.orbit_dir), '--out', str(config.output_dir),
            '--systems', config.systems,
            '--products', ','.join(products),
            '--save-output', 'hdf5' if config.save_hdf5 else 'none',
            '--figure-dpi', str(config.figure_dpi), '--plot-min-elevation', str(config.plot_min_elevation)]
    if config.medusa_root is not None:
        args += ['--medusa-root', str(config.medusa_root)]
    args += ['--mean-min-satellites', str(config.mean_min_satellites), '--mean-method', config.mean_method, '--product-figure-color-mode', config.product_figure_color_mode]
    if not config.mean_line:
        args.append('--no-mean-line')
    if config.glonass_channels is not None:
        args += ['--glonass-channels', str(config.glonass_channels)]
    if config.primary_sp3 is not None:
        args += ['--sp3', str(config.primary_sp3)]
    for path in config.supplemental_sp3:
        args += ['--supplemental-sp3', str(path)]
    if config.qc_elevation:
        args.append('--qc-elevation')
    if config.qc_zoom:
        args.append('--qc-zoom')
    if config.figures:
        args.append('--figures')
    if not config.overview_figures:
        args.append('--no-overview-figures')
    if config.qc_figures:
        args.append('--qc-figures')
    if config.figure_products is not None:
        args += ['--figure-products', ','.join(config.figure_products)]
    if config.max_qc_figures is not None:
        args += ['--max-qc-figures', str(config.max_qc_figures)]
    if config.overwrite_output:
        args.append('--overwrite-output')
    if config.require_complete_geometry:
        args.append('--require-complete-geometry')
    if config.offline:
        args.append('--no-download')
    parser = build_parser()
    options = parser.parse_args(args)
    validate_inputs(options, parser)
    return run(options)
