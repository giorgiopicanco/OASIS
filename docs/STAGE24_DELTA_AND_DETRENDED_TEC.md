# Stage 24 — scientific and software update

## Scope

This is an incremental update to Stage 23. RNX_QC, the original MEDUSA historical QC engine, SP3 geometry, ROTI V6, SIDX, and the existing TEC calibration are not scientifically modified. The previous DTEC calculation is renamed DELTATEC (display name ΔTEC), and its numerical values are unchanged. The public detrended TEC product is dTEC (ASCII machine key DTEC). Both coexist in HDF5 schema 2.1.

## ΔTEC (legacy OASIS index)

Source: user-supplied DTEC_CALC.py and Stage 23 implementation. The legacy 150-second rolling-mean difference, factor of ten, QC rules, polynomial residual screen and original elevation metadata remain unchanged. The multiplier is dimensionless; values remain labeled as scaled TECU amplitude, not TECU/hour. No conversion to a physical derivative has been introduced.

## dTEC (new detrended VTEC)

```
background_VTEC = savgol_filter(VTEC, window_samples, polyorder=1)
dTEC = VTEC - background_VTEC
```

The intended window is 900 s (15 minutes), centered at each original valid VTEC observation. The number of samples is `round(900 / native_cadence_seconds)`, increased by one if even and constrained to at least three. The actual interval from first to last sample is `(window_samples - 1) * cadence`. For 15 s or 30 s data the interval is exactly 900 s (61 or 31 samples). For 60 s cadence the selected 15-point interval spans 840 s; this is the nearest odd count under this convention. No interpolated or padded sample is retained. The first/last half-window of every segment is excluded from dTEC. Native time sampling is preserved. The centered linear fit is equivalent to a moving average on a complete symmetric uniform grid; it is not an ideal brick-wall high-pass filter.

Eligibility: valid calibrated VTEC, original RNX_QC arc ID, final good flags, nonnegative model VTEC, and elevation >= 30 degrees. Segment boundaries include changes of original arc, non-contiguous source indices, missing or rejected samples, cadence changes, final ambiguity/corrected-slip flags, and the original final UPDARC slip indices. The filter never spans these boundaries. A negative dTEC value is valid; negative model VTEC is excluded upstream, not clipped. Segments shorter than a full window yield no output. Original VTEC and STEC products preserve Stage 23 negative values with flags. No new calibration method is introduced. Missing orbital geometry prevents a calibrated VTEC and therefore dTEC result for that satellite; LGF and geometry-independent indices remain available under their existing policies.

The method is inspired by Zhang's use of a linear Savitzky–Golay background subtraction per satellite-receiver link. The specific 15-minute window was requested by the user; the 30-degree cutoff and no-interpolation/edge policy are explicit pyOASIS decisions. Published Zhang examples use other windows and cutoffs (e.g. 30 minutes and 15 degrees), and their TEC pipeline is different. This is not exact replication of the full Madrigal system.

References:
- Zhang et al. (2019), *Traveling Ionospheric Disturbances and Ionospheric Perturbations Associated With Solar Flares in September 2017*, https://doi.org/10.1029/2019JA026585.
- Zhang et al., *Mesoscale Ionospheric Irregularity Oval at High Latitudes Observed by Global GNSS Networks (2010–2024)*, https://www.janss.kr/archive/view_article?pid=jass-42-4-119.
- Dinsmore et al. (2021), *General resource for ionospheric transient investigations (GRITI)*, https://pmc.ncbi.nlm.nih.gov/articles/PMC8374629/ (odd window sample rule and order-one Savitzky–Golay).

## HDF5

Default filename: STATION_YEAR_DOY_ION.h5. Same station/day file name regardless of selected products; distinct output folders are recommended for independent runs. Explicit --output-file remains supported. The date identifies the input RINEX GPST day; plotting uses UTC.

```
/products/DELTATEC/G01/value                 # previous OASIS index
/products/DTEC/G01/value                     # dTEC in TECU
/products/DTEC/G01/vtec_tecu
/products/DTEC/G01/background_vtec_tecu
/products/DTEC/G01/utc_ms
/products/DTEC/G01/arc_id
/products/DTEC/G01/segment_id
/products/DTEC/G01/source_epoch_index
/products/DTEC/G01/window_samples
/products/DTEC/G01/cadence_ms
/products/DTEC/G01/actual_window_span_ms
/products/DTEC/G01/elevation_deg
/products/DTEC/G01/ipp_lat_deg
/products/DTEC/G01/ipp_lon_deg
/products/DTEC/G01/ipp_height_km
```

The new dTEC group has display_name='dTEC', algorithm='VTEC_MINUS_SAVITZKY_GOLAY_900S_LINEAR', units='TECU', and window/edge/elevation/calibration attributes. DELTATEC has display_name='ΔTEC' and the unchanged legacy algorithm metadata. Both ASCII group keys and figure filenames avoid Unicode path issues. Unicode display names and CLI ΔTEC are supported. The old DTEC key is deliberately redefined at schema 2.1; use algorithm and version metadata when reading older files. Plotting recognizes the previous known algorithm, not the name alone. The root retains the legacy format identifier pyOASIS-LGF for reader compatibility; format_version=2.1 identifies this schema.

## Compact RNX_QC

Five panels (a–e): combined raw/corrected phase LGF; raw/screened MW; raw PGF and levelled carrier-phase LGF; residuals; five QC event categories. Optional (f) elevation. Shade accepted arc extents and show boundary lines, arc IDs and original arc bias. Annotate original reported slip magnitudes directly at the event (candidates are not asserted to be corrections). Header summarizes observations, outliers, candidates, corrections and arcs. No removed data is deleted from HDF5. The plotting reader marks unavailable MW diagnostic masks N/A rather than treating them as zero. Zoom can restrict the display to the pass. Scientific overview plots preserve 12x6 inches, UTC, two-hour ticks, circular thin markers, and GPS blue / GLONASS red / Galileo lime / BeiDou violet in GREC order. Output filenames deltaTEC.png and dTEC.png remain distinct even on case-insensitive Windows filesystems.

## Validation and limitations

17 unit/regression tests passed (Stage 22, updated Stage 23, Stage 24): native cadence and comparison with scipy, linear trend removal, arc and final-slip boundaries, rejected/missing/negative/low-elevation samples, empty and invalid input, new HDF5 schema, backward-compatible old ΔTEC plotting, switches and CLI product selection.

Integrated BOAV RINEX 2.11/GPS run, matching SP3, original MEDUSA, 32 satellites and 5,700 epochs: 62,336 LGF, 62,334 IPP, 5,877 ROTI, 6,099 ΔTEC, 15,496 SIDX, 40,043 STEC, 40,043 VTEC and 20,750 dTEC records. One HDF5, JSON, CSV, and nine figures (eight overviews + one six-panel QC diagnostic) were created. The earlier six product record counts are unchanged. The actual dTEC HDF5 was checked for numerical identity with stored VTEC minus background (zero maximum absolute arithmetic difference for the stored arrays). The final visualization revision was additionally tested by replotting this actual HDF5.

The real BOAV model still contains 3,731 negative STEC estimates. Large dTEC excursions may reflect ionospheric activity, measurement or calibration errors; no independent scientific validation is claimed. CHPI RINEX and its 2025 SP3 are not available in this execution environment, so full Stage 24 CHPI processing and the Windows installer execution must be verified by the user. Existing geometry C++ parity and original live MEDUSA V6 context validation remain pending. See Stage 23 specification for prior algorithm details.
