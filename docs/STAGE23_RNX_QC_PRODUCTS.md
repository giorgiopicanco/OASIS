# Stage 23 scientific and software specification

## Scope and provenance

The original MEDUSA historical B111 engine is retained unchanged. `RNX_QC` is a public application name for the existing quality-control and leveling integration. Internal original function/class names remain for source compatibility. The `medusa_b111.py` and `medusa_b111_sp3.py` changes add public property aliases only; the original MEDUSA source remains external. The Stage 22 MEDUSA ION V6 `window_records` kernel is unchanged; the adapter changes only satellite iteration order. No claim of complete Bernese or compiled MEDUSA geometry parity is made.

Source algorithms supplied by the user: `DTEC_CALC.py`, `SIDX_CALC.py`, `TEC_CAL.py`, `RNX_LEVELLING.py`, `ROTI_CALC.py`, `settings.py`. SHA-256 references are provided in `stage23_validation/SOURCE_HASHES.txt`. Original OASIS plotting settings are the starting point; the requested GREC palette and thinner markers override earlier colors/widths.

## Public data names and time

- `lgf_raw_m = lambda1 * phase1_cycles - lambda2 * phase2_cycles`.
- `pgf_raw_m = pseudorange2_m - pseudorange1_m` (code / pseudorange GF).
- `lgf_corrected_m` is the original cycle-slip-corrected phase GF.
- `lgf_leveled_m` is phase GF leveled to code: corrected LGF minus the mean phase-minus-code bias of the accepted arc. PGF itself is not leveled.
- Flags are original Bernese-style bit masks: bit 0 bad/outlier, bit 1 new ambiguity, bit 2 corrected cycle slip; -1 not processed. A stage-reported slip candidate is not automatically a confirmed correction. Full masks, intermediate flags and stage-reported MW/first-phase/second-phase slip sizes are stored on the common grid. Missing or excluded epochs remain missing, not fabricated.
- GPST is kept for SP3 and RINEX indexing. UTC is independently converted using the existing historical leap-second table (through the 2017 offset of 18 seconds); datetime64 cannot represent the leap second label itself. The GPST nanosecond grid is preserved, and the UTC millisecond grid and nominal UTC MJD are exported. Time axes are UTC. The filename year/DOY still identifies the input RINEX day in GPST, which can include a small portion of the previous UTC day.
- The source remains the actual RINEX GPST reader. Non-GPST inputs must not be silently relabeled.

## Products

### ROTI

MEDUSA ION V6 fixed UTC slots: 150-second windows, two per 300-second block; 15/30-second observations, complete slot coverage, valid continuous arc, post-QC absolute ROT <= 5 TECU/min and ROTI <= 10 TECU/min, population standard deviation. Elevation >= 30 degrees and the 0–5 axis range are display-only. Missing IPP does not invalidate a numerical ROTI result. Historical B111 context parity with live block processing remains unverified.

### DTEC

Source: `DTEC_CALC.py`. Within each 150-second window, use the original centered rolling short and long mean rule, with short = 15 samples for >30 observations, otherwise 5; long = four times short. The original last-element difference is reproduced without constructing temporary pandas frames. Apply the original multiplier 10. The multiplier is dimensionless, so this is a legacy scaled TEC amplitude, not a physical time derivative in TECU/hour. The value is preserved and explicitly labeled as a scaled amplitude in TECU.

### SIDX

Source: `SIDX_CALC.py`. Within each 60-second window, compute mean absolute ROT in TECU/min, using actual inter-epoch seconds. Do not duplicate the L1-L2 result as a nonexistent second frequency-pair result. Station SIDX is the equal-weight mean of the available one-per-satellite values for each UTC minute, using the original 20-degree elevation cutoff. The native HDF5 value is TECU/min; the station plot and the derived field use mTECU/s via the physically correct factor 1000/60, not the original unexplained factor 10. Thin lines are broken at missing minutes.

### Shared DTEC/SIDX improvements

Use the real pair-specific frequency factor `40.3e16 * (1/f2**2 - 1/f1**2)`, including GLONASS FDMA and Galileo/BeiDou. The legacy code applied GPS factors to all systems. Windows are aligned to UTC rather than to the first observation of each satellite. Windows require at least two observations, continuous cadence, one QC arc, and no interior final ambiguity/slip flag. No interpolation, gap bridging, or artificial observations. Third-degree polynomial residual screening uses the original five-standard-deviation threshold separately within existing QC arcs; undersampled arcs are not fit. A numerical tolerance prevents rejection of constant series due solely to floating-point roundoff. Numeric windows without geometry are retained with `geometry_available=False`; `original_elevation_ok` preserves the legacy DTEC (30 degree) and SIDX (20 degree) criteria. The figures use these cutoffs. DTEC/SIDX are adaptations, not a bitwise reproduction of the original text-file pipeline.

### STEC and VTEC / TEC

Source: `TEC_CAL.py`, six local ionospheric model coefficients per UTC hour and one offset per accepted satellite arc. Each arc's LGF-equivalent STEC is normalized at its first eligible observation. With vertical factor `m = 1 / slant_mapping_factor`, solve the original linear observation model:

```
m * normalized_STEC = a0 + a1*u + a2*dLon + a3*u*dLon
                            + a4*dLat + a5*u*dLat + m*arc_offset
STEC_model = normalized_STEC - arc_offset
VTEC_model = STEC_model * m
```

`u` is fractional UTC hour, angular offsets are radians. Unlike the original fixed GPS conversion and fixed GLONASS slot table, actual selected frequencies are used. The MEDUSA-method 350 km IPP and mapping factor are used (not the legacy 450 km model). Use existing RNX_QC arcs and the original 20-degree calibration cutoff. No PCHIP/Akima gap interpolation, global all-satellite reconnection, or post-calibration plot-only vertical shifts. The old fixed-size normal matrix is replaced by a sparse observation design and scaled rank-checked solve. Unsupported hours (e.g. 18 seconds of preceding UTC day) and one-sample arcs are excluded; rank-deficient or oversized systems yield no invented TEC results. Hour coefficients, per-arc model offsets, rank, condition and residual RMS are saved in the report. Negative estimates are preserved with `physical_nonnegative=False`, not clipped. These offsets are model estimates, not independently measured DCBs. The output is **experimental model-calibrated TEC**, not validated absolute TEC. Direct uncalibrated `lgf_equivalent_stec_tecu` remains available in the QC group even for satellites without geometry when TEC is requested.

## HDF5 v2.0 schema

One file per run, all successfully QC-processed satellites, input-only skipped satellites documented. No old L4/P4/B111 dataset names in newly generated HDF5 files. Legacy HDF5 readers in the plotting adapter accept both old and new field names; old pre-V6 ROTI data is rejected for replotting rather than mislabeled.

```
/metadata/                        source, receiver ECEF, signals, report
/epochs/time_gpst_ns
/epochs/datetime_gpst_iso
/epochs/time_utc_ms
/epochs/datetime_utc_iso
/epochs/mjd_utc
/epochs/gpst_minus_utc_s
/satellites/Gxx/rnx_qc/lgf_raw_m
/satellites/Gxx/rnx_qc/lgf_corrected_m
/satellites/Gxx/rnx_qc/lgf_leveled_m
/satellites/Gxx/rnx_qc/pgf_raw_m
/satellites/Gxx/rnx_qc/stage_diagnostics/
/satellites/Gxx/rnx_qc/masks/
/satellites/Gxx/geometry/         XYZ, range, elevation, azimuth, IPP, mapping factor
/satellites/Gxx/selected_rinex_observables/  original values, LLI, SSI
/products/ROTI/150s/Gxx/          original V6 records
/products/DTEC/Gxx/
/products/SIDX/Gxx/
/products/STEC/Gxx/
/products/VTEC/Gxx/
/station_products/SIDX/G/        UTC-minute station mean and satellite counts
```

The same structure is used for R, E, C. Requested-but-empty products are marked by zero record counts and not listed as available. HDF5 is written atomically. A missing SP3 position never produces fabricated IPP; default behavior preserves other satellites, with an optional stricter zero-IPP satellite policy. A supplemental real SP3 can be supplied; no broadcast ephemeris fallback has been implemented.

## Figures

Constellation order is G/R/E/C: blue/red/lime/violet. All scientific overview plots are 12x6 inches, 300 DPI by default, UTC axis, 2-hour ticks, external legend, dashed gray grid and smaller marker sizes. Per-satellite diagnostic panels include raw phase LGF, raw/screened Melbourne-Wubbena, corrected phase LGF, raw PGF and leveled phase LGF, leveling residual, counted QC masks, original reported slip sizes, and arc ID. Outlier/slip markers come from original flags/candidates, not arbitrary detection thresholds. Stage diagnostics absent in old HDF5 files are identified as unavailable. QC and overview plots are separately switchable.

## Validation

- Eleven automated tests passed (eight Stage 23 tests plus three unchanged Stage 22 V6 tests), including exact recovery of a synthetic known TEC model, rank-deficiency rejection, GREC frequency conversion, time conversion, gap/slip rejection, HDF5 schema, diagnostic flags, aggregation, and plot switches.
- Full real BOAV RINEX 2.11 GPS run with original MEDUSA and matching SP3: 5,700 epochs, 32 QC satellites, 62,336 LGF, 62,334 IPP, 5,877 ROTI, 6,099 DTEC, 15,496 SIDX, 40,043 each model STEC/VTEC; HDF5, CSV, JSON and 8 figures saved (one QC panel selected at 150 DPI). The TEC system had rank 189/189, RMS about 3.227 TECU and 3,731 negative model STEC estimates, flagged and warned. This is a functional integration test, not an absolute TEC validation.
- Separate test of all four constellations on the supplied earlier Stage 18 CHPI HDF5 (111 satellites) generated the new DTEC/SIDX/model TEC products and GREC plots. The input contained the earlier preliminary geometry; this was NOT a full CHPI Stage 23 RINEX/SP3 run or a ROTI V6 comparison. The complete CHPI RINEX and corresponding SP3 were not available in the execution environment.
- Windows installer syntax is modeled on the earlier successful patch installers; execution in the user's Windows workspace has not been performed here.
- Compiled MEDUSA/RTKLIB geometry equivalence, live-block V6 parity, absolute TEC calibration, and regression against original OASIS product values remain scientific acceptance tasks.
