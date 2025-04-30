<p align="center">
  <img src="img/logo_bar.png" alt="OASIS Logo" width="900"/>
</p>

# OASIS: Open-Access System for Ionospheric Studies

**OASIS** is a modular and open-access Python toolbox for processing multi-frequency GNSS data and computing ionospheric indices. It was developed to overcome limitations of proprietary or non-standardized tools and to promote scientific transparency and reproducibility in ionospheric research.

OASIS automates the detection and correction of cycle slips and outliers, performs arc-wise geometry-free (GF) leveling, and derives ionospheric indices directly from calibrated observations, without relying on external products such as Differential Code Biases (DCBs) or vertical TEC maps.

---

## Key Features

- Processes RINEX v2 and v3 files with 15- or 30-second sampling rates.
- Supports GPS and GLONASS constellations (Galileo and BeiDou coming soon).
- Fully autonomous detection of data gaps, cycle slips, and outliers.
- Arc-wise geometry-free leveling of carrier-phase combinations.
- Computation of ionospheric indices: **ROTI**, **ΔTEC**, and **SIDX**.
- No dependency on DCBs or global TEC maps.

---

## Installation Requirements

- Python 3.8 or higher
- Required libraries:
  - `numpy`, `pandas`, `matplotlib`, `scipy`, `astropy`, `georinex`, `pyproj`

---

## Workflow Summary

1. **Input**: RINEX and SP3 files.
2. **Orbit Interpolation**: SP3 files are parsed and interpolated.
3. **IPP Calculation**: Computes the coordinates of the ionospheric piercing point.
4. **MW Combination**: Melbourne-Wübbena combination is used to identify cycle slips.
5. **Screening**:
   - Initial residual screening using ∆MW and polynomial fitting.
   - Refined residual analysis defines mini-arcs by sign changes.
6. **Geometry-Free Leveling**: Performed arc-wise on valid combinations.
7. **Index Derivation**:
   - **ROTI**: Standard deviation of ROT (rate of TEC).
   - **ΔTEC**: Difference between 15-min and 60-min moving averages of GF combinations.
   - **SIDX**: Mean absolute ROT in a 1-minute window.

---

## How to Run

1. Define the station code, year, DOY, and directories in `main.py`.
2. Run the complete pipeline:

```bash
python3 main.py
```

3. Results are saved in structured output folders, organized by station and satellite.

---

## Directory Structure

- `RNX_CLEAN.py` – Prepares and organizes RINEX data.
- `RNX_SCREENING.py` – Detects outliers and cycle slips using MW combinations.
- `RNX_LEVELLING.py` – Performs arc-wise geometry-free leveling.
- `DTEC_CALC.py` – Calculates ΔTEC index.
- `SIDX_CALC.py` – Calculates the Sudden Ionospheric Disturbance Index.
- `SP3_INTERPOLATE.py` – Extracts and organizes SP3 orbit data.
- `linear_combinations.py`, `gnss_freqs.py`, `settings.py`, etc. – Support modules for GNSS frequency handling and coordinate transformation.

---

## Outputs

- Time series of leveled GF combinations.
- ROT, ROTI, ΔTEC and SIDX per satellite and station.
- Intermediate figures for screening and quality control.

---

## License and Citation

The OASIS toolbox is open-source and free to use under the MIT license.

If you use this software in your research, please cite:

**Picanço, G.A.S. et al. (2025). Introducing OASIS: an Open-Access System for Ionospheric Studies. GPS Solutions.** *(submitted)*

---

## Contact

For questions or contributions, please visit:
[https://github.com/giorgiopicanco/OASIS](https://github.com/giorgiopicanco/OASIS)
