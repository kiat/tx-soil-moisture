# Soil Data Pipeline



## Features

1. **Load and parse raw data**  
   - Soil data (`SM_{station}.dat`) and MET data (`MET_{station}.dat`) are read from configurable directories.  
   - Dates are converted to a `DateTimeIndex` and numeric columns are coerced to floats (invalid parse → NaN).

2. **Merge datasets**  
   - Soil and MET data are joined on the timestamp index (inner join).  
   - When both stations report precipitation (`Ppt_soil` & `Ppt_met`), the MET value is kept and soil’s is dropped.

3. **Validate and clean**  
   - **Missing data**: scans each column for NaNs and records their timestamps.  
   - **Invalid data**: applies realistic bounds to physical measurements, replacing out-of-range values with NaN:  
     - Soil moisture (`SWC_*`): 0–0.6  
     - Precipitation (`Ppt`): ≥ 0  
     - Relative humidity (`RH`): 0–100%  
     - Wind speed: 0–25 m/s  
     - Wind direction: 0–360°  
     - Solar radiation (`Srad`): ≥ 0  
     - Temperature (`T_*`, `Tair`): –30–60 °C

4. **Generate summaries and outputs**  
   - **Raw merged data** saved to `raw_merged_data/raw_merged_station_{station}.csv`.  
   - **Missing/invalid summary** grouped by consecutive timestamp runs, exported to `missing_data/Station{station}_missing_data.csv` or custom path.  
   - **Cleaned full-timeline data** reindexed to an hourly grid (from first to last timestamp), with all invalid/missing replaced by NaN, saved to `cleaned_data/Station{station}_cleaned_data.csv` or custom path.

## Requirements

- Python 3.7 or higher  
- `pandas`  
- `numpy`

Install dependencies via:

```bash
pip install pandas numpy
```

## Usage

```bash
python soil_data_pipeline.py --station 1
```

**Optional arguments:**

- `-s`, `--station` (int, default=1): Station ID (1–6)  
- `--soil-base-dir` (str): Path to soil `.dat` files (default `../datasets/TX-Data/soil_station`)  
- `--met-base-dir` (str): Path to MET `.dat` files (default `../datasets/TX-Data/met_station`)  
- `--raw-output-dir` (str): Directory for raw merged CSVs (default `raw_merged_data`)  
- `--missing-output` (str): Filename for missing/invalid summary CSV (default `missing_data/Station{station}_missing_data.csv`)  
- `--cleaned-output` (str): Filename for cleaned full-timeline CSV (default `cleaned_data/Station{station}_cleaned_data.csv`)

### Batch processing example

You can run multiple stations in sequence:

```bash
python soil_data_pipeline.py --station 1
python soil_data_pipeline.py --station 2
python soil_data_pipeline.py --station 3
python soil_data_pipeline.py --station 4
python soil_data_pipeline.py --station 5
python soil_data_pipeline.py --station 6
...
```
