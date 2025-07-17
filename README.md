# Rainfall and Soil Moisture Response Analysis

## Overview

This part of the project aims to investigate how soil moisture (SWC) responds to rainfall events across different stations, based on hourly environmental data. The accompanying Python script helps preprocess station data to identify rainfall events and quantify how long it takes for soil moisture at various depths to return to pre-rain conditions. The results can support broader environmental and hydrological research, including soil classification, climate comparisons, and predictive modeling.

---

## Research Objectives

The following key research questions guide this analysis:

1. **Rainfall Event Frequency**
   - How often do rainfall events occur at each station?
   - Can differences in event frequency be attributed to the climate zones of the stations?

2. **SWC Recovery Lag**
   - After a rainfall event, how long does it take for deeper soil layers (especially at 20 cm and 50 cm) to return to pre-rain SWC levels?

3. **Impact of Rainfall Intensity**
   - How does the intensity of precipitation influence the change in soil moisture content?

4. **Soil Type Inference**
   - Can we infer soil type by observing how quickly or slowly SWC levels change and recover, using resources such as [this soil water content guide](https://extension.okstate.edu/fact-sheets/understanding-soil-water-content-and-thresholds-for-irrigation-management.html)?

5. **Seasonality and Predictive Modeling (Future Work)**
   - Does seasonality influence SWC responses?
   - Can we eventually predict soil moisture dynamics based on past rainfall patterns and climatological variables?

---

## Script Description: `swc_analysis_script.py`

This script processes a CSV file containing station data with the following key tasks:

### Functionality

- **Input Validation**
  - Prompts the user for a CSV file path.
  - Checks for presence of required columns: `Timestamp`, `Ppt` (precipitation), and at least one soil moisture column like `SWC_5`, `SWC_20`, etc.

- **Rainfall Event Detection**
  - Identifies discrete rainfall events based on dry intervals (default threshold: 6 hours).

- **SWC Recovery Time Calculation**
  - For each rainfall event and each SWC depth:
    - Computes baseline SWC in the hours before rain.
    - Determines how many hours after the rain it takes for SWC to return to baseline.

- **Output**
  - Saves the results to a CSV file named `<original_filename>_SWC_return_times.csv`.

### Input Format

The CSV file should meet the following conditions:
- A column named `Timestamp` (parsed as datetime index).
- A column named `Ppt` indicating hourly precipitation in mm.
- One or more columns beginning with `SWC_` (e.g. `SWC_5`, `SWC_20`, `SWC_50`) representing volumetric water content at various soil depths.

### How to Use It

Run the script:

```bash
python swc_analysis_script.py
```

You'll be asked for the path to your data file. The results will be saved in the same folder as a new CSV file.

Example:

```
Enter the path to the station CSV file: data/Station_A.csv
Output saved to: data/Station_A_SWC_return_times.csv
```

---

## Application

This script is a foundational preprocessing tool for a broader soil moisture and climate interaction analysis pipeline. The output data can be used for:
- Climatological comparisons between stations.
- Time series analysis of rainfall and moisture.
- Soil classification based on hydrological behavior.
- Feature engineering for machine learning models.

---

## Contact

For questions or suggestions, please open an issue to the repository.

