## Overview

This notebook provides an interactive interface to explore and visualize the revised final data for TX-SOIL-MOISTURE. 
Used Data in `../datasets/Revised_Final_Data` with file names like `Station1_Revised_Final_Data.csv` and so forth for stations 1 through 6.
The key features include:

- **Soil Moisture Data:**  
  Visualizations for soil moisture on a monthly (interactive) and yearly basis using the columns `SWC_5`, `SWC_10`, `SWC_20`, and `SWC_50`.

- **Soil Temperature Data:**  
  Visualizations for soil temperature on both a monthly and yearly basis using the columns `T_5`, `T_10`, `T_20`, and `T_50`.

- **Meteorological Data:**  
  Visualizations for variables such as `Tair`, `RH`, `Wind speed`, and `Srad` (with station location provided via `Latitude` and `Longitude` if available) in both monthly and yearly views.


## Notebook Structure

1. **Data Loader**  
   - **Function:** `load_revised_data(station_id)`  
   - **Purpose:** Reads the revised CSV file for a given station, parsing the first column as a datetime index.

2. **Soil Moisture Plotting Functions**  
   - **`plot_soil_moisture_interactive`** (monthly)  
   - **`plot_soil_moisture_yearly`**  
   - **Purpose:** Display four soil moisture depths (`SWC_5`, `SWC_10`, `SWC_20`, `SWC_50`) for a selected station and time range.

3. **Soil Temperature Plotting Functions**  
   - **`plot_soil_temperature_interactive`** (monthly)  
   - **`plot_soil_temperature_yearly`**  
   - **Purpose:** Show four soil temperature depths (`T_5`, `T_10`, `T_20`, `T_50`) for a selected station and time range.

4. **MET Variable Plotting Functions**  
   - **`plot_met_variable_interactive`** (monthly)  
   - **`plot_met_variable_yearly`**  
   - **Purpose:** Plot a chosen meteorological variable (e.g., `Tair`, `RH`, `Wind speed`, `Srad`) for a selected station and time range, including station coordinates if available.

5. **Update Plot Function**  
   - **`update_plot`**  
   - **Purpose:** Calls the correct plotting function based on widget selections.

6. **Interactive Interface**  
   - Uses `ipywidgets` to provide dropdown selectors for Plot Type, Station, Year, Month, MET Variable, and Time Type (Monthly/Yearly).  
   - Dynamically updates and displays the chosen figure.

## Dependencies

- **Python 3**  
- **pandas** 
- **plotly** 
- **ipywidgets** 
- **pathlib** 