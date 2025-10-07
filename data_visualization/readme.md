# TX-SOIL-MOISTURE Data Visualization & Analysis

This repository contains a suite of scripts and interactive notebooks to explore, visualize, and analyze the revised final data for TX-SOIL-MOISTURE. The data are stored in the folder `datasets/Revised_Final_Data` with file names like `Station1_Revised_Final_Data.csv`, `Station2_Revised_Final_Data.csv`, etc., for stations 1 through 6.


---

## Static Plotting Scripts

These scripts generate static charts and save them as PDF files.

### Script 1. `script_plot_swc_yearly.py`
- **Purpose:**  
  Plots a line chart for a specified SWC parameter (e.g. `SWC_10`) for each year within a given range.  
  Each year is plotted on a separate page in a single PDF.
- **Usage Example:**
  ```bash
  python3 script_plot_swc_yearly.py --param SWC_10 --station 1 --start_year 2015 --end_year 2020
  ```

---

### Script 2: `script_plot_swc_monthly.py`
- **Purpose:**  
  Computes and plots the monthly average of a chosen SWC parameter for each year.  
  Each year’s monthly averages are displayed as a bar chart on a separate page of a PDF.
- **Usage Example:**  
  ```bash
  python3 script_plot_swc_monthly.py --param SWC_10 --station 1 --start_year 2015 --end_year 2020
  ```

---

### Script 3: `script_plot_swc_violin.py`
- **Purpose:**  
  Visualizes the distribution of a specified SWC parameter (e.g. `SWC_10`) for each year using violin plots.  
  The script produces a single PDF where each page shows one year’s violin plot or arranges all years side by side for horizontal comparison.
- **Usage Example (side-by-side):**  
  ```bash
  python3 script_plot_swc_violin.py --param SWC_10 --station 1 --start_year 2015 --end_year 2020
  ```
  
---

### Script 4: `script_plot_ppt_monthly.py`
- **Purpose:**  
  Calculates the monthly total precipitation (Ppt) for each year and plots it as a bar chart.  
  Each year’s bar chart is saved on a separate page in a PDF.
- **Usage Example:**  
  ```bash
  python3 script_plot_ppt_monthly.py --station 1 --start_year 2015 --end_year 2020
  ```

---

## Dynamic Plotting

### `Dynamic_Data_Visualization.ipynb`

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
