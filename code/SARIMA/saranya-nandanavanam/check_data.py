import pandas as pd
import os

# 1. Load data
data_dir = "../datasets/New_Revised_Final_Data/"
file_path = os.path.join(data_dir, "Station1_filled_Data.csv")
df = pd.read_csv(file_path)

depths = ['T_5', 'T_10', 'T_20', 'T_50']

# --- CHECK 1: ABSOLUTE MIN / MAX ---
print("=======================================================")
print("🔍 SENSOR SANITY CHECK RESULTS (MIN/MAX) 🔍")
print("=======================================================")
print(df[depths].agg(['min', 'max']))
print("=======================================================\n")

# --- CHECK 2: SUDDEN HOURLY SHOCKS ---
print("=======================================================")
print("⚡️ MAXIMUM HOURLY TEMPERATURE SHOCKS ⚡️")
print("=======================================================")
for depth in depths:
    # .diff() calculates the difference between current hour and previous hour
    hourly_changes = df[depth].diff().abs()
    max_jump = hourly_changes.max()
    print(f"Largest sudden 1-hour jump at depth {depth}: {round(max_jump, 2)}°C")
print("=======================================================\n")

# --- CHECK 3: STATISTICAL OUTLIERS (IQR METHOD) ---
print("=======================================================")
print("📊 STATISTICAL OUTLIER COUNT (IQR METHOD) 📊")
print("=======================================================")
for depth in depths:
    Q1 = df[depth].quantile(0.25)
    Q3 = df[depth].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[depth] < lower_bound) | (df[depth] > upper_bound)]
    print(f"Number of unusual values in {depth}: {len(outliers)}")
print("=======================================================")