import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings('ignore') # This keeps our terminal clean from math warnings

# 1. Load data
data_dir = "../datasets/New_Revised_Final_Data/"
file_path = os.path.join(data_dir, "Station1_filled_Data.csv")
df = pd.read_csv(file_path)

df.rename(columns={'Unnamed: 0': 'DateTime'}, inplace=True)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)
df = df.asfreq('h')

# The 4 depths 
depths = ['T_5', 'T_10', 'T_20', 'T_50']
one_week = 168

print("--- Running Baseline ARIMA(1,1,1) across ALL Depths ---")
print("This will loop through all depths. Processing, please wait...\n")

# Final summary table data
summary_data = []

for depth in depths:
    print(f"Training model for depth: {depth}...")
    
    # Split data for this specific depth
    train = df[depth].iloc[:-one_week]
    test_1week = df[depth].iloc[-one_week:]
    test_72h = df[depth].iloc[-one_week:-one_week+72]
    test_24h = df[depth].iloc[-one_week:-one_week+24]
    
    # Fit ARIMA(1,1,1)
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Forecast
    forecast_1week = model_fit.forecast(steps=168)
    forecast_72h = forecast_1week.iloc[:72]
    forecast_24h = forecast_1week.iloc[:24]
    
    # Calculate Errors (MAE)
    mae_24h = np.mean(np.abs(forecast_24h.values - test_24h.values))
    mae_72h = np.mean(np.abs(forecast_72h.values - test_72h.values))
    mae_1week = np.mean(np.abs(forecast_1week.values - test_1week.values))
    
    # Save results
    summary_data.append({
        'Depth': depth,
        '24h Error (°C)': round(mae_24h, 2),
        '72h Error (°C)': round(mae_72h, 2),
        '1-Week Error (°C)': round(mae_1week, 2)
    })

# 2. Display the final results 
results_df = pd.DataFrame(summary_data)
print("\n=======================================================")
print("🎯 FINAL BASELINE ARIMA(1,1,1) RESULTS SCOREBOARD 🎯")
print("=======================================================")
print(results_df.to_string(index=False))
print("=======================================================")