import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# 1. Direct path to Station 1 data
data_dir = "../../datasets/New_Revised_Final_Data/"
file_path = os.path.join(data_dir, "Station1_filled_Data.csv")

print("🚀 Starting Baseline SARIMA Evaluation for Soil Moisture (Station 1)...\n")

# 2. Load dataset and establish hourly time series index
df = pd.read_csv(file_path)
df.index = pd.date_range(start="2022-01-01 00:00:00", periods=len(df), freq="h")

# Target columns for Soil Water Content (Moisture)
moisture_cols = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']
horizons = [24, 72, 168] # 1 day, 3 days, 1 week

# 3. Loop through each moisture depth layer
for col in moisture_cols:
    if col not in df.columns:
        print(f"⚠️ Column {col} not found in dataset. Skipping.")
        continue
        
    print(f"--- Training Model for Moisture Depth: {col} ---")
    
    # Split data: Hold out the last week (168 hours) for testing
    train_data = df[col].iloc[:-168]
    test_data = df[col].iloc[-168:]
    
    # Using the stable 'bfgs' solver to protect your Mac's memory usage
    model = ARIMA(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    results = model.fit(method_kwargs={'method': 'bfgs'}) 
    
    # Generate the full 1-week forecast window
    full_forecast = results.forecast(steps=168)
    
    print(f" Moisture Results for {col}:")
    for h in horizons:
        actual = test_data.iloc[:h]
        pred = full_forecast.iloc[:h]
        mae = mean_absolute_error(actual, pred)
        
        horizon_name = f"{h} hours" if h != 168 else "1 week"
        # Displaying to 5 decimal places since SWC tracks small decimal fractions
        print(f"   ↳ MAE for {horizon_name}: {mae:.5f} m³/m³")
    print("\n")
    
    # Memory clean-up after each layer loop
    del model
    del results