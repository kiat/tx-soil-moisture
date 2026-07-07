import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# 1. Path setups for BOTH stations
data_dir = "../datasets/New_Revised_Final_Data/"
station1_path = os.path.join(data_dir, "Station1_filled_Data.csv")
station2_path = os.path.join(data_dir, "Station2_filled_Data.csv")

print("🚀 Starting Cross-Station Generalization Evaluation...")
print("📦 Training on Station 1 ➡️ Testing on Station 2 (Soil Moisture)\n")

# 2. Load datasets and set explicit hourly datetime indexes
df_st1 = pd.read_csv(station1_path)
df_st1.index = pd.date_range(start="2022-01-01 00:00:00", periods=len(df_st1), freq="h")

df_st2 = pd.read_csv(station2_path)
df_st2.index = pd.date_range(start="2022-01-01 00:00:00", periods=len(df_st2), freq="h")

# Target columns are Soil Water Content (Moisture)
moisture_cols = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']
horizons = [24, 72, 168] # 1 day, 3 days, 1 week

# 3. Cross-Station Training Loop
for col in moisture_cols:
    print(f"--- Training on Station 1 {col} | Testing on Station 2 {col} ---")
    
    # Train data comes entirely from Station 1 history
    train_data = df_st1[col].iloc[:-168]
    
    # Test data comes entirely from Station 2 future window
    test_data = df_st2[col].iloc[-168:]
    
    # Fit the SARIMA baseline model on Station 1's historical patterns
    model = ARIMA(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    results = model.fit(method_kwargs={'method': 'bfgs'})
    
    # Forecast out blindly into Station 2's timeline
    full_forecast = results.forecast(steps=168)
    
    print(f" Cross-Station Moisture Metrics for {col}:")
    for h in horizons:
        actual = test_data.iloc[:h]
        pred = full_forecast.iloc[:h]
        mae = mean_absolute_error(actual, pred)
        
        horizon_name = f"{h} hours" if h != 168 else "1 week"
        # Adjusted print to show 5 decimal places and correct SWC units
        print(f"   ↳ Station 2 MAE for {horizon_name}: {mae:.5f} m³/m³")
    print("\n")
    
    # Clean up memory cache
    del model
    del results