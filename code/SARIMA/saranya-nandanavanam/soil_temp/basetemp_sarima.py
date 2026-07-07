import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Load only Station 1 data
data_dir = "../datasets/New_Revised_Final_Data/"
file_path = os.path.join(data_dir, "Station1_filled_Data.csv")

print(" Starting Complete Baseline SARIMA Evaluation for Station 1 Only...\n")

df = pd.read_csv(file_path)
df.index = pd.date_range(start="2022-01-01 00:00:00", periods=len(df), freq="h")

depths = ['T_5', 'T_10', 'T_20', 'T_50']
horizons = [24, 72, 168]

for depth in depths:
    print(f"--- Training Model for Depth: {depth} ---")
    
    train_data = df[depth].iloc[:-168]
    test_data = df[depth].iloc[-168:]
    
    # Using 'bfgs' solver via method_kwargs to drastically reduce memory overhead
    model = ARIMA(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    results = model.fit(method_kwargs={'method': 'bfgs'}) 
    
    full_forecast = results.forecast(steps=168)
    
    print(f"📊 Results for {depth}:")
    for h in horizons:
        actual = test_data.iloc[:h]
        pred = full_forecast.iloc[:h]
        mae = mean_absolute_error(actual, pred)
        
        horizon_name = f"{h} hours" if h != 168 else "1 week"
        print(f"   ↳ MAE for {horizon_name}: {mae:.2f}°C")
    print("\n")
    
    # Explicitly clear memory
    del model
    del results