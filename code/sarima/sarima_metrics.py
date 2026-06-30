import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (CHANGE DEPTH HERE)
# ==========================================
# TARGET_DEPTH = 'SWC_20'   # or 'SWC_50'
TARGET_DEPTH = 'SWC_50'
# SARIMA order selected via prior grid search
ORDER = (0,1,2)
SEASONAL_ORDER = (0, 0, 0, 24)

# Minimum hours of history before first forecast
INIT_TRAIN_HOURS = 30 * 24  # 720 h

# ==========================================
# 2. PATH CONFIGURATION & DATA LOADING
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = CURRENT_DIR
while os.path.basename(BASE_DIR) != 'tx-soil-moisture' and BASE_DIR != os.path.dirname(BASE_DIR):
    BASE_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'datasets', 'New_Revised_Final_Data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(BASE_DIR, 'soil_predict', 'datasets', 'New_Revised_Final_Data')

STATIONS = [
    'Station1_filled_Data.csv', 'Station2_filled_Data.csv', 'Station3_filled_Data.csv',
    'Station4_filled_Data.csv', 'Station5_filled_Data.csv', 'Station6_filled_Data.csv',
]

print(f"Initializing data loader for 2020 datasets [{TARGET_DEPTH}]...")
station_data = {}

for s in STATIONS:
    path = os.path.join(DATA_DIR, s)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df_2020 = df.loc['2020'].asfreq('h').ffill().bfill()
    station_data[s] = df_2020[TARGET_DEPTH]

print("✓ Data loaded.")

# ==========================================
# 3. BUILD TRAINING SIGNAL & TEST TARGET
# ==========================================
# Training signal: hourly mean across Stations 2-6
train_series = pd.concat(
    [station_data[s] for s in STATIONS[1:]], axis=1
).mean(axis=1)

# Test target: Station 1 actuals
test_series = station_data['Station1_filled_Data.csv']

print(f"Train signal length : {len(train_series)} hours  (mean of Stations 2-6)")
print(f"Test  target length : {len(test_series)} hours  (Station 1)")

# ==========================================
# 4. ROLLING H-STEP-AHEAD SARIMA EVALUATION
# ==========================================
horizons = {'24h': 24, '48h': 48, '72h': 72, '1-Week': 168}
results = []

for name, H in horizons.items():
    print(f"\nEvaluating horizon {name} (H={H})...")
    preds, actuals = [], []

    # Refit every H steps; advance the window in H-hour blocks
    steps = range(INIT_TRAIN_HOURS, len(train_series) - H, H)
    total_steps = len(steps)

    for i, t in enumerate(steps):
        if i % 20 == 0:
            print(f"  [{i+1}/{total_steps}] t={t}")

        # Fit on the growing training window up to t
        # enforce_stationarity=True keeps AR roots inside the unit circle,
        # preventing the optimizer from drifting to explosive params that
        # cause NaN forecasts with higher-order AR terms like AR(4).
        model = SARIMAX(
            train_series.iloc[:t],
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        res = model.fit(disp=False, maxiter=500)
        

        # Produce H-step-ahead out-of-sample forecast
        fc = res.forecast(steps=H)
        h_step_pred = float(fc.iloc[-1])
        h_step_actual = float(test_series.iloc[t + H - 1])

        preds.append(h_step_pred)
        actuals.append(h_step_actual)

    preds = np.array(preds)
    actuals = np.array(actuals)

    mse = float(np.mean((actuals - preds) ** 2))
    mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)
    results.append({'Horizon': name, 'MSE': mse, 'MAPE (%)': mape})
    print(f"  → MSE={mse:.6f}  MAPE={mape:.4f}%")

# ==========================================
# 5. DISPLAY RESULTS & AUTO-SAVE TO CSV
# ==========================================
print("\n" + "=" * 45)
print(f"      SARIMA METRICS ({TARGET_DEPTH})     ")
print("=" * 45)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("=" * 45)

output_dir = os.path.join(BASE_DIR, 'outputs')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f'sarima_metrics_{TARGET_DEPTH}.csv')
df_results.to_csv(output_file, index=False)
print(f"\nResults saved to:\n{output_file}")
