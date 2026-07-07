import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
TARGET_DEPTH = 'SWC_50'       # Options: 'SWC_5', 'SWC_10', 'SWC_20', 'SWC_50'
PPT_COLUMN = 'Ppt'           

# Pre-selected structural settings
ORDER = (0, 1, 2)
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

print(f"Initializing data loader for 2020 datasets [{TARGET_DEPTH} + {PPT_COLUMN}]...")
station_swc = {}
station_ppt = {}

for s in STATIONS:
    path = os.path.join(DATA_DIR, s)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df_2020 = df.loc['2020'].asfreq('h').ffill().bfill()
    
    station_swc[s] = df_2020[TARGET_DEPTH]
    station_ppt[s] = df_2020[PPT_COLUMN]

print("✓ Data loaded.")

# ==========================================
# 3. BUILD TRAINING SIGNALS & TEST TARGET
# ==========================================
# Target training signal: hourly mean SWC across Stations 2-6
train_swc = pd.concat([station_swc[s] for s in STATIONS[1:]], axis=1).mean(axis=1)

# Exogenous feature training signal: hourly mean Precipitation across Stations 2-6
train_ppt = pd.concat([station_ppt[s] for s in STATIONS[1:]], axis=1).mean(axis=1)

# Test target: Station 1 actual SWC
test_swc = station_swc['Station1_filled_Data.csv']

print(f"Train SWC signal length : {len(train_swc)} hours (mean of Stations 2-6)")
print(f"Train PPT feature length: {len(train_ppt)} hours (mean of Stations 2-6)")
print(f"Test SWC target length  : {len(test_swc)} hours (Station 1)")

# ==========================================
# 4. ROLLING H-STEP-AHEAD SARIMAX EVALUATION
# ==========================================
horizons = {'24h': 24, '48h': 48, '72h': 72, '1-Week': 168}
results = []

for name, H in horizons.items():
    print(f"\nEvaluating horizon {name} (H={H}) with Exogenous Ppt...")
    preds, actuals = [], []

    steps = range(INIT_TRAIN_HOURS, len(train_swc) - H, H)
    total_steps = len(steps)

    for i, t in enumerate(steps):
        if i % 20 == 0:
            print(f"  [{i+1}/{total_steps}] t={t}")

        # Pass the historical precipitation data as the exogenous variable (exog)
        model = SARIMAX(
            train_swc.iloc[:t],
            exog=train_ppt.iloc[:t],
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        res = model.fit(disp=False, maxiter=500)
        
        # Pull future precipitation data for the H-step forecast horizon window
        future_exog = train_ppt.iloc[t : t + H]
        
        # Produce H-step-ahead out-of-sample forecast passing the future rain data
        fc = res.forecast(steps=H, exog=future_exog)
        h_step_pred = float(fc.iloc[-1])
        h_step_actual = float(test_swc.iloc[t + H - 1])

        preds.append(h_step_pred)
        actuals.append(h_step_actual)

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Calculate metrics
    mse = float(np.mean((actuals - preds) ** 2))
    rmse = float(np.sqrt(mse))                     
    mae = float(np.mean(np.abs(actuals - preds)))  
    
    results.append({
        'Horizon': name, 
        'RMSE': rmse, 
        'MAE': mae
    })
    print(f"  → RMSE={rmse:.6f}  MAE={mae:.6f}")

# ==========================================
# 5. DISPLAY RESULTS & AUTO-SAVE TO CSV
# ==========================================
print("\n" + "=" * 55)
print(f"   SARIMAX EXOGENOUS METRICS ({TARGET_DEPTH} + PPT)   ")
print("=" * 55)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("=" * 55)

output_dir = os.path.join(BASE_DIR, 'outputs')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f'sarimax_ppt_metrics_{TARGET_DEPTH}.csv')
df_results.to_csv(output_file, index=False)
print(f"\nResults saved to:\n{output_file}")