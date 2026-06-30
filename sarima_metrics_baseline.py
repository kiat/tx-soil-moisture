import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ==========================================
# 1. CONFIGURATION (CHANGE DEPTH HERE)
# ==========================================
TARGET_DEPTH = 'SWC_50' 

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

stations = [
    'Station1_filled_Data.csv', 'Station2_filled_Data.csv', 'Station3_filled_Data.csv',
    'Station4_filled_Data.csv', 'Station5_filled_Data.csv', 'Station6_filled_Data.csv'
]

print(f"Initializing data loader for 2020 datasets [{TARGET_DEPTH}]...")
station_data = {}

for s in stations:
    path = os.path.join(DATA_DIR, s)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
        
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df_2020 = df.loc['2020'].asfreq('h').ffill().bfill()
    station_data[s] = df_2020[TARGET_DEPTH]

print("✓ Data loaded perfectly.")

# ==========================================
# 3. PREPARE TRAINING DATA (Stations 2-6) WITH LAGS
# ==========================================
print("\nFormatting lag features from Stations 2-6...")
X_train_list, y_train_list = [], []

# Use a 24-hour lookback window to predict future states
for s in ['Station2_filled_Data.csv', 'Station3_filled_Data.csv', 'Station4_filled_Data.csv', 'Station5_filled_Data.csv', 'Station6_filled_Data.csv']:
    series = station_data[s].values
    for t in range(24, len(series) - 168):
        X_train_list.append(series[t-24:t])
        y_train_list.append(series[t])

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

# ==========================================
# 4. TRAIN THE FAST BASELINE MODEL
# ==========================================
print("Fitting Fast Linear Baseline Model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✓ Model training complete!")

# ==========================================
# 5. TESTING HORIZONS EVALUATION (Station 1)
# ==========================================
print(f"\nEvaluating multi-horizon forecasts on Station 1 for {TARGET_DEPTH}...")
test_values = station_data['Station1_filled_Data.csv'].values
total_len = len(test_values)

horizons = {'24h': 24, '48h': 48, '72h': 72, '1-Week': 168}
results = []

# Build test matrices out right away using vectorization
for name, steps in horizons.items():
    X_test_list, y_test_list = [], []
    
    for t in range(24, total_len - steps):
        X_test_list.append(test_values[t-24:t])
        y_test_list.append(test_values[t + steps])
        
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    # Predict everything instantly in one vectorized calculation pass
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    results.append({'Horizon': name, 'MSE': mse, 'MAPE (%)': mape})

# ==========================================
# 6. DISPLAY RESULTS & AUTO-SAVE TO CSV
# ==========================================
print("\n" + "="*40)
print(f"       BASELINE METRICS ({TARGET_DEPTH})     ")
print("="*40)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("="*40)

output_dir = os.path.join(BASE_DIR, 'soil_predict', 'outputs')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f'linreg_baseline_{TARGET_DEPTH}_metrics.csv')
df_results.to_csv(output_file, index=False)
print(f"\n📊 Results automatically saved to:\n➡️ {output_file}")