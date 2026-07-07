import os
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string
import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate
import warnings
warnings.filterwarnings("ignore")

# 1. Setup the data pathway
data_dir = "../../datasets/New_Revised_Final_Data/"
file_path = os.path.join(data_dir, "Station1_filled_Data.csv")

print("🚀 Starting StatsForecast AutoARIMA Optimization for Station 1...")

# 2. Load the dataset
raw_df = pd.read_csv(file_path)

# Create a clean, hourly timeline for the data
raw_df['ds'] = pd.date_range(start="2022-01-01 00:00:00", periods=len(raw_df), freq="h")

# We will target SWC_5 (the shallow layer) since it has the most sudden changes
target_depth = 'SWC_5'

# 3. Transform the data into the strict 3-column format StatsForecast requires!
df = pd.DataFrame({
    'unique_id': 'Station_1',
    'ds': raw_df['ds'],
    'y': raw_df[target_depth]
})

# Split data: Save the final week (168 hours) for the test
Y_train_df = df.iloc[:-168]
Y_test_df = df.iloc[-168:]

# 4. Set up the AutoARIMA engine
season_length = 24  # Look back 24 hours for daily cycles
models = [AutoARIMA(season_length=season_length)]

# Tell StatsForecast to run using hourly frequency ('h')
sf = StatsForecast(models=models, freq='h', n_jobs=-1)

print(f"⚙️ Running automated model search on {target_depth}... please wait...")
sf.fit(df=Y_train_df)

# 5. Print the optimal combination
optimal_model = arima_string(sf.fitted_[0, 0].model_)
print(f"\n🏆 THE OPTIMAL MODEL CONFIGURATION IS: {optimal_model.strip()}")

# 6. Generate forecasts for the final 1 week window
print("🔮 Generating predictions...")
Y_hat_df = sf.forecast(df=Y_train_df, h=168)

# 7. Check the scores (MAE and RMSE) across our 3 timelines
horizons = [24, 72, 168]
print(f"\n📊 Performance Metrics for {target_depth}:")

for h in horizons:
    horizon_name = f"{h} hours" if h != 168 else "1 week"
    
    # Grab the slices for this timeline
    actuals_slice = Y_test_df.iloc[:h]
    preds_slice = Y_hat_df.iloc[:h]
    
    # Merge them together to pass to the evaluation function
    merged_slice = pd.merge(actuals_slice, preds_slice, on=['unique_id', 'ds'])
    
    # Let the evaluation function calculate the scores
    metrics_summary = evaluate(
        df=merged_slice,
        metrics=[ufl.mae, ufl.rmse],
        train_df=Y_train_df
    )
    
    mae_score = metrics_summary.loc[metrics_summary['metric'] == 'mae', 'AutoARIMA'].values[0]
    rmse_score = metrics_summary.loc[metrics_summary['metric'] == 'rmse', 'AutoARIMA'].values[0]
    
    print(f" ↳ Horizon [{horizon_name}]:")
    print(f"    - MAE (Chill Judge) : {mae_score:.5f} m³/m³")
    print(f"    - RMSE (Strict Judge): {rmse_score:.5f} m³/m³")