import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Load all result CSVs
path = './'
all_files = glob.glob(os.path.join(path, "results_ws168_offset*_SWC_10*.csv"))
df_list = [pd.read_csv(file) for file in all_files]
df_all = pd.concat(df_list, ignore_index=True)

# Metrics to analyze
metrics = ['MSE', 'MAE', 'MAPE', 'SMAPE', 'RSE']

# Group by Feature combination and take mean across all models/offsets
df_grouped = df_all.groupby('Features')[metrics].mean().sort_values('MSE')

# Plot each metric
for metric in metrics:
    plt.figure(figsize=(12, 6))
    df_grouped[metric].plot(kind='bar')
    plt.title(f'{metric} by Feature Combination (NO CORR)')
    plt.ylabel(metric)
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(f'feature_effect_{metric.lower()}_NO_CORR.png', dpi=300)
    plt.close()
