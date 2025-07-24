import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Step 1: Load all files
path = './'  # folder where your CSVs are located
all_files = glob.glob(os.path.join(path, "results_ws168_offset*_SWC_10*.csv"))

df_list = [pd.read_csv(file) for file in all_files]
df_all = pd.concat(df_list, ignore_index=True)

# Step 2: Filter and sort
metrics = ['MSE', 'MAE', 'MAPE', 'SMAPE', 'RSE']
df_all['Offset'] = df_all['Offset'].astype(int)
df_all = df_all.sort_values(by='Offset')

# Step 3: Plot performance over offset for each model (averaged over features)
models = df_all['Model'].unique()

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for model in models:
        df_model = df_all[df_all['Model'] == model]
        avg_per_offset = df_model.groupby('Offset')[metric].mean()
        plt.plot(avg_per_offset.index, avg_per_offset.values, label=model, marker='o')
    
    plt.title(f'{metric} vs Offset (NO CORR)')
    plt.xlabel('Offset')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{metric.lower()}_vs_offset_NO_CORR.png', dpi=300)
    plt.close()
