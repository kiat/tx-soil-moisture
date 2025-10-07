import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg') 

# Load your CSV
file_path = 'results_ws48_offset168_SWC_10.csv'
df = pd.read_csv(file_path)

# Set the model names as the row index
df.set_index('Model', inplace=True)

# Select only the desired metric columns
metrics = ['MSE', 'MAE', 'SMAPE', 'RSE', 'CORR']
df_metrics = df[metrics]

# Truncate to 4 decimal places (no rounding)
df_metrics_trunc = df_metrics.applymap(lambda x: f"{x:.4f}")

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, len(df_metrics_trunc) * 0.6))

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.axis('off')

# Create table
table = ax.table(
    cellText=df_metrics_trunc.values,
    rowLabels=df_metrics_trunc.index,
    colLabels=df_metrics_trunc.columns,
    cellLoc='center',
    loc='center'
)

table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(12)

plt.title('Model Evaluation Metrics', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('model_metrics_table.png', dpi=300)
plt.close()
