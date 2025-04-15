import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
file_path = 'results_ws168_offset168_SWC_10_Ppt_Tair_YearSin_YearCos.csv'
df = pd.read_csv(file_path)

# Select metrics to include in the radar chart
metrics = ['MSE', 'MAE', 'MAPE', 'SMAPE', 'RSE', 'CORR']
models = df['Model'].tolist()
data = df[metrics]

# Normalize metrics (invert lower-is-better ones)
def normalize(column, inverse=False):
    col = column.copy()
    if inverse:
        col = col.max() - col
    return (col - col.min()) / (col.max() - col.min())

# Normalize each column appropriately
normalized_data = pd.DataFrame()
for metric in metrics:
    inverse = metric != 'CORR'  # CORR is higher-is-better
    normalized_data[metric] = normalize(df[metric], inverse=inverse)

# Setup radar chart
labels = metrics
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, model in enumerate(models):
    values = normalized_data.iloc[i].tolist()
    values += values[:1]
    ax.plot(angles, values, label=model, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

# Format
ax.set_title('Model Performance Radar Chart', size=16, pad=30)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

# Save to file
plt.savefig('radar_chart_model_metrics'+file_path+'.png', dpi=300)
plt.close()
