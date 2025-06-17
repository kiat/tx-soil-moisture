import matplotlib
matplotlib.use('Agg')  # headless-safe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'loss_history_ws168_offset168_SWC_10_YearSin_YearCos.csv'
df = pd.read_csv(file_path)
sns.set(style="whitegrid")

models = df['Model'].unique()

# === 1. One plot per model ===
for model in models:
    model_df = df[df['Model'] == model]
    min_val_epoch = model_df['Validation Loss'].idxmin()
    min_epoch = model_df.loc[min_val_epoch, 'Epoch']
    min_val = model_df.loc[min_val_epoch, 'Validation Loss']

    plt.figure(figsize=(8, 5))
    plt.plot(model_df['Epoch'], model_df['Loss'], label='Train Loss', color='blue')
    plt.plot(model_df['Epoch'], model_df['Validation Loss'], label='Validation Loss', color='orange', linestyle='--')

    # Marker for min val loss
    plt.scatter(min_epoch, min_val, color='red', zorder=5)
    plt.text(min_epoch, min_val, f'Min: {min_val:.4f}', color='red', fontsize=9, ha='right')

    plt.title(f'{model} - Training vs Validation Loss', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    filename = f'loss_curve_{model}.png'.replace(" ", "_")
    plt.savefig(filename, dpi=300)
    plt.close()

# === 2. Summary Train Loss Plot ===
plt.figure(figsize=(10, 6))
for model in models:
    model_df = df[df['Model'] == model]
    plt.plot(model_df['Epoch'], model_df['Loss'], label=model)
plt.title('Training Loss Across Models', pad=20)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('train_loss_across_models.png', dpi=300)
plt.close()

# === 3. Summary Validation Loss Plot ===
plt.figure(figsize=(10, 6))
for model in models:
    model_df = df[df['Model'] == model]
    min_val_epoch = model_df['Validation Loss'].idxmin()
    min_epoch = model_df.loc[min_val_epoch, 'Epoch']
    min_val = model_df.loc[min_val_epoch, 'Validation Loss']

    plt.plot(model_df['Epoch'], model_df['Validation Loss'], label=model, linestyle='--')
    plt.scatter(min_epoch, min_val, color='red', zorder=5)
    plt.text(min_epoch, min_val, f'{model} Min: {min_val:.4f}', fontsize=8, color='red', ha='right')

plt.title('Validation Loss Across Models', pad=20)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('validation_loss_across_models.png', dpi=300)
plt.close()
