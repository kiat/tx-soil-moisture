import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()
    plt.close()  # Clear the figure after saving or showing

def plot_predictions(y_true, y_pred, target_col, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(y_pred, label='Predictions')
    plt.plot(y_true, label='Actuals')
    plt.xlabel('Time Step')
    plt.ylabel(target_col)
    plt.legend()
    plt.title('Test Predictions vs Actuals')
    if save_path:
        plt.savefig(save_path)
        print(f"Predictions plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
