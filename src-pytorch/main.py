import argparse
import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Assuming 'models.py' contains the fully converted PyTorch models.
import models as model_module

# --- Helper for Early Stopping ---
class EarlyStopping:
    """
    Monitors validation loss and stops training when it stops improving.
    """
    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        """
        Checks if training should be stopped.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("--- Early stopping triggered ---")
            if self.restore_best_weights and self.best_model_state is not None:
                print("Restoring best model weights.")
                model.load_state_dict(self.best_model_state)
            return True
        return False

# --- Comprehensive Evaluation Function (FINAL FIX) ---
def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates a PyTorch model and returns metrics with keys that EXACTLY match
    the final CSV headers.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            
            if y_pred.shape != y_batch.shape:
                y_pred = y_pred.view_as(y_batch)
            
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
            
    all_preds = torch.cat(all_preds, dim=0).squeeze()
    all_targets = torch.cat(all_targets, dim=0).squeeze()
    
    epsilon = 1e-8
    num_samples = len(all_targets)

    mse = total_loss / num_samples
    mae = torch.mean(torch.abs(all_targets - all_preds)).item()
    mape = (torch.mean(torch.abs((all_targets - all_preds) / (all_targets + epsilon))) * 100) / num_samples
    smape_numerator = torch.abs(all_preds - all_targets)
    smape_denominator = torch.abs(all_targets) + torch.abs(all_preds) + epsilon
    smape = torch.mean(2 * smape_numerator / smape_denominator) * 100
    rse_numerator = torch.sum((all_preds - all_targets) ** 2)
    rse_denominator = torch.sum((all_targets - torch.mean(all_targets)) ** 2) + epsilon
    rse = (rse_numerator / rse_denominator).item()
    vx = all_targets - torch.mean(all_targets)
    vy = all_preds - torch.mean(all_preds)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + epsilon)

    # FINAL FIX: The keys now EXACTLY match the CSV header names.
    return {
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape.item(),
        'SMAPE': smape.item(),
        'RSE': rse,
        'CORR': corr.item()
    }

def main(args):
    """
    Main function to run the data processing, model training, and evaluation pipeline.
    """
    # Input validation
    if args.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {args.window_size}")
    if args.offset <= 0:
        raise ValueError(f"offset must be positive, got {args.offset}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.patience <= 0:
        raise ValueError(f"patience must be positive, got {args.patience}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preparation ---
    try:
        from core.data_helpers import read_and_process_csvs, engineer_features, split_and_stack_data, normalize_features, data_to_X_y
        from core.evaluation_helpers import write_loss_history_to_csv, write_model_results_to_csv
    except ImportError:
        print("Warning: 'core' module not found. Using placeholder data and dummy output functions.")
        def read_and_process_csvs(): return {f'Station{i+1}': pd.DataFrame(np.random.rand(100, 6), columns=['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']) for i in range(6)}
        def engineer_features(dfs): return dfs
        def split_and_stack_data(dfs, test_station_name): return dfs, dfs[test_station_name], dfs[test_station_name]
        def normalize_features(df, features): return df, None
        def data_to_X_y(df, window, offset): return np.random.rand(50, window, len(df.columns)), np.random.rand(50)
        def write_loss_history_to_csv(*args): pass
        # This dummy function will now receive the correct keys
        def write_model_results_to_csv(station, model_name, ws, offset, perf, features):
            print(f"DUMMY_WRITE: Writing results for {model_name}...")
            # This will now correctly find the keys
            print(f"  -> MSE: {perf.get('MSE', 'N/A')}, CORR: {perf.get('CORR', 'N/A')}")

    stations = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6']
    target_station = stations[-1]
    raw_dfs = read_and_process_csvs()
    engineered_dfs = engineer_features(raw_dfs)
    engineered_dfs, val_df, test_df = split_and_stack_data(engineered_dfs, test_station_name=target_station)
    all_features = args.features.split(',') if args.features else ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']

    # --- Prepare data and convert to PyTorch Tensors ---
    scaled_val, _ = normalize_features(val_df, all_features)
    X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)
    scaled_test, _ = normalize_features(test_df, all_features)
    X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)
    train_data_x, train_data_y = [], []
    for train_station in [s for s in stations if s != target_station]:
        scaled_train, _ = normalize_features(engineered_dfs[train_station], all_features)
        X_train_part, y_train_part = data_to_X_y(scaled_train, args.window_size, args.offset)
        train_data_x.append(X_train_part)
        train_data_y.append(y_train_part)
    X_train = np.concatenate(train_data_x, axis=0)
    y_train = np.concatenate(train_data_y, axis=0).reshape(-1, 1)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size)
    model_dir = "saved_models_pytorch"
    os.makedirs(model_dir, exist_ok=True)

    # --- Model Selection ---
    def normalize_id(name: str) -> str: return name.lower().replace("_", "").replace("model", "")
    requested_ids = set(normalize_id(n) for n in args.model_names.split(","))
    model_map = {
        'ar': model_module.AR, 'autoregressive': model_module.AR, 'lstm': model_module.LSTMModel, 
        'bilstm': model_module.BiLSTMModel, 'rnn': model_module.RNNModel, 'cnn': model_module.CNNModel,
        'attentionlstm': model_module.AttentionLSTM, 'attentiononly': model_module.AttentionOnly,
        'transformer': model_module.TransformerModel, 'multiheadlstm': model_module.MultiHeadLSTM,
        'baseline': model_module.Baseline, 'movingaverage': model_module.MovingAverageBaseline,
    }
    process_queue = {name: model_class for name, model_class in model_map.items() if name in requested_ids}
    print(f"Models to be processed: {list(process_queue.keys())}")
    
    # --- Training and Evaluation Loop ---
    for model_name, model_class in process_queue.items():
        print(f"\n===== Processing {model_name.upper()} =====")
        input_dim = X_train_t.shape[2]
        criterion = nn.MSELoss()
        
        if model_name in ["baseline", "movingaverage"]:
            print(f"Evaluating non-trainable baseline model: {model_name.upper()}")
            model = model_class().to(device)
            performance = evaluate_model(model, test_loader, criterion, device)
        else:
            print(f"Training {model_name.upper()}...")
            model = model_class(input_dim=input_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopper = EarlyStopping(patience=args.patience, restore_best_weights=True)
            history = {'loss': [], 'val_loss': []}
            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * X_batch.size(0)
                
                val_metrics = evaluate_model(model, val_loader, criterion, device)
                train_loss = epoch_loss / len(train_loader.dataset)
                val_loss = val_metrics['MSE'] 
                history['loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                if early_stopper(val_loss, model):
                    break
            
            print("\nFinal Evaluation on Test Set...")
            performance = evaluate_model(model, test_loader, criterion, device)
            feature_str = '_'.join(all_features)
            main_name = f"model_{model_name}_ws{args.window_size}_offset{args.offset}_{feature_str}"
            model_path = os.path.join(model_dir, f"{main_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"{model_name.upper()} model saved at {model_path}")
            write_loss_history_to_csv(target_station, model_name, args.window_size, args.offset, history, feature_str)

        print(f"\n--- {model_name.upper()} Final Test Metrics ---")
        for key, value in performance.items():
            print(f"{key}: {value:.6f}")
        
        # feature_str already calculated above - no need to recalculate
        write_model_results_to_csv(target_station, model_name, args.window_size, args.offset, performance, feature_str)
        print("-------------------------------------\n")

    print("All runs complete! Results have been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate time series models using PyTorch.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data sequences.')
    parser.add_argument('--offset', type=int, default=24, help='Prediction offset from the end of the window.')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument("--features", type=str, default="SWC_20,T_20,Ppt,Tair,Wx,Wy", help="Comma-separated list of features to use.")
    parser.add_argument("--model_names", type=str, default="LSTM,CNN,BiLSTM,RNN,AttentionLSTM,AR,Baseline", help="Comma-separated list of model names to run.")
    
    args = parser.parse_args()
    main(args)
