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

# Assuming original data helpers and model_entry are compatible or adapted
from core.data_helpers import read_and_process_csvs, engineer_features, split_and_stack_data, normalize_features, data_to_X_y, concatenate_with_gaps, plot_split_timeline
from core.evaluation_helpers import write_loss_history_to_csv, write_model_results_to_csv
import models as model_module

# Helper for Early Stopping
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping triggered.")
            if self.restore_best_weights and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
            return True
        return False

# Evaluation function for PyTorch models
def evaluate_model_pytorch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_mape = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            
            # Metrics
            rmse = torch.sqrt(loss)
            mape = torch.mean(torch.abs((y_batch - y_pred) / y_batch)) * 100
            total_rmse += rmse.item() * X_batch.size(0)
            total_mape += mape.item() * X_batch.size(0)
            
    num_samples = len(dataloader.dataset)
    return {
        'mean_squared_error': total_loss / num_samples,
        'root_mean_squared_error': total_rmse / num_samples,
        'mean_absolute_percentage_error': total_mape / num_samples,
    }

def main(args):
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preparation (largely unchanged) ---
    stations = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6']
    target_station = stations[-1]
    raw_dfs = read_and_process_csvs()
    engineered_dfs = engineer_features(raw_dfs)
    engineered_dfs, val_df, test_df = split_and_stack_data(engineered_dfs, test_station_name=target_station)
    all_features = args.features.split(',') if args.features else ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']

    if args.visualize:
        # Visualization logic remains the same
        # ...
        return

    # Prepare data and convert to PyTorch Tensors
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

    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size)

    model_dir = "saved_models_pytorch"
    os.makedirs(model_dir, exist_ok=True)

    # --- Model Selection (adapted for PyTorch classes) ---
    def normalize_id(name: str) -> str:
        return name.lower().replace("_", "").replace("model", "")
    
    requested_ids = set(normalize_id(n) for n in args.model_names.split(","))
    process_queue = {}
    
    # Map normalized names to PyTorch classes
    model_map = {
        'ar': model_module.AR, 'autoregressive': model_module.AR,
        'lstm': model_module.LSTMModel,
        'bilstm': model_module.BiLSTMModel,
        'rnn': model_module.RNNModel,
        'cnn': model_module.CNNModel,
        'attentionlstm': model_module.AttentionLSTM,
        'attentiononly': model_module.AttentionOnly,
        'transformer': model_module.TransformerModel,
        'multiheadlstm': model_module.MultiHeadLSTM,
        'baseline': model_module.Baseline,
        'movingaverage': model_module.MovingAverageBaseline,
    }

    for name, model_class in model_map.items():
        if name in requested_ids:
            process_queue[name] = model_class

    print(f"Models to be processed: {list(process_queue.keys())}")
    
    # --- Training and Evaluation Loop ---
    for model_name, model_class in process_queue.items():
        print(f"\nTraining {model_name.upper()} across stations...\n")
        
        # Handle Baseline models
        if model_name in ["baseline", "movingaverage"]:
            model = model_class()
            y_pred = model.predict(X_test_t)
            y_true = y_test_t.numpy().flatten()
            performance = {
                'mean_squared_error': np.mean((y_true - y_pred)**2)
            }
            write_model_results_to_csv(target_station, model_name, args.window_size, args.offset, performance, '_'.join(all_features))
            print(f"{model_name.upper()} Final Test Loss: {performance['mean_squared_error']}\n")
            continue

        # Instantiate model
        input_dim = X_train_t.shape[2]
        if model_name == 'cnn':
            model = model_class(seq_len=args.window_size, input_dim=input_dim).to(device)
        else:
            model = model_class(input_dim=input_dim).to(device)

        criterion = nn.MSELoss()
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
            
            # Validation
            val_metrics = evaluate_model_pytorch(model, val_loader, criterion, device)
            train_loss = epoch_loss / len(train_loader.dataset)
            val_loss = val_metrics['mean_squared_error']
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if early_stopper(val_loss, model):
                break
        
        print("\nFinal Evaluation on Test Set...\n")
        performance = evaluate_model_pytorch(model, test_loader, criterion, device)

        # Save model and results
        feature_str = '_'.join(all_features)
        main_name = f"model_{model_name}_ws{args.window_size}_offset{args.offset}_{feature_str}"
        model_path = os.path.join(model_dir, f"{main_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"{model_name.upper()} saved at {model_path}")
        
        write_model_results_to_csv(target_station, model_name, args.window_size, args.offset, performance, feature_str)
        write_loss_history_to_csv(target_station, model_name, args.window_size, args.offset, history, feature_str)
        
        print(f"{model_name.upper()} Final Test Loss (MSE): {performance['mean_squared_error']:.6f}\n")

    print("All Runs Complete! All results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train various models for time series prediction using PyTorch.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data')
    parser.add_argument('--offset', type=int, default=24, help='Offset for prediction')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument("--features", type=str, default="SWC_20,T_20,Ppt,Tair,Wx,Wy", help="Comma-separated list of features to use in training")
    parser.add_argument("--model_names", type=str, default="LSTM,CNN,BiLSTM,RNN,AttentionLSTM,AR,Baseline", help="Comma-separated list of model names")
    parser.add_argument('--visualize', action='store_true', help='If true, plots train/val/test splits instead of running models')

    args = parser.parse_args()
    main(args)