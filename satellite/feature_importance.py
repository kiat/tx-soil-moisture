import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
import os

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :-1]
        target = data[i+seq_length, -1]
        sequences.append(seq)
        targets.append(target)
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)


# Load and preprocess data
data = pd.read_csv('satellite/all_merged_data/Station1_AMSR_SMAP_Merged.csv', index_col=0, parse_dates=True)

# Drop all NaN values
data = data.ffill()

# Change winddirection to 0 when windspeed is 0
data.loc[data['Windspeed'] == 0, 'Winddirection'] = 0

# Define features and targets
features = ['Ppt', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'Srad']
targets = ['Sat_SM_SMAP', 'Sat_SM_AMSR']

# Standardize the data (divide by standard deviation)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features + targets])
scaled_df = pd.DataFrame(scaled_data, columns=features + targets, index=data.index)

seq_length = 24  
X, y_smap = create_sequences(scaled_data[:, :-1], seq_length)
_, y_amsr = create_sequences(scaled_data[:, [0, 1, 2, 3, 4, 5, 7]], seq_length)

# Ensure y is 2D
y_smap = y_smap.view(-1, 1)
y_amsr = y_amsr.view(-1, 1)

# RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# CNN model
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 11, 50) 
        self.fc2 = nn.Linear(50, output_size)
    
    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Function to train and evaluate a model
def train_and_evaluate(model_type, X_train, y_train, X_test, y_test):
    if model_type == 'rnn':
        model = RNNModel(X_train.shape[2])
    elif model_type == 'cnn':
        model = CNNModel(X_train.shape[2])
    else:
        raise ValueError("Invalid model type")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):  # You can adjust the number of epochs
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    mse = mean_squared_error(y_test.numpy(), predictions.numpy())
    mae = mean_absolute_error(y_test.numpy(), predictions.numpy())
    mae_percentage = mae / np.std(y_test.numpy()) * 100  # Use standard deviation instead of mean
    r2 = max(r2_score(y_test.numpy(), predictions.numpy()), 0)  # Ensure R² is never negative
    return mse, mae, mae_percentage, r2

def evaluate_feature_importance(model_type, X, y_smap, y_amsr, features):
    X_train, X_test, y_train_smap, y_test_smap, y_train_amsr, y_test_amsr = train_test_split(X, y_smap, y_amsr, test_size=0.2, random_state=42)
    
    results_smap = train_and_evaluate(model_type, X_train, y_train_smap, X_test, y_test_smap)
    results_amsr = train_and_evaluate(model_type, X_train, y_train_amsr, X_test, y_test_amsr)
    
    base_mse_smap, base_mae_smap, base_mae_percentage_smap, base_r2_smap = results_smap
    base_mse_amsr, base_mae_amsr, base_mae_percentage_amsr, base_r2_amsr = results_amsr
    
    importance_smap = {metric: {} for metric in ['mse', 'mae', 'mae_percentage', 'r2']}
    importance_amsr = {metric: {} for metric in ['mse', 'mae', 'mae_percentage', 'r2']}
    
    print(f"{model_type.upper()} Base Performance:")
    print(f"SMAP - MSE: {base_mse_smap:.4f}, MAE: {base_mae_smap:.4f}, MAE%: {base_mae_percentage_smap:.2f}%, R²: {base_r2_smap:.4f}")
    print(f"AMSR - MSE: {base_mse_amsr:.4f}, MAE: {base_mae_amsr:.4f}, MAE%: {base_mae_percentage_amsr:.2f}%, R²: {base_r2_amsr:.4f}")
    
    for feature in features:
        X_without_feature = X.clone()
        X_without_feature[:, :, features.index(feature)] = 0
        X_train_wf, X_test_wf, y_train_smap_wf, y_test_smap_wf, y_train_amsr_wf, y_test_amsr_wf = train_test_split(X_without_feature, y_smap, y_amsr, test_size=0.2, random_state=42)
        
        results_smap_wf = train_and_evaluate(model_type, X_train_wf, y_train_smap_wf, X_test_wf, y_test_smap_wf)
        results_amsr_wf = train_and_evaluate(model_type, X_train_wf, y_train_amsr_wf, X_test_wf, y_test_amsr_wf)
        
        for i, metric in enumerate(['mse', 'mae', 'mae_percentage', 'r2']):
            importance_smap[metric][feature] = results_smap_wf[i] - results_smap[i]
            importance_amsr[metric][feature] = results_amsr_wf[i] - results_amsr[i]
    
    return importance_smap, importance_amsr, results_smap, results_amsr

# Modify the custom_ar_importance function to handle both targets
def custom_ar_importance(data, features, targets, lag=3):
    importance = {target: {metric: {} for metric in ['mse', 'mae', 'mae_percentage', 'r2']} for target in targets}
    results = {target: None for target in targets}
    target_indices = [len(features) + i for i in range(len(targets))]
    
    for target, target_idx in zip(targets, target_indices):
        X = np.column_stack([data[lag-i:-i, :-len(targets)] for i in range(1, lag+1)])
        y = data[lag:, target_idx]
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        mae_percentage = mae / np.std(y) * 100
        r2 = max(r2_score(y, predictions), 0)  # Ensure R² is never negative
        
        results[target] = (mse, mae, mae_percentage, r2)
        
        for feature_idx, feature in enumerate(features):
            X_without_feature = X.copy()
            X_without_feature[:, feature_idx::len(features)] = 0
            
            predictions_wf = model.predict(X_without_feature)
            mse_wf = mean_squared_error(y, predictions_wf)
            mae_wf = mean_absolute_error(y, predictions_wf)
            mae_percentage_wf = mae_wf / np.std(y) * 100
            r2_wf = max(r2_score(y, predictions_wf), 0)  # Ensure R² is never negative
            
            importance[target]['mse'][feature] = mse_wf - mse
            importance[target]['mae'][feature] = mae_wf - mae
            importance[target]['mae_percentage'][feature] = mae_percentage_wf - mae_percentage
            importance[target]['r2'][feature] = r2 - r2_wf
    
    print("Custom AR Base Performance:")
    for target in targets:
        mse, mae, mae_percentage, r2 = results[target]
        print(f"{target} - MSE: {mse:.4f}, MAE: {mae:.4f}, MAE%: {mae_percentage:.2f}%, R²: {r2:.4f}")
    
    return importance, results

# Function to display feature importance
def display_feature_importance(importance_dict, model_name, metric_name, target):
    if importance_dict is None or len(importance_dict) == 0:
        print(f"No feature importance data available for {model_name} ({metric_name}) - {target}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(list(importance_dict[metric_name].items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=False)
    
    # Display tabular data
    print(f"\n{model_name} Feature Importance for {target} ({metric_name}):")
    print(df.to_string(index=False))

# Main execution block
if __name__ == "__main__":
    # List of station files
    station_files = [
        'Station1_AMSR_SMAP_Merged.csv',
        'Station2_AMSR_SMAP_Merged.csv',
        'Station3_AMSR_SMAP_Merged.csv',
        'Station4_AMSR_SMAP_Merged.csv',
        'Station5_AMSR_SMAP_Merged.csv',
        'Station6_AMSR_SMAP_Merged.csv'
    ]

    # Create a single CSV file for all results
    csv_filename = os.path.join('satellite', 'all_stations_satellite_sm_results.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Dictionary to store summary results
        summary_results = {}
        
        for station_file in station_files:
            print(f"\nProcessing {station_file}...")
            station_name = station_file.split('_')[0]
            summary_results[station_name] = {}
            
            # Load and preprocess data
            data = pd.read_csv(os.path.join('satellite', 'all_merged_data', station_file), index_col=0, parse_dates=True)

            # ... rest of the data preprocessing ...

            # Evaluate feature importance for RNN
            print("Starting RNN feature importance calculation...")
            try:
                rnn_importance_smap, rnn_importance_amsr, rnn_results_smap, rnn_results_amsr = evaluate_feature_importance('rnn', X, y_smap, y_amsr, features)
                print("RNN feature importance calculation completed.")
            except Exception as e:
                print(f"Error in RNN feature importance calculation: {str(e)}")
                rnn_importance_smap, rnn_importance_amsr, rnn_results_smap, rnn_results_amsr = None, None, None, None

            # Evaluate feature importance for CNN
            print("Starting CNN feature importance calculation...")
            try:
                cnn_importance_smap, cnn_importance_amsr, cnn_results_smap, cnn_results_amsr = evaluate_feature_importance('cnn', X, y_smap, y_amsr, features)
                print("CNN feature importance calculation completed.")
            except Exception as e:
                print(f"Error in CNN feature importance calculation: {str(e)}")
                cnn_importance_smap, cnn_importance_amsr, cnn_results_smap, cnn_results_amsr = None, None, None, None

            # Calculate custom AR importance and scores
            print("Starting Custom AR importance calculation...")
            ar_importance, ar_results = custom_ar_importance(scaled_data, features, targets)

            # Write results to CSV file
            for model_name, importance_smap, importance_amsr, results_smap, results_amsr in [
                ("RNN", rnn_importance_smap, rnn_importance_amsr, rnn_results_smap, rnn_results_amsr),
                ("CNN", cnn_importance_smap, cnn_importance_amsr, cnn_results_smap, cnn_results_amsr),
                ("Custom AR", ar_importance['Sat_SM_SMAP'], ar_importance['Sat_SM_AMSR'], ar_results['Sat_SM_SMAP'], ar_results['Sat_SM_AMSR'])
            ]:
                for target, importance, results in [("Sat_SM_SMAP", importance_smap, results_smap), ("Sat_SM_AMSR", importance_amsr, results_amsr)]:
                    if importance is not None:
                        for metric in ['mse', 'mae', 'mae_percentage', 'r2']:
                            for feature, value in importance[metric].items():
                                csvwriter.writerow([station_name, model_name, target, metric, feature, f"{value:.4f}"])
                
                if results_smap is not None and results_amsr is not None:
                    csvwriter.writerow([station_name, model_name, 'Sat_SM_SMAP', 'performance'] + [f"{value:.4f}" for value in results_smap])
                    csvwriter.writerow([station_name, model_name, 'Sat_SM_AMSR', 'performance'] + [f"{value:.4f}" for value in results_amsr])
                    summary_results[station_name][f"{model_name}_SMAP"] = results_smap
                    summary_results[station_name][f"{model_name}_AMSR"] = results_amsr

    print(f"\nAll results saved to {csv_filename}")

    # Print final summary
    print("\nFinal Summary:")
    for station, results in summary_results.items():
        print(f"\n{station}:")
        for model_target, metrics in results.items():
            print(f"  {model_target}:")
            print(f"    MSE: {metrics[0]:.4f}")
            print(f"    MAE: {metrics[1]:.4f}")
            print(f"    MAE%: {metrics[2]:.2f}%")
            print(f"    R²: {metrics[3]:.4f}")
