import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

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
data = pd.read_csv('/Users/ayaannazir/Desktop/Research/tx-soil-moisture/datasets/Revised_Final_Data/Station4_Revised_Final_Data.csv', index_col=0, parse_dates=True)

# Drop all NaN values
data = data.dropna()

# Define features and target
features = ['Ppt', 'T_20', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'Srad']
target = ['SWC_20']

# Prepare data for SWC_20 prediction
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + target])

seq_length = 24  
X, y = create_sequences(scaled_data, seq_length)

# Ensure y is 2D
y = y.view(-1, 1)

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
    mae_percentage = mae / np.mean(y_test.numpy()) * 100
    return mse, mae, mae_percentage

def evaluate_feature_importance(model_type, X, y, features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_mse, base_mae, base_mae_percentage = train_and_evaluate(model_type, X_train, y_train, X_test, y_test)
    importance_mse = {}
    importance_mae = {}
    importance_mae_percentage = {}
    
    print(f"{model_type.upper()} Base Performance:")
    print(f"MSE: {base_mse:.4f}")
    print(f"MAE: {base_mae:.4f}")
    print(f"MAE Percentage: {base_mae_percentage:.2f}%")
    
    for feature in features:
        X_without_feature = X.clone()
        X_without_feature[:, :, features.index(feature)] = 0
        X_train_wf, X_test_wf, y_train_wf, y_test_wf = train_test_split(X_without_feature, y, test_size=0.2, random_state=42)
        mse_without_feature, mae_without_feature, mae_percentage_without_feature = train_and_evaluate(model_type, X_train_wf, y_train_wf, X_test_wf, y_test_wf)
        importance_mse[feature] = mse_without_feature - base_mse
        importance_mae[feature] = mae_without_feature - base_mae
        importance_mae_percentage[feature] = mae_percentage_without_feature - base_mae_percentage
    
    return importance_mse, importance_mae, importance_mae_percentage, base_mse, base_mae, base_mae_percentage

# Evaluate feature importance for RNN
print("Starting RNN feature importance calculation...")
try:
    rnn_importance_mse, rnn_importance_mae, rnn_importance_mae_percentage, rnn_mse, rnn_mae, rnn_mae_percentage = evaluate_feature_importance('rnn', X, y, features)
    print("RNN feature importance calculation completed.")
    print("RNN Feature Importance (MSE):")
    for feature, importance in sorted(rnn_importance_mse.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
except Exception as e:
    print(f"Error in RNN feature importance calculation: {str(e)}")
    rnn_importance_mse, rnn_importance_mae, rnn_importance_mae_percentage, rnn_mse, rnn_mae, rnn_mae_percentage = None, None, None, None, None, None

# Evaluate feature importance for CNN
print("Starting CNN feature importance calculation...")
try:
    cnn_importance_mse, cnn_importance_mae, cnn_importance_mae_percentage, cnn_mse, cnn_mae, cnn_mae_percentage = evaluate_feature_importance('cnn', X, y, features)
    print("CNN feature importance calculation completed.")
    print("CNN Feature Importance (MSE):")
    for feature, importance in sorted(cnn_importance_mse.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
except Exception as e:
    print(f"Error in CNN feature importance calculation: {str(e)}")
    cnn_importance_mse, cnn_importance_mae, cnn_importance_mae_percentage, cnn_mse, cnn_mae, cnn_mae_percentage = None, None, None, None, None, None

# Custom autoregressive model for feature importance
def custom_ar_importance(data, features, target, lag=3):
    importance = {}
    target_idx = len(features)
    
    for feature in features:
        feature_idx = features.index(feature)
        X = np.column_stack([data[lag-i:-i, feature_idx] for i in range(1, lag+1)])
        y = data[lag:, target_idx]
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        importance[feature] = mse
    
    return importance

# Calculate custom AR importance
ar_importance = custom_ar_importance(scaled_data, features, 0)  # 0 is the index of SWC_20 in the target list

# Function to adjust values
def adjust_values(x):
    return x.replace({0: 0.000001, 1: 0.999999})

# Function to display feature importance
def display_feature_importance(importance_dict, model_name, metric_name):
    if importance_dict is None or len(importance_dict) == 0:
        print(f"No feature importance data available for {model_name} ({metric_name})")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=False)
    
    # Display tabular data
    print(f"\n{model_name} Feature Importance for SWC_20 ({metric_name}):")
    print(df.to_string(index=False))

# Display feature importance for all metrics
for model_name, importance_mse, importance_mae, importance_mae_percentage in [
    ("RNN", rnn_importance_mse, rnn_importance_mae, rnn_importance_mae_percentage),
    ("CNN", cnn_importance_mse, cnn_importance_mae, cnn_importance_mae_percentage)
]:
    display_feature_importance(importance_mse, model_name, "MSE")
    display_feature_importance(importance_mae, model_name, "MAE")
    display_feature_importance(importance_mae_percentage, model_name, "MAE Percentage")

# Compare all models
if all(importance is not None for importance in [rnn_importance_mse, cnn_importance_mse, ar_importance]):
    all_models_mse = pd.DataFrame({
        'RNN (MSE)': pd.Series(rnn_importance_mse),
        'CNN (MSE)': pd.Series(cnn_importance_mse),
        'Custom AR': pd.Series(ar_importance)
    })
    all_models_mae = pd.DataFrame({
        'RNN (MAE)': pd.Series(rnn_importance_mae),
        'CNN (MAE)': pd.Series(cnn_importance_mae),
    })
    all_models_mae_percentage = pd.DataFrame({
        'RNN (MAE %)': pd.Series(rnn_importance_mae_percentage),
        'CNN (MAE %)': pd.Series(cnn_importance_mae_percentage),
    })

    # Normalize and adjust the values for each model
    all_models_normalized = pd.concat([
        all_models_mse.apply(lambda x: adjust_values((x - x.min()) / (x.max() - x.min()))),
        all_models_mae.apply(lambda x: adjust_values((x - x.min()) / (x.max() - x.min()))),
        all_models_mae_percentage.apply(lambda x: adjust_values((x - x.min()) / (x.max() - x.min())))
    ], axis=1)

    print("\nComparison of Feature Importance Across Models for 20 Prediction (Normalized and Adjusted):")
    print(all_models_normalized.to_string())
else:
    print("Not all model importance data is available for comparison")

# Model Performance Summary
print("\nModel Performance Summary:")
print(f"RNN - MSE: {rnn_mse:.4f}, MAE: {rnn_mae:.4f}, MAE Percentage: {rnn_mae_percentage:.2f}%")
print(f"CNN - MSE: {cnn_mse:.4f}, MAE: {cnn_mae:.4f}, MAE Percentage: {cnn_mae_percentage:.2f}%")

# After all calculations, add this section to create the CSV file
print("Saving results to CSV file...")

with open('station4_swc_20_results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write headers
    csvwriter.writerow(['Model', 'Metric', 'Feature', 'Importance'])
    
    # Write RNN results
    for metric, importance_dict in [('MSE', rnn_importance_mse), ('MAE', rnn_importance_mae), ('MAE Percentage', rnn_importance_mae_percentage)]:
        if importance_dict:
            for feature, importance in importance_dict.items():
                csvwriter.writerow(['RNN', metric, feature, f"{importance:.4f}"])
    
    # Write CNN results
    for metric, importance_dict in [('MSE', cnn_importance_mse), ('MAE', cnn_importance_mae), ('MAE Percentage', cnn_importance_mae_percentage)]:
        if importance_dict:
            for feature, importance in importance_dict.items():
                csvwriter.writerow(['CNN', metric, feature, f"{importance:.4f}"])
    
    # Write Custom AR results
    if ar_importance:
        for feature, importance in ar_importance.items():
            csvwriter.writerow(['Custom AR', 'MSE', feature, f"{importance:.4f}"])
    
    # Write model performance summary
    csvwriter.writerow([])
    csvwriter.writerow(['Model Performance Summary'])
    csvwriter.writerow(['Model', 'MSE', 'MAE', 'MAE Percentage'])
    csvwriter.writerow(['RNN', f"{rnn_mse:.4f}", f"{rnn_mae:.4f}", f"{rnn_mae_percentage:.2f}%"])
    csvwriter.writerow(['CNN', f"{cnn_mse:.4f}", f"{cnn_mae:.4f}", f"{cnn_mae_percentage:.2f}%"])

print("Results saved to station4_swc_20_results.csv")