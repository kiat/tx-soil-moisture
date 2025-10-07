import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import os, csv

model_dir = './saved_models/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Load your dataset
def load_data(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = data[~data.index.duplicated(keep='first')]
    return data

# Load the data
filepath = './datasets/Revised_Final_Data/Station3_Revised_Final_Data.csv'
df = load_data(filepath)

# Set the DateTime index and drop unnecessary columns (e.g., Latitude, Longitude)
#df.set_index('Ppt', inplace=True)
df = df.drop(['Latitude', 'Longitude'], axis=1)
wv = df.pop('Windspeed')

# # Convert to radians.
wd_rad = df.pop('Winddirection')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

timestamp_s = (df.index).map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
# Normalize the data
train_mean = df.mean()
train_std = df.std()

def normalize(data):
    return (data - train_mean) / train_std

df = normalize(df)
# Define the WindowGenerator class
class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    def plot(self, model_name, config, model=None, plot_col='SWC_5', max_subplots=3):
        """
        Plots the actual vs predicted values for the given plot column.
        """
        inputs, labels = next(iter(self.test))  # Use the test dataset for evaluation
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_subplots, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            label_col_index = self.label_columns_indices.get(plot_col, plot_col_index)

            # Plot actual labels
            plt.scatter(self.label_indices, labels[n, :, label_col_index], label='Labels', edgecolors='k', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)
                # Plot predictions
                plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.title(model_name + " " + config)
        plt.show()

# Split data into train, validation, and test sets
n = len(df)
train_df = df.iloc[0:int(n*0.7)]  # First 70% for training
val_df = df.iloc[int(n*0.7):int(n*0.9)]  # Next 20% for validation
test_df = df.iloc[int(n*0.9):]  # Last 10% for testing

  
num_features = df.shape[1]
label_features = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']


configurations = [
       {"features": 'SWC_5', "input_steps": 48, "output_steps": 12},
        {"features": 'SWC_10', "input_steps": 48, "output_steps": 12},
        {"features": 'SWC_20', "input_steps": 48, "output_steps": 12},
    ]


#Second version plotting comparison of performance
def create_window(input_width, label_width, shift, train_df, val_df, test_df, label_columns):
    return WindowGenerator(
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=label_columns
    )

# Function to train and evaluate all models for a given configuration
def train_and_evaluate_models(config, models, train_df, val_df, test_df):
    # Adjust the window based on the current configuration
    window = create_window(
        input_width=config['input_steps'],
        label_width=config['output_steps'],
        shift=config['output_steps'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=[config['features']]
    )
    
    performance = {}
    val_performance = {}
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train, evaluate, and store losses for each model
    for name, model in models.items():
    
        print(f'\nTraining {name} model for configuration: {config}')
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])
        history = model.fit(
            window.train,
            validation_data=window.val,
            epochs=10,
            callbacks=[early_stopping]
        )
        
        # Evaluate the model on the test set
        performance[name] = model.evaluate(window.test, return_dict = True)
        val_performance[name] = model.evaluate(window.val, return_dict = True)
        #model_losses[name] = test_loss
        print(f'{name} Model Test Loss: {performance[name]}')
    
    # min_loss_model = min(performance, key=lambda k: performance[k]['mean_absolute_error'])
    # max_loss_model = max(performance, key=lambda k: performance[k]['mean_absolute_error'])

    # print(f'\nModel with the lowest MAE: {min_loss_model} - MAE: {performance[min_loss_model]}')
    # print(f'Model with the highest MAE: {max_loss_model} - MAE: {performance[max_loss_model]}')
    return performance, val_performance

# Function to plot model performance
# Maybe pass in Config name and also store config name in configurations dictionary
def plot_performance(multi_val_performance, multi_test_performance, title='Model Comparison'):
    x = np.arange(len(multi_test_performance))
    width = 0.3

    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in multi_val_performance.values()]
    test_mae = [v[metric_name] for v in multi_test_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=multi_test_performance.keys(), rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    plt.title(title)
    plt.legend()
    plt.show()

def calculate_original_performance(models, config, train_df, val_df, test_df):
    original_performance = {}
    window = create_window(
        input_width=config['input_steps'],
        label_width=config['output_steps'],
        shift=config['output_steps'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=[config['features']]
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    for model_name, model in models.items():
        print(f"\nEvaluating original performance for model: {model_name}")

        # Compile and train each model using the full dataset
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])
        history = model.fit(
            window.train,
            validation_data=window.val,
            epochs=10,
            callbacks=[early_stopping]
        )

        # Evaluate on the full test set to get the original MAE
        performance = model.evaluate(window.test, return_dict=True)
        original_performance[model_name] = performance
        write_results_to_csv(config['features'], 'Baseline', model_name, performance)
        print(f"Original MAE for {model_name}: {performance['mean_absolute_error']:.4f}")

    return original_performance

def write_results_to_csv(label_feature, dropped_feature, model_name, metrics, csv_filename='feature_importance_results.csv'):
    """
    Writes the results of a model to a CSV file.
    
    Parameters:
    - label_feature (str): The label feature for the model
    - dropped_feature (str): The feature that was dropped for the model
    - model_name (str): The name of the model
    - metrics (dict): Dictionary containing the metrics for the model, e.g., test MSE, test MAE, validation MSE, validation MAE
    - csv_filename (str): The name of the CSV file to save results to (default is 'model_results.csv')
    
    Metrics dict format:
    {
        'test_mse': value,
        'test_mae': value,
        'val_mse': value,
        'val_mae': value
    }
    """

    # Ensure the metrics dictionary contains the necessary keys
    required_keys = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
    for key in required_keys:
        if key not in metrics:
            raise ValueError(f"Missing required metric: {key}")

    # Write results to the CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            label_feature,                # Label feature
            dropped_feature,              # Dropped feature
            model_name,                   # Model name
            metrics['mean_squared_error'],          # Test MSE
            metrics['mean_absolute_error'],          # Test MAE
            metrics['mean_absolute_percentage_error']
        ])

def drop_feature_and_evaluate(config, original_mae, train_df, val_df, test_df, features, target):
    feature_importance = {}
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    num_features = df_copy.shape[1]
    # Iterate over all features and drop one feature at a time
    for feature in features:
        print(f"\nEvaluating the effect of dropping feature: {feature}")
        
        # Drop the feature from the dataframe
        df_dropped = train_df.drop(columns=[feature])
        
        # Update the window for training without the feature
        new_window = create_window(
            input_width=config['input_steps'],
            label_width=config['output_steps'],
            shift=config['output_steps'],
            train_df=df_dropped,  # Use the dataframe with the feature dropped
            val_df=val_df.drop(columns=[feature]),  # Similarly for validation data
            test_df=test_df.drop(columns=[feature]),  # Similarly for test data
            label_columns=[target]
        )
        label_width = config['output_steps']
        num_features = df_dropped.shape[1]
        # Initialize dictionary to store the MAE changes for each model
        mae_diffs = {}
        models = {
            'Baseline': tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: x[:, -label_width:, 1])  # predicts last swc_5 value
            ]),
            'Multi-step Linear': tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                # Shape => [batch, 1, out_steps*features]
                tf.keras.layers.Dense(label_width*num_features,
                                    kernel_initializer=tf.initializers.zeros()),
                layers.Reshape([label_width, num_features])
            ]),
            'Multi-step Dense': tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                layers.Dense(units=512, activation='relu'),
                tf.keras.layers.Dense(label_width*num_features,
                                kernel_initializer=tf.initializers.zeros()),
                layers.Reshape([label_width, num_features])
            ]),
            'CNN': tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
                # Shape => [batch, 1, conv_units]
                tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
                # Shape => [batch, 1,  out_steps*features]
                tf.keras.layers.Dense(label_width*num_features,
                                    kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([label_width, num_features])
            ]),
            
            'RNN': tf.keras.Sequential([
                layers.SimpleRNN(64, return_sequences=False),
                tf.keras.layers.Dense(label_width * num_features,
                                kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([label_width, num_features])
            ]),
            'LSTM': tf.keras.Sequential([
                layers.LSTM(64, return_sequences=False),
                tf.keras.layers.Dense(label_width * num_features,
                                kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([label_width, num_features])
            ]),
            'Autoregressive': tf.keras.Sequential([
                layers.Dense(units=128, activation='relu'),
                layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(label_width * num_features,
                                kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([label_width, num_features])
            ]),
            'Bi-LSTM': tf.keras.models.Sequential([
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation = 'relu', return_sequences=True)),
                #tf.keras.layers.Flatten(),  # Flatten the input to 1D
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
                tf.keras.layers.Dense(32, activation='relu'),
                #tf.keras.layers.Dense(units=label_width, activation='tanh'),
                tf.keras.layers.Dense(label_width * num_features,
                                kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([label_width, num_features])
            ]),
        }
        for model_name, model in models.items():
            print(f"\nRetraining model: {model_name} after dropping feature: {feature}")
            # Recompile and retrain the model with the updated data (feature dropped)
            model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])
            history = model.fit(
                new_window.train,
                validation_data=new_window.val,
                epochs=10,
                callbacks=[early_stopping]
            )

            # Evaluate the model on the test data without the dropped feature
            performance = model.evaluate(new_window.test, return_dict=True)
            write_results_to_csv(config['features'], feature, model_name, performance)
            # Calculate the change in MAE
            mae_diff = performance['mean_absolute_error'] - original_mae[model_name]['mean_absolute_error']
            mae_diffs[model_name] = mae_diff
            model_save_path = os.path.join(model_dir, f"{model_name}_drop_{feature}.keras")
            model.save(model_save_path)
            print(f"Model: {model_name} - Original MAE: {original_mae[model_name]['mean_absolute_error']:.4f}, New MAE: {performance['mean_absolute_error']:.4f}, MAE Change: {mae_diff:.4f}")

        # Store the results of this feature drop in the feature_importance dictionary
        feature_importance[feature] = mae_diffs

    return feature_importance



# Initialize a dictionary to hold losses across configurations
all_losses = {}
csv_filename = 'feature_importance_results.csv'
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow([
            'Label Feature', 'Dropped Feature', 'Model Name', 
            'Test MSE', 'Test MAE', 'Test MAPE'
        ])
# Loop through each configuration and compare models
for config in configurations:
    print(f"\nEvaluating models for configuration: {config}")
    CONV_WIDTH = 3
    # Define models in a dictionary
    label_width = config['output_steps']
    n = len(df)
    df_copy = df.copy()
    label = config['features']
    features_to_drop = [col for col in label_features if col != label]

    # Drop the features
    df_copy = df_copy.drop(columns=features_to_drop)
    train_df = df_copy.iloc[0:int(n*0.7)]
    val_df = df_copy.iloc[int(n*0.7):int(n*0.9)]
    test_df = df_copy.iloc[int(n*0.9):]
    num_features = df_copy.shape[1]
    models = {
        'Baseline': tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -label_width:, 1])  # predicts last swc_5 value
        ]),
        'Multi-step Linear': tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(label_width*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            layers.Reshape([label_width, num_features])
        ]),
        'Multi-step Dense': tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(label_width*num_features,
                            kernel_initializer=tf.initializers.zeros()),
            layers.Reshape([label_width, num_features])
        ]),
        'CNN': tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(label_width*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([label_width, num_features])
        ]),
        
        'RNN': tf.keras.Sequential([
            layers.SimpleRNN(64, return_sequences=False),
            tf.keras.layers.Dense(label_width * num_features,
                            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([label_width, num_features])
        ]),
        'LSTM': tf.keras.Sequential([
            layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(label_width * num_features,
                            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([label_width, num_features])
        ]),
        'Autoregressive': tf.keras.Sequential([
            layers.Dense(units=128, activation='relu'),
            layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(label_width * num_features,
                            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([label_width, num_features])
        ]),
        'Bi-LSTM': tf.keras.models.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation = 'relu', return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
            tf.keras.layers.Dense(32, activation='relu'),
            #tf.keras.layers.Dense(units=label_width, activation='tanh'),
            tf.keras.layers.Dense(label_width * num_features,
                            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([label_width, num_features])
        ]),
    }

    # Example usage:
    print("\nDetermining feature importance...")
    # Assume 'all_features' contains the list of features in your dataset
    all_features = train_df.columns.tolist()
    target_feature = config['features']
    all_features.remove(target_feature)

    # Create a dictionary to store the original MAE values before feature dropping
    original_mae = calculate_original_performance(models, config, train_df, val_df, test_df)

    # Call the function
    feature_importance_results = drop_feature_and_evaluate(config, original_mae, train_df, val_df, test_df, all_features, target_feature)

    # Print the feature importance results
    for feature, importance in feature_importance_results.items():
        print(f"\nFeature: {feature}")
        for model_name, mae_diff in importance.items():
            print(f"Model: {model_name} - MAE Change: {mae_diff:.4f}")


# def print_model_summaries(models, all_losses):
#     print("\nModel Performance Summary Across All Configurations:")
    
#     for config_name, config_result in all_losses.items():
#         print(f"\nConfiguration: {config_name}")
#         test_results, val_results = config_result
#         for model_name, performance in test_results.items():            

#             # Print the performance metrics for this model
#             if 'loss' in performance:  # Neural network models have 'loss'
#                 print(f"Model: {model_name} - Loss: {performance['loss']:.4f}, MAE: {performance['mean_absolute_error']:.4f}")
#             else:  # Non-neural models might have different metrics
#                 print(f"Model: {model_name} - MSE: {performance['mse']:.4f}, RMSE: {performance['rmse']:.4f}, "
#                       f"MAE: {performance['mean_absolute_error']:.4f}")
#         min_loss_model = min(test_results, key=lambda k: test_results[k]['mean_absolute_error'])
#         max_loss_model = max(test_results, key=lambda k: test_results[k]['mean_absolute_error'])

#         min_result = test_results[min_loss_model]['mean_absolute_error']
#         max_result = test_results[max_loss_model]['mean_absolute_error']
#         print(f'\nModel with the lowest MAE: {min_loss_model} - MAE: {min_result}')
#         print(f'Model with the highest MAE: {max_loss_model} - MAE: {max_result}')

# # Call the function to print the summaries
# print_model_summaries(models, all_losses)
# for config_name, losses in all_losses.items():
#     print(f"\nPlotting performance for configuration: {config_name}")
    
#     # Plot the performance for this configuration
#     plot_performance(losses[0], losses[1], title=config_name)