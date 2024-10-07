import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import csv


# Load your dataset
def load_data(filepath):

    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = data[~data.index.duplicated(keep='first')]
    return data

# Load the data
filepath = './datasets/Revised_Final_Data/Station3_Revised_Final_Data.csv'
df = load_data(filepath)

# Set the DateTime index and drop unnecessary columns (e.g., Latitude, Longitude)
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
# n = len(df)
# train_df = df.iloc[0:int(n*0.7)]
# val_df = df.iloc[int(n*0.7):int(n*0.9)]
# test_df = df.iloc[int(n*0.9):]

  
num_features = df.shape[1] - 3
label_features = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']

configurations = [
        {"features": 'SWC_5', "input_steps": 24, "output_steps": 1},
        {"features": 'SWC_5', "input_steps": 24, "output_steps": 6},
        {"features": 'SWC_5', "input_steps": 48, "output_steps": 12},
        {"features": 'SWC_5', "input_steps": 7*24, "output_steps": 24},
        {"features": 'SWC_5', "input_steps": 7*24, "output_steps": 48},
        {"features": 'SWC_10', "input_steps": 24, "output_steps": 1},
        {"features": 'SWC_10', "input_steps": 24, "output_steps": 6},
        {"features": 'SWC_10', "input_steps": 48, "output_steps": 12},
        {"features": 'SWC_10', "input_steps": 7*24, "output_steps": 24},
        {"features": 'SWC_10', "input_steps": 7*24, "output_steps": 48},
        {"features": 'SWC_20', "input_steps": 24, "output_steps": 1},
        {"features": 'SWC_20', "input_steps": 24, "output_steps": 6},
        {"features": 'SWC_20', "input_steps": 48, "output_steps": 12},
        {"features": 'SWC_20', "input_steps": 7*24, "output_steps": 24},
        {"features": 'SWC_20', "input_steps": 7*24, "output_steps": 48},
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
        if name == 'ARIMA':  # Handle ARIMA separately
            print(f'\nTraining {name} model for configuration: {config}')
            y_train = train_df['SWC_5'].values
            y_test = test_df['SWC_5'].values
            arima_model = ARIMA(y_train, order=(24, 1, 0))
            arima_model_fit = arima_model.fit()
            
            predictions = arima_model_fit.forecast(steps=len(y_test))
            
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)

            performance[name] = {
                'mean_absolute_error': mae,
                'mse': mse,
                'rmse': rmse
            }
            print(f"ARIMA Model Test MAE: {performance[name]['mean_absolute_error']:.4f}")
        else:
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
            # test_results = model.evaluate(window.test, return_dict=True)
            # val_results = model.evaluate(window.val, return_dict=True)
            
            # test_predictions = model.predict(window.test)
            # test_labels = next(iter(window.test))[1].numpy()  # Get true labels
            
            # mse = mean_squared_error(test_labels.flatten(), test_predictions.flatten())
            # mae = mean_absolute_error(test_labels.flatten(), test_predictions.flatten())
            # mape = mean_absolute_percentage_error(test_labels.flatten(), test_predictions.flatten())
            
            # performance[name] = {'mae': mae, 'mse': mse, 'mape': mape}
            # val_performance[name] = val_results
            performance[name] = model.evaluate(window.test, return_dict = True)
            val_performance[name] = model.evaluate(window.val, return_dict = True)
            print(f'{name} Model Test Loss: {performance[name]}')
    
    min_loss_model = min(performance, key=lambda k: performance[k]['mean_absolute_error'])
    max_loss_model = max(performance, key=lambda k: performance[k]['mean_absolute_error'])

    print(f'\nModel with the lowest MAE: {min_loss_model} - MAE: {performance[min_loss_model]}')
    print(f'Model with the highest MAE: {max_loss_model} - MAE: {performance[max_loss_model]}')
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


# print summaries for each configuration and each model
def print_model_summaries(models, all_losses):
    print("\nModel Performance Summary Across All Configurations:")
    
    for config_name, config_result in all_losses.items():
        print(f"\nConfiguration: {config_name}")
        test_results, val_results = config_result
        for model_name, performance in test_results.items():            

            # Print the performance metrics for this model
            if 'loss' in performance:  # Neural network models have 'loss'
                print(f"Model: {model_name} - Loss: {performance['loss']:.4f}, MAE: {performance['mean_absolute_error']:.4f}")
            else:
                print(f"Model: {model_name} - MSE: {performance['mse']:.4f}, RMSE: {performance['rmse']:.4f}, "
                      f"MAE: {performance['mean_absolute_error']:.4f}")
        min_loss_model = min(test_results, key=lambda k: test_results[k]['mean_absolute_error'])
        max_loss_model = max(test_results, key=lambda k: test_results[k]['mean_absolute_error'])

        min_result = test_results[min_loss_model]['mean_absolute_error']
        max_result = test_results[max_loss_model]['mean_absolute_error']
        print(f'\nModel with the lowest MAE: {min_loss_model} - MAE: {min_result}')
        print(f'Model with the highest MAE: {max_loss_model} - MAE: {max_result}')

def write_results_to_csv(all_losses, filename='model_results.csv'):
    fieldnames = ['Configuration', 'Model', 'MAE', 'MSE', 'MAPE']
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for config_name, (test_results, _) in all_losses.items():
            for model_name, metrics in test_results.items():
                writer.writerow({
                    'Configuration': config_name,
                    'Model': model_name,
                    'MAE': metrics['mean_absolute_error'],
                    'MSE': metrics['mean_squared_error'],
                    'MAPE': metrics['mean_absolute_percentage_error']
                })

# Initialize a dictionary to hold losses across configurations
all_losses = {}

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
    # make way to drop other features besides one in config
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
            tf.keras.layers.Dense(label_width * num_features,
                            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([label_width, num_features])
        ]),
        #6'ARIMA': 'arima_model'
    }
    # Train and evaluate models for the current configuration
    performance, val_performance = train_and_evaluate_models(config, models, train_df, val_df, test_df)
    model_losses = (performance, val_performance)
    # Store the losses for this configuration
    all_losses[f"{config['features']} - {config['input_steps']} input / {config['output_steps']} output"] = model_losses


# Call the function to print the summaries
print_model_summaries(models, all_losses)
write_results_to_csv(all_losses, filename='model_results.csv')

for config_name, losses in all_losses.items():
    print(f"\nPlotting performance for configuration: {config_name}")
    
    # Plot the performance for this configuration
    plot_performance(losses[0], losses[1], title=config_name)