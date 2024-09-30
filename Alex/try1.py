import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


# Load your dataset
def load_data(filepath):
    """
    Loads the dataset from the specified CSV file, with the first column 
    parsed as datetime and used as the index. Removes duplicate indices.
    """
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)  # Parse first column as datetime and use as index
    data = data[~data.index.duplicated(keep='first')]  # Remove duplicate indices
    return data

# Load the data
filepath = '../datasets/Revised_Final_Data/Station1_Revised_Final_Data.csv'
df = load_data(filepath)

# Set the DateTime index and drop unnecessary columns (e.g., Latitude, Longitude)
df.set_index('Ppt', inplace=True)
df = df.drop(['Latitude', 'Longitude'], axis=1)

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

    def plot(self, model=None, plot_col='SWC_5', max_subplots=3):
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
        plt.show()

# Split data into train, validation, and test sets
n = len(df)
train_df = df.iloc[0:int(n*0.7)]  # First 70% for training
val_df = df.iloc[int(n*0.7):int(n*0.9)]  # Next 20% for validation
test_df = df.iloc[int(n*0.9):]  # Last 10% for testing

# Create a WindowGenerator instance for 24*7 hours of inputs and 24 hours for prediction
input_width = 24 * 7  
label_width = 24       
shift = 1            

window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, 
                         train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['SWC_5'])


# Define models in a dictionary
models = {
    'Dense': tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=24)
    ]),
    'Multi-step Dense': tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(units=512, activation='relu'),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=24)
    ]),
    'CNN': tf.keras.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_width, len(train_df.columns))),
        layers.GlobalAveragePooling1D(),  # This layer reduces the time dimension to a single vector
        layers.Dense(units=24)  # Output 24-hour predictions
    ]),
    'RNN': tf.keras.Sequential([
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(24)
    ]),
    'LSTM': tf.keras.Sequential([
        layers.LSTM(64, return_sequences=False),
        layers.Dense(24)
    ]),
    'Autoregressive': tf.keras.Sequential([
        layers.Dense(units=128, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(units=24)
    ]),
    'Linear': tf.keras.Sequential([
        layers.GlobalAveragePooling1D(),
        layers.Dense(units=24)  # Output a vector of length 24 for 24-hour predictions
    ])
}

# Compile all models in the dictionary
for name, model in models.items():
    model.compile(optimizer='adam', loss='mae')

# Early stopping callback to stop training when validation loss does not improve
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model_losses = {}

# Train, evaluate, and visualize all models
for name, model in models.items():
    print(f'\nTraining {name} model...')
    # Train the model
    history = model.fit(
        window.train,
        validation_data=window.val,
        epochs=10,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(window.test)
    model_losses[name] = test_loss
    print(f'{name} Model Test Loss: {test_loss}')
    
    # Plot the predictions for each model
    # window.plot(model=model, plot_col='SWC_5')


min_loss_model = min(model_losses, key=model_losses.get)
max_loss_model = max(model_losses, key=model_losses.get)

print(f'\nModel with the least loss: {min_loss_model} - Loss: {model_losses[min_loss_model]}')
print(f'Model with the most loss: {max_loss_model} - Loss: {model_losses[max_loss_model]}')
