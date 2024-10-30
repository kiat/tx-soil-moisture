import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

csv_path = './datasets/Revised_Final_Data/Station3_Revised_Final_Data.csv'
#df = df[5::6]
print(os.getcwd())
df = pd.read_csv(csv_path, parse_dates = ["Date"], index_col="Date")
# Slice [start:stop:step], starting from index 5 take every 6th record.
#df = df[5::6]
print(df.head())
wv = df.pop('Windspeed')
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

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

def plot(self, model=None, plot_col='SWC_5', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')
  plt.show()


WindowGenerator.plot = plot

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

OUT_STEPS = 24 * 2
# multi_window = WindowGenerator(input_width=24 * 7,
#                                label_width=OUT_STEPS,
#                                shift=OUT_STEPS)

# multi_window.plot()
# print(multi_window)

# multi_val_performance = {}
# multi_performance = {}

CONV_WIDTH = 3
# multi_conv_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
#     # Shape => [batch, 1, conv_units]
#     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
#     # Shape => [batch, 1,  out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros()),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])

# history = compile_and_fit(multi_conv_model, multi_window)

# IPython.display.clear_output()

# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val, return_dict=True)
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)
# multi_window.plot(multi_conv_model)

# metric_name = 'mean_absolute_error'
# val_mae = [v[metric_name] for v in multi_val_performance.values()]
# test_mae = [v[metric_name] for v in multi_performance.values()]

# for name, value in multi_performance.items():
#   print(f'{name:8s}: {value[metric_name]:0.4f}')

# Store original features
df = df.drop(columns=['Latitude', 'Longitude'])
original_features = df.columns

# Dictionary to store MAE for each feature
mae_results = {}

# Function to remove a feature and rerun the model
def evaluate_feature_importance(feature_to_remove, original_df):
    # Create a new dataframe without the selected feature
    temp_df = original_df.drop(columns=[feature_to_remove])
    
    # Split data into train, validation, and test sets again
    train_df = temp_df[0:int(n*0.7)]
    val_df = temp_df[int(n*0.7):int(n*0.9)]
    test_df = temp_df[int(n*0.9):]

    # Standardize data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # Create a new WindowGenerator with the updated dataset
    window = WindowGenerator(input_width=24*7, label_width=OUT_STEPS, shift=OUT_STEPS,
                             train_df=train_df, val_df=val_df, test_df=test_df)
    
    # Compile and fit the model
    CONV_WIDTH = 3
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        tf.keras.layers.Dense(OUT_STEPS*(len(temp_df.columns)),
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, len(temp_df.columns)])
    ])
    
    # Train the model
    history = compile_and_fit(model, window)

    # Evaluate model on validation set and return MAE
    performance = model.evaluate(window.val, return_dict=True)
    return performance['mean_absolute_error']

# Loop through each feature, remove it, and store the MAE
for feature in original_features:
    mae = evaluate_feature_importance(feature, df)
    mae_results[feature] = mae
    print(f"Feature: {feature}, MAE: {mae}")

# Find most and least important features
most_important_feature = max(mae_results, key=mae_results.get)
least_important_feature = min(mae_results, key=mae_results.get)


for feature in original_features:
    print(f"Feature: {feature}, MAE: {mae_results[feature]}")
print(f"Most important feature: {most_important_feature}, MAE: {mae_results[most_important_feature]}")
print(f"Least important feature: {least_important_feature}, MAE: {mae_results[least_important_feature]}")