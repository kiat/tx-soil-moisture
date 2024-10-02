import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import operator


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


# all models here
models = {
    # 'Dense': tf.keras.Sequential([
    #     layers.Flatten(),
    #     layers.Dense(units=128, activation='relu'),
    #     layers.Dense(units=24)
    # ]),
    # 'Multi-step Dense': tf.keras.Sequential([
    #     layers.Flatten(),
    #     layers.Dense(units=512, activation='relu'),
    #     layers.Dense(units=128, activation='relu'),
    #     layers.Dense(units=24)
    # ]),
    # 'CNN': tf.keras.Sequential([
    #     layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_width, len(train_df.columns))),
    #     layers.GlobalAveragePooling1D(),  
    #     layers.Dense(units=24)
    # ]),
    # 'RNN': tf.keras.Sequential([
    #     layers.SimpleRNN(64, return_sequences=False),
    #     layers.Dense(24)
    # ]),
    # 'LSTM': tf.keras.Sequential([
    #     layers.LSTM(64, return_sequences=False),
    #     layers.Dense(24)
    # ]),
    # 'Autoregressive': tf.keras.Sequential([
    #     layers.Dense(units=128, activation='relu'),
    #     layers.GlobalAveragePooling1D(),
    #     layers.Dense(units=24)
    # ]),
    # 'Linear': tf.keras.Sequential([
    #     layers.GlobalAveragePooling1D(),
    #     layers.Dense(units=24)
    # ]), 
    'Bi-LSTM': tf.keras.Sequential([
        layers.Bidirectional(layers.LSTM(128, activation='relu', input_shape=(shape[0], 21))),
        layers.Dense(64, activation='relu'),
        layers.Dense(units=24)  # Output a vector of length 24 for 24-hour predictions
    ])
}

for name, model in models.items():
    model.compile(optimizer='adam', loss='mae')

# Early stopping callback to stop training when validation loss does not improve
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_losses = {}

# Dictionary to store the feature importance for each model
model_feature_importance = {}

all_features = list(train_df.columns)
all_features.remove('SWC_5')  # Assuming 'SWC_5' is the label column

# Dictionary to store the overall performance of each model (with all features)
overall_model_losses = {}



# # Train and evaluate each model with all features (to get the baseline loss)
# print("\nTraining models with all features (baseline)...")
# for name, model in models.items():
#     print(f'\nTraining {name} model with all features...')
    
#     # Reinitialize model
#     model = tf.keras.models.clone_model(model)
#     model.compile(optimizer='adam', loss='mae')
    
#     # Train the model with early stopping
#     history = model.fit(
#         window.train,
#         validation_data=window.val,
#         epochs=10,
#         callbacks=[early_stopping]
#     )
    
#     # Evaluate the model on the test set and store the test loss as the baseline
#     baseline_loss = model.evaluate(window.test)
#     overall_model_losses[name] = baseline_loss
    
#     print(f'{name} Model Baseline Test Loss (with all features): {baseline_loss}')

# # Now perform feature ablation
# for name, model in models.items():
#     print(f'\nEvaluating feature importance for model: {name}')
    
#     # Dictionary to track how much the loss increases when each feature is removed
#     feature_loss_diffs = {}
    
#     # Iterate through each feature, removing it from the dataset
#     for feature_to_remove in all_features:
#         print(f'\nTraining {name} model without feature: {feature_to_remove}')
        
#         # Remove the selected feature from the dataset
#         train_df_mod = train_df.drop(columns=[feature_to_remove])
#         val_df_mod = val_df.drop(columns=[feature_to_remove])
#         test_df_mod = test_df.drop(columns=[feature_to_remove])

#         # Create a new WindowGenerator instance with the modified dataset
#         window_mod = WindowGenerator(
#             input_width=input_width,
#             label_width=label_width,
#             shift=shift,
#             train_df=train_df_mod,
#             val_df=val_df_mod,
#             test_df=test_df_mod,
#             label_columns=['SWC_5']
#         )
        
#         # Reinitialize the model to ensure fresh training
#         model = tf.keras.models.clone_model(model)
#         model.compile(optimizer='adam', loss='mae')
        
#         # Train the model with early stopping
#         history = model.fit(
#             window_mod.train,
#             validation_data=window_mod.val,
#             epochs=10,
#             callbacks=[early_stopping]
#         )
        
#         # Evaluate the model on the test set without the removed feature
#         ablation_loss = model.evaluate(window_mod.test)
        
#         # Calculate the difference in loss (how much worse the model performed without the feature)
#         loss_diff = ablation_loss - overall_model_losses[name]
#         feature_loss_diffs[feature_to_remove] = loss_diff

#         print(f'{name} Model Test Loss without {feature_to_remove}: {ablation_loss} (Loss Difference: {loss_diff})')
    
#     # Sort features by their importance (features with the largest loss increase are most important)
#     sorted_feature_importance = sorted(feature_loss_diffs.items(), key=operator.itemgetter(1), reverse=True)
    
#     # Store the sorted feature importance for this model
#     model_feature_importance[name] = sorted_feature_importance
    
#     print(f'\nFeature importance for {name} model (most to least important):')
#     for feature, loss_diff in sorted_feature_importance:
#         print(f'  {feature}: {loss_diff}')



# # Ensure that the CNN model's baseline loss is computed and stored first
# print("\nTraining CNN model with all features (baseline)...")

# # Reinitialize the CNN model
# cnn_model = tf.keras.Sequential([
#     layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_width, len(train_df.columns))),
#     layers.GlobalAveragePooling1D(),
#     layers.Dense(units=24)
# ])
# cnn_model.compile(optimizer='adam', loss='mae')

# # Train the CNN model with early stopping
# history = cnn_model.fit(
#     window.train,
#     validation_data=window.val,
#     epochs=10,
#     callbacks=[early_stopping]
# )

# # Evaluate the CNN model on the test set and store the test loss as the baseline
# baseline_loss_cnn = cnn_model.evaluate(window.test)
# overall_model_losses['CNN'] = baseline_loss_cnn  # Store the baseline loss for CNN

# print(f'CNN Model Baseline Test Loss (with all features): {baseline_loss_cnn}')

# # Now perform feature ablation for CNN model only
# print(f'\nEvaluating feature importance for CNN model')

# # Dictionary to track how much the loss increases when each feature is removed
# feature_loss_diffs = {}

# # Iterate through each feature, removing it from the dataset
# for feature_to_remove in all_features:
#     print(f'\nTraining CNN model without feature: {feature_to_remove}')
    
#     # Remove the selected feature from the dataset
#     train_df_mod = train_df.drop(columns=[feature_to_remove])
#     val_df_mod = val_df.drop(columns=[feature_to_remove])
#     test_df_mod = test_df.drop(columns=[feature_to_remove])

#     # Dynamically set the input shape based on the remaining number of features
#     num_features = len(train_df_mod.columns)

#     # Create a new WindowGenerator instance with the modified dataset
#     window_mod = WindowGenerator(
#         input_width=input_width,
#         label_width=label_width,
#         shift=shift,
#         train_df=train_df_mod,
#         val_df=val_df_mod,
#         test_df=test_df_mod,
#         label_columns=['SWC_5']
#     )
    
#     # Reinitialize the CNN model with the correct input shape
#     model = tf.keras.Sequential([
#         layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_width, num_features)),
#         layers.GlobalAveragePooling1D(),
#         layers.Dense(units=24)
#     ])
#     model.compile(optimizer='adam', loss='mae')
    
#     # Train the model with early stopping
#     history = model.fit(
#         window_mod.train,
#         validation_data=window_mod.val,
#         epochs=10,
#         callbacks=[early_stopping]
#     )
    
#     # Evaluate the model on the test set without the removed feature
#     ablation_loss = model.evaluate(window_mod.test)
    
#     # Calculate the difference in loss (how much worse the model performed without the feature)
#     loss_diff = ablation_loss - overall_model_losses['CNN']  # Use the stored baseline CNN loss
#     feature_loss_diffs[feature_to_remove] = loss_diff

#     print(f'CNN Model Test Loss without {feature_to_remove}: {ablation_loss} (Loss Difference: {loss_diff})')

# # Sort features by their importance (features with the largest loss increase are most important)
# sorted_feature_importance = sorted(feature_loss_diffs.items(), key=operator.itemgetter(1), reverse=True)

# # Store the sorted feature importance for CNN model
# model_feature_importance['CNN'] = sorted_feature_importance

# print(f'\nFeature importance for CNN model (most to least important):')
# for feature, loss_diff in sorted_feature_importance:
#     print(f'  {feature}: {loss_diff}')













# Train and evaluate each model with all features (to get the baseline loss)
print("\nTraining models with all features (baseline)...")
for name, model in models.items():
    print(f'\nTraining {name} model with all features...')
    
    # Reinitialize model
    model = tf.keras.models.clone_model(model)
    model.compile(optimizer='adam', loss='mae')
    
    # Train the model with early stopping
    history = model.fit(
        window.train,
        validation_data=window.val,
        epochs=10,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model on the test set and store the test loss as the baseline
    baseline_loss = model.evaluate(window.test)
    overall_model_losses[name] = baseline_loss  # Store baseline loss for each model
    
    print(f'{name} Model Baseline Test Loss (with all features): {baseline_loss}')


# Now perform feature ablation for each model
name = ''

for name, model in models.items():

    print(f'\nEvaluating feature importance for model: {name}')
    
    # Dictionary to track how much the loss increases when each feature is removed
    feature_loss_diffs = {}
    
    # Iterate through each feature, removing it from the dataset
    for feature_to_remove in all_features:
        print(f'\nTraining {name} model without feature: {feature_to_remove}')
        
        # Remove the selected feature from the dataset
        train_df_mod = train_df.drop(columns=[feature_to_remove])
        val_df_mod = val_df.drop(columns=[feature_to_remove])
        test_df_mod = test_df.drop(columns=[feature_to_remove])

        # Dynamically set the input shape based on the remaining number of features
        num_features = len(train_df_mod.columns)

        # Create a new WindowGenerator instance with the modified dataset
        window_mod = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df_mod,
            val_df=val_df_mod,
            test_df=test_df_mod,
            label_columns=['SWC_5']
        )
        
        # Reinitialize the model to ensure fresh training with updated input shape
        model = tf.keras.models.clone_model(models[name])
        model.compile(optimizer='adam', loss='mae')
        
        # Train the model with early stopping
        history = model.fit(
            window_mod.train,
            validation_data=window_mod.val,
            epochs=10,
            callbacks=[early_stopping]
        )
        
        # Evaluate the model on the test set without the removed feature
        ablation_loss = model.evaluate(window_mod.test)
        
        # Calculate the difference in loss (how much worse the model performed without the feature)
        loss_diff = ablation_loss - overall_model_losses[name]
        feature_loss_diffs[feature_to_remove] = loss_diff

        print(f'{name} Model Test Loss without {feature_to_remove}: {ablation_loss} (Loss Difference: {loss_diff})')
    
    # Sort features by their importance (features with the largest loss increase are most important)
    sorted_feature_importance = sorted(feature_loss_diffs.items(), key=operator.itemgetter(1), reverse=True)
    
    # Store the sorted feature importance for this model
    model_feature_importance[name] = sorted_feature_importance
    
    print(f'\nFeature importance for {name} model (most to least important):')
    for feature, loss_diff in sorted_feature_importance:
        print(f'  {feature}: {loss_diff}')


# Final ranking of models by baseline test loss
sorted_models_by_performance = sorted(overall_model_losses.items(), key=operator.itemgetter(1))

print("\nFinal Ranking of Models (Best to Worst based on baseline test loss):")
for model_name, loss in sorted_models_by_performance:
    print(f'  {model_name}: {loss}')

# Print the most important features for each model
print("\nMost important features for each model:")
for model_name, feature_importance in model_feature_importance.items():
    print(f'\n{model_name} model feature importance (most to least important):')
    for feature, importance in feature_importance:
        print(f'  {feature}: {importance}')
