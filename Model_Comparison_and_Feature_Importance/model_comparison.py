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
import argparse

from helpers import load_data, preprocess, normalize, create_window, train_and_evaluate_models, \
    plot_performance, print_model_summaries, write_model_results_to_csv, WindowGenerator, \
    baseline, linear, dense, simple_rnn, cnn, lstm, autoregressive, bi_lstm, load_all_data, create_csv, \
    calculate_original_performance, drop_feature_and_evaluate, create_feature_csv, plot_training_history, \
    evaluate_single_feature_models, create_loss_csv, lstm_attention, bi_lstm_attention, run_evaluation_and_save_results

def main():
    parser = argparse.ArgumentParser(description="Train model on a specific configuration.")
    parser.add_argument("--features", type=str, required=True, help="The feature to predict (e.g., 'SWC_5')")
    parser.add_argument("--input_steps", type=int, required=True, help="Number of input steps")
    parser.add_argument("--output_steps", type=int, required=True, help="Number of output steps")


    args = parser.parse_args()
    model_dir = './saved_models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Load your dataset
    
    # selected_features = ["Ppt", "Tair",  "SWC_5"]


    # TODO: load in all datasets into a dictionary

    # Load the data

    label_features = ['SWC_5']
    # temp_features = ['T_5', 'T_10', 'T_20', 'T_50']

    configurations = [
            {"features": args.features,
       "input_steps": args.input_steps,
       "output_steps": args.output_steps},
            # {"features": 'SWC_5', "input_steps": 24, "output_steps": 6},
            # {"features": 'SWC_5', "input_steps": 48, "output_steps": 12},
            # {"features": 'SWC_5', "input_steps": 7*24, "output_steps": 24},
            # {"features": 'SWC_5', "input_steps": 7*24, "output_steps": 48},
            # {"features": 'SWC_10', "input_steps": 24, "output_steps": 1},
            # {"features": 'SWC_10', "input_steps": 24, "output_steps": 6},
            # {"features": 'SWC_10', "input_steps": 48, "output_steps": 12},
            # {"features": 'SWC_10', "input_steps": 7*24, "output_steps": 24},
            # {"features": 'SWC_10', "input_steps": 7*24, "output_steps": 48},
            # {"features": 'SWC_20', "input_steps": 24, "output_steps": 1},
            # {"features": 'SWC_20', "input_steps": 24, "output_steps": 6},
            # {"features": 'SWC_20', "input_steps": 48, "output_steps": 12},
            # {"features": 'SWC_20', "input_steps": 7*24, "output_steps": 24},
            # {"features": 'SWC_20', "input_steps": 7*24, "output_steps": 48},
        ]

    # Feature Importance - Dropping Features

    feature_configurations = [
        {"features": 'SWC_5', "input_steps": 48, "output_steps": 12},
            # {"features": 'SWC_10', "input_steps": 48, "output_steps": 12},
            # {"features": 'SWC_20', "input_steps": 48, "output_steps": 12},
        ]

    # csv_filename = 'feature_importance_results.csv'
    # if not os.path.exists(csv_filename):
    #     with open(csv_filename, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         # Write the header row
    #         writer.writerow([
    #             'Label Feature', 'Dropped Feature', 'Model Name', 
    #             'Test MSE', 'Test MAE', 'Test MAPE'
    #         ])

    dfs = load_all_data()
    model_filename = f"{args.features}_{args.input_steps}_{args.output_steps}_model_results.csv"
    loss_filename = f"{args.features}_{args.input_steps}_{args.output_steps}_loss_history.csv"
    create_csv(model_filename)
    create_loss_csv(loss_filename)
    create_feature_csv('feature_importance_results.csv')
    for station, df in dfs.items():
        df = preprocess(df)
        # df = df[selected_features]

        df = normalize(df)

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
            label_index = label_features.index(label)

            # Identify the temperature feature to keep based on the label feature
            # temp_to_keep = temp_features[label_index]
            # features_to_drop = [col for col in label_features if col != label]
            # features_to_drop += [col for col in temp_features if col != temp_to_keep]
            # Drop the features
            # df_copy = df_copy.drop(columns=features_to_drop)


            train_df = df_copy.iloc[0:int(n*0.7)]
            val_df = df_copy.iloc[int(n*0.7):int(n*0.9)]
            test_df = df_copy.iloc[int(n*0.9):]
            print(train_df.isnull().sum())
            print(val_df.isnull().sum())
            print(test_df.isnull().sum())
            num_features = df_copy.shape[1]
            models = {
                    'Baseline': baseline(label_width, num_features),
                    'Multi-step Linear': linear(label_width, num_features),
                    'Multi-step Dense': dense(label_width, num_features),
                    'CNN': cnn(label_width, num_features, CONV_WIDTH),
                    'RNN': simple_rnn(label_width, num_features),
                    'LSTM': lstm(label_width, num_features),
                    'Autoregressive': autoregressive(label_width, num_features),
                    'Bi-LSTM': bi_lstm(label_width, num_features),
                    'LSTM_Attention': lstm_attention(label_width, num_features),
                    'BiLSTM_Attention': bi_lstm_attention(label_width, num_features)
                }

            # make way to drop other features besides one in config
            # Train and evaluate models for the current configuration
            performance, val_performance, history_dicts = train_and_evaluate_models(station, config, models, train_df, val_df, test_df, model_dir, model_filename, loss_filename  )
            model_losses = (performance, val_performance)
            # Store the losses for this configuration
            all_losses[f"{config['features']} - {config['input_steps']} input / {config['output_steps']} output"] = model_losses
            #plot_training_history(history_dicts)
            

        # Call the function to print the summaries
        print_model_summaries(models, all_losses)
        

    # for config_name, losses in all_losses.items():
    #     print(f"\nPlotting performance for configuration: {config_name}")
        
    #     # Plot the performance for this configuration
    #     plot_performance(losses[0], losses[1], title=config_name)


    # Loop through each configuration and compare models
        for config in feature_configurations:
            print(f"\nEvaluating models for configuration: {config}")
            CONV_WIDTH = 3
            # Define models in a dictionary
            label_width = config['output_steps']
            n = len(df)
            df_copy = df.copy()
            label = config['features']
            #eatures_to_drop = [col for col in label_features if col != label]

            # Drop the features
            #df_copy = df_copy.drop(columns=features_to_drop)
            train_df = df_copy.iloc[0:int(n*0.7)]
            val_df = df_copy.iloc[int(n*0.7):int(n*0.9)]
            test_df = df_copy.iloc[int(n*0.9):]
            num_features = df_copy.shape[1]
            models = {
                        'Baseline': baseline(label_width, num_features),
                        'Multi-step Linear': linear(label_width, num_features),
                        'Multi-step Dense': dense(label_width, num_features),
                        'CNN': cnn(label_width, num_features, CONV_WIDTH),
                        'RNN': simple_rnn(label_width, num_features),
                        'LSTM': lstm(label_width, num_features),
                        'Autoregressive': autoregressive(label_width, num_features),
                        'Bi-LSTM': bi_lstm(label_width, num_features),
                        'LSTM_Attention': lstm_attention(label_width, num_features),
                        'BiLSTM_Attention': bi_lstm_attention(label_width, num_features)
                    }

            # Example usage:
            print("\nDetermining feature importance...")
            # Assume 'all_features' contains the list of features in your dataset
            all_features = train_df.columns.tolist()
            target_feature = config['features']
            all_features.remove(target_feature)

            # Create a dictionary to store the original MAE values before feature dropping
            original_mae = calculate_original_performance(models, config, train_df, val_df, test_df)
            # feature_importance_results = drop_feature_and_evaluate(config, original_mae, train_df, val_df, test_df, all_features, target_feature, CONV_WIDTH, model_dir)
            # feature_importance_singular = feature_importance_singular(config, original_mae, train_df, val_df, test_df, all_features, target_feature, CONV_WIDTH, model_dir)
            
            run_evaluation_and_save_results(
                config=config,
                original_performance=original_mae,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                features=all_features,
                target=target_feature,
                CONV_WIDTH=CONV_WIDTH,
                model_dir=model_dir,
                output_csv="evaluation_results.csv"
            )
            # # Print the feature importance results
            # for feature, importance in feature_importance_results.items():
            #     print(f"\nFeature: {feature}")
            #     for model_name, mae_diff in importance.items():
            #         print(f"Model: {model_name} - MAE Change: {mae_diff:.4f}")

if __name__ == '__main__':
    main()



