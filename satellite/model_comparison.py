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

def validate_data_and_config(df, args, input_features):
    """Validate data and configuration before running models"""
    validation_errors = []
    
    # Check data shape
    if len(df) < (args.input_steps + args.output_steps):
        validation_errors.append(f"Not enough data points ({len(df)}) for specified input_steps ({args.input_steps}) and output_steps ({args.output_steps})")
    
    # Check for NaN values
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        validation_errors.append(f"NaN values found in columns: {nan_cols}")
    
    # Check feature availability
    missing_features = [f for f in input_features if f not in df.columns]
    if missing_features:
        validation_errors.append(f"Missing features: {missing_features}")
    
    # Check data types
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    if len(non_numeric_cols) > 0:
        validation_errors.append(f"Non-numeric columns found: {non_numeric_cols}")
    
    # Check for infinite values
    inf_cols = df.columns[np.isinf(df).any()].tolist()
    if inf_cols:
        validation_errors.append(f"Infinite values found in columns: {inf_cols}")
    
    # Validate window sizes
    if args.input_steps < 1:
        validation_errors.append(f"Invalid input_steps: {args.input_steps}")
    if args.output_steps < 1:
        validation_errors.append(f"Invalid output_steps: {args.output_steps}")
    
    return validation_errors

def main():
    parser = argparse.ArgumentParser(description="Train model on satellite soil moisture data.")
    parser.add_argument("--target", type=str, required=True, choices=['Sat_SM_AMSR', 'Sat_SM_SMAP'], 
                      help="The target feature to predict")
    parser.add_argument("--input_steps", type=int, required=True, help="Number of input steps")
    parser.add_argument("--output_steps", type=int, required=True, help="Number of output steps")
    parser.add_argument("--station", type=int, required=True, choices=[1,2,3,4,5,6],
                      help="Station number to process")

    args = parser.parse_args()
    
    # Load and preprocess data first
    data_path = f'satellite/met_merged_satelite_data/Station{args.station}_Met_AMSR_SMAP.csv'
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Initial data processing
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.drop(['distance_AMSR', 'distance_SMAP'], axis=1)
    
    # Preprocess the data first
    print("Preprocessing data...")
    df = preprocess(df)
    
    # Define input features
    other_target = 'Sat_SM_SMAP' if args.target == 'Sat_SM_AMSR' else 'Sat_SM_AMSR'
    input_features = [col for col in df.columns if col not in [args.target, other_target]]
    
    # Now validate the preprocessed data
    validation_errors = validate_data_and_config(df, args, input_features)
    
    if validation_errors:
        print("\nValidation Errors:")
        for error in validation_errors:
            print(f"- {error}")
        print("\nPlease fix these issues before running the models.")
        return
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print("-" * 50)
    print(f"Target Variable: {args.target}")
    print(f"Input Steps: {args.input_steps}")
    print(f"Output Steps: {args.output_steps}")
    print(f"Station: {args.station}")
    print(f"Input Features: {input_features}")
    print(f"Training Data Shape: {df.shape}")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with these configurations? (y/n): ")
    if response.lower() != 'y':
        print("\nProcess cancelled.")
        return
    
    # Define configurations based on command line arguments
    configurations = [{
        'features': args.target,
        'input_steps': args.input_steps,
        'output_steps': args.output_steps
    }]

    model_dir = './saved_models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filename = f"{args.target}_station{args.station}_{args.input_steps}_{args.output_steps}_model_results.csv"
    loss_filename = f"{args.target}_station{args.station}_{args.input_steps}_{args.output_steps}_loss_history.csv"
    
    create_csv(model_filename)
    create_loss_csv(loss_filename)
    create_feature_csv('feature_importance_results.csv')

    # Normalize the data
    df = normalize(df)

    # Print dataset information
    print("\nDataset Information:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Target variable: {args.target}")
    print(f"Input features: {input_features}")
    
    # Check for any remaining NaN values
    if df.isna().any().any():
        print("\nWarning: Dataset still contains NaN values after preprocessing")
        print(df.isna().sum())
        print(f"Samples after removing NaN: {len(df)}")

    # Initialize a dictionary to hold losses across configurations
    all_losses = {}

    # Loop through each configuration and compare models
    for config in configurations:
        print(f"\nEvaluating models for configuration: {config}")
        CONV_WIDTH = 3
        label_width = config['output_steps']
        n = len(df)
        df_copy = df.copy()
        
        # Split the data
        train_df = df_copy.iloc[0:int(n*0.7)]
        val_df = df_copy.iloc[int(n*0.7):int(n*0.9)]
        test_df = df_copy.iloc[int(n*0.9):]
        
        # Print split sizes
        print(f"\nData split sizes:")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        num_features = df_copy.shape[1]
        print(f"Number of features: {num_features}")
        
        # Create window generator with specific label columns
        window = create_window(
            input_width=config['input_steps'],
            label_width=config['output_steps'],
            shift=config['output_steps'],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=[args.target]
        )
        
        # Initialize models
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

        # Train and evaluate models
        performance, val_performance, history_dicts = train_and_evaluate_models(
            args.station, config, models, train_df, val_df, test_df, 
            model_dir, model_filename, loss_filename
        )
        
        model_losses = (performance, val_performance)
        all_losses[f"{config['features']} - {config['input_steps']} input / {config['output_steps']} output"] = model_losses

        # Print model summaries
        print_model_summaries(models, all_losses)

        # Feature importance analysis
        original_mae = calculate_original_performance(models, config, train_df, val_df, test_df)
        
        run_evaluation_and_save_results(
            config=config,
            original_performance=original_mae,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            features=input_features,
            target=args.target,
            CONV_WIDTH=CONV_WIDTH,
            model_dir=model_dir,
            output_csv=f"evaluation_results_station{args.station}_{args.target}.csv"
        )

if __name__ == '__main__':
    main()


