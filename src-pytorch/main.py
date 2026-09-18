import argparse
import os
import torch
import torch.nn as nn

from models import get_model_class, is_registered, list_models
from utils import Trainer, Evaluator, prepare_dataloaders, get_output_helpers
from core.evaluation_helpers import save_experiment_results, write_model_results_to_csv

def main(args):
    """Main training and evaluation pipeline."""
    # Input validation
    if args.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {args.window_size}")
    if args.offset < 0:
        raise ValueError(f"offset must be non-negative, got {args.offset}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.patience <= 0:
        raise ValueError(f"patience must be positive, got {args.patience}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse features - prefer predictors over features for backward compatibility
    if args.predictors:
        predictors_list = args.predictors.split(",")
    else:
        predictors_list = ["SWC_20", "T_20", "Ppt", "Tair", "Wx", "Wy"]

    # Prepare data
    stations = ["Station1", "Station2", "Station3", "Station4", "Station5", "Station6"]

    # Set target station based on argument (1-6 maps to Station1-Station6)
    if args.target_station < 1 or args.target_station > 6:
        raise ValueError(
            f"target_station must be between 1 and 6, got {args.target_station}"
        )
    target_station = stations[args.target_station - 1]
    print(f"Target station: {target_station}")

    # Prepare prediction feature names (with _daily_avg suffix if needed)
    # When daily_average_output=True:
    #   - User specifies: --predict_features "SWC_20,Ppt"
    #   - engineer_features creates: SWC_20_daily_avg, Ppt_daily_avg columns
    #   - Models predict: SWC_20_daily_avg, Ppt_daily_avg (daily means)
    #   - Logs/files saved with: "SWC_20_daily_avg_Ppt_daily_avg" suffix
    predict_features_list = (
        args.predict_features.split(",") if args.predict_features else ["SWC_20"]
    )
    if args.daily_average_output:
        predict_features_display = [f"{f}_daily_avg" for f in predict_features_list]
    else:
        predict_features_display = predict_features_list

    predict_features_str = "_".join(predict_features_display)

    train_loader, val_loader, test_loader, all_features, input_dim, data_shape = (
        prepare_dataloaders(
            stations=stations,
            target_station=target_station,
            window_size=args.window_size,
            offset=args.offset,
            batch_size=args.batch_size,
            predictors=args.predictors,
            predict_features=args.predict_features,
            predict_avg=args.daily_average_output,
            label_width=args.label_width,
            training_stride=args.training_stride,
            validation_stride=args.validation_stride,
        )
    )


    # Setup directories
    model_dir = "saved_models_pytorch"
    os.makedirs(model_dir, exist_ok=True)

    # Build process queue using registry
    requested_names = [name.strip() for name in args.model_names.split(",")]
    process_queue = {}
    for name in requested_names:
        if is_registered(name):
            process_queue[name.lower()] = get_model_class(name)
        else:
            print(
                f"Warning: Model '{name}' not found in registry. Available: {list_models()}"
            )

    if not process_queue:
        raise ValueError(f"No valid models found. Available models: {list_models()}")

    print(f"Models to be processed: {list(process_queue.keys())}")

    # Training and evaluation loop
    criterion = nn.MSELoss()
    for model_name, model_class in process_queue.items():
        print(f"\n===== Processing {model_name.upper()} =====")

        if model_name in ["baseline", "movingaverage"]:
            # Non-trainable baseline models
            print(f"Evaluating non-trainable baseline model: {model_name.upper()}")
            model = model_class().to(device)
            evaluator = Evaluator(criterion, device)
            performance = evaluator.evaluate(model, test_loader)
        else:
            # Trainable models
            print(f"Training {model_name.upper()}...")

            if model_name == "ilstmsoil":
                T = data_shape["time_steps"]
                N = data_shape["num_features"]
                output_dim = data_shape["output_dim"]
                print(
                    f"Input shape for ILSTM_Soil: T={T}, N={N}, output_dim={output_dim}"
                )
                model = model_class(time_steps=T, num_features=N, output_dim=output_dim)
            else:
                output_dim = data_shape["output_dim"]
                model = model_class(input_dim=input_dim, output_dim=output_dim)

            trainer = Trainer(
                model,
                criterion,
                device,
                model_name=model_name,
                log_dir="logs",
                lr=0.001,
                patience=args.patience,
                window_size=args.window_size,
                offset=args.offset,
                predictors="_".join(predictors_list),
                predict_features=predict_features_str,
            )
            history = trainer.fit(train_loader, val_loader, epochs=args.epochs)

            print("\nFinal Evaluation on Test Set...")
            performance = trainer.evaluator.evaluate(trainer.model, test_loader)

            # Save model
            feature_str = "_".join(predictors_list)
            main_name = f"model_{model_name}_ws{args.window_size}_offset{args.offset}_{feature_str}"
            model_path = os.path.join(model_dir, f"{main_name}.pth")
            torch.save(trainer.model.state_dict(), model_path)
            print(f"{model_name.upper()} model saved at {model_path}")


        # Print and save metrics
        print(f"\n--- {model_name.upper()} Final Test Metrics ---")
        for key, value in performance.items():
            print(f"{key}: {value:.6f}")

        feature_str = "_".join(predictors_list)

        if model_name in ["baseline", "movingaverage"]:
            print("Skipping loss history save for non-trainable baseline model.")

            write_model_results_to_csv(
                station=target_station,
                model_name=model_name,
                window_size=args.window_size,
                offset=args.offset,
                performance=performance,
                feature_str=feature_str,
                label_str=predict_features_str,
                epochs=args.epochs,
                patience=args.patience,
            )
        else:
            save_experiment_results(
                station=target_station,
                model_name=model_name,
                window_size=args.window_size,
                offset=args.offset,
                history=history,
                performance=performance,
                feature_str=feature_str,
                label_str=predict_features_str,
                epochs=args.epochs,
                patience=args.patience,
            )
        print("-------------------------------------\n")

    print("All runs complete! Results have been saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate time series models using PyTorch."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=48,
        help="Window size for input data sequences.",
    )
    parser.add_argument(
        "--label_width",
        type=int,
        default=1,
        help="Width of the label/output window.",
    )
    parser.add_argument(
        "--training_stride",
        type=int,
        default=8,
        help="Stride for creating training sequences.",
    )
    parser.add_argument(
        "--validation_stride",
        type=int,
        default=8,
        help="Stride for creating validation sequences.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=48,
        help="Prediction offset from the end of the window.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--predictors",
        type=str,
        default="SWC_20,T_20,Ppt,Wx,Wy,Srad,DayCos,DaySin,MonthCos,MonthSin",
        help="Comma-separated list of predictor features to use.",
    )
    parser.add_argument(
        "--predict_features",
        type=str,
        default="SWC_20",
        help="Comma-separated list of features to predict.",
    )
    parser.add_argument(
        "--daily_average_output",
        action="store_true",
        default=False,
        help="Whether to output daily average predictions.",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        default="bilstm",
        help="Comma-separated list of model names to run.",
    )
    parser.add_argument(
        "--target_station",
        type=int,
        default=6,
        help="Target station number (1-6, corresponding to Station1-Station6).",
    )

    args = parser.parse_args()
    main(args)
