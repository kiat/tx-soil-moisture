# src/helpers/arg_parser.py
import argparse
from helpers.data_helpers import load_config

def get_config_parameters():
    """
    Parses command-line arguments and integrates them with default values from config.yaml.
    
    Returns:
        dict: Dictionary containing experiment parameters.
    """
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Run soil moisture prediction experiments.")
    parser.add_argument("--split_strategy", type=str, choices=["split_by_year", "leave_one_out"], default="split_by_year",
                        help="Choose how to split data: 'split_by_year' (default) or 'leave_one_out'.")
    parser.add_argument("--window_sizes", nargs="+", type=int, help="List of window sizes to test.")
    parser.add_argument("--offset", type=int, help="Forecasting offset.")
    parser.add_argument("--swc_list", nargs="+", help="List of SWC columns.")
    parser.add_argument("--station_list", nargs="+", help="List of stations.")
    parser.add_argument("--test_year", type=int, help="Year to exclude for testing.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")

    args = parser.parse_args()

    params = {
        "SPLIT_STRATEGY": args.split_strategy,  # Added split strategy
        "WINDOW_SIZES": args.window_sizes if args.window_sizes else cfg["experiment"]["window_sizes"],
        "OFFSET": args.offset if args.offset else cfg["experiment"]["offset"],
        "SWC_LIST": args.swc_list if args.swc_list else cfg["experiment"]["swc_list"],
        "STATION_LIST": args.station_list if args.station_list else cfg["experiment"]["station_list"],
        "TEST_YEAR": args.test_year if args.test_year else 2020,  # Default test year
        "EPOCHS": args.epochs if args.epochs else cfg["hyperparameters"]["epochs"],
        "BATCH_SIZE": args.batch_size if args.batch_size else cfg["hyperparameters"]["batch_size"],
    }

    return params, cfg
