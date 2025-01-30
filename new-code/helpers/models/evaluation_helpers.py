from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluates the model using RMSE and MAE.
    """
    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{model_name} Evaluation:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    return rmse, mae
