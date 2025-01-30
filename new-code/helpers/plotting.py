import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_data(df, station_name):
    """
    Plots a heatmap of missing values in the dataset.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
    plt.title(f"Missing Data Heatmap - {station_name}")
    plt.show()

def plot_soil_moisture_trends(df, station_name):
    """
    Plots soil moisture trends over time for a given station.
    """
    plt.figure(figsize=(14, 6))
    for col in ["SWC_5", "SWC_10", "SWC_20", "SWC_50"]:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.title(f"Soil Moisture Trends - {station_name}")
    plt.xlabel("Date")
    plt.ylabel("Soil Moisture")
    plt.show()

def plot_arima_predictions(test, predictions, target_col):
    """
    Plots ARIMA model predictions vs. actual values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label="Actual")
    plt.plot(test.index, predictions, label="Predicted", linestyle="dashed")
    plt.legend()
    plt.title(f"ARIMA Predictions for {target_col}")
    plt.show()
