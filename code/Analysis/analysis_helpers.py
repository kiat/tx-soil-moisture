import plotly.graph_objects as go
from ipywidgets import widgets, VBox
from IPython.display import display
import plotly.io as pio
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



pio.renderers.default = 'notebook_connected'  # Use notebook renderer for dynamic updates

def interactive_plot(df):
    """
    Creates an interactive plot with checkboxes for each column in the DataFrame.
    Only the columns that are checked will be displayed in the plot, without regenerating the graph.
    The plot is made larger vertically for better visualization.

    Parameters:
    - df (pd.DataFrame): The DataFrame to plot.
    """
    # Initialize a FigureWidget with traces for all columns, but make them invisible
    fig = go.FigureWidget(
        data=[go.Scatter(x=df.index, y=df[column], name=column, visible=True) for column in df.columns]
    )

    # Adjust the layout to increase the plot size, especially the height
    fig.update_layout(
        height=600,  # Set the height of the plot to 600 pixels (adjust as needed)
        margin=dict(l=20, r=20, t=20, b=20)  # Adjust margins if necessary
    )

    # Function to update the plot based on checkbox states
    def update_plot(change):
        # Update the visibility of each trace based on the checkbox value
        for i, checkbox in enumerate(checkboxes):
            fig.data[i].visible = checkbox.value

    # Create a checkbox for each column in the DataFrame, linked to the plot
    checkboxes = [widgets.Checkbox(value=True, description=column) for column in df.columns]
    for checkbox in checkboxes:
        checkbox.observe(update_plot, names='value')

    # Display checkboxes
    checkbox_container = VBox(checkboxes)
    display(checkbox_container)

    # Display the plot
    display(fig)

def create_dictionaries_from_csv(files):
    """
    Parameters:
    - files (list of str): List of paths to CSV files.
    Returns:
    - dict: A dictionary containing dictionaries for each CSV file.
    """
    master_dict = {}
    for file_path in files:
        df = pd.read_csv("/Users/benjamincartwright/tx-soil-moisture/code/Analysis/Results_Data/" + file_path + "/results.csv")
        #column_dict = {column: df[column].tolist() for column in df.columns}
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        master_dict[file_name] = df
        
    return master_dict

def generate_error_table(df):
    """
    Creates an error table comparing each column (excluding 'y_true') in the DataFrame
    against the 'y_true' column, calculating mean squared error, mean absolute error,
    and mean absolute percentage error.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the predictions and the true values ('y_true').

    Returns:
    - pd.DataFrame: A DataFrame with the error metrics for each column compared to 'y_true'.
    """
    # Initialize an empty list to store error metrics for each column
    errors = []

    # Exclude 'y_true' and iterate over the remaining columns
    for col in df.columns:
        if col != 'y_true':
            mse = mean_squared_error(df['y_true'], df[col])
            mae = mean_absolute_error(df['y_true'], df[col])
            mape = mean_absolute_percentage_error(df['y_true'], df[col] )
            errors.append({'column': col, 'mean_squared_error': mse, 'mean_absolute_error': mae, 'mean_absolute_percentage_error': mape})

    # Convert the list of dictionaries into a DataFrame
    error_df = pd.DataFrame(errors)
    error_df.set_index('column', inplace=True)

    return error_df

def update_y_true(master_dict):
    """
    Iterates through each DataFrame in master_dict and updates the y_true column
    by replacing zero values with the next smallest value in that column.

    Parameters:
    - master_dict (dict): A dictionary containing DataFrames as values.
    """
    for file_name, df in master_dict.items():
        # Check if 'y_true' column exists
        zero_indexes = df["y_true"] <= 0
        min_val = df["y_true"][df["y_true"] > 0].min()
        df.loc[zero_indexes, "y_true"] = min_val

    return master_dict