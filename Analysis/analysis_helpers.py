import plotly.graph_objects as go
from ipywidgets import widgets, VBox
from IPython.display import display
import plotly.io as pio
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

pio.renderers.default = 'notebook_connected'  # Use notebook renderer for dynamic updates

class Trial:

    def __init__(self, trial_name):

        self.trial_name = trial_name
        self.result_df = self.get_df(trial_name)
        

    def get_df(self, trial_name):

        df = pd.read_csv("Analysis/Results_1_3_5_7/" + trial_name + "/results.csv")
        cols = df.columns
        if "Unnamed: 0" in cols:
            df.pop('Unnamed: 0')
        if "Unnamed: 0.1" in cols:
            df.pop("Unnamed: 0.1")

        zero_indexes = df["y_true"] <= 0
        min_val = df["y_true"][df["y_true"] > 0].min()
        df.loc[zero_indexes, "y_true"] = min_val

        return df


    def show_plot(self):
        df = self.result_df
        fig = go.Figure()
        fig.add_scatter(x=df.index, y=df['y_true'], mode='lines', name='Observed Values')
        fig.add_scatter(x=df.index, y=df['ar_bi_lstm'], mode='lines', name='AR Bi-LSTM Predictions')
        fig.add_scatter(x=df.index, y=df['ar_lstm'], mode='lines', name='AR LSTM Predictions')
        fig.add_scatter(x=df.index, y=df['bi_lstm'], mode='lines', name='Bi-LSTM Predictions')
        fig.add_scatter(x=df.index, y=df['lstm'], mode='lines', name='LSTM Predictions')
        fig.add_scatter(x=df.index, y=df['cnn'], mode='lines', name='CNN Predictions')
        fig.add_scatter(x=df.index, y=df['shift_bl'], mode='lines', name='Shifted Baseline')
        fig.update_layout(
                title=f' {self.trial_name} Results',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Soil Moisture'}
            )
        fig.show()

    def generate_error_table(self):
        """
        Creates an error table comparing each column (excluding 'y_true') in the DataFrame
        against the 'y_true' column, calculating mean squared error, mean absolute error,
        and mean absolute percentage error.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the predictions and the true values ('y_true').

        Returns:
        - pd.DataFrame: A DataFrame with the error metrics for each column compared to 'y_true'.
        """

        df = self.result_df
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



def total_avg_df(results):

    error_df_list = [trial.generate_error_table() for trial in results]
    average_df = pd.concat(error_df_list, axis=1).groupby(level=0, axis=1).mean()

    return average_df

def total_avg_df(results):

    error_df_list = [trial.generate_error_table() for trial in results]
    average_df = pd.concat(error_df_list, axis=1).groupby(level=0, axis=1).mean()

    return average_df

