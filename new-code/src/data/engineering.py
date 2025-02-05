# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def engineer_data(df):
    import numpy as np
    # Consolidate wind vectors only if both 'Windspeed' and 'Winddirection' exist
    if 'Windspeed' in df.columns and 'Winddirection' in df.columns:
        wv = df.pop('Windspeed')
        wd_rad = df.pop('Winddirection') * np.pi / 180
        df['Wx'] = wv * np.cos(wd_rad)
        df['Wy'] = wv * np.sin(wd_rad)
    
    # Remove Latitude and Longitude if they exist
    if 'Latitude' in df.columns:
        df.pop('Latitude')
    if 'Longitude' in df.columns:
        df.pop('Longitude')
    
    # Add periodic time features based on the index (timestamp in seconds)
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    return df
