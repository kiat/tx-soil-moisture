import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#Reads in Simulated-Cleaned-Merged-Data and drops all indexes not withi the intersection of all 6 stations
def read_data():
    dfs = {}
    for index in range(0, 6):
        df = pd.read_csv('../datasets/finalized_data/Station' + str(index + 1) + '_Final_Version.csv', sep=",",
                         parse_dates=["Date"], index_col="Date")
        dfs['Station' + str(index + 1)] = df

    index_union = pd.Index([])
    for station, df in dfs.items():
        index_union = index_union.union(df.index)

    index_int = index_union
    for station, df in dfs.items():
        index_int = index_int.intersection(df.index)
    for key in dfs.keys():
        dfs[key] = dfs[key].loc[index_int]

    return dfs

#vectorizes wind, changes time to trig values, converts longitude and latitude to trig values
def engineer_data(dfs):
    day = 24 * 60 * 60
    year = 365.2425 * day
    for station, df in dfs.items():
        wv = df.pop('Windspeed')
        timestamp_s = (df.index).map(pd.Timestamp.timestamp)
        lat = df.pop('Latitude')
        lon = df.pop('Longitude')
        wd_rad = df.pop('Winddirection') * np.pi / 180
        df['Wx'] = wv * np.cos(wd_rad)
        df['Wy'] = wv * np.sin(wd_rad)
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        df["x_cord"] = np.cos(lat) * np.cos(lon)
        df["y_cord"] = np.sin(lat) * np.cos(lon)
        df["z_cord"] = np.sin(lon)
        dfs[station] = df

#scales all the station dfs
def scale_data(dfs):
    for station, df in dfs.items():
        cur_df = df.copy()
        d_sin = cur_df.pop("Day sin")
        d_cos = cur_df.pop("Day cos")
        y_sin = cur_df.pop("Year sin")
        y_cos = cur_df.pop("Year cos")
        x = cur_df.pop("x_cord")
        y = cur_df.pop("y_cord")
        z = cur_df.pop("z_cord")
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(data=scaler.fit_transform(cur_df), columns=cur_df.columns, index=cur_df.index)
        scaled_df["Day sin"] = d_sin.values
        scaled_df["Day cos"] = d_cos.values
        scaled_df["Year sin"] = y_sin.values
        scaled_df["Year cos"] = y_cos.values
        scaled_df["x_cord"] = x.values
        scaled_df["y_cord"] = y.values
        scaled_df["z_cord"] = z.values
        dfs[station] = scaled_df

#takes the last year of the target station and makes that the test set
# the remaining values from that station are used as validation and 
# the rest is training 
def split_and_stack_data(dfs, test_station_name = "Station6", remove_met = False):
    cut = '2021-01-01 00:00:00'

    for key in dfs.keys():
        dfs[key] = dfs[key][dfs[key].index < cut]
        if remove_met:
            dfs[key] = dfs[key][["SWC_5","SWC_10","SWC_20","SWC_50"]]

    test_df = dfs[test_station_name].loc['2020-01-01 00:00:00':]
    val_df = dfs[test_station_name].loc[:'2020-12-31 23:00:00']
    dfs[test_station_name].drop(test_df.index, inplace = True)

    frame_array = []
    for key in dfs.keys():
        if key != test_station_name:
            frame_array.append(dfs[key].values)
    train_df = pd.DataFrame(np.vstack(tuple(frame_array)))

    train_df.columns = dfs["Station1"].columns
    val_df.columns = dfs["Station1"].columns
    test_df.columns = dfs["Station1"].columns
    return train_df, val_df, test_df


#Returns data frames for training, validation and testing
def get_data(test_station_name = "Station6", remove_met = False):
    dfs = read_data()
    engineer_data(dfs)
    for key in dfs.keys():
        dfs[key].pop('Unnamed: 0')
    scale_data(dfs)
    return split_and_stack_data(dfs, test_station_name, remove_met)


#creates windows
def generate_windows(data, window_size=24, shift=24, target_idx = 0):
    labels = data.values[:, target_idx]
    X = []
    y = []
    for i in range(len(data) - window_size - shift):
        # get window based on input width
        window = data[i: i + window_size]

        # keep track of label associated with current window
        window_label = labels[i + window_size + shift]

        X.append(window)
        y.append(window_label)

    # in new dataset, each element is a data window, and window label is single value
    return np.array(X), np.array(y)

#creates batches
def generate_batches(X, y, batch_size=64):
    # divides data into batches, drops any remainder batches smaller than specified batch size.
    # allows models to run with any batch size
    tf_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    tf_dataset = tf_dataset.repeat().batch(batch_size=batch_size, drop_remainder=True)

    # tf_dataset repeats indefinitely, need to compute number of step updates to complete 1 epoch
    steps_per_epoch = len(X) // batch_size

    return tf_dataset, steps_per_epoch

#Saves the dataframes as parquet files
def set_data(target_station = "Station6", target_col = "SWC_5", remove_met = False):
    try:
        train = pd.read_parquet('train_df.parquet.gzip')
    except: 
        train, val, test = get_data(target_station, remove_met)
        test.to_parquet('test_df.parquet.gzip',compression='gzip', index=True)
        val.to_parquet('val_df.parquet.gzip',compression='gzip', index=True)
        train.to_parquet('train_df.parquet.gzip',compression='gzip', index=True)

    return train.columns.get_loc(target_col)

#retrieves the saved data
def retrieve_data(test = False):
    if test:
        return pd.read_parquet('test_df.parquet.gzip')

    else:
        return (pd.read_parquet('train_df.parquet.gzip'),pd.read_parquet('val_df.parquet.gzip'))
        
#breaks a data frame into windows and batches, if the test parameter is true it 
# also returns the X and y for that set to use in predictions accuracy metrics
def create_sets(df, window_size, shift_amt, target_idx, batch_size, test = False):
    X, y = generate_windows(df, window_size=window_size,
                                    shift=shift_amt, target_idx=target_idx)

    dataset, steps = generate_batches(X, y, batch_size=batch_size)
    
    if test == True: 
        return dataset, steps, X, y
    return dataset, steps