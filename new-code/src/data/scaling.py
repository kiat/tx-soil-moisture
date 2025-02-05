from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_df(df):
    periodic_cols = ['Day sin', 'Day cos', 'Year sin', 'Year cos']
    non_periodic = df.drop(columns=periodic_cols)
    scaler = MinMaxScaler()
    scaled_np = scaler.fit_transform(non_periodic)
    scaled_df = pd.DataFrame(scaled_np, columns=non_periodic.columns, index=df.index)
    for col in periodic_cols:
        scaled_df[col] = df[col].values
    return scaled_df, scaler


def inverse_scale(preds, base_df, target_col):
    periodic_cols = ['Day sin', 'Day cos', 'Year sin', 'Year cos']
    non_periodic = base_df.drop(columns=periodic_cols)
    scaler_temp = MinMaxScaler().fit(non_periodic)
    placeholder = base_df.iloc[-len(preds):].copy()
    placeholder[target_col] = preds.reshape(-1)
    non_periodic_inv = scaler_temp.inverse_transform(placeholder.drop(columns=periodic_cols))
    inv_df = pd.DataFrame(non_periodic_inv, columns=non_periodic.columns, index=placeholder.index)
    for col in periodic_cols:
        inv_df[col] = placeholder[col].values
    return inv_df