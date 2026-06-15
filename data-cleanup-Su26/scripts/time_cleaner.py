import pandas as pd
 
def time_cleaner(df):
    """
    expects a DatetimeIndexed df and returns a DatetimeIndexed df with all missing 
    timestamps filled in with NaN measurements.
    """
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex (run dup_cleaner.py first).")
    if df.empty:
        return df
 
    # --- case 1: insert missing timestamps ---
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='h')
    return df.reindex(full_range)

    