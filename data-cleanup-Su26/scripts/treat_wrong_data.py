import pandas as pd

def find_and_replace_wrong_data(dfc):
    """
    Replaces values outside valid bounds with NaN for final export.
    CREDITS: HANNAH AND ZUN
    """
    for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']:
        if col in dfc.columns:
            dfc.loc[(dfc[col] < 0) | (dfc[col] > 0.6), col] = pd.NA
    if 'Ppt' in dfc.columns:
        dfc.loc[dfc['Ppt'] < 0, 'Ppt'] = pd.NA
    if 'RH' in dfc.columns:
        dfc.loc[(dfc['RH'] < 0) | (dfc['RH'] > 100), 'RH'] = pd.NA
    if 'Wind speed' in dfc.columns:
        dfc.loc[(dfc['Wind speed'] < 0) | (dfc['Wind speed'] > 25), 'Wind speed'] = pd.NA
    if 'Wind direction' in dfc.columns:
        dfc.loc[(dfc['Wind direction'] < 0) | (dfc['Wind direction'] > 360), 'Wind direction'] = pd.NA
    if 'Srad' in dfc.columns:
        dfc.loc[dfc['Srad'] < 0, 'Srad'] = pd.NA
    for col in ['T_5', 'T_10', 'T_20', 'T_50', 'Tair']:
        if col in dfc.columns:
            dfc.loc[(dfc[col] < -30) | (dfc[col] > 60), col] = pd.NA
    return dfc