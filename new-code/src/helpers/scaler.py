
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# --- Use MinMaxScaler with Pipeline ---
class MinMaxScalerWrapper:
    """Wraps MinMaxScaler to handle periodic columns correctly."""
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.periodic_cols = ['Day sin', 'Day cos', 'Year sin', 'Year cos']
        
    def fit(self, X, y=None):
        """Fit scaler on non-periodic features only."""
        non_periodic = X.drop(columns=self.periodic_cols, errors='ignore')
        self.scaler.fit(non_periodic)
        return self
    
    def transform(self, X):
        """Scale non-periodic features, retain periodic ones."""
        non_periodic = X.drop(columns=self.periodic_cols, errors='ignore')
        scaled_np = self.scaler.transform(non_periodic)
        scaled_df = pd.DataFrame(scaled_np, columns=non_periodic.columns, index=X.index)
        for col in self.periodic_cols:
            if col in X:
                scaled_df[col] = X[col].values  # Preserve periodic features
        return scaled_df
    
    def inverse_transform(self, preds, base_df, target_col):
        """Inverse scale for non-periodic features while preserving structure."""
        non_periodic = base_df.drop(columns=self.periodic_cols, errors='ignore')
        
        # Ensure predictions match original shape
        placeholder = base_df.iloc[-len(preds):].copy()
        placeholder[target_col] = preds.reshape(-1)

        # Apply inverse scaling
        # Ensure the column order and count match exactly what was originally fit
        expected_columns = self.scaler.feature_names_in_  # Get original columns used in fit()
        non_periodic = placeholder.drop(columns=self.periodic_cols, errors='ignore')

        # Ensure same column order and add missing columns if necessary
        for col in expected_columns:
            if col not in non_periodic:
                non_periodic[col] = 0  # Add missing columns with dummy values

        # Reorder columns to match the original fit()
        non_periodic = non_periodic[expected_columns]

        # Apply inverse transformation
        non_periodic_inv = self.scaler.inverse_transform(non_periodic)

        inv_df = pd.DataFrame(non_periodic_inv, columns=non_periodic.columns, index=placeholder.index)

        # Preserve periodic features
        for col in self.periodic_cols:
            if col in base_df:
                inv_df[col] = placeholder[col].values
        
        return inv_df