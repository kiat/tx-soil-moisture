"""
Attention-based models for time-series prediction.

This module provides:
- AttentionLSTM: LSTM with multi-head attention between layers
- AttentionOnly: Pure attention model without recurrence
- PredictorAttention: Attention mechanism for predictor features
- TemporalAttention: Attention mechanism for temporal features
- ILSTM_Soil: Advanced interpretable LSTM for soil moisture prediction

Purpose:
    Attention mechanisms allow models to focus on relevant parts of the
    input sequence, improving interpretability and performance. These
    models use various attention patterns optimized for time-series data.

Usage:
    >>> model = AttentionLSTM(input_dim=10)
    >>> predictions = model(input_tensor)
"""

import torch
import torch.nn as nn
from .base import BaseModel


class AttentionLSTM(BaseModel):
    """
    LSTM model with multi-head self-attention between LSTM layers.

    Architecture:
    - LSTM layer 1 → Multi-head attention → LSTM layer 2 → MLP decoder

    Features:
    - First LSTM processes the sequence
    - Attention refines representations by focusing on important timesteps
    - Second LSTM further processes attended features
    - Uses final hidden state for prediction

    Args:
        input_dim (int): Number of input features

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, 1)
    """

    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Input projection
        self.proj_inp = nn.Linear(input_dim, 64)

        # First LSTM processes input
        self.lstm1 = nn.LSTM(64, 32, batch_first=True)

        # Multi-head attention refines representations
        self.attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )

        # Second LSTM processes attended features
        self.lstm2 = nn.LSTM(32, 32, batch_first=True)

        # MLP decoder
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the AttentionLSTM model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, 1)
        """
        x = self.proj_inp(x)  # (batch, seq_len, 64)
        lstm_out, _ = self.lstm1(x)  # (batch, seq_len, 32)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)  # Self-attention
        _, (hn, _) = self.lstm2(attn_output)  # (1, batch, 32)
        x = hn.squeeze(0)  # (batch, 32)
        x = self.relu(self.fc1(x))  # (batch, 8)
        x = self.fc2(x)  # (batch, 1)
        return x


class AttentionOnly(BaseModel):
    """
    Pure attention model without recurrent layers.

    Features:
    - Multi-head self-attention directly on input
    - Layer normalization with residual connection
    - Adaptive pooling across sequence
    - Fast parallel processing (no sequential dependencies)

    Args:
        input_dim (int): Number of input features

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, 1)

    Note:
        Number of attention heads is automatically adjusted to evenly
        divide input_dim for proper multi-head attention operation.
    """

    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Ensure num_heads divides input_dim evenly
        num_heads = min(4, input_dim) if input_dim < 4 else 4
        while input_dim % num_heads != 0:
            num_heads -= 1
        if num_heads == 0:
            num_heads = 1

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        # Layer normalization for stable training
        self.norm = nn.LayerNorm(input_dim)

        # Pooling and decoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the AttentionOnly model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, 1)
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)  # (batch, seq_len, input_dim)
        x = self.norm(attn_output + x)  # Residual + LayerNorm

        # Pool across sequence dimension
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.pool(x)  # (batch, input_dim, 1)
        x = x.squeeze(-1)  # (batch, input_dim)

        # Decode to prediction
        x = self.relu(self.fc1(x))  # (batch, 64)
        x = self.fc2(x)  # (batch, 1)
        return x


class PredictorAttention(nn.Module):
    """
    Attention mechanism for predictor features.

    This module computes attention weights over combined features
    (hidden state + predictor features) to emphasize important predictors.

    Args:
        input_dim (int): Combined dimension of input features
        dropout (float): Dropout probability (default: 0.1)

    Input:
        last_hidden_state: Tensor of shape (batch, num_features, hidden_dim)
        pred_features: Tensor of shape (batch, num_features, hidden_dim)

    Output:
        enriched_features: Weighted sum (batch, hidden_dim)
    """

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.attn_mech = nn.Sequential(nn.Linear(input_dim, 1), nn.Softmax(dim=1))

    def forward(self, last_hidden_state, pred_features):
        """
        Apply attention to predictor features.

        Args:
            last_hidden_state: Hidden state from LSTM (batch, num_features, hidden_dim)
            pred_features: Predictor features (batch, num_features, hidden_dim)

        Returns:
            Enriched features (batch, hidden_dim)
        """
        fp = torch.cat((last_hidden_state, pred_features), dim=-1)  # Concatenate
        fp_transformed = self.fc(fp)  # Transform
        attn_weights = self.attn_mech(fp)  # Compute attention
        enriched_features = torch.sum(
            fp_transformed * attn_weights, dim=1
        )  # Weighted sum
        return enriched_features


class TemporalAttention(nn.Module):
    """
    Attention mechanism for temporal features.

    This module computes attention weights over time steps to emphasize
    important temporal patterns.

    Args:
        num_features (int): Number of input features
        hidden_size (int): Size of hidden state
        dropout (float): Dropout probability (default: 0.1)

    Input:
        input_data: Original input (batch, seq_len, num_features)
        temp_features: Temporal features (batch, seq_len, hidden_size)

    Output:
        enriched_features: Weighted sum (batch, num_features + hidden_size)
    """

    def __init__(self, num_features, hidden_size, dropout=0.1):
        super().__init__()
        combined_dim = num_features + hidden_size
        self.fc = nn.Linear(combined_dim, combined_dim)
        self.attn_mech = nn.Sequential(nn.Linear(combined_dim, 1), nn.Softmax(dim=1))

    def forward(self, input_data, temp_features):
        """
        Apply attention to temporal features.

        Args:
            input_data: Original input (batch, seq_len, num_features)
            temp_features: Temporal features (batch, seq_len, hidden_size)

        Returns:
            Enriched features (batch, num_features + hidden_size)
        """
        ft = torch.cat((input_data, temp_features.squeeze(2)), dim=-1)  # Concatenate
        ft_transformed = self.fc(ft)  # Transform
        attn_weights = self.attn_mech(ft)  # Compute attention
        enriched_features = torch.sum(
            ft_transformed * attn_weights, dim=1
        )  # Weighted sum
        return enriched_features


class ILSTM_Soil(BaseModel):
    """
    Interpretable LSTM for soil moisture prediction.

    This advanced model uses:
    - Per-feature LSTMs to process each input feature independently
    - Multi-feature attention to weigh feature importance
    - Predictor attention for spatial/feature relationships
    - Temporal attention for time-based patterns
    - Combined features fed to final MLP

    Architecture designed for interpretability in soil moisture forecasting.

    Args:
        time_steps (int): Sequence length
        num_features (int): Number of input features
        output_dim (int): Number of output features (default: 1)
        hidden_size (int): Hidden state dimension (default: 128)
        dropout (float): Dropout probability (default: 0)

    Input shape:
        (batch_size, time_steps, num_features)

    Output shape:
        (batch_size, output_dim)

    Note:
        This model creates separate LSTM branches for each feature,
        allowing it to learn feature-specific temporal patterns.
    """

    def __init__(
        self, time_steps, num_features, output_dim=1, hidden_size=128, dropout=0
    ):
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.num_features = num_features

        # Create one LSTM per feature for feature-specific processing
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=5,
                    batch_first=True,
                )
                for _ in range(num_features)
            ]
        )

        # Normalization and regularization
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Multi-feature attention
        self.multi_feat_attn = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Softmax(dim=1)
        )

        # Linear projections for feature reduction
        self.lin2 = nn.Linear(time_steps, 1)
        self.lin3 = nn.Linear(num_features, 1)

        # Custom attention modules
        self.predictor_attention = PredictorAttention(2 * hidden_size)
        self.temporal_attention = TemporalAttention(num_features, hidden_size)

        # Final MLP decoder
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_size + num_features, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        """
        Forward pass through the ILSTM_Soil model.

        Args:
            x: Input tensor (batch, time_steps, num_features)

        Returns:
            Predictions (batch, output_dim)
        """
        batch_size, time_steps, num_features = x.shape

        # Process each feature independently with its own LSTM
        multi_feature_vectors = []
        for i in range(num_features):
            feature_vector, (_, _) = self.lstms[i](x[:, :, i].unsqueeze(-1))
            feature_vector = self.norm(feature_vector)
            feature_vector = self.dropout(feature_vector)
            multi_feature_vectors.append(feature_vector)

        # Stack feature vectors: (batch, time, num_features, hidden_size)
        multi_feature_vectors = torch.stack(multi_feature_vectors, dim=2)

        # Apply multi-feature attention
        mf_a = self.multi_feat_attn(multi_feature_vectors)
        mf_xt = multi_feature_vectors * mf_a

        # Extract predictor features (aggregated across time)
        predictor_features = (
            (self.lin2(mf_xt.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).squeeze(1)
        )

        # Extract temporal features (aggregated across features)
        temporal_features = (
            self.lin3(mf_xt.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).squeeze(2)
        )

        # Apply attention mechanisms
        enriched_pred_features = self.predictor_attention(
            multi_feature_vectors[:, -1, :, :], predictor_features
        )
        enriched_temp_features = self.temporal_attention(x, temporal_features)

        # Combine all enriched features
        combined = torch.cat([enriched_pred_features, enriched_temp_features], dim=-1)

        # Final prediction
        output = self.mlp(combined)
        return output
