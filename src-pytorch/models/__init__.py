"""
Models package for time-series prediction.

This package provides a collection of PyTorch models for soil moisture
and other time-series prediction tasks. Models range from simple baselines
to advanced architectures with attention mechanisms.

Package Structure:
    base.py         - BaseModel class with save/load utilities
    baseline.py     - Simple baseline models (persistence, moving average)
    lstm.py         - LSTM-based models (LSTM, BiLSTM, MultiHeadLSTM)
    rnn_cnn.py      - RNN and CNN models
    attention.py    - Attention-based models (AttentionLSTM, ILSTM_Soil, etc.)
    other.py        - Other models (AR, Transformer)
    registry.py     - Model registration and factory functions

Quick Start:
    >>> from models import get_model, list_models
    >>>
    >>> # See available models
    >>> print(list_models())
    >>>
    >>> # Create a model by name
    >>> model = get_model("lstm", input_dim=10, hidden_size=64)
    >>>
    >>> # Or import directly
    >>> from models import LSTMModel
    >>> model = LSTMModel(input_dim=10)

All Models:
    Baselines:
        - Baseline: Last-value persistence
        - MovingAverageBaseline: Moving average predictor

    LSTM Family:
        - LSTMModel: Standard LSTM
        - BiLSTMModel: Bidirectional LSTM
        - MultiHeadLSTM: LSTM with multi-head attention

    Attention Models:
        - AttentionLSTM: LSTM with attention between layers
        - AttentionOnly: Pure attention (no recurrence)
        - ILSTM_Soil: Interpretable LSTM for soil moisture

    Other Models:
        - RNNModel: Simple RNN
        - CNNModel: 1D CNN
        - AR: Autoregressive model
        - TransformerModel: Transformer encoder

Migration from models.py:
    This package replaces the single models.py file. Old code like:
        import models as model_module
        model = model_module.LSTMModel(input_dim=10)

    Can now use:
        import models
        model = models.LSTMModel(input_dim=10)

    Or use the registry:
        from models import get_model
        model = get_model("lstm", input_dim=10)
"""

# Import base class
from .base import BaseModel

# Import all model classes
from .baseline import Baseline, MovingAverageBaseline
from .lstm import LSTMModel, BiLSTMModel, MultiHeadLSTM
from .rnn_cnn import RNNModel, CNNModel
from .attention import (
    AttentionLSTM,
    AttentionOnly,
    PredictorAttention,
    TemporalAttention,
    ILSTM_Soil,
)
from .other import AR, TransformerModel

# Import registry functions
from .registry import register, get_model, get_model_class, list_models, is_registered

# Register all models
register("baseline", Baseline)
register("movingaverage", MovingAverageBaseline)
register("lstm", LSTMModel)
register("bilstm", BiLSTMModel)
register("multiheadlstm", MultiHeadLSTM)
register("rnn", RNNModel)
register("cnn", CNNModel)
register("attentionlstm", AttentionLSTM)
register("attentiononly", AttentionOnly)
register("ilstmsoil", ILSTM_Soil)
register("ar", AR)
register("autoregressive", AR)  # Alias
register("transformer", TransformerModel)

# Define what gets exported with "from models import *"
__all__ = [
    # Base
    "BaseModel",
    # Baseline models
    "Baseline",
    "MovingAverageBaseline",
    # LSTM models
    "LSTMModel",
    "BiLSTMModel",
    "MultiHeadLSTM",
    # RNN and CNN
    "RNNModel",
    "CNNModel",
    # Attention models
    "AttentionLSTM",
    "AttentionOnly",
    "PredictorAttention",
    "TemporalAttention",
    "ILSTM_Soil",
    # Other models
    "AR",
    "TransformerModel",
    # Registry functions
    "register",
    "get_model",
    "get_model_class",
    "list_models",
    "is_registered",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "TX Soil Moisture Team"
