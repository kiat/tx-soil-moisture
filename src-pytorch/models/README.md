# Models Package Documentation

## Overview

The `models/` package provides a clean, organized structure for PyTorch time-series models. This replaces the single `models.py` file with a modular package that's easier to maintain, test, and extend.

## Package Structure

```
models/
├── __init__.py         # Package entry point, exports all models and registry
├── base.py            # BaseModel class with save/load utilities
├── baseline.py        # Simple baseline models
├── lstm.py            # LSTM-based models
├── rnn_cnn.py         # RNN and CNN models
├── attention.py       # Attention-based models
├── other.py           # AR, Transformer, and other models
└── registry.py        # Model registration and factory system
```

## File Descriptions

### `base.py` - Foundation

**Purpose**: Provides `BaseModel` class that all models inherit from.

**Key Features**:

-   `save_checkpoint()`: Save model with optimizer state and metadata
-   `load_checkpoint()`: Load saved checkpoints
-   `count_parameters()`: Count trainable parameters
-   `get_device()`: Get the device model is on

**Why it matters**: Standardizes model lifecycle operations across all models. No more duplicate save/load code.

### `baseline.py` - Simple Baselines

**Purpose**: Non-trainable baseline models for comparison.

**Models**:

-   `Baseline`: Last-value persistence (naive forecast)
-   `MovingAverageBaseline`: Moving average predictor

**Why it matters**: These establish performance floors. Any ML model should beat these baselines.

### `lstm.py` - LSTM Family

**Purpose**: LSTM-based sequence models.

**Models**:

-   `LSTMModel`: Standard multi-layer LSTM with projection
-   `BiLSTMModel`: Bidirectional LSTM with layer normalization
-   `MultiHeadLSTM`: LSTM + multi-head attention

**Why it matters**: LSTMs are the workhorse of sequence modeling. These implementations include modern best practices (projection layers, layer norm, dropout).

### `rnn_cnn.py` - Alternative Architectures

**Purpose**: Non-LSTM sequence models.

**Models**:

-   `RNNModel`: Simple RNN (faster, simpler than LSTM)
-   `CNNModel`: 1D CNN with adaptive pooling

**Why it matters**:

-   RNN: Good for short sequences, faster training
-   CNN: Parallel processing, captures local patterns

### `attention.py` - Attention Mechanisms

**Purpose**: Models with attention for interpretability and performance.

**Models**:

-   `AttentionLSTM`: LSTM with attention between layers
-   `AttentionOnly`: Pure attention (no recurrence)
-   `ILSTM_Soil`: Specialized interpretable LSTM for soil moisture
    -   Per-feature LSTMs
    -   Multi-feature attention
    -   Predictor and temporal attention

**Helper Modules**:

-   `PredictorAttention`: Attention over predictor features
-   `TemporalAttention`: Attention over time steps

**Why it matters**: Attention mechanisms improve performance and provide interpretability (which features/timesteps matter most).

### `other.py` - Advanced Models

**Purpose**: Specialized and state-of-the-art architectures.

**Models**:

-   `AR`: Autoregressive model with adaptive pooling
-   `TransformerModel`: Transformer encoder

**Why it matters**:

-   AR: Simple but effective for stationary series
-   Transformer: State-of-the-art, good for long sequences

### `registry.py` - Model Factory

**Purpose**: Dynamic model selection by name.

**Key Functions**:

-   `register(name, model_class)`: Register a model
-   `get_model(name, **kwargs)`: Get model instance by name
-   `list_models()`: List all registered models
-   `is_registered(name)`: Check if model exists

**Why it matters**: Enables string-based model selection (great for CLI args, config files). No more giant if/elif chains.

### `__init__.py` - Package Entry Point

**Purpose**: Exposes all models and functions at package level.

**What it does**:

-   Imports all model classes
-   Registers all models with the registry
-   Defines `__all__` for clean exports

**Why it matters**: Users can `from models import LSTMModel` or use the registry.

## Usage Examples

### Basic Usage (Direct Import)

```python
from models import LSTMModel

# Create model directly
model = LSTMModel(input_dim=10, hidden_size=64, num_layers=3)
```

### Registry-Based Usage (Recommended)

```python
from models import get_model, list_models

# See available models
print(list_models())
# ['ar', 'attentionlstm', 'baseline', 'bilstm', 'cnn', ...]

# Create model by name (case-insensitive, flexible)
model = get_model("lstm", input_dim=10, hidden_size=64)
model = get_model("BiLSTM", input_dim=10)  # Same as "bilstm"
model = get_model("LSTM_Model", input_dim=10)  # Also works
```

### Using BaseModel Features

```python
from models import LSTMModel
import torch.optim as optim

model = LSTMModel(input_dim=10)
optimizer = optim.Adam(model.parameters())

# Save checkpoint
model.save_checkpoint(
    "checkpoints/model.pth",
    optimizer=optimizer,
    epoch=10,
    metadata={"val_loss": 0.05}
)

# Load checkpoint
metadata = model.load_checkpoint("checkpoints/model.pth", optimizer=optimizer)
print(f"Resumed from epoch {metadata['epoch']}")

# Count parameters
print(f"Model has {model.count_parameters():,} parameters")
```

### Migration from Old Code

**Old code (single models.py)**:

```python
import models as model_module

model = model_module.LSTMModel(input_dim=10)
```

**New code (Option 1 - Direct)**:

```python
from models import LSTMModel

model = LSTMModel(input_dim=10)
```

**New code (Option 2 - Registry)**:

```python
from models import get_model

model = get_model("lstm", input_dim=10)
```

**Backward compatible (no changes needed)**:

```python
import models as model_module

# Works exactly as before!
model = model_module.LSTMModel(input_dim=10)
```

## Model Selection Pattern (from notebook)

The notebook currently uses a `model_map` dictionary. With the registry, this becomes cleaner:

**Old pattern**:

```python
model_map = {
    "lstm": model_module.LSTMModel,
    "bilstm": model_module.BiLSTMModel,
    # ... etc
}
process_queue = {
    name: model_class
    for name, model_class in model_map.items()
    if name in requested_ids
}
```

**New pattern (simpler)**:

```python
from models import get_model_class, is_registered

process_queue = {}
for name in requested_ids:
    if is_registered(name):
        process_queue[name] = get_model_class(name)
    else:
        print(f"Warning: Model '{name}' not found")
```

## Logging Structure

When training models, TensorBoard logs are automatically organized in a hierarchical directory structure for easy experiment tracking:

```
logs/
├── ws48_offset24/              # Window size 48, offset 24
│   ├── lstm/                   # Model type
│   │   ├── SWC_20-SWC_20/     # Predictors-Predictions
│   │   ├── Ppt,Tair-SWC_20/   # Multiple predictors
│   │   └── ...
│   ├── bilstm/
│   │   └── ...
│   └── ...
├── ws72_offset24/              # Different window configuration
│   └── ...
└── ...
```

### Directory Hierarchy

**Level 1: Window Configuration** - `ws{window_size}_offset{offset}/`

-   Groups experiments by temporal settings
-   Example: `ws48_offset24` = 48-hour window, 24-hour prediction offset

**Level 2: Model Architecture** - `{model}/`

-   Organizes by model type (lstm, bilstm, cnn, etc.)
-   All models registered in the registry

**Level 3: Feature Configuration** - `{predictors}-{predictions}/`

-   Separates experiments by input/output features
-   Predictors: comma-separated list of input features
-   Predictions: comma-separated list of target features
-   Examples:
    -   `SWC_20-SWC_20`: Autoregressive (predict SWC_20 from itself)
    -   `Ppt,Tair-SWC_20`: Predict soil moisture from precipitation and air temperature
    -   `SWC_20,Ppt,Tair-SWC_20,SWC_40`: Multiple predictors and targets

### Usage with TensorBoard

View all experiments:

```bash
tensorboard --logdir=logs/
```

View specific window configuration:

```bash
tensorboard --logdir=logs/ws48_offset24/
```

Compare models for specific feature set:

```bash
tensorboard --logdir=logs/ws48_offset24/*/SWC_20-SWC_20/
```

### Daily Average Outputs

When `daily_average_output=True`, predicted feature names include `_daily_avg` suffix:

-   Input: `predict_features="SWC_20"`
-   Logged as: `SWC_20_daily_avg-SWC_20_daily_avg`
-   This indicates the model predicts daily averaged values rather than instantaneous measurements

### Benefits

1. **Organization**: Easy to navigate hundreds of experiments
2. **Comparison**: TensorBoard can compare runs within same directory
3. **Reproducibility**: Path clearly indicates configuration
4. **Filtering**: Use wildcards to view specific subsets
5. **Scalability**: Structure handles arbitrary feature combinations

## Adding New Models

### Step 1: Create the model class

Add your model to an appropriate file (or create a new one):

```python
# models/lstm.py (or create models/custom.py)
from .base import BaseModel
import torch.nn as nn

class MyNewModel(BaseModel):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        # Define layers...

    def forward(self, x):
        # Define forward pass...
        return output
```

### Step 2: Register in `__init__.py`

```python
# models/__init__.py
from .lstm import LSTMModel, MyNewModel  # Add import
# ...
register("mynewmodel", MyNewModel)  # Add registration
```

### Step 3: Use it

```python
from models import get_model
model = get_model("mynewmodel", input_dim=10)
```

## Testing

### Quick smoke test

```python
# Test all models can be instantiated
from models import list_models, get_model

for name in list_models():
    try:
        if name in ["baseline", "movingaverage"]:
            model = get_model(name, label_width=1)
        elif name == "ilstmsoil":
            model = get_model(name, time_steps=48, num_features=10)
        else:
            model = get_model(name, input_dim=10)
        print(f"✓ {name}: {model.count_parameters()} params")
    except Exception as e:
        print(f"✗ {name}: {e}")
```

### Forward pass test

```python
import torch

x = torch.randn(4, 48, 10)  # (batch=4, seq_len=48, features=10)

for name in ["lstm", "bilstm", "cnn"]:
    model = get_model(name, input_dim=10)
    y = model(x)
    print(f"{name}: input {x.shape} -> output {y.shape}")
```

## Benefits of This Structure

1. **Modularity**: Each model type in its own file (easier to navigate)
2. **Maintainability**: Smaller files, focused responsibilities
3. **Testability**: Can test each module independently
4. **Extensibility**: Add new models without touching existing code
5. **Collaboration**: Multiple people can work on different model files
6. **Registry Pattern**: Dynamic model selection without hardcoding
7. **Backward Compatible**: Old code keeps working via shim
8. **Documentation**: Each file has focused documentation
9. **Code Review**: Smaller diffs when adding/modifying models
10. **Reusability**: Common functionality in BaseModel

## Next Steps

1. **Run existing code**: Test with your notebook (should work unchanged)
2. **Migrate gradually**: Start using `get_model()` in new code
3. **Add tests**: Create `tests/test_models.py` for automated testing
4. **Remove shim eventually**: Once all code migrated, remove `models.py` shim
5. **Add new models**: Follow the pattern to add domain-specific models
