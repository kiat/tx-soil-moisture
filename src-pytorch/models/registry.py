"""
Model registry for dynamic model selection.

This module provides:
- register(): Register a model class with a name
- get_model(): Retrieve and instantiate a model by name
- list_models(): List all registered model names

Purpose:
    The registry pattern allows code to select models by string names
    (e.g., from command-line arguments or config files) without hardcoding
    imports and conditionals. It makes the codebase more extensible.

Usage:
    >>> from models.registry import register, get_model
    >>> register("lstm", LSTMModel)
    >>> model = get_model("lstm", input_dim=10, hidden_size=64)

    Or with normalized names:
    >>> model = get_model("LSTM", input_dim=10)  # Case-insensitive
    >>> model = get_model("lstm_model", input_dim=10)  # Ignores underscores
"""

# Global registry dictionary
_registry = {}


def normalize_name(name: str) -> str:
    """
    Normalize model name for case-insensitive, flexible matching.

    Converts to lowercase and removes underscores and "model" suffix.

    Examples:
        "LSTM" -> "lstm"
        "lstm_model" -> "lstm"
        "BiLSTM_Model" -> "bilstm"

    Args:
        name (str): Model name to normalize

    Returns:
        str: Normalized name
    """
    return name.lower().replace("_", "").replace("model", "")


def register(name: str, model_class):
    """
    Register a model class with a name.

    Args:
        name (str): Name to register the model under (will be normalized)
        model_class: Model class (not an instance)

    Example:
        >>> register("lstm", LSTMModel)
        >>> register("bi_lstm", BiLSTMModel)
    """
    normalized = normalize_name(name)
    _registry[normalized] = model_class


def get_model(name: str, **kwargs):
    """
    Get a model instance by name.

    Args:
        name (str): Name of the model (will be normalized)
        **kwargs: Arguments to pass to the model constructor

    Returns:
        Model instance

    Raises:
        KeyError: If model name is not registered

    Example:
        >>> model = get_model("lstm", input_dim=10, hidden_size=64)
        >>> model = get_model("BiLSTM", input_dim=10)
    """
    normalized = normalize_name(name)
    if normalized not in _registry:
        raise KeyError(
            f"Model '{name}' (normalized: '{normalized}') not found in registry. "
            f"Available models: {list(_registry.keys())}"
        )

    model_class = _registry[normalized]
    return model_class(**kwargs)


def get_model_class(name: str):
    """
    Get a model class (not instantiated) by name.

    Args:
        name (str): Name of the model (will be normalized)

    Returns:
        Model class

    Raises:
        KeyError: If model name is not registered

    Example:
        >>> LSTMClass = get_model_class("lstm")
        >>> model = LSTMClass(input_dim=10)
    """
    normalized = normalize_name(name)
    if normalized not in _registry:
        raise KeyError(
            f"Model '{name}' (normalized: '{normalized}') not found in registry. "
            f"Available models: {list(_registry.keys())}"
        )

    return _registry[normalized]


def list_models():
    """
    List all registered model names.

    Returns:
        list: Sorted list of registered model names

    Example:
        >>> list_models()
        ['ar', 'attentionlstm', 'baseline', 'bilstm', 'cnn', ...]
    """
    return sorted(_registry.keys())


def is_registered(name: str) -> bool:
    """
    Check if a model name is registered.

    Args:
        name (str): Model name to check

    Returns:
        bool: True if registered, False otherwise

    Example:
        >>> is_registered("lstm")
        True
        >>> is_registered("nonexistent")
        False
    """
    normalized = normalize_name(name)
    return normalized in _registry
