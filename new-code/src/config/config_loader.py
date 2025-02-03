import yaml
from pathlib import Path

def load_config(config_path=None):
    """
    Loads a YAML config file and returns it as a dictionary.
    If no path is specified, defaults to src/config/config.yaml
    """
    if config_path is None:
        # Move up from this file's directory to find src/config/
        default_path = Path(__file__).parent / "config.yaml"
        config_path = default_path.resolve()

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
