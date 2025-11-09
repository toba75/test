"""Configuration management utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 402
    seq_len: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.2


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    num_steps: int = 10000
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    use_amp: bool = True
    seed: int = 42


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "~/stockGPT/data"
    prepared_path: str | None = None
    returns_output: str | None = None
    train_start: str = "1926-01-01"
    train_end: str = "2000-12-31"
    val_start: str = "1991-01-01"
    val_end: str = "2000-12-31"
    test_start: str = "2001-01-01"
    test_end: str = "2023-12-31"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], config_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML config file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
