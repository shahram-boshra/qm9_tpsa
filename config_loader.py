# config_loader.py
import yaml
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List, Optional, Any
import logging
from exceptions import ConfigLoadError

logger = logging.getLogger(__name__)

class DataConfig(BaseModel):
    """
    Configuration for data loading and preprocessing.
    """
    root_dir: str = Field(..., description="Root directory of the QM9 dataset.")
    target_indices: List[int] = Field([0], description="List of target indices to use from QM9.")
    use_cache: bool = Field(True, description="Whether to use cached processed data.")
    train_split: float = Field(..., description="Ratio of the dataset to use for training.")
    valid_split: float = Field(..., description="Ratio of the dataset to use for validation.")
    subset_size: Optional[int] = Field(None, description="Size of subset to use. If None, use entire dataset.")

    @field_validator("train_split", "valid_split")
    def validate_splits(cls, v: float) -> float:
        """
        Validates that split ratios are between 0 and 1.

        Args:
            v (float): The split ratio to validate.

        Returns:
            float: The validated split ratio.

        Raises:
            ValueError: If the split ratio is not between 0 and 1.
        """
        if not 0 < v < 1:
            raise ValueError("Split ratios must be between 0 and 1.")
        return v

    @field_validator("train_split")
    def validate_split_sum(cls, train_split: float, info: ValidationInfo) -> float:
        """
        Validates that the sum of train and validation splits is less than 1.

        Args:
            train_split (float): The training split ratio.
            info (ValidationInfo): Validation information.

        Returns:
            float: The validated training split ratio.

        Raises:
            ValueError: If the sum of train and validation splits is greater than or equal to 1.
        """
        valid_split: Optional[float] = info.data.get("valid_split")
        if valid_split is not None and train_split + valid_split >= 1:
            raise ValueError("train_split + valid_split must be less than 1.")
        return train_split

class ModelConfig(BaseModel):
    """
    Configuration for the model training process.
    """
    batch_size: int = Field(32, description="Batch size for training.")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")
    weight_decay: float = Field(0.0001, description="Weight decay for the optimizer.")
    step_size: int = Field(50, description="Step size for the learning rate scheduler.")
    gamma: float = Field(0.5, description="Gamma for the learning rate scheduler.")
    reduce_lr_factor: float = Field(0.5, description="Factor for reducing learning rate on plateau.")
    reduce_lr_patience: int = Field(10, description="Patience for reducing learning rate on plateau.")
    early_stopping_patience: int = Field(20, description="Patience for early stopping.")
    early_stopping_delta: float = Field(0.001, description="Minimum change in validation loss to qualify as an improvement.")
    l1_regularization_lambda: float = Field(0.001, description="Lambda for L1 regularization.")
    first_layer_type: Optional[str] = Field("custom_mp", description="Type of the first layer.")
    hidden_channels: int = Field(256, description="Number of hidden channels in the model.")
    second_layer_type: Optional[str] = Field("custom_mp", description="Type of the second layer.")
    dropout_rate: float = Field(0.5, description="Dropout rate in the model.")

class Config(BaseModel):
    """
    Main configuration class that combines data and model configurations.
    """
    data: DataConfig = Field(..., description="Data loading configuration.")
    model: ModelConfig = Field(..., description="Model training configuration.")

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """
        Loads configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Config: An instance of the Config class.

        Raises:
            ConfigLoadError: If there is an error loading the configuration.
        """
        try:
            with open(config_path, 'r') as f:
                config_dict: Any = yaml.safe_load(f)
            return cls(**config_dict)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise ConfigLoadError(f"Failed to load config from {config_path}: {e}")

def load_config(config_path: str) -> Config:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Config: An instance of the Config class.

    Raises:
        ConfigLoadError: If there is an error loading the configuration.
    """
    try:
        config: Config = Config.from_yaml(config_path)
        return config
    except ConfigLoadError as e:
        logger.error(f"Config loading failed: {e}")
        raise
