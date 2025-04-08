# main.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np
import logging
from dataset import QM9Dataset
from models import MGModel
from config_loader import load_config, Config
from training_utils import EarlyStopping, Trainer, Plot
from sklearn.model_selection import train_test_split
from exceptions import DataSplitError, ModelInitializationError, TrainingError, TestingError, DeviceInitializationError
from device_utils import get_device
from torch_geometric.data import Dataset




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_dataset(dataset: QM9Dataset, train_split: float, valid_split: float) -> tuple[QM9Dataset, QM9Dataset, QM9Dataset]:
    """
    Splits a QM9Dataset into training, validation, and test datasets.

    Args:
        dataset (QM9Dataset): The input dataset to be split.
        train_split (float): The proportion of the dataset to be used for training.
        valid_split (float): The proportion of the remaining data (after training split) to be used for validation.

    Returns:
        tuple[QM9Dataset, QM9Dataset, QM9Dataset]: A tuple containing the training, validation, and test datasets.

    Raises:
        DataSplitError: If an error occurs during the dataset splitting process.
    """
    try:
        train_idx, temp_idx = train_test_split(range(len(dataset)), train_size=train_split, random_state=42)
        valid_idx, test_idx = train_test_split(temp_idx, train_size=valid_split / (1 - train_split), random_state=42)

        train_dataset = dataset[list(train_idx)]
        valid_dataset = dataset[list(valid_idx)]
        test_dataset = dataset[list(test_idx)]

        return train_dataset, valid_dataset, test_dataset
    except Exception as e:
        logger.error(f"Error during dataset splitting: {e}")
        raise DataSplitError(f"Failed to split dataset: {e}")

if __name__ == '__main__':
    data_dir = Path('C:/Chem_Data/qm9')
    main_dir = Path(__file__).parent

    try:
        config_path = main_dir / 'config.yaml'
        config: Config = load_config(config_path)

        dataset: QM9Dataset = QM9Dataset(root=config.data.root_dir, target_indices=config.data.target_indices, subset_size=config.data.subset_size)


        torch.manual_seed(11)

        train_dataset, valid_dataset, test_dataset = split_dataset(dataset, config.data.train_split, config.data.valid_split)

        in_channels: int = dataset[0].x.shape[1]
        out_channels: int = train_dataset[0].y.shape[1]

        train_loader: DataLoader = DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True)
        valid_loader: DataLoader = DataLoader(valid_dataset, batch_size=config.model.batch_size, shuffle=False)
        test_loader: DataLoader = DataLoader(test_dataset, batch_size=config.model.batch_size, shuffle=False)

        model: MGModel = MGModel(
            in_channels=in_channels,
            out_channels=out_channels,
            first_layer_type=config.model.first_layer_type,
            second_layer_type=config.model.second_layer_type,
            hidden_channels=config.model.hidden_channels,
            dropout_rate=config.model.dropout_rate,
            gat_heads=1,
            transformer_heads=1,
        )

        logger.info(f'Model Architecture {model}')

        criterion: nn.HuberLoss = nn.HuberLoss(reduction='mean', delta=0.1)
        optimizer: optim.Adam = optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

        step_lr: StepLR = StepLR(optimizer, step_size=config.model.step_size, gamma=config.model.gamma)
        red_lr: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, mode='min', factor=config.model.reduce_lr_factor, patience=config.model.reduce_lr_patience)

        early_stopping: EarlyStopping = EarlyStopping(patience=config.model.early_stopping_patience, verbose=True, delta=config.model.early_stopping_delta)

        try:
            device: torch.device = get_device()
            logger.info(f"Using device: {device}")
        except DeviceInitializationError as e:
            logger.error(f"Device initialization failed: {e}")
            raise

        trainer: Trainer = Trainer(model, criterion, optimizer, step_lr, red_lr, early_stopping, config, device)

        try:
            train_losses: list[float]
            valid_losses: list[float]
            maes: list[float]
            mses: list[float]
            r2s: list[float]
            explained_variances: list[float]
            train_losses, valid_losses, maes, mses, r2s, explained_variances = trainer.train_and_validate(train_loader, valid_loader)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise TrainingError(f"Training failed: {e}")

        try:
            test_loss: float
            metrics: dict[str, float]
            test_targets: list[torch.Tensor]
            test_predictions: list[torch.Tensor]
            test_loss, metrics, test_targets, test_predictions = trainer.test_epoch(test_loader, return_predictions=True)
            test_mae: float = metrics['mae']
            test_mse: float = metrics['mse']
            test_r2: float = metrics['r2']
            test_explained_variance: float = metrics['explained_variance']
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            raise TestingError(f"Testing failed: {e}")

        np.save('test_targets.npy', np.array(test_targets))
        np.save('test_predictions.npy', np.array(test_predictions))

        logger.info(f'Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, R2: {test_r2:.4f}, Explained Variance: {test_explained_variance:.4f}')

        Plot.plot_losses(train_losses, valid_losses)
        Plot.plot_metrics_vs_epoch(maes, mses, r2s, explained_variances)

    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


        
