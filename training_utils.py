# training_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np
import matplotlib.pyplot as plt
from models import MGModel
from config_loader import Config
import logging
from exceptions import TrainingError, TestingError, EarlyStoppingError, PlottingError
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0, path: str = 'chk_learn.pt'):
        """
        Initializes EarlyStopping.

        Args:
            patience (int): How many epochs to wait after last validation loss improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score: float = None
        self.early_stop: bool = False
        logger.debug("EarlyStopping initialized.")

    def __call__(self, valid_loss: float, model: nn.Module) -> None:
        """
        Checks if the validation loss has improved.

        Args:
            valid_loss (float): Current validation loss.
            model (nn.Module): Model to save if the validation loss has improved.

        Raises:
            EarlyStoppingError: If an error occurs during early stopping.
        """
        try:
            if self.best_score is None:
                self.best_score = valid_loss
                self.save_model_state(valid_loss, model)
            elif valid_loss > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = valid_loss
                self.save_model_state(valid_loss, model)
                self.counter = 0
        except Exception as e:
            logger.error(f"Error in EarlyStopping: {e}")
            raise EarlyStoppingError(f"EarlyStopping failed: {e}")

    def save_model_state(self, valid_loss: float, model: nn.Module) -> None:
        """
        Saves the model state if the validation loss has improved.

        Args:
            valid_loss (float): Current validation loss.
            model (nn.Module): Model to save.
        """
        if self.verbose:
            logger.info(f"Validation loss decreased ({self.best_score:.6f} --> {valid_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)

def calculate_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Calculates evaluation metrics.

    Args:
        targets (np.ndarray): True target values.
        predictions (np.ndarray): Predicted values.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    return {
        "mae": mean_absolute_error(targets, predictions),
        "mse": mean_squared_error(targets, predictions),
        "r2": r2_score(targets, predictions),
        "explained_variance": explained_variance_score(targets, predictions),
    }

class TrainingLoop:
    """
    Encapsulates the training loop logic.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, step_lr: StepLR, device: torch.device, l1_lambda: float):
        """
        Initializes TrainingLoop.

        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Optimizer.
            step_lr (StepLR): Learning rate scheduler.
            device (torch.device): Device to use for training.
            l1_lambda (float): L1 regularization lambda.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.device = device
        self.l1_lambda = l1_lambda
        logger.debug("TrainingLoop initialized.")

    def _process_epoch(self, data_loader: DataLoader, train: bool = False) -> Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray]]:
        """
        Processes a single epoch (train, validate, or test).

        Args:
            data_loader (DataLoader): DataLoader for the epoch.
            train (bool): If True, performs training; otherwise, evaluation.

        Returns:
            Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray]]: Average loss, metrics, all targets, and all predictions.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        num_graphs = 0
        all_targets: List[np.ndarray] = []
        all_predictions: List[np.ndarray] = []

        with torch.set_grad_enabled(train):
            for batch in data_loader:
                batch = batch.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                out, l1_reg = self.model(batch.x, batch.edge_index, batch.batch)
                target = batch.y
                loss = self.criterion(out, target)
                if train:
                    loss += l1_reg * self.l1_lambda
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                num_graphs += batch.num_graphs
                all_targets.append(target.detach().cpu().numpy())
                all_predictions.append(out.detach().cpu().numpy())

        avg_loss = total_loss / num_graphs
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        metrics = calculate_metrics(all_targets, all_predictions)
        return avg_loss, metrics, all_targets, all_predictions

    def validate_epoch(self, valid_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validates the model for one epoch.

        Args:
            valid_loader (DataLoader): DataLoader for validation data.

        Returns:
            Tuple[float, Dict[str, float]]: Average validation loss and metrics.
        """
        avg_loss, metrics, _, _ = self._process_epoch(valid_loader)
        logger.debug(f"Validation Epoch Loss: {avg_loss:.4f}, {metrics}")
        return avg_loss, metrics

    def test_epoch(self, test_loader: DataLoader, return_predictions: bool = False) -> Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray]]:
        """
        Tests the model for one epoch.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            return_predictions (bool): If True, returns predictions and targets.

        Returns:
            Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray]]: Average test loss, metrics, all targets, and all predictions.
        """
        avg_loss, metrics, all_targets, all_predictions = self._process_epoch(test_loader)
        logger.info(f"Test Epoch Loss: {avg_loss:.4f}, {metrics}")
        if return_predictions:
            return avg_loss, metrics, all_targets, all_predictions
        return avg_loss, metrics, [], []

class Trainer:
    """
    Manages the training and validation process.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, step_lr: StepLR, red_lr: ReduceLROnPlateau, early_stopping: EarlyStopping, config: Config, device: torch.device):
        """
        Initializes Trainer.

        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Optimizer.
            step_lr (StepLR): Step learning rate scheduler.
            red_lr (ReduceLROnPlateau): Reduce learning rate on plateau scheduler.
            early_stopping (EarlyStopping): Early stopping object.
            config (Config): Configuration object.
            device (torch.device): Device to use for training.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.red_lr = red_lr
        self.early_stopping = early_stopping
        self.config = config
        self.device = device
        self.training_loop = TrainingLoop(self.model, self.criterion, self.optimizer, self.step_lr, self.device, self.config.model.l1_regularization_lambda)
        logger.debug("Trainer initialized.")

    def train_and_validate(self, train_loader: DataLoader, valid_loader: DataLoader) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        """
        Trains and validates the model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader): DataLoader for validation data.

        Returns:
            Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]: Lists of training losses, validation losses, MAEs, MSEs, R2s, and explained variances.

        Raises:
            TrainingError: If an error occurs during training and validation.
        """
        try:
            train_losses: List[float] = []
            valid_losses: List[float] = []
            maes: List[float] = []
            mses: List[float] = []
            r2s: List[float] = []
            explained_variances: List[float] = []
            for epoch in range(self.config.model.early_stopping_patience * 2):
                avg_loss, metrics, _, _ = self.training_loop._process_epoch(train_loader, train=True)
                self.training_loop.step_lr.step()
                train_loss = avg_loss
                valid_loss, metrics = self.training_loop.validate_epoch(valid_loader)

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                maes.append(metrics['mae'])
                mses.append(metrics['mse'])
                r2s.append(metrics['r2'])
                explained_variances.append(metrics['explained_variance'])

                self.red_lr.step(valid_loss)
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    logger.info("Early stopping triggered.")
                    break
            return train_losses, valid_losses, maes, mses, r2s, explained_variances
        except Exception as e:
            logger.error(f"Error during training and validation: {e}")
            raise TrainingError(f"Training and validation failed: {e}")

    def test_epoch(self, test_loader: DataLoader, return_predictions: bool = False) -> Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray]]:
        """
        Tests the model for one epoch.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            return_predictions (bool): If True, returns predictions and targets.

        Returns:
            Tuple[float, Dict[str, float], List[np.ndarray], List[np.ndarray]]: Average test loss, metrics, all targets, and all predictions.
        """
        return self.training_loop.test_epoch(test_loader, return_predictions)

class Plot:
    """
    Handles plotting of training and validation results.
    """
    @staticmethod
    def plot_losses(train_losses: List[float], valid_losses: List[float]) -> None:
        """
        Plots training and validation losses.

        Args:
            train_losses (List[float]): List of training losses.
            valid_losses (List[float]): List of validation losses.

        Raises:
            PlottingError: If an error occurs during plotting.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting losses: {e}")
            raise PlottingError(f"Failed to plot losses: {e}")

    @staticmethod
    def plot_metrics_vs_epoch(maes: List[float], mses: List[float], r2s: List[float], explained_variances: List[float]) -> None:
        """
        Plots metrics against epochs.

        Args:
            maes (List[float]): List of MAEs.
            mses (List[float]): List of MSEs.
            r2s (List[float]): List of R2s.
            explained_variances (List[float]): List of explained variances.

        Raises:
            PlottingError: If an error occurs during plotting.
        """
        try:
            epochs = range(1, len(maes) + 1)

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.plot(epochs, maes, label='MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('MAE vs. Epoch')
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(epochs, mses, label='MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('MSE vs. Epoch')
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(epochs, r2s, label='R2')
            plt.xlabel('Epoch')
            plt.ylabel('R2')
            plt.title('R2 vs. Epoch')
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(epochs, explained_variances, label='Explained Variance')
            plt.xlabel('Epoch')
            plt.ylabel('Explained Variance')
            plt.title('Explained Variance vs. Epoch')
            plt.legend()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")
            raise PlottingError(f"Failed to plot metrics: {e}")
