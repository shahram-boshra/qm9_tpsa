# data_utils.py
import torch
from torch_geometric.datasets import QM9
from typing import List, Callable
import logging
from exceptions import DataLoadingError
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class SelectQM9Targets(object):

    def __init__(self, target_indices: List[int]):

        self.target_indices = target_indices

    def __call__(self, data: Data) -> Data:

        try:
            data.y = data.y[:, self.target_indices]
            return data
        except Exception as e:
            logger.error(f"Error selecting QM9 targets: {e}")
            raise DataLoadingError(f"Failed to select QM9 targets: {e}")

def load_qm9_data(root: str, target_indices: List[int]) -> List[Data]:
    try:
        dataset = QM9(root=root)
        transform = SelectQM9Targets(target_indices=target_indices)
        dataset.transform = transform
        data_list = [dataset[i] for i in range(len(dataset))]
        return data_list
    except Exception as e:
        logger.error(f"Error loading QM9 data: {e}")
        raise DataLoadingError(f"Failed to load QM9 data: {e}")
