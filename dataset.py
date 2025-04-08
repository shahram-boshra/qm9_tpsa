# dataset.py
import torch
import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
import logging
import os
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors
from data_utils import SelectQM9Targets
from exceptions import DatasetLoadingError, SDFFileNotFoundError, SDFFileLoadError, MoleculeLoadError

logger = logging.getLogger(__name__)


class QM9Dataset(torch_geometric.data.Dataset):
    """
    A PyTorch Geometric Dataset for the QM9 molecular dataset.

    This dataset loads molecular graphs from pre-processed PyTorch save files
    or directly from the raw SDF file. It includes functionality to select
    specific target properties and optionally limit the size of the dataset.
    It also calculates and adds the Topological Polar Surface Area (TPSA)
    as an additional target property.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (Optional[callable], optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to None.
        target_indices (Optional[List[int]], optional): A list of indices
            indicating which target properties from the original QM9 dataset
            should be included. If None, all original targets are used.
            Defaults to None.
        subset_size (Optional[int], optional): If specified, the dataset will
            be limited to the first `subset_size` molecules. Defaults to None.
    """
    def __init__(self, root: str, transform: Optional[callable] = None, target_indices: Optional[List[int]] = None, subset_size: Optional[int] = None):
        """
        Initializes the QM9Dataset.
        """
        self.root = root
        self.target_indices = target_indices
        self.subset_size = subset_size
        try:
            self.data_list: List[Data] = self._load_data()
        except DatasetLoadingError as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        super().__init__(self.root, transform)
        logger.info(f"QM9Dataset initialized. Loaded {len(self.data_list)} data points from: {os.path.join(self.processed_dir, self.processed_file_names[0])}")

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the names of the raw files required to process the dataset.
        """
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the names of the processed files that the dataset relies on.
        """
        return ['qm9_v3.pt']

    def process(self) -> None:
        """
        Processes the raw data and saves it into a processed file.
        Currently, this method is a placeholder as the data loading is handled
        directly in the `_load_data` method for efficiency.
        """
        logger.info("Processing QM9 data (currently a placeholder)...")
        pass

    def _load_data(self) -> List[Data]:
        """
        Loads the QM9 dataset either from a pre-processed PyTorch file or
        by reading and processing the raw SDF file. It also calculates and
        adds the TPSA as a target property.

        Returns:
            List[Data]: A list of PyTorch Geometric Data objects representing
                         the molecular graphs with selected targets and TPSA.

        Raises:
            DatasetLoadingError: If there is a general error during data loading.
            SDFFileNotFoundError: If the raw SDF file is not found.
            SDFFileLoadError: If there is an error loading the SDF file using RDKit.
            MoleculeLoadError: If an individual molecule cannot be loaded from the SDF file.
        """
        processed_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        logger.info(f"Attempting to load processed data from: {processed_path}")
        if os.path.exists(processed_path):
            try:
                loaded_data = torch.load(processed_path)
                logger.info(f"Successfully loaded processed data from: {processed_path}")
            except Exception as e:
                error_msg = f"Error loading processed data from {processed_path}: {e}"
                logger.error(error_msg)
                raise DatasetLoadingError(error_msg)
        else:
            logger.info(f"Processed data file not found at: {processed_path}. Loading raw data and processing.")
            loaded_data = []

        data_list: List[Data] = []
        sdf_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        if not os.path.exists(sdf_path):
            error_msg = f"SDF file not found at: {sdf_path}"
            logger.error(error_msg)
            raise SDFFileNotFoundError(error_msg)
        logger.info(f"Loading SDF file from: {sdf_path}")

        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            if suppl is None:
                error_msg = f"Error loading SDF file from {sdf_path}: Chem.SDMolSupplier returned None."
                logger.error(error_msg)
                raise SDFFileLoadError(error_msg)

            for i, data in enumerate(loaded_data):
                try:
                    mol = suppl[i]
                    if mol is None:
                        raise MoleculeLoadError(f"Error loading molecule {i+1} from SDF.")

                    tpsa = Descriptors.TPSA(mol)
                    y = data['y'][:, self.target_indices]


                    tpsa_tensor = torch.tensor([[tpsa]], dtype=y.dtype)
                    y_with_tpsa = torch.cat([y, tpsa_tensor], dim=1)

                    graph_data = torch_geometric.data.Data(x=data['x'], edge_index=data['edge_index'], y=y_with_tpsa, pos=data['pos'])
                    data_list.append(graph_data)
                except MoleculeLoadError as e:
                    logger.error(e) 
                except Exception as e:
                    error_msg = f"An unexpected error occurred while processing molecule {i+1}: {e}"
                    logger.error(error_msg)

            if self.subset_size is not None:
                data_list = data_list[:self.subset_size]
                logger.info(f"Subset of data selected: {len(data_list)} samples.")

            return data_list

        except SDFFileNotFoundError:
            raise 
        except SDFFileLoadError:
            raise 
        except Exception as e:
            error_msg = f"An unexpected error occurred during data loading: {e}"
            logger.error(error_msg)
            raise DatasetLoadingError(error_msg)

    def len(self) -> int:
        """
        Returns the number of molecules in the dataset.
        """
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        """
        Gets the data object at the specified index.

        Args:
            idx (int): The index of the data object to retrieve.

        Returns:
            Data: The PyTorch Geometric Data object at the given index.
        """
        return self.data_list[idx]

    
