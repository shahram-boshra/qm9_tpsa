# TPSA-Augmented GNN for Molecular Property Prediction on QM9

This repository contains Python code for training a graph neural network (GNN) model to predict molecular properties from the QM9 dataset, with a specific focus on integrating the Topological Polar Surface Area (TPSA) as an additional target. The model leverages PyTorch Geometric for graph handling and supports various graph convolutional layers.

## Overview

The project is structured into several Python modules:

- **`data_utils.py`**: Contains utility functions for loading and preprocessing the QM9 dataset, including selecting specific target properties.
- **`dataset.py`**: Defines a custom PyTorch Geometric Dataset class for the QM9 dataset. It loads molecular graphs from raw SDF files or pre-processed PyTorch files and crucially, calculates and adds the Topological Polar Surface Area (TPSA) as an additional target property for prediction.
- **`device_utils.py`**: Provides a utility function to automatically determine and return the appropriate computational device (CPU, GPU, or TPU).
- **`exceptions.py`**: Defines custom exception classes for better error handling throughout the project.
- **`models.py`**: Implements the graph neural network model (`MGModel`) using PyTorch and PyTorch Geometric. It supports various graph convolutional layers like GCN, GAT, SAGE, GIN, GraphConv, and TransformerConv, as well as a custom message passing layer.
- **`training_utils.py`**: Contains classes and functions for managing the training loop, early stopping, evaluation metric calculation, and plotting of training/validation results.
- **`config_loader.py`**: (Although not explicitly a `.py` module, the `config.yaml` file serves this purpose) Loads and manages configuration parameters from the `config.yaml` file.

The `config.yaml` file provides a centralized way to configure the dataset loading, model hyperparameters, and training settings.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You might need to create a `requirements.txt` file with the following content if it doesn't exist):
    ```
    torch
    torch-scatter -f [https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html](https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html) # Adjust torch and CUDA version if needed
    torch-sparse -f [https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html](https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html) # Adjust torch and CUDA version if needed
    torch-geometric
    rdkit
    scikit-learn
    matplotlib
    pyyaml
    ```
    **Note:** Ensure that your PyTorch installation is compatible with your CUDA version if you intend to use a GPU. Refer to the [PyTorch website](https://pytorch.org/) for installation instructions.

3.  **Download the QM9 dataset:**
    The code expects the raw QM9 SDF file (`gdb9.sdf`) to be present in the `raw` directory within the specified `root_dir` in `config.yaml`. The `torch_geometric.datasets.QM9` class will automatically download the necessary files if they are not found. You might need to adjust the `root_dir` in `config.yaml` to your desired location.

## Configuration

The training and model parameters are configured in the `config.yaml` file:

```yaml
data:
  root_dir: C:/Chem_Data/qm9    # Root directory for the QM9 dataset
  target_indices: [0, 3, 5]      # Indices of target properties to predict
  use_cache: true
  train_split: 0.8
  valid_split: 0.1
  subset_size: 500             # Optional: Limit the dataset size

model:
  batch_size: 256
  learning_rate: 0.0070779431649418655
  weight_decay: 1.0908657690794923e-05
  step_size: 50
  gamma: 0.5
  reduce_lr_factor: 0.5
  reduce_lr_patience: 10
  early_stopping_patience: 20
  early_stopping_delta: 0.001
  l1_regularization_lambda: 0.006
  first_layer_type: "transformer_conv"
  hidden_channels: 512
  second_layer_type: "transformer_conv"
  dropout_rate: 0.176
data: Specifies the location of the QM9 dataset, which original target properties to use, data splitting ratios, and optional subsetting. Importantly, the dataset loading process in dataset.py automatically adds TPSA as an additional target.
model: Defines the hyperparameters for the graph neural network model, including batch size, learning rate, weight decay, learning rate scheduler parameters, early stopping criteria, regularization strength, GCN layer types, hidden channel dimensions, and dropout rate.
Make sure to adjust the root_dir in config.yaml to the actual path where you want to store the QM9 dataset.

Usage
Ensure the config.yaml file is properly configured with the correct paths and hyperparameters.

Create a main.py script that imports the necessary modules (config_loader, dataset, models, training_utils, etc.) and orchestrates the data loading, model initialization, training, and evaluation processes. Refer to the individual .py files for their functionalities.

Run the training script:

Bash
python main.py

Acknowledgements
This project relies on the following excellent open-source libraries and datasets:

PyTorch (https://pytorch.org/): For the deep learning framework and tensor computations.
PyTorch Geometric (https://pytorch-geometric.readthedocs.io/): For the graph neural network layers and graph data handling.
RDKit (https://www.rdkit.org/): For cheminformatics tasks, specifically for calculating the Topological Polar Surface Area (TPSA).
QM9 Dataset (https://www.nature.com/articles/sdata201422): For the molecular structures and property data used for training and evaluation.
Scikit-learn (https://scikit-learn.org/): For evaluation metrics and data splitting utilities.
Matplotlib (https://matplotlib.org/): For creating visualizations of the training and validation results.
PyYAML (https://pyyaml.org/): For parsing the configuration file.
We are grateful to the developers and maintainers of these resources for making their work publicly available.

Contributing
Contributions to this project are welcome. Please feel free to submit pull requests or open issues for any bugs or enhancements.

License
[Specify your license here, e.g., MIT License]

