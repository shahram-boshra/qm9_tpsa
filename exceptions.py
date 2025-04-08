# --- exceptions.py ---

class ConfigLoadError(Exception):
    """
    Custom exception raised when there are issues loading or parsing configuration files.

    This exception indicates problems such as:
    - File not found.
    - Invalid file format (e.g., incorrect JSON or YAML syntax).
    - Missing required configuration parameters.
    - Invalid data types for configuration parameters.

    Example:
        try:
            config = load_config("config.yaml")
        except ConfigLoadError as e:
            print(f"Error loading configuration: {e}")
            # Handle the error appropriately, e.g., exit or use default values.
    """
    pass

class DataSplitError(Exception):
    """
    Custom exception raised when there are problems splitting the dataset.

    This exception indicates issues such as:
    - Invalid split ratios (e.g., sum of ratios not equal to 1).
    - Insufficient data for the requested split.
    - Errors during the splitting process (e.g., incorrect indexing).

    Example:
        try:
            train_set, val_set, test_set = split_data(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        except DataSplitError as e:
            print(f"Error splitting data: {e}")
            # Handle the error appropriately, e.g., use a different splitting strategy.
    """
    pass

class ModelInitializationError(Exception):
    """
    Custom exception raised when there are problems creating or initializing the machine learning model.

    This exception indicates issues such as:
    - Invalid model architecture specification.
    - Missing required model parameters.
    - Errors during weight initialization.
    - Incompatible model parameters with the dataset.

    Example:
        try:
            model = create_model(input_dim=10, output_dim=5)
        except ModelInitializationError as e:
            print(f"Error initializing model: {e}")
            # Handle the error appropriately, e.g., use a different model or adjust parameters.
    """
    pass

class TrainingError(Exception):
    """
    Custom exception raised when there are problems during the model training process.

    This exception indicates issues such as:
    - Loss function errors (e.g., NaN loss).
    - Optimizer errors (e.g., invalid learning rate).
    - Gradient issues (e.g., exploding or vanishing gradients).
    - Memory errors during training.
    - Unexpected interruptions during training.

    Example:
        try:
            train_model(model, train_loader, optimizer, loss_fn)
        except TrainingError as e:
            print(f"Error during training: {e}")
            # Handle the error appropriately, e.g., adjust hyperparameters or debug the training loop.
    """
    pass

class TestingError(Exception):
    """
    Custom exception raised when there are problems during the model testing or evaluation process.

    This exception indicates issues such as:
    - Errors during prediction on test data.
    - Incorrect calculation or application of evaluation metrics.
    - Invalid or corrupted test data.
    - Device-related errors during testing.

    Example:
        try:
            evaluate_model(model, test_loader)
        except TestingError as e:
            print(f"Error during testing: {e}")
            # Handle the error appropriately, e.g., check evaluation metrics or debug the testing loop.
    """
    pass

class DeviceInitializationError(Exception):
    """
    Custom exception raised when there are problems initializing the computational device (CPU or GPU).

    This exception indicates issues such as:
    - Requested GPU not being available.
    - Errors related to CUDA or other device-specific libraries.
    - Insufficient memory on the chosen device.
    - Issues with device drivers.

    Example:
        try:
            device = initialize_device()
        except DeviceInitializationError as e:
            print(f"Error initializing device: {e}")
            # Handle the error appropriately, e.g., fall back to CPU or exit.
    """
    pass

class DatasetLoadingError(Exception):
    """
    Custom base exception for errors that occur during the process of loading a dataset.

    This is a general exception from which more specific dataset loading errors are derived.
    It indicates a fundamental problem in accessing or preparing the data.

    Example:
        try:
            dataset = load_my_dataset("path/to/data")
        except DatasetLoadingError as e:
            print(f"Error loading dataset: {e}")
            # Handle the error appropriately, e.g., check data path or format.
    """
    pass

class SDFFileNotFoundError(DatasetLoadingError):
    """
    Custom exception raised specifically when the specified SDF (Structure-Data File) is not found at the given path.

    This is a subclass of DatasetLoadingError, indicating a problem with locating the necessary chemical structure data file.

    Example:
        try:
            dataset = QM9Dataset(root='./data')
        except SDFFileNotFoundError as e:
            print(f"Error: SDF file not found. Please check the file path. Details: {e}")
    """
    pass

class SDFFileLoadError(DatasetLoadingError):
    """
    Custom exception raised when there is an error encountered while attempting to load or read the content of the SDF file.

    This is a subclass of DatasetLoadingError, indicating a problem with the integrity or format of the SDF file itself.

    Example:
        try:
            dataset = QM9Dataset(root='./data')
        except SDFFileLoadError as e:
            print(f"Error: Could not load the SDF file. It might be corrupted or in an invalid format. Details: {e}")
    """
    pass

class MoleculeLoadError(DatasetLoadingError):
    """
    Custom exception raised when there is a problem loading or processing a specific molecule record within the SDF file.

    This is a subclass of DatasetLoadingError, indicating an issue with the data for an individual chemical structure.

    Example:
        try:
            dataset = QM9Dataset(root='./data')
        except MoleculeLoadError as e:
            print(f"Error: Failed to load one or more molecules from the SDF file. Details: {e}")
    """
    pass

class ModelLayerInitializationError(Exception):
    """
    Custom exception raised when there are issues initializing individual layers within a neural network model.

    This exception indicates problems such as:
    - Providing invalid parameters to a layer.
    - Incompatible input or output dimensions for a layer.
    - Errors occurring within custom layer implementations during initialization.

    Example:
        try:
            layer = CustomLinear(in_features=10, out_features='abc')
        except ModelLayerInitializationError as e:
            print(f"Error initializing model layer: {e}")
            # Handle the error appropriately, e.g., correct layer parameters.
    """
    pass

class EarlyStoppingError(Exception):
    """
    Custom exception raised when there are issues related to the early stopping mechanism during training.

    This exception indicates problems such as:
    - Invalid configuration of early stopping parameters (e.g., negative patience).
    - Errors occurring within the early stopping logic.
    - Unexpected behavior or termination due to the early stopping criteria.

    Example:
        try:
            early_stopper = EarlyStopper(patience=-1)
        except EarlyStoppingError as e:
            print(f"Error in early stopping configuration: {e}")
            # Handle the error appropriately, e.g., adjust early stopping parameters.
    """
    pass

class PlottingError(Exception):
    """
    Custom exception raised when there are problems generating plots or visualizations.

    This exception indicates issues such as:
    - Invalid or incompatible data for plotting.
    - Errors encountered by the plotting library (e.g., matplotlib, seaborn).
    - Missing required plotting libraries.
    - Errors during the saving of plot files.

    Example:
        try:
            plot_results(history)
        except PlottingError as e:
            print(f"Error plotting results: {e}")
            # Handle the error appropriately, e.g., check data or install necessary libraries.
    """
    pass

class DataLoadingError(DatasetLoadingError):
    """
    Alias to DatasetLoadingError for backward compatibility.

    This alias is provided to ensure that existing code that uses `DataLoadingError`
    continues to function correctly. It maps directly to `DatasetLoadingError`.
    """
    pass
