import numpy as np

def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalizes the features of a dataset.
    :param X: The dataset to normalize.
    :return: The normalized dataset.
    """

    # Calculate the mean and standard deviation of each feature
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    # Normalize the dataset
    X_normalized = (X - X_mean) / X_std

    return X_normalized

