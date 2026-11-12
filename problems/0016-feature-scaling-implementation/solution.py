import numpy as np

def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=0)

    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # Avoid division by zero for constant columns
    std = np.where(std == 0, 1, std)
    range_val = np.where(max_val - min_val == 0, 1, max_val - min_val)

    standardized_data = (data - mean) / std
    normalized_data = (data - min_val) / range_val

    return (
        np.round(standardized_data, 4),
        np.round(normalized_data, 4)
    )