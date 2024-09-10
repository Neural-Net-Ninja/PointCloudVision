import pandas as pd
from typing import Dict, Union

def normalize_dataset(dataset: pd.DataFrame,
                      min_values: Dict[str, float],
                      max_values: Dict[str, float]) -> pd.DataFrame:
    """
    Normalizes all fields of the point cloud that require normalization to 0-1.

    :param dataset: A point cloud.
    :type dataset: pandas.DataFrame
    :param min_values: A dictionary mapping column/attribute names to their minimum values.
    :type min_values: Dict[string, float]
    :param max_values: A dictionary mapping column/attribute names to their maximum values.
    :type max_values: Dict[string, float]
    :return: Point cloud with the supported columns normalized.
    :rtype: pandas.DataFrame
    """
    excluded_columns = {"x", "y", "z", "nx", "ny", "nz", "semclassid", "specificclassid", "original_semclassid"}
    
    def normalize_value(value, min_val, max_val):
        if (max_val - min_val) == 0:
            return 0
        return (value - min_val) / (max_val - min_val)
    
    for col in dataset.columns:
        if col not in excluded_columns and col in min_values:
            dataset[col] = dataset[col].apply(normalize_value, args=(min_values[col], max_values[col]))
    
    return dataset

def remap_normalized_values(dataset: pd.DataFrame, min_values: Dict[str, Union[int, float]],
                            max_values: Dict[str, Union[int, float]]) -> pd.DataFrame:
    """
    Reads the given metadata and remaps the normalized (0-1) values (e.g. intensity) to their original range using
    the min/max values saved in the metadata.

    :param dataset: point cloud
    :type dataset: pandas.DataFrame
    :param min_values: minimum values of columns before processing point cloud
    :type min_values: dict
    :param max_values: maximum values of columns before processing point cloud
    :type max_values: dict
    :return: point cloud without any normalized values
    :rtype: pandas.DataFrame
    """
    assert set(min_values.keys()) == set(max_values.keys()), "Min and max keys must match."
    
    def remap_value(value, min_val, max_val):
        return value * (max_val - min_val) + min_val
    
    for col in dataset.columns:
        if col in min_values:
            dataset[col] = dataset[col].apply(remap_value, args=(min_values[col], max_values[col]))
    
    return dataset

# Example test case
if __name__ == "__main__":
    # Example dataset
    data = {
        'intensity': [10, 55, 100, 150, 200],
        'color': [0.1, 0.5, 0.9, 0.3, 0.7],
        'temperature': [300, 350, 400, 450, 500],
        'pressure': [1.0, 1.5, 2.0, 2.5, 3.0],
        'humidity': [30, 40, 50, 60, 70]
    }
    dataset = pd.DataFrame(data)

    # Min and max values for normalization
    min_values = {
        'intensity': 10,
        'color': 0.1,
        'temperature': 300,
        'pressure': 1.0,
        'humidity': 30
    }
    max_values = {
        'intensity': 200,
        'color': 0.9,
        'temperature': 500,
        'pressure': 3.0,
        'humidity': 70
    }
    print(dataset)

    # Normalize the dataset
    normalized_dataset = normalize_dataset(dataset.copy(), min_values, max_values)
    print("\nNormalized Dataset:")
    print(normalized_dataset)

    # Denormalize the dataset
    denormalized_dataset = remap_normalized_values(normalized_dataset.copy(), min_values, max_values)
    print("\nDenormalized Dataset:")
    print(denormalized_dataset)

    # Convert denormalized dataset to match original data types
    for col in dataset.columns:
        denormalized_dataset[col] = denormalized_dataset[col].astype(dataset[col].dtype)

    # Check if denormalized values match original values
    if dataset.equals(denormalized_dataset):
        print("\nDenormalization successful, values match.")
    else:
        print("\nDenormalization failed, values do not match.")