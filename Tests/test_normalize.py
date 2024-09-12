import pandas as pd
from typing import Dict, Union
import unittest
import numpy as np
from remap import remap_normalized_values

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

# test_remap.py


class TestRemapNormalizedValues(unittest.TestCase):

    def setUp(self):
        # Example dataset
        self.data = {
            'intensity': [0.1, 0.5, 0.9],
            'height': [0.2, 0.6, 1.0]
        }
        self.dataset = pd.DataFrame(self.data)

        # Example min, max values and data types
        self.min_values = {'intensity': 0, 'height': 0}
        self.max_values = {'intensity': 100, 'height': 10}
        self.data_types = {'intensity': 'int', 'height': 'float32'}  # Note the use of a NumPy type

    def test_remap_normalized_values(self):
        # Remap normalized values
        remapped_dataset = remap_normalized_values(self.dataset, self.min_values, self.max_values, self.data_types)

        # Expected results
        expected_data = {
            'intensity': [10, 50, 90],
            'height': [2.0, 6.0, 10.0]
        }
        expected_dataset = pd.DataFrame(expected_data)

        # Check if the remapped dataset matches the expected dataset
        pd.testing.assert_frame_equal(remapped_dataset, expected_dataset)

    def test_dtype_conversion(self):
        # Test dtype conversion directly
        for key in self.min_values:
            dtype = getattr(np, self.data_types[key])
            self.assertTrue(callable(dtype), f"{self.data_types[key]} should be a callable type")

if __name__ == '__main__':
    unittest.main()
        
        
        
import pandas as pd
from typing import Dict, Union

def remap_normalized_values(dataset: pd.DataFrame, min_values: Dict[str, Union[int, float]],
                            max_values: Dict[str, Union[int, float]],
                            precision_mapping: Dict[str, Union[int, float]]) -> pd.DataFrame:
    """Reads the given metadata and remaps the normalized (0-1) values (e.g. intensity) to their original range using
    the min/max values saved in the metadata.

    :param dataset: point cloud
    :type dataset: pandas.DataFrame
    :param min_values: minimum values of columns before processing point cloud
    :type min_values: dict
    :param max_values: maximum values of columns before processing point cloud
    :type max_values: dict
    :param precision_mapping: mapping of attributes to their desired precision (int or float with decimal places)
    :type precision_mapping: dict
    :return: point cloud without any normalized values
    :rtype: pandas.DataFrame
    """
    assert min_values.keys() == max_values.keys(), "This needs to be the case or the values are broken."
    for key in min_values:
        if key in dataset:
            min_value = min_values[key]
            max_value = max_values[key]
            precision = precision_mapping.get(key, 9)  # Default to 9 decimal places if not specified

            if isinstance(precision, int) and precision == 0:
                # Round to integer
                dataset[key] = dataset[key].map(lambda val: round(val * (max_value - min_value) + min_value))
            else:
                # Round to specified number of decimal places
                dataset[key] = dataset[key].map(lambda val: round(val * (max_value - min_value) + min_value, precision))
    return dataset

# Example usage
dataset = pd.DataFrame({
    'intensity': [0.1, 0.5, 0.9],
    'distancetodtm': [0.1, 0.5, 0.9]
})
min_values = {'intensity': 0, 'distancetodtm': 0}
max_values = {'intensity': 100, 'distancetodtm': 1}
precision_mapping = {'intensity': 0, 'distancetodtm': 6}  # 'intensity' as int, 'distancetodtm' as float with 6 decimals

remapped_dataset = remap_normalized_values(dataset, min_values, max_values, precision_mapping)
print(remapped_dataset)
